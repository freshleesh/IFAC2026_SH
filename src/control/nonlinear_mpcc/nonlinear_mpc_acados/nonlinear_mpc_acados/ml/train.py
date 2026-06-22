#!/usr/bin/env python3
"""Train the WeightScaleMLP on logged MPC CSV data.

Usage:
    python3 -m nonlinear_mpc_acados.ml.train [--csv ~/mpc_logs/mpc_*.csv]
                                              [--epochs 200]
                                              [--out saved/weight_scaler.pt]

Pipeline:
    1. glob CSVs from ~/mpc_logs/
    2. extract (state, target_scale) pairs:
         state         = [kappa_abs, kappa_signed, v_actual, ref_v]
         target_scale  = [q_cte_scale, q_lag_scale, q_v_scale, q_drate_scale]
       (= B heuristic's output → MLP imitates B as a starting baseline)
    3. filter bad cycles: feasible==0, opti_value > 200 (cost spike),
       v_actual<0.1 (stuck) — these get DROPPED to avoid teaching bad behavior
    4. train MLP, MSE loss, Adam
    5. save as TorchScript (.pt) — mpc_node loads with torch.jit.load
"""
from __future__ import annotations
import argparse, csv, glob, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Allow running as both module (`python -m`) and script
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nonlinear_mpc_acados.ml.model import build_model, INPUT_DIM, OUTPUT_DIM
else:
    from .model import build_model, INPUT_DIM, OUTPUT_DIM


# Column names required from the CSV header. Indices are resolved per-file
# from the header row so the loader is robust against DBG_FIELDS reorders.
REQUIRED = [
    "v_actual", "ref_v", "opti_value", "feasible",
    "kappa_abs", "kappa_signed",
    "q_cte_scale", "q_lag_scale", "q_v_scale", "q_drate_scale",
    "v_max_cost",
]


def load_csv_data(csv_glob: str) -> tuple[np.ndarray, np.ndarray]:
    """Glob CSVs → (X, Y) numpy arrays. Drops bad cycles."""
    paths = sorted(glob.glob(os.path.expanduser(csv_glob)))
    if not paths:
        raise SystemExit(f"no CSVs found at: {csv_glob}")
    print(f"[train] loading {len(paths)} CSVs:")
    for p in paths:
        print(f"  - {p}")

    Xs, Ys = [], []
    dropped = {"infeasible": 0, "opti_high": 0, "stuck": 0, "parse": 0, "miss_col": 0}
    for p in paths:
        with open(p) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                print(f"  skip (empty): {p}")
                continue
            try:
                col = {name: header.index(name) for name in REQUIRED}
            except ValueError as e:
                missing = [n for n in REQUIRED if n not in header]
                print(f"  skip (missing cols {missing}): {p}")
                continue
            need = max(col.values()) + 1
            kept_here = 0
            for row in reader:
                if len(row) < need:
                    dropped["miss_col"] += 1
                    continue
                try:
                    feasible = int(float(row[col["feasible"]]))
                    opti = float(row[col["opti_value"]])
                    v_actual = float(row[col["v_actual"]])
                except (ValueError, IndexError):
                    dropped["parse"] += 1
                    continue
                if feasible == 0:
                    dropped["infeasible"] += 1; continue
                if opti > 200.0:
                    dropped["opti_high"] += 1; continue
                if v_actual < 0.1:
                    dropped["stuck"] += 1; continue
                try:
                    k_abs = float(row[col["kappa_abs"]])
                    x = [k_abs,
                         float(row[col["kappa_signed"]]),
                         v_actual,
                         float(row[col["ref_v"]]),
                         float(row[col["v_max_cost"]])]
                    # B target 를 인라인 재계산 (CSV 의 q_*_scale 열은 무시 — 그건 옛 공식이라).
                    # 강화된 B (코너 anti-shake): q_drate 계수 2.5 → 4.0.
                    k = min(max(k_abs, 0.0), 1.0)
                    y = [max(0.3, 1.0 - 2.0 * k),   # q_cte_scale: 코너 path 자유도 ↑
                         max(0.5, 1.0 - 1.5 * k),   # q_lag_scale: 코너 progress 압박 ↓
                         1.5 + 1.5 * k,             # q_v_scale: ref_v 추종 강화
                         1.5 + 4.0 * k]             # q_drate_scale: 강화. κ=0.82 → 4.78 (SCALE_MAX=5.0 cap)
                except (ValueError, IndexError):
                    dropped["parse"] += 1; continue
                Xs.append(x); Ys.append(y); kept_here += 1
            print(f"    kept {kept_here:6d} from {os.path.basename(p)}")
    X = np.array(Xs, dtype=np.float32)
    Y = np.array(Ys, dtype=np.float32)
    print(f"[train] loaded {len(X)} valid samples; dropped {dropped}")
    return X, Y


def train(X: np.ndarray, Y: np.ndarray, epochs: int = 200,
          batch_size: int = 256, lr: float = 1e-3,
          hidden: int = 32, val_frac: float = 0.15) -> torch.nn.Module:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    n_val = max(1, int(len(ds) * val_frac))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)

    model = build_model(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None
    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            pred = model(xb)
            l = loss_fn(pred, yb)
            l.backward(); opt.step()
            tr_loss += l.item() * xb.size(0)
        tr_loss /= n_tr

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                va_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        va_loss /= n_val

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == 1 or ep == epochs:
            print(f"  ep {ep:4d}: train MSE={tr_loss:.5f}  val MSE={va_loss:.5f}  "
                  f"(best val={best_val:.5f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="~/mpc_logs/mpc_*.csv",
                   help="glob pattern for CSVs")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--out", default=None,
                   help="output .pt path (default: ml/saved/weight_scaler.pt)")
    args = p.parse_args()

    X, Y = load_csv_data(args.csv)
    if len(X) < 100:
        raise SystemExit(f"too few samples ({len(X)}) — drive more first")

    print(f"[train] starting ({args.epochs} epochs, hidden={args.hidden})")
    t0 = time.time()
    model = train(X, Y, epochs=args.epochs, batch_size=args.batch,
                  lr=args.lr, hidden=args.hidden)
    print(f"[train] done in {time.time()-t0:.1f}s")

    out_path = args.out
    if out_path is None:
        here = Path(__file__).resolve().parent
        out_path = here / "saved" / "weight_scaler.pt"
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as TorchScript so mpc_node can `torch.jit.load` without our model.py
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(out_path))
    print(f"[train] saved TorchScript → {out_path}")


if __name__ == "__main__":
    main()
