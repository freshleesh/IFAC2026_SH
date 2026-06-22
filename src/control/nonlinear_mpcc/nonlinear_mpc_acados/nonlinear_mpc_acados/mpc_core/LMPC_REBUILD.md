# LMPC apex rebuild — joint α-in-solver (acados-native)

Branch `lmpc-joint-alpha`. main = verified N=25 deploy (dc2fe84), untouched.

## Why a rebuild (what the decoupled attempt proved, by instrumentation)
- LMPC target (= SS·α from an external QP) repeatedly sat **outside the corridor**
  (IQP raceline ±0.97 vs reachable ±0.60; 48% of the raceline is past the ±0.75
  corridor). Hard-feeding that target pulled x_N past the wall → crash on respawn.
- Clamping the target to ±0.60 fixed *that* layer but revealed the next: the target
  jumps ±0.6 side-to-side as the apex alternates (discrete SS) → car overshoots →
  stalls → **SQP_RTI prediction diverges** (ec_xN → −3 m). Layered instability.
- Root verdict: **decoupled target + SQP_RTI is fragile.** The references
  (TC-LMPC WEVJ-2023, Berkeley Racing-LMPC-ROS2) BOTH optimize α JOINTLY with x/u
  inside one solve, with a **soft** terminal (never hard). [[lmpc-mpcc-references]]

## Architecture — α as a constant-dynamics STATE (parameter-as-state trick)
acados has no control at the terminal node, so α can't be a control. Instead make
α part of the **state**, optimized jointly with x/u in the single RTI solve:

- **State**: `x_aug = [x(8), α(K)]`, nx 8 → 8+K (K=lmpc_K_points, 10). `nx_phys=8`.
- **Dynamics**: `f_aug = [f_dyn(x,u); 0_K]`  (α constant over the horizon).
- **Initial constraint (idxbx_0)**: fix only x_aug[0:8] = current physical state;
  **α[8:8+K] FREE** → the solver chooses α (joint optimization). acados supports
  partial initial-state bounds via `constraints.idxbx_0` / lbx_0 / ubx_0.
- **α simplex**: `Σα = 1` (linear constraint, con_h or a dedicated eq), `0 ≤ α ≤ 1`
  (state bounds idxbx on the α components, all stages — cheap).
- **Terminal cost (cost_e, NONLINEAR_LS, SOFT)**:
  `r_costtogo = sqrt(w_Q) · sqrt(Σ αᵢ·Qᵢ + ε)`   (pull toward low cost-to-go)
  `r_slack    = sqrt(w_s) · W^½ · (x_N[pos] − Σ αᵢ·SSᵢ)`   (anchor to SS hull, SOFT)
  SS (K×4: px,py,ψ,vx) and Q (K) enter as **parameters** (already wired: p[18:68]).
- **Stage cost**: KEEP the contouring/lag/progress (MPCC) — but REDUCE terminal
  contour/yaw (reference-free terminal) so the α-cost-to-go sets the line.
- **Corridor**: KEEP as a stage hard(-ish, slacked) constraint. Because the
  terminal SS-anchor is SOFT, when the SS lies outside the corridor the **corridor
  wins** and the car cuts apex only up to the corridor edge — no wall crash. This is
  the key robustness the decoupled hard-target lacked.

## Why this is stable where decoupled wasn't
- α optimized WITH x/u in one QP → no alternating-loop whipsaw, no target jumps.
- SS-anchor SOFT + corridor hard → conflicts resolve toward the drivable corridor,
  never past the wall. Recursive feasibility preserved via the slack.

## Companion fix (needed for real apex gain, multi-map)
The fixed ±0.75 corridor is NARROWER than the track (±~1 m) and the raceline. For
apex speed the corridor must follow the **actual per-point track width** (d_left/
d_right − R_car), not a constant. Without it the apex is capped at ±0.60 regardless.
→ widen/var-corridor is a parallel workstream (track_loader corridor from d_l/d_r).

## Implementation order (each step builds + runs)
1. **State augmentation**: _build_dynamic_model returns x_aug/f_aug (8+K); setup_MPC
   nx, idxbx_0 (x[0:8] fixed, α free), α bounds [0,1], Σα=1 constraint.
2. **Terminal cost** rewrite to α form (cost_e = costtogo + soft slack).
3. **Solve loop**: set lbx_0/ubx_0 for x[0:8]=state, leave α; SS/Q params (have it);
   extract traj[:, 0:8] for control/viz (ignore α rows).
4. **Seed + filters**: keep grip-clamped IQP seed (in lmpc_wip patch); SS query forward.
5. **Verify** with lmpc_probe2 (robust, fixed-duration+CSV): lap-by-lap + ec, must
   hold 16 laps STUCK=0 and ideally < 17.72 (apex gain) without drift.
6. **Corridor var-width** (companion) → unlock full apex.

## Exact edit points (acados_kinematic.py) — Step 1 state augmentation
- `_build_dynamic_model` (~390): after `x=vertcat(...8...)`, add `alpha=SX.sym('alpha',K)`,
  `x_aug=vertcat(x,alpha)`, `f_aug=vertcat(f_expl, SX.zeros(K))`, `xdot_aug=SX.sym('xdot',8+K)`.
  Return x_aug/f_aug (and keep phys symbols for cost).
- setup_MPC dynamic branch (~555-572): `x=dyn['x_aug']`, `f_expl=dyn['f_aug']`, `nx=8+K`.
  Cost residuals still index phys symbols (x_,y_,psi,vx...) — UNCHANGED (they're the
  same SX leaves inside x_aug).
- `ocp.constraints.x0 = np.zeros(nx)` (~1086): REPLACE with partial initial constraint —
  `idxbx_0 = arange(8)`, `lbx_0=ubx_0=zeros(8)` (set per-cycle). α NOT in idxbx_0 → free.
- α bounds + simplex: extend `idxbx` (~1110) to include α indices [8..8+K-1] with
  lbx=0,ubx=1; add `con_h` row `sum(alpha)-1` bounded [0,0] (eq) — or a linear constraint C.
- Solve loop (~1704): `self.solver.set(0,"lbx_0",state8); set(0,"ubx_0",state8)` for the 8
  phys indices (was `set(0,"lbx",initial_state)` with full x0). α left free.
- traj extraction unchanged: `traj[:,0:8]` is the physical state (viz/control).
- Terminal cost_e: α form (see Architecture). Needs `tgt=SS@alpha` in-graph (SS=p[18:58]
  reshaped 4×K, alpha=x_aug[8:8+K]). cost-to-go `Σ alpha*Q` (Q=p[58:68]).

## ★ Progress log (branch lmpc-joint-alpha)
- **Step 1 (committed 3faf664)**: state aug nx 8→18, α free at t=0, +1ms solve. ✓ verified LMPC-off.
- **Step 2 (25291cc)**: joint-α soft terminal cost. **Step 3 (8ffb7e1)**: IQP apex seed grip-clamped.
- **Steps 4-9 UNCOMMITTED** (reference-free terminal W_e, query fix, seed-penalty removal, αdbg
  instrumentation, seed lateral-clamp, dial-back params).
- ★★ **CRITICAL BUG found via data (Step 8b)**: `_lmpc_query_state = traj[-1]` was 18-wide
  (incl α) but SS stores 8-dim → **SS query failed EVERY cycle → LMPC never activated
  (lmpc_w_live=0) → pure-MPCC centerline through Steps 2-8.** That's why every knob (lmpc_w,
  q_cte, CTG_COEF, seed-penalty) showed NO change — LMPC was OFF the whole time. Fixed:
  `traj[-1, :8]`. → query 0 fails, αdbg fires, **tgt_lat=-0.95 (α DOES target the apex!)**.
- After the fix, with the OVER-CRANKED params (set blindly while LMPC was off: lmpc_w=3.0,
  CTG_COEF=0.5, q_cte=0.2) the now-active terminal OVERSHOT the corridor (ec=1.02 > 0.75 →
  wall, lap 49s). → dial-back to moderate (lmpc_w=0.5, CTG_COEF=0.1, q_cte=0.45, q_v=0.3) +
  **seed lateral-clamp to corridor** (raceline ±0.97 → ±0.6 so SS⊂corridor → no overshoot).
  This config is BUILT but UNTESTED (next run).
- **αdbg tool**: logs tgt_lat/car_lat/α-argmax/ssLat each 0.5s (mpc_node ~1499) — KEEP for tuning.
- Next: test the dial-back+clamp config (LMPC active) → expect apex cut to ±0.6, stable, ≤17.7.
  Then tune lmpc_w/CTG/q_cte for actual speed gain. Then max_speed↑ (5→6/7, user-requested).

## Gotchas (learned this session)
- `_poll_lap_count` stale-latch → use lmpc_probe2.py (fixed-duration + CSV).
- gym CSV appends across runs → archive old CSVs before analysis.
- pkill no-match exits 1 → `set -e` aborts; guard with `|| true`.
- W_e is baked at codegen; LMPC-mode terminal weights need the `_lmpc_codegen` flag
  set on the mpc object BEFORE setup_MPC.
- Experimental decoupled code preserved: `~/bo_results/lmpc_wip/*.patch`.
