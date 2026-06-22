#!/usr/bin/env python3
"""acados port of EVO-MPCC base — 4-state Cartesian kinematic, EXTERNAL cost.

Identical model to Nonlinear_MPC.py (IPOPT) so the ROS node can swap
backends without re-tuning. Difference is the solver: SQP_RTI + HPIPM
runs in 1-5 ms per cycle vs IPOPT's 18-70 ms.

State (4):  [x, y, psi, s]
Input (3):  [v, delta, p]   (direct controls, NOT rates)

Per-stage parameters (model.p), set each cycle from the ROS node:
  [obs_dmin, obs_x, obs_y, side_pref,
   q_cte, q_lag, q_d_delta, R_safe, M_slack,
   left_x, left_y, right_x, right_y]

Track / centerline / boundaries supplied as CasADi spline interpolants
(self.center_lut_x(s) etc.) — same interface the IPOPT version uses, set
via set_track_data() before setup_MPC().
"""
import os
import math
import numpy as np
import casadi as ca
import scipy.linalg
from ._ros_compat import NullLogger, monotonic_now, yaw_to_quat
from .model_policy import (A_MIN_DYN, clamp_a_lat_to_grip, decide_side_window,
                           grip_a_lat_limit)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


def codegen_paths(use_dynamic, lmpc_joint, nx_solver, dyn_mu=None):
    """Return (export_dir, json_path) keyed to the OCP structure.

    Switching kinematic<->dynamic (2026-06-10 unified layout: BOTH nx=8 — the
    dyn/kin tag alone separates f_kin vs blended-Pacejka codegens) or toggling
    LMPC joint-α (nx 8<->18) changes the generated solver. A FIXED export dir
    made acados re-use the previous codegen and silently run the WRONG model —
    the recurring "stale codegen" trap. Keying the dir/json to (model, nx, lmpc)
    makes that axis safe (and same config warm-reuses its build instead of
    regenerating). Tags: dyn8 / kin8 / dyn18_lmpc / kin18_lmpc.

    NOTE: only the model/nx/lmpc axis is keyed. Other codegen-time knobs —
    N_horizon, dT, speed_target, lookahead_m, baked cost weights — are NOT in
    the tag, so changing only those reuses the same dir. Callers that vary them
    (BO/sweep harnesses) must still `rm -rf /tmp/acados_codegen_evompcc*`
    between runs to force regeneration.
    dyn_mu IS keyed (2026-06-10).
    """
    tag = f"{'dyn' if use_dynamic else 'kin'}{int(nx_solver)}{'_lmpc' if lmpc_joint else ''}"
    if dyn_mu is not None:
        # μ is baked into the tanh tire + friction-ellipse codegen — key it
        # so switching dyn_mu can never reuse a stale build (mu0p600 etc.).
        tag += f"_mu{float(dyn_mu):.3f}".replace('.', 'p')
    return (f"/tmp/acados_codegen_evompcc_{tag}",
            f"/tmp/acados_ocp_evompcc_{tag}.json")


class MPC:
    """acados backend mirroring the IPOPT EVO-MPCC interface."""

    # Forced-reverse (STUCK release) is terminated as soon as the car has
    # backed away this far from the wedge point, instead of always running the
    # full ~0.5 s countdown. Bounds the backward travel so the car does not
    # keep creeping rearward after it has already separated from the wall.
    STUCK_REVERSE_DIST = 0.20   # [m]

    def __init__(self, cost_type, system_model, logger=None):
        self._log = logger if logger is not None else NullLogger("[MPC-acados]")
        self.arch = "acados_evompcc"

        # Sizing — set in set_initial_params()
        self.N = None
        self.dT = None
        self.n_states = 4
        self.n_controls = 3
        # Wheelbase from yaml
        self.L = 0.307

        # ──────────────── Dynamic model (Phase 1: toggle + params) ────────
        # Default OFF: kinematic bicycle (current working configuration).
        # Switch via env var or class override; full dynamic adapter
        # comes online in Phase 2 (cost/output rewiring).
        self.use_dynamic = False
        # nlp_solver_iters (2026-06-19): 1 → SQP_RTI (default, real-time stable).
        # >1 → SQP with CAPPED max_iter + MERIT_BACKTRACKING globalization, to
        # escape the documented "1-iter RTI can't rebuild the dual active set"
        # low-speed trap WITHOUT the unbounded-step startup divergence that
        # uncapped multi-iter SQP hit. Set onto self before setup_MPC (ROS param).
        self.nlp_solver_iters = 1
        # Phase B3: joint-α LMPC — α as constant-dynamics STATE (nx 8→8+K),
        # optimized jointly with x/u in the RTI solve. OFF → nx=8 baseline.
        self._lmpc_joint = False
        self._lmpc_cog_w = 0.02   # B3 Step2: linear cost-to-go weight (Qᵀα) in ψ_e
        # B4' error-dynamics regression: affine velocity correction added to
        # f_expl rows [vx,vy,r]. Default OFF → p[76:79]=0 → baseline f_expl.
        self._err_regr = False
        self._e_corr = np.zeros(3)   # filled per-cycle by mpc_node (B4'.3)
        # ── Phase D closed-form CasADi GP residual (dynamics-only) ─────────
        # Independent of the l4acados `use_gp_residual` path. Default OFF.
        # Set by mpc_node from the `use_gp_casadi` ROS param BEFORE setup_MPC.
        self.use_gp_casadi = False
        self.gp_casadi_ckpt = '~/bo_results/gp_residual_realvy.pt'
        self.gp_casadi_train_data = '~/bo_results/gp_train_data_realvy.pt'
        # Vehicle dynamics — values match `stack_master/config/SIM/
        # SIM_pacejka.yaml` so the MPC's internal model is identical to
        # what the f110-simulator's std_kinematics::update_pacejka()
        # actually integrates. Identical params → MPC predictions match
        # simulator behaviour to numerical precision.
        # 2026-05-28 #17: align EXACTLY with gym sim_params.yaml (was 6 params
        # off — m+2%, Iz+23%★, lf+2%, lr-15%★, h_cg-81%★, mu-5%). This
        # mismatch caused MPCC to predict slower yaw than gym actually
        # delivered → corner over-rotate → outside slip into wall.
        self.dyn_m    = 3.47      # mass [kg]                  (sim: m)
        self.dyn_Iz   = 0.04712   # yaw inertia [kg·m²]        (sim: I)
        self.dyn_lf   = 0.15875   # CG to front axle [m]        (sim: lf)
        self.dyn_lr   = 0.17145   # CG to rear axle [m]         (sim: lr)
        self.dyn_h_cg = 0.074     # CG height [m] (load transfer) (sim: h)
        self.dyn_mu   = 1.0489    # friction coefficient        (sim: mu)
        self.ellipse_frac = 0.95  # friction ellipse headroom η (a_lim = μ·g·η)
        # Linear tire stiffness — sim-matched (f110-simulator params.yaml).
        # F_y = μ · C_S · F_z · α (linear in slip angle).
        # F1Tenth realistic; dynamic effects are inherently small at
        # this scale + speed → looks kinematic-like. Pure visual
        # similarity to kinematic is EXPECTED for low-speed F1Tenth.
        self.dyn_Csf = 4.718
        self.dyn_Csr = 5.4562
        # Tire model selector. Three options:
        #   'linear'  : F_y = μ·C_S·F_z·α                — fastest, no saturation
        #   'tanh'    : F_y = μ·D·F_z·tanh(B·α)          — smooth saturation
        #   'pacejka' : F_y = μ·D·F_z·sin(C·atan(B·α−…)) — most accurate, unstable in RTI
        self.dyn_tire_model = 'linear'
        # Legacy flag kept for backward compat (read-only proxy).
        self.dyn_use_pacejka = (self.dyn_tire_model == 'pacejka')
        # STANDARD F1Tenth Pacejka (literature: TUM/AMZ/Liniger):
        #   B (stiffness): 10  — typical
        #   C (shape):     1.4 — magic number for tire curves
        #   D (peak):      1.0 — peak friction coefficient
        #   E (curvature): 0   — symmetric
        # Numerically stable for SQP_RTI single-iter (C ≈ 1.4 keeps
        # sin·atan derivative well-conditioned across operating range).
        self.dyn_Bf, self.dyn_Cf, self.dyn_Df, self.dyn_Ef = (
            10.0, 1.4, 1.0, 0.0)
        self.dyn_Br, self.dyn_Cr, self.dyn_Dr, self.dyn_Er = (
            10.0, 1.4, 1.0, 0.0)
        # Hybrid blend tuned for HIGH-SPEED DYNAMIC mode:
        # vx ≥ 0.5 m/s → essentially 100% Pacejka dynamic.
        # Below 0.5 (startup / stop only) blend toward kinematic so the
        # atan2-slip singularity is well-conditioned. ICRA / racing usage:
        # car spends ≪1% of time below 0.5 m/s, so this is effectively
        # always dynamic.
        #   vx=0.0: w_std = 0.07  (93% kinematic — startup only)
        #   vx=0.5: w_std = 0.50  (50/50 — passing through this is brief)
        #   vx=1.0: w_std = 0.95  (95% dynamic)
        #   vx=2.0: w_std = 1.00  (~100% dynamic)
        self.dyn_v_b   = 0.5       # 2026-06-03 0.3→0.5 (ifac_mpcc 검증값, ICRA·타이트)
        self.dyn_v_s   = 0.3       # 2026-06-03 0.1→0.3 (블렌드 폭↑=tanh 부드럽게, stiff Hessian 방지)
        self.dyn_v_min = 0.2       # below: kinematic-dominated [m/s]
        self.dyn_a_max = 7.5       # max accel (matches sim max_accel)
        # Singularity epsilon for atan2 denominator. 1.0 m/s = robust for
        # SQP_RTI (proven stable on ICRA + F maps). Smaller (0.5) gives
        # better low-speed accuracy but causes IPM step collapse on
        # certain map start states. Stability over fidelity.
        # 2026-05-29: lowered 1.0 -> 0.5 for better low-speed yaw-rate
        # fidelity. cold_start_vx_floor=2.0 feeds solver vx>=2.0 at startup,
        # which should avoid the singular start regime that previously
        # caused IPM collapse. Verified empirically.
        self.dyn_v_eps = 1.0       # 2026-06-03 0.5→1.0 (ifac: 0.5는 IPM step collapse, 1.0 안정)

        # Bounds
        self.v_max = 6.0
        # R3 — decouple max_speed: v_max stays the HARD CAP (vx ubx, ubu, per-stage
        # caps). These two are set independently (None → derive from v_max so
        # behaviour is unchanged when unset) so raising the cap does NOT make the
        # progress target more aggressive or lengthen the κ-lookahead (the things
        # that caused the high-speed trajectory shake).
        self.speed_target = None   # q_p progress reward target (None → v_max)
        self.lookahead_m  = None   # ref_v κ-lookahead window [m] (None → max(6, v_max²/6))
        self.v_min = 0.5
        self.theta_max = 0.4
        self.theta_min = -0.4
        # C-STEER: hard steering-RATE limit [rad/s]. Gym servo slew is 3.2, but
        # the MPC is constrained tighter (≈paper Kim&Park 2025 uses 1.0) so it
        # plans SMOOTH, servo-feasible steering → kills the documented sign-flip
        # hunting at root. Enforced as an OCP h-constraint on BOTH stage 0
        # (published δ) and stages 1..N-1.
        self.sv_max = 2.0
        self.s_min, self.s_max = 0.0, 1e3
        self.p_min, self.p_max = 0.0, 6.0

        self.is_ot = False
        self.vheid = {}
        self.param = {}

        # Track splines (set via set_track_data)
        self.center_lut_x = None
        self.center_lut_y = None
        self.center_lut_dx = None
        self.center_lut_dy = None
        self.right_lut_x = None
        self.right_lut_y = None
        self.left_lut_x = None
        self.left_lut_y = None
        self.element_arc_lengths = None
        self.arc_lengths_orig_l = None
        self.path_length = None
        self.ref_v = None    # CasADi interpolant ref_v(s)

        # Live (rqt-tunable) weights — same names the IPOPT MPC uses.
        self.q_cte_live     = 8.0
        self.q_lag_live     = 200.0
        self.q_d_delta_live = 25.0
        # R_safe=0.3 (was 0.8 → 0.4 → 0.3): with corridor h-constraint now
        # also active (R_CAR=0.10 from each wall, ≈0.35 m usable each side
        # of centerline), the 0.4 m keepout combined with a 0.4 m D_DETOUR
        # forced the car to a position 0.535 m off-center to clear an
        # on-centerline obstacle — outside the corridor. Solver couldn't
        # satisfy both → braked to 2.9 m/s every lap to soften slack.
        # 0.3 m matches car_R 0.135 + obs_marker 0.135 + 0.03 m margin and
        # keeps the required offset (0.435 m) inside the corridor.
        self.R_safe_live    = 0.3
        self.a_lat_safe_live   = 6.0    # rqt: curvature-speed cap headroom
        self.D_detour_live     = 0.15   # rqt: side-cost detour offset
        self.D_apex_live       = 0.22   # rqt: apex-bias lateral target offset.
                                        # Pulls e_c toward inside of corner.
                                        # 0 = no apex bias (pure centerline).
                                        # 0.4 = aggressive racing line.
        self.R_car_live        = 0.0    # rqt: corridor + obs h-constraint margin
                                        # (0.0 = wall touchable for racing line)
        self.commit_dist_live  = 10.0   # rqt: obstacle commit trigger distance
        self.cost_spike_thr_live = 500.0  # rqt: fallback threshold
        self.alpha_steer_live  = 0.6    # rqt: steer EMA blend (Python only)
        # Baked codegen-time cost weights (must match q_*_def in
        # setup_MPC). Used to convert rqt-set absolute weights into the
        # scale multiplier pushed via p_sym slots. ALL major cost weights
        # are scale-wired so BO / rqt can sweep without acados rebuild.
        self.Q_CTE_BAKED    = 9.0
        self.Q_LAG_BAKED    = 45.0
        self.Q_PSI_BAKED    = 10.0
        self.Q_V_BAKED      = 8.0
        self.Q_DD_BAKED     = 5.0    # steer regularisation |δ|²
        self.Q_P_BAKED      = 4.0    # progress (p_v - v_max)²
        self.Q_DRATE_BAKED  = 80.0   # steer rate (Δδ)²
        self.q_cte_scale_live   = 1.0
        self.q_lag_scale_live   = 1.0
        self.q_psi_scale_live   = 1.0
        self.q_v_scale_live     = 1.0
        self.q_dd_scale_live    = 1.0
        self.q_p_scale_live     = 1.0
        self.q_drate_scale_live = 1.0
        self.q_dv_scale_live    = 1.0   # 2026-05-27 #8 — a_x penalty
        self.M_slack_live   = 2.0e4
        # Static cost weights (set via param dict)
        self.q_v       = 15.0
        self.q_dv      = 15.0
        self.q_mu      = 0.05
        self.q_vp_proj = 60.0   # progress reward gamma

        # Solver storage
        self.solver = None
        self.X0 = None
        self.u0 = None
        self.WARM_START = False

        # Lap tracking — keep solver-internal s unbounded so the dynamics
        # constraint ṡ = p_v never sees a wrap discontinuity. Spline lookup
        # in cost uses s_periodic = s − L·floor(s/L) for [0, L) wrap.
        self.lap_count = 0
        self._last_sensor_s = None
        # Same persistent-offset trick for yaw (mirrors lap_count for s).
        # Without this, sensor yaw ∈ [−π, π] wraps every full revolution
        # and the cycle-by-cycle "shift initial_state[2] ± 2π" detection
        # fires repeatedly in the wrap zone, each time triggering multi-
        # iter SQP and disrupting the solve → cost spike (observed: 974
        # at the exact wrap moment + ~5 cycles of elevated cost).
        # Tracking yaw_offset as multiples of 2π keeps the solver-internal
        # ψ monotonically continuous, so no wrap event ever reaches the
        # solver.
        self._yaw_offset = 0.0
        self._last_sensor_yaw = None

        # Debug — same names IPOPT version uses, ROS node reads these
        self.dbg_n_obs_input = 0
        self.dbg_sel_dmin = float('inf')
        self.dbg_sel_x = float('inf')
        self.dbg_sel_y = float('inf')
        self.dbg_side_pref = 0.0
        self.dbg_solver_status = ""
        self.boundary_hook = None  # callable(shifted_points: list[(x,y)]) or None

    # ------------------------------------------------------------------
    # Setters (mirror IPOPT MPC interface so node code reuses)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_time_steps(mode: str, dT: float, N: int) -> list[float]:
        """multi-dt time_steps grid 생성.
        - uniform   : [dT] * N
        - pyramidal : 가까이 sharp control + 중간 medium + 멀리 long planning.
                      N=80, dT=0.04 가정 시 [0.04]*20 + [0.06]*30 + [0.10]*30 = 5.6s.
                      N 다른 경우 비율 조정.
        """
        mode = str(mode).lower()
        if mode == 'pyramidal':
            n1 = max(1, N // 4)         # 가까이 (25%) dt = dT
            n3 = max(1, (N * 3) // 8)   # 멀리   (37.5%) dt = dT*2.5
            n2 = N - n1 - n3            # 중간 (37.5%) dt = dT*1.5
            return [dT] * n1 + [dT * 1.5] * n2 + [dT * 2.5] * n3
        # default uniform
        return [dT] * N

    def set_initial_params(self, param, vheid, is_ot):
        self.vheid = vheid
        self.param = param
        self.dT = param['dT']
        self.N = param['N']
        # multi-dt time_steps grid (pyramidal: 가까이 sharp + 멀리 long planning).
        # uniform 이면 [dT]*N. pyramidal 이면 [dT]*20 + [dT*1.5]*30 + [dT*2.5]*30 (N=80 가정).
        self.time_steps_mode = param.get('time_steps_mode', 'uniform')
        self.time_steps = self._build_time_steps(self.time_steps_mode, self.dT, self.N)
        self.L = vheid.get('l_wb', param.get('L', 0.307))
        self.theta_max = param['theta_max']
        self.theta_min = -self.theta_max
        self.v_max = param['v_max']
        self.v_min = 0.5
        self.x_min, self.x_max = param['x_min'], param['x_max']
        self.y_min, self.y_max = param['y_min'], param['y_max']
        self.psi_min, self.psi_max = param['psi_min'], param['psi_max']
        self.s_min, self.s_max = param['s_min'], param['s_max']
        self.p_min, self.p_max = param['p_min'], param['p_max']
        self.is_ot = is_ot
        # Param overrides for static weights (LIVE ones come from defaults below)
        self.q_v       = param.get('mpc_v_track',     self.q_v)
        self.q_dv      = param.get('mpc_w_accel',     self.q_dv)
        self.q_mu      = param.get('q_mu',            self.q_mu)
        self.q_vp_proj = param.get('mpc_vp_project',  self.q_vp_proj)

    def a_lat_safe_eff(self):
        """a_lat_safe_live clamped to physical grip μ·g·η.

        BO/yaml/rqt 가 어떤 값을 요청하든 속도 프로필·vcap·a_lat 제약이
        타이어가 못 내는 그립 위에 세워지지 않게 하는 단일 게이트.
        Live-safe: 호출 시점의 a_lat_safe_live/dyn_mu 로 매번 계산.
        """
        eff, clamped = clamp_a_lat_to_grip(
            self.a_lat_safe_live, self.dyn_mu, self.ellipse_frac)
        if clamped and not getattr(self, '_alat_clamp_warned', False):
            self._alat_clamp_warned = True
            self._log.warning(
                f"[MPC-acados] a_lat_safe_live {float(self.a_lat_safe_live):.2f} > "
                f"μgη {eff:.2f} (mu={float(self.dyn_mu):.3f}) → clamped. "
                "BO 탐색 상한이 물리 그립에 묶임 (의도된 동작).")
        return eff

    def set_track_data(self, c_x, c_y, c_dx, c_dy, r_x, r_y, l_x, l_y,
                       element_arc_lengths, original_arc_length_total, ref_v):
        self.center_lut_x, self.center_lut_y = c_x, c_y
        self.center_lut_dx, self.center_lut_dy = c_dx, c_dy
        self.right_lut_x, self.right_lut_y = r_x, r_y
        self.left_lut_x, self.left_lut_y = l_x, l_y
        self.element_arc_lengths = element_arc_lengths
        self.arc_lengths_orig_l = original_arc_length_total
        self.path_length = original_arc_length_total
        self.ref_v = ref_v

        # Precompute curvature κ(s) on a fine grid for runtime v-cap lookup.
        # κ = dθ/ds where θ = atan2(dy/ds, dx/ds). Sharp corners (hairpin)
        # have |κ| > 2 rad/m on f1tenth tracks; pure-Cartesian MPCC's
        # contour/lag cost geometry breaks there, so we cap v per stage by
        # v ≤ sqrt(a_lat_safe / |κ|) to keep the QP well-posed.
        self.kappa_ds = 0.05
        L = float(original_arc_length_total)
        s_grid = np.arange(0.0, L, self.kappa_ds)
        dx_g = np.array([float(c_dx(float(s))) for s in s_grid])
        dy_g = np.array([float(c_dy(float(s))) for s in s_grid])
        theta_g = np.unwrap(np.arctan2(dy_g, dx_g))
        self.kappa_grid = np.gradient(theta_g, self.kappa_ds)
        self._log.info("[MPC-acados] kappa grid: |k|_max=%.2f rad/m, |k|_p95=%.2f",
                      float(np.max(np.abs(self.kappa_grid))),
                      float(np.percentile(np.abs(self.kappa_grid), 95)))

        # Make |κ| available to CasADi as a linear interpolant — enables
        # per-stage curvature-aware speed cap inside the cost. Track is
        # closed-loop so append the s=0 value at s=L for clean wrap-around.
        # FORWARD-LOOKING: instead of |κ(s)|, use max(|κ(s')| for s' ∈ [s,
        # s+LOOKAHEAD]). At s=60 (just before the upper-left U-turn) the
        # raw |κ| is only 0.04 (v_kin=11 m/s, cap inactive), but at s=63
        # the peak |κ|=0.71 (v_kin=2.9 m/s). MPC's horizon (~2.5 m at
        # v=5) barely reaches the peak so per-stage cap fires too late
        # to brake. Forward-max over LOOKAHEAD = 4 m makes the cap kick
        # in 4 m before any sharp corner, giving MPC enough time to slow.
        abs_k_arr = np.abs(self.kappa_grid)
        # Gaussian-ish smoothing of |κ| profile before forward-max. Without
        # it, |κ| has high-freq variation from spline derivative noise →
        # ref_v_cap jitters cycle-to-cycle as car advances, manifesting as
        # speed surge after corner exit. 11-tap rolling avg ≈ 0.5 m smooth.
        smooth_win = 11
        kernel = np.ones(smooth_win) / float(smooth_win)
        abs_k_arr = np.convolve(abs_k_arr, kernel, mode='same')
        # 2026-06-02 속도비례: 고속일수록 코너 전 제동거리↑ 필요. v=8 면
        # 8→2.3 감속에 ~10m. per-stage hard cap 이 u[0] 을 √(a_lat/κ_fwd) 로
        # 묶으므로, forward-max 가 코너를 이만큼 일찍 봐야 cap 이 제때 내려감.
        LOOKAHEAD_M = float(self.lookahead_m) if self.lookahead_m else max(6.0, float(self.v_max) ** 2 / 6.0)   # R3: fixed window decouples shake from cap
        n_look = int(LOOKAHEAD_M / self.kappa_ds)
        n_grid = len(abs_k_arr)
        # R4 (refv_smoothing.distance_tapered_forward_max) was tried here to remove
        # the ref_v step when a corner crosses the window edge. On this ultra-tight
        # map the forward window is SATURATED (a corner is always within ~6 m →
        # p95≈max), so there is no edge step to smooth and R4 is a no-op here; it
        # only helps open maps with sparse corners + long straights. Kept as a
        # tested module (mpc_core/refv_smoothing.py) for that case; baseline uses
        # the plain hard max.
        # 2026-06-08 (사용자 통찰): plain forward-MAX 가 완만(저-κ) 구간서도 멀리
        # 있는 tight 코너 κ로 ref_v 를 눌러 가속을 막음(twisty 맵서 p95≈max 포화).
        # → brake-distance-aware 등가 κ 로 교체: 거리 d 만큼 떨어진 코너는 제동거리
        # 가 있으니 지금은 빨라도 됨 → 등가-κ 를 거리로 할인.
        #   v_allow(s)² = v_corner(s')² + 2·a_brake·d  (a_brake=a_lat 가정)
        #   = a_lat/κ(s') + 2·a_lat·d = a_lat·(1/κ' + 2d)
        #   κ_eq = a_lat/v_allow² = 1/(1/κ' + 2d) = κ'/(1+2d·κ')   ← a_lat 소거
        # κ_eq[d=0]=κ'(local, 코너서 full), d↑ → κ_eq→0(제한 풀림). window 안
        # 모든 upcoming 코너 중 가장 binding(max κ_eq) 채택 → 완만 구간 가속 허용,
        # tight 코너는 제때 감속.
        _ds = float(self.kappa_ds)
        # bf = a_brake/a_lat (κ_eq 유도식 그대로). 고정 0.7은 a_lat_safe=5 에서
        # 제동 3.5 m/s² 가정 = 솔버 한계(3.0) 초과 낙관 — 솔버 값으로 일관화.
        _bf = min(1.0, abs(A_MIN_DYN) / max(1e-3, self.a_lat_safe_eff()))
        abs_k_fwd = np.empty(n_grid, dtype=float)
        for i in range(n_grid):
            j = min(i + n_look, n_grid)
            seg = abs_k_arr[i:j]
            dist = np.arange(len(seg)) * _ds
            k_eq = seg / (1.0 + 2.0 * _bf * dist * seg)   # brake-aware equivalent κ (margin)
            abs_k_fwd[i] = float(k_eq.max()) if len(seg) else float(abs_k_arr[i])
        self._log.info("[MPC-acados] brake-aware |κ_eq| (lookahead=%.1f m): max=%.3f p95=%.3f "
                      "(was plain forward-max — 사용자 통찰: 완만구간 가속)",
                      LOOKAHEAD_M,
                      float(abs_k_fwd.max()),
                      float(np.percentile(abs_k_fwd, 95)))
        if self.use_dynamic:
            self._log.info("[MPC-acados] MODEL = DYNAMIC (%s tire) (n_states=8, "
                          "n_controls=3, LM=1.0, RTI 1-iter)",
                          self.dyn_tire_model)
        else:
            self._log.info("[MPC-acados] MODEL = KINEMATIC-UNIFIED 8-state (f_kin, "
                          "n_states=8, n_controls=3 [a_x, δ, p_v], LM=0.2)")
        s_grid_ext = list(s_grid) + [L]
        abs_k_ext  = abs_k_fwd.tolist() + [abs_k_fwd[0]]
        self.abs_kappa_lut = ca.interpolant(
            'abs_kappa_lut', 'linear', [s_grid_ext], abs_k_ext
        )

        # Signed κ LUT for apex-biased lateral reference. Used by the cost
        # to shift the e_c target toward the INSIDE of the upcoming corner,
        # so centerline tracking produces racing-line behaviour. Sign:
        # κ_signed > 0 = left turn (CCW) → inside is LEFT.
        #
        # Profile design — emulate the global IQP raceline's behaviour
        # where straights aren't strictly tracked: the car drifts
        # diagonally toward the inside of the next corner during the
        # approach straight, hits apex, and decays back to centerline
        # gradually on the exit. Without this, centerline mode produces
        # straight-on-straight then sharp corner attack — slow.
        #
        # Construction:
        #  1. Gaussian kernel σ=0.8 m (3σ ≈ 2.4 m support) — covers a
        #     racing-line-like wide influence zone around each apex.
        #  2. Forward-shift output by 0.6 m (np.roll on closed loop).
        #     At s on the approach straight, the LUT returns the
        #     smoothed κ at s+0.6 m → bias kicks in BEFORE apex.
        # Net bias profile (peak-normalised) around an apex:
        #   apex-2.5 m: 0.10  (drift starts on approach straight)
        #   apex-1.5 m: 0.40  (clear diagonal on straight)
        #   apex-0.6 m: 1.00  (peak — slight before geometric apex)
        #   apex      : 0.75
        #   apex+0.5 m: 0.30  (sustain on exit)
        #   apex+1.5 m: 0.05  (almost back to centerline)
        sigma_m_apex   = 0.8
        forward_bias_m = 0.6
        half_idx = int(3 * sigma_m_apex / self.kappa_ds)
        kx = np.arange(-half_idx, half_idx + 1) * self.kappa_ds
        ker_apex = np.exp(-kx * kx / (2.0 * sigma_m_apex * sigma_m_apex))
        ker_apex /= ker_apex.sum()
        signed_k_smooth = np.convolve(self.kappa_grid, ker_apex, mode='same')
        forward_bias_idx = int(forward_bias_m / self.kappa_ds)
        signed_k_arr = np.roll(signed_k_smooth, -forward_bias_idx)
        signed_k_ext = signed_k_arr.tolist() + [signed_k_arr[0]]
        self.signed_kappa_lut = ca.interpolant(
            'signed_kappa_lut', 'linear', [s_grid_ext], signed_k_ext
        )

    # ------------------------------------------------------------------
    # Dynamic-bicycle model builder (Phase 1: helper, NOT YET WIRED)
    # ------------------------------------------------------------------
    def _build_dynamic_model(self):
        """Build the Pacejka single-track dynamic model expressions.

        Mirrors `f110-simulator/src/std_kinematics.cpp::update_pacejka`
        verbatim so MPC predictions match what the simulator integrates
        (assuming MPC and sim are configured with the same SIM_pacejka
        parameters — which `__init__` does by default).

        Returns a dict (intentionally NOT installed into `setup_MPC` yet).
        Phase 2 wiring will:
          - replace the kinematic `x` / `u` / `f_expl` with these
          - swap the cost residual `v − ref_v` → `vx − ref_v`
          - swap the h-constraint `a_lat = v² tan(δ)/L` → `a_lat = vx · r`
          - change `solve_step` initial state to include vx, vy, r
          - change `solve_step` output to use vx[1] as v_cmd
        """
        # ── States (8): x, y, ψ, vx, vy, r, s, δ_prev ─────────────────
        x_pos      = ca.SX.sym('x_pos')
        y_pos      = ca.SX.sym('y_pos')
        psi        = ca.SX.sym('psi')
        vx         = ca.SX.sym('vx')
        vy         = ca.SX.sym('vy')
        r_yaw      = ca.SX.sym('r_yaw')
        s          = ca.SX.sym('s')
        delta_prev = ca.SX.sym('delta_prev')
        x = ca.vertcat(x_pos, y_pos, psi, vx, vy, r_yaw, s, delta_prev)

        # ── Inputs (3): a_x, δ, p_v ───────────────────────────────────
        # Replaces kinematic's `v` with `a_x` (longitudinal accel).
        # F1Tenth's ackermann_drive accepts `drive.speed` (m/s); we'll
        # send `vx[1]` (next-stage predicted vx) as v_cmd in Phase 2.
        a_x   = ca.SX.sym('a_x')
        delta = ca.SX.sym('delta')
        p_v   = ca.SX.sym('p_v')
        u = ca.vertcat(a_x, delta, p_v)

        xdot = ca.SX.sym('xdot', 8)

        # ── Pacejka dynamic equations ─────────────────────────────────
        # Slip angles. Singularity at v_x → 0 in atan2 denominator: use
        # ca.fmax(vx, ε) so derivatives stay finite. Kinematic blend
        # below dyn_v_b further regularises (low-speed mode).
        vx_safe = ca.fmax(vx, self.dyn_v_eps)
        # sim convention: alpha_f = atan2(-vy - lf·r, vx) + delta
        alpha_f = ca.atan2(-vy - self.dyn_lf * r_yaw, vx_safe) + delta
        alpha_r = ca.atan2(-vy + self.dyn_lr * r_yaw, vx_safe)

        # Vertical loads with longitudinal load transfer
        # (sim: F_zf = m · (-a_x · h_cg + g · l_r) / L)
        g_const = 9.81
        L_wb = self.dyn_lf + self.dyn_lr
        F_zf = self.dyn_m * (-a_x * self.dyn_h_cg + g_const * self.dyn_lr) / L_wb
        F_zr = self.dyn_m * ( a_x * self.dyn_h_cg + g_const * self.dyn_lf) / L_wb

        # Tire force model selector — see __init__ self.dyn_tire_model.
        if self.dyn_tire_model == 'pacejka':
            # Full Pacejka Magic Formula. NOTE: SQP_RTI single iter
            # struggles with this — sin·atan derivative variation
            # between cycles → MINSTEP cascades. Use only with multi-
            # iter solve enabled (self.dyn_multi_iter).
            pacejka_arg_f = (self.dyn_Bf * alpha_f
                             - self.dyn_Ef * (self.dyn_Bf * alpha_f
                                              - ca.atan(self.dyn_Bf * alpha_f)))
            pacejka_arg_r = (self.dyn_Br * alpha_r
                             - self.dyn_Er * (self.dyn_Br * alpha_r
                                              - ca.atan(self.dyn_Br * alpha_r)))
            F_yf = (self.dyn_mu * self.dyn_Df * F_zf
                    * ca.sin(self.dyn_Cf * ca.atan(pacejka_arg_f)))
            F_yr = (self.dyn_mu * self.dyn_Dr * F_zr
                    * ca.sin(self.dyn_Cr * ca.atan(pacejka_arg_r)))
        elif self.dyn_tire_model == 'tanh':
            # tanh tire = SIMPLIFIED Pacejka. saturation built-in but
            # derivative is well-behaved everywhere (sech², bounded).
            # F_y = μ·D·F_z·tanh(B·α). Reaches saturation D·F_z at α≫0,
            # linear for small α with slope D·F_z·μ·B.
            # Used in: Hewing/Liniger ETH, Carrau et al, several
            # racing MPC papers. Sweet spot of accuracy + numerics.
            F_yf = self.dyn_mu * self.dyn_Df * F_zf * ca.tanh(self.dyn_Bf * alpha_f)
            F_yr = self.dyn_mu * self.dyn_Dr * F_zr * ca.tanh(self.dyn_Br * alpha_r)
        else:  # 'linear'
            # F_y = μ · C_S · F_z · α. No saturation.
            F_yf = self.dyn_mu * self.dyn_Csf * F_zf * alpha_f
            F_yr = self.dyn_mu * self.dyn_Csr * F_zr * alpha_r

        # Dynamic state derivatives (sim std_kinematics.cpp:82-88).
        # F_xf = 0 (RWD); F_xr is folded into a_x (sim's `accel` arg).
        f_dyn = ca.vertcat(
            vx * ca.cos(psi) - vy * ca.sin(psi),                         # ẋ
            vx * ca.sin(psi) + vy * ca.cos(psi),                         # ẏ
            r_yaw,                                                        # ψ̇
            a_x + (1.0 / self.dyn_m) * (-F_yf * ca.sin(delta)) + vy * r_yaw,
            (1.0 / self.dyn_m) * (F_yr + F_yf * ca.cos(delta)) - vx * r_yaw,
            (1.0 / self.dyn_Iz) * (F_yf * self.dyn_lf * ca.cos(delta)
                                    - F_yr * self.dyn_lr),
            p_v,
            (delta - delta_prev) / self.dT,
        )

        # ── Kinematic equations (for low-vx blend) ────────────────────
        # When vx ≪ v_b, the dynamic atan2 and small lateral forces
        # produce noisy derivatives. Blend toward kinematic single-track
        # so prediction stays well-conditioned at start-up / near-stop.
        # Kinematic regime forces vy and r toward their kinematic-
        # consistent values via fast first-order decay (τ_kin small).
        beta_kin   = ca.atan(self.dyn_lr * ca.tan(delta) / L_wb)
        vy_target  = vx * ca.tan(beta_kin)
        r_target   = (vx / L_wb) * ca.tan(delta) * ca.cos(beta_kin)
        tau_kin    = 0.05  # rapid relaxation toward kinematic (50 ms)
        f_kin = ca.vertcat(
            vx * ca.cos(psi + beta_kin),                                  # ẋ
            vx * ca.sin(psi + beta_kin),                                  # ẏ
            r_target,                                                      # ψ̇ (set by kin geom)
            a_x,                                                           # v̇x
            (vy_target - vy) / tau_kin,                                    # v̇y → kin
            (r_target  - r_yaw) / tau_kin,                                 # ṙ   → kin
            p_v,
            (delta - delta_prev) / self.dT,
        )

        # Smooth blend: w_std rises from 0 to 1 across vx ∈ [v_b−v_s, v_b+v_s]
        # (sim: w_std = 0.5·(1 + tanh((vx − v_b)/v_s)))
        # 2026-06-10 unified layout: KINEMATIC mode = this same 8-state model
        # with f_expl = f_kin at ALL speeds (w_std ≡ 0). Only the vehicle
        # model changes between modes — state/input layout, costs, bounds,
        # LMPC, GP/B4' residual hooks and BO search space stay IDENTICAL.
        if self.use_dynamic:
            w_std  = 0.5 * (1.0 + ca.tanh((vx - self.dyn_v_b) / self.dyn_v_s))
            f_expl = w_std * f_dyn + (1.0 - w_std) * f_kin
        else:
            w_std  = ca.SX(0.0)
            f_expl = f_kin   # pure kinematic single-track, 8-state

        # Real lateral acceleration (replaces v² tan(δ)/L kinematic form).
        a_lat_expr = vx * r_yaw

        # ── Phase B3 Step 1: α as constant-dynamics STATE (joint-α LMPC) ──
        # x_aug = [x(8), α(K)],  f_aug = [f_expl; 0_K]  (α constant over the
        # horizon → solver optimizes α jointly with x/u in one RTI solve).
        # Built here as AVAILABLE-BUT-UNUSED; setup_MPC wires x_aug/f_aug/
        # xdot_aug in only when joint-α LMPC is enabled (use_lmpc + lmpc_joint).
        # Convex-α terminal (Σαᵢ·Qᵢ + soft SS-anchor) replaces the softmin.
        # Refs: Racing-LMPC-ROS2 racing_mpc.cpp:479-504; LMPC_REBUILD.md.
        K_aug    = 10  # = K_LMPC (lmpc_ss slots p[18:68])
        alpha_aug = ca.SX.sym('alpha_lmpc', K_aug)
        x_aug    = ca.vertcat(x, alpha_aug)
        f_aug    = ca.vertcat(f_expl, ca.SX.zeros(K_aug))
        xdot_aug = ca.SX.sym('xdot_aug', 8 + K_aug)

        return dict(
            x_aug=x_aug, f_aug=f_aug, xdot_aug=xdot_aug,
            alpha_aug=alpha_aug, K_aug=K_aug,
            x=x, u=u, xdot=xdot, f_expl=f_expl,
            x_pos=x_pos, y_pos=y_pos, psi=psi,
            vx=vx, vy=vy, r=r_yaw, s=s, delta_prev=delta_prev,
            a_x=a_x, delta=delta, p_v=p_v,
            F_yf=F_yf, F_yr=F_yr, F_zf=F_zf, F_zr=F_zr,
            alpha_f=alpha_f, alpha_r=alpha_r,
            f_dyn=f_dyn, f_kin=f_kin, w_std=w_std,
            a_lat_expr=a_lat_expr,
        )

    # ------------------------------------------------------------------
    # OCP setup
    # ------------------------------------------------------------------
    def setup_MPC(self):
        ocp = AcadosOcp()
        model_ac = AcadosModel()
        model_ac.name = 'mpcc_evompcc_acados'

        # ---- States: x, y, psi, s, delta_prev ----
        # delta_prev is augmented to enable a per-stage steer-rate cost
        # (residual = delta - delta_prev). NONLINEAR_LS y(x,u) can only
        # see one stage at a time, so we shadow the previous applied δ
        # as a state. Dynamics: δ_prev_dot = (δ − δ_prev)/dt → after one
        # Euler step, δ_prev_new = δ_old (the control just applied).
        # Without rate cost the solver picks bimodal δ patterns
        # ([+0.22, +0.08, +0.22, +0.08, ...] alternating) that satisfy
        # the same average heading change with similar |δ|² total but
        # produce a visibly wavy prediction line.
        # ── Unified 8-state layout (2026-06-10) ─────────────────────────
        # BOTH modes: 8 states [x, y, ψ, vx, vy, r, s, δ_prev], 3 inputs
        # [a_x, δ, p_v]. Only the vehicle model (f_expl) differs:
        #   dynamic   : Pacejka/tanh tire forces, kinematic blend at low vx
        #   kinematic : pure kinematic single-track (f_kin, no tire forces)
        # Everything downstream — costs (incl. vterm + q_dv), bounds, LMPC
        # safe-set (slot 3 = vx), joint-α, GP/B4' residual rows [3,4,5],
        # warm-start, BO 9D search space — is shared with NO model branch.
        dyn = self._build_dynamic_model()
        u          = dyn['u']
        # B3: joint-α swaps in the augmented state x_aug=[x(8),α(K)].
        if self._lmpc_joint:
            x        = dyn['x_aug']
            f_expl   = dyn['f_aug']
            xdot     = dyn['xdot_aug']
            alpha_sym = dyn['alpha_aug']
            K_aug    = dyn['K_aug']
            nx       = 8 + K_aug
        else:
            x        = dyn['x']
            f_expl   = dyn['f_expl']
            xdot     = dyn['xdot']
            alpha_sym = None
            nx       = 8
        x_         = dyn['x_pos']
        y_         = dyn['y_pos']
        psi        = dyn['psi']
        s          = dyn['s']
        delta_prev = dyn['delta_prev']
        delta      = dyn['delta']
        p_v        = dyn['p_v']
        vx_for_cost  = dyn['vx']
        a_lat_expr   = dyn['a_lat_expr']
        a_x_input    = dyn['a_x']
        nu = 3
        # ── Phase D (closed-form CasADi GP residual) ──────────────────
        # Adds the offline-trained sparse-GP posterior MEAN (as a pure
        # CasADi expression) to the dynamics ONLY. Independent of the
        # l4acados `use_gp_residual` path (which rebuilds the OCP). The
        # cost / W / p_sym are left UNTOUCHED — the GP is mapped through
        # B_d to the velocity-state rows [vx, vy, r] = indices 3,4,5 of
        # the 8-state vector [x, y, psi, vx, vy, r, s, delta_prev].
        # Works for BOTH models (unified layout) — train the GP per-model
        # (kinematic GP learns kinematic-model residuals).
        # Default OFF (self.use_gp_casadi); guarded so any failure falls
        # back to the plain f_expl.
        if getattr(self, 'use_gp_casadi', False) and getattr(self, '_err_regr', False):
            self._log.warn(
                "[MPC-acados] use_gp_casadi AND _err_regr both set — skipping "
                "GP residual (e_corr error-regression takes precedence; both add "
                "to f_expl velocity rows = double correction).")
        if getattr(self, 'use_gp_casadi', False) and not getattr(self, '_err_regr', False):
            try:
                from .gp_casadi_residual import (
                    load_gp_casadi_params, build_casadi_residual,
                )
                gp_ckpt = os.path.expanduser(
                    getattr(self, 'gp_casadi_ckpt',
                            '~/bo_results/gp_residual_realvy.pt'))
                gp_train = os.path.expanduser(
                    getattr(self, 'gp_casadi_train_data',
                            '~/bo_results/gp_train_data_realvy.pt'))
                gp_p = load_gp_casadi_params(gp_ckpt, gp_train)
                # SAME symbols already in scope (unified-model build).
                gp_z = ca.vertcat(dyn['vx'], dyn['vy'], dyn['r'],
                                  dyn['delta'], dyn['a_x'])
                mu = build_casadi_residual(gp_z, gp_p)
                # B_d: 3 GP outputs -> state rows vx(3), vy(4), r(5).
                # Pad residual to f_expl width: 8 (plain) or 18 (joint-α
                # LMPC appends 10 α-states AFTER the 8 physical, so rows
                # 3/4/5 = vx/vy/r in both; α rows get 0 residual).
                _nx_full = f_expl.shape[0]
                f_expl = f_expl + ca.vertcat(
                    0, 0, 0, mu[0], mu[1], mu[2], *([0] * (_nx_full - 6)))
                self._log.info(
                    "[MPC-acados] Phase D CasADi-GP residual ACTIVE "
                    "(M=%d inducing, ckpt=%s) — added to f_expl rows "
                    "[vx,vy,r]", gp_p.M, gp_ckpt)
            except Exception as _gp_e:
                self._log.warn(
                    "[MPC-acados] Phase D CasADi-GP residual FAILED "
                    "(%s) — falling back to plain f_expl", _gp_e)
        # B3: split physical state dim (ROS interface) from solver state dim.
        # joint-α: solver carries nx=8+K but the ROS-facing physical state is 8.
        self._nx_solver = nx
        self.n_states   = 8
        self.n_controls = nu

        # ---- Per-cycle parameters (constant across all stages) ----
        # idx: 0=obs_dmin, 1=obs_x, 2=obs_y, 3=side_pref,
        #      4=q_cte, 5=q_lag, 6=q_d_delta, 7=R_safe, 8=M_slack,
        #      9=e_c_obs (obstacle's lateral offset from centerline, Frenet)
        # 10 reserved (unused).
        # The half-plane uses e_c_obs (Frenet) instead of a tangent-anchored
        # Cartesian normal. The previous tangent-anchored form (obs_tan_sin/
        # cos) was stable per cycle but ignored track curvature — over a
        # curving section the prediction had to choose between following
        # the curve (e_c roughly tracks centerline) and respecting a
        # straight-line half-plane that rotated relative to the curve, so
        # prediction shot off-track. Frenet-frame e_c naturally follows
        # the curve, and a Cartesian-distance proximity gate disables the
        # constraint far from the obstacle.
        # Slot map (per-cycle constants):
        #   0  obs_dmin (debug)        9  e_c_obs (Frenet obstacle offset)
        #   1  obs_x                   10 a_lat_safe   (rqt)
        #   2  obs_y                   11 D_apex       (rqt)
        #   3  side_pref               12 q_psi_scale  (rqt)
        #   4  D_DETOUR    (rqt)       13 q_v_scale    (rqt)
        #   5  R_CAR       (rqt)       14 q_dd_scale   (rqt)
        #   6  q_cte_scale (rqt)       15 q_p_scale    (rqt)
        #   7  R_safe      (rqt)       16 q_drate_scale(rqt)
        #   8  q_lag_scale (rqt)       17 q_dv_scale   (rqt) — 2026-05-27 #8
        # 2026-05-28 #18 LMPC slots (per-cycle constants — terminal cost only):
        #   18..57  : K=10 × (x*, y*, ψ*, vx*) — SS nearest 점들 (4 floats each)
        #   58..67  : K=10 × Q* — cost-to-go (step-count to lap end)
        #   68      : lmpc_w     (default 0 → LMPC OFF; backward-compat 보존)
        #   69      : lmpc_alpha (distance penalty in softmin)
        #   70      : lmpc_beta  (softmin sharpness; reviewer 권장 0.05)
        #   71      : lmpc_reg_w (regularization 가중치 — best SS 점 attractor)
        # Per-stage (4):
        #   72 left_x  73 left_y  74 right_x  75 right_y
        # B4' error regression (3, const across horizon):
        #   76 e_corr_vx  77 e_corr_vy  78 e_corr_r
        n_p_const = 18 + 54 + 3   # 18 기존 + 50 SS + 4 LMPC scalars + 3 B4' e_corr (const across horizon)
        n_p_stage = 4             # 4 corridor (per-stage)
        n_p_total = n_p_const + n_p_stage   # 79
        K_LMPC = 10
        p_sym = ca.SX.sym('p_sym', n_p_total)
        e_corr_sym = p_sym[76:79]   # B4' velocity-row correction (vx,vy,r)
        obs_dmin = p_sym[0]; obs_x = p_sym[1]; obs_y = p_sym[2]
        side_pref = p_sym[3]
        D_detour_p     = p_sym[4]   # side-cost detour offset (rqt)
        R_car_p        = p_sym[5]   # corridor + obs h margin (rqt)
        q_cte_scale_p  = p_sym[6]   # × q_cte_def (rqt)
        R_safe_p       = p_sym[7]
        q_lag_scale_p  = p_sym[8]   # × q_lag_def (rqt)
        e_c_obs_p      = p_sym[9]
        a_lat_safe_p   = p_sym[10]  # curvature speed-cap headroom (rqt)
        D_apex_p       = p_sym[11]  # apex-bias offset (rqt) — ACTIVE: drives e_c_ref (~L888)
        q_psi_scale_p   = p_sym[12]  # × q_psi_def
        q_v_scale_p     = p_sym[13]  # × q_v_def
        q_dd_scale_p    = p_sym[14]  # × q_dd_def
        q_p_scale_p     = p_sym[15]  # × q_p_def
        q_drate_scale_p = p_sym[16]  # × q_d_rate_def
        q_dv_scale_p    = p_sym[17]  # × q_dv_def (longitudinal accel a_x penalty)
        # LMPC SS slots: 50 floats packed as (K, 4) state + (K,) cost-to-go.
        # ss_states[k] = (x*, y*, ψ*, vx*); padding 점들의 Q* = 1e6 → exp(-β·1e6) ≈ 0.
        lmpc_ss_states = ca.reshape(p_sym[18:18 + K_LMPC * 4], 4, K_LMPC)  # 4 x K
        lmpc_ss_Q      = p_sym[18 + K_LMPC * 4 : 18 + K_LMPC * 5]            # (K,)
        lmpc_w_p     = p_sym[68]
        lmpc_alpha_p = p_sym[69]
        lmpc_beta_p  = p_sym[70]
        lmpc_reg_w_p = p_sym[71]
        left_x  = p_sym[72]; left_y  = p_sym[73]
        right_x = p_sym[74]; right_y = p_sym[75]

        # ---- Reference geometry ----
        # s_periodic = s − L·floor(s/L) ∈ [0, L). The solver-internal s
        # state grows unboundedly across laps (dynamics ṡ = p_v stays
        # smooth — no wrap discontinuity → no lap-rollover ACADOS_MINSTEP).
        # ca.floor's derivative is 0, so ∂s_periodic/∂s = 1 throughout,
        # which means the cost gradient w.r.t. s is correct everywhere
        # except exactly at multiples of L (a measure-zero set; for a
        # closed-loop track the spline values at s=L⁻ and s=0⁺ are nearly
        # equal anyway, so the cost-value jump is tiny).
        L_track = float(self.path_length)
        s_periodic = s - L_track * ca.floor(s / L_track)
        ref_x  = self.center_lut_x(s_periodic)
        ref_y  = self.center_lut_y(s_periodic)
        dxt    = self.center_lut_dx(s_periodic)
        dyt    = self.center_lut_dy(s_periodic)
        # Compute sin/cos of the centerline tangent DIRECTLY from dxt, dyt
        # — never form t_angle = atan2(dyt, dxt) as an intermediate
        # variable. atan2 has a branch cut at the −x axis: when the
        # centerline tangent crosses (dxt<0, dyt=0±ε) — which happens on
        # any leftward straight or curve in trackf at s≈56.7 — atan2
        # jumps from +π to −π even though the geometric direction is
        # smooth. That value-jump in t_angle, even with downstream
        # atan2(sin·,cos·) wrap-safety in yaw_err, creates a ~43k cost
        # spike at exactly that s every lap (observed on trackf at
        # (6-7, 18) wrap location). Forming sin/cos directly via
        # normalization is smooth across the jump.
        norm_t = ca.sqrt(dxt * dxt + dyt * dyt + 1e-12)
        sin_t = dyt / norm_t
        cos_t = dxt / norm_t

        # Contour (lateral) and lag (along-track) errors
        e_c = sin_t * (x_ - ref_x) - cos_t * (y_ - ref_y)
        e_l = -cos_t * (x_ - ref_x) - sin_t * (y_ - ref_y)
        # Reference velocity at s, capped by kinematic curvature limit
        # v ≤ √(a_lat_safe / |κ(s)|). Without this cap the centerline
        # ref_v=6 m/s applies even at the hairpins (κ_max≈0.84) where
        # the kinematically feasible v is sqrt(8/0.84) ≈ 3.08 m/s. MPC
        # was forced to choose between (a) tracking ref_v=6 → saturated
        # steer + a_lat slack absorbing huge violation → wall-stuck, or
        # (b) braking only on the q_v term, which competes with q_lag and
        # never wins enough. With the cap, ref_v becomes ~3 m/s in tight
        # corners → MPC naturally slows for the apex; the rest of the
        # cost machinery just tracks the new (kinematically-safe) target.
        # The hard a_lat backstop is now derived as a_lat_safe + 1.0 (see
        # a_lat_max below), so it sits just above this soft cap and is rarely
        # binding — tracking ref_v=√(a_lat_safe/κ) yields steady a_lat≈a_lat_safe,
        # which stays under the backstop (no per-corner slack spike).
        # Smooth fmin via 0.5·(a+b−sqrt((a−b)²+ε)) — differentiable
        # everywhere, kink rounded over ~0.03 m/s.
        ref_v_raw  = self.ref_v(s_periodic)
        kappa_at_s = self.abs_kappa_lut(s_periodic)
        v_kin_max  = ca.sqrt(a_lat_safe_p / (kappa_at_s + 1e-3))
        EPS_VM     = 1e-3
        # CiMPCC g(κ) 원복 (alpha=2 도 너무 강함, mpc 가 vx=0 local min 갇힘).
        # Heilmeier CSV ref_v 이미 곡률+brake 포함 → 충분.
        diff_v     = ref_v_raw - v_kin_max
        _rv1       = 0.5 * (ref_v_raw + v_kin_max
                            - ca.sqrt(diff_v * diff_v + EPS_VM))
        # ref_v = Heilmeier CSV profile smooth-min'd with the curvature cap
        # √(a_lat_safe/κ).  (A width-aware cap was tried 2026-06-02 but was
        # neutralized — K_WIDTH=100 made the smooth-min always pick _rv1 — and
        # removed 2026-06-03: it solved a non-existent width problem (map min
        # width 0.89 m, not 0.36 m) and only ever slowed the car.)
        ref_v_expr = _rv1

        # ---- Distance² to obstacle (used by h constraint only) ----
        d2 = (x_ - obs_x) ** 2 + (y_ - obs_y) ** 2

        # Heading error — wrap to [-π, π] via atan2(sin, cos) of the
        # difference. Use the angle-subtraction identities so the inputs
        # to atan2 are computed from sin_t, cos_t (smooth) rather than
        # going through t_angle (would have the same branch-cut spike):
        #   sin(ψ − t) = sin(ψ)·cos(t) − cos(ψ)·sin(t)
        #   cos(ψ − t) = cos(ψ)·cos(t) + sin(ψ)·sin(t)
        sin_psi = ca.sin(psi); cos_psi = ca.cos(psi)
        sin_diff = sin_psi * cos_t - cos_psi * sin_t
        cos_diff = cos_psi * cos_t + sin_psi * sin_t
        yaw_err = ca.atan2(sin_diff, cos_diff)

        # ---- NONLINEAR_LS cost form ----
        # y(x,u) = [e_c, e_l, yaw_err, v − ref_v, δ, p − p_max, side_term]
        # The side_term re-introduces obstacle-side preference but in a
        # PSD form (multiplied by sqrt(proximity)·|side_pref|). When no
        # obstacle (sentinel) → proximity≈0 → cost contribution≈0.
        # σ=1.3 (was 0.6 → 1.0 → 1.3; 1.5 caused cost=3337 IPM blowup with
        # the old q_side=25, but with q_side=12 and D_DETOUR_SIDE=0.25
        # there's headroom). Earlier engagement so prediction starts to
        # bend laterally before the obstacle is in the dynamics horizon —
        # avoids the "pred line stuck against obstacle, brake at the last
        # moment" pattern (pred_endpoint within R_safe of obs in 14% of
        # cycles in the σ=1.0 run). Proximity table:
        #   d=2  m: σ=1.0→0.14   σ=1.3→0.31
        #   d=3  m: σ=1.0→0.01   σ=1.3→0.07
        #   d=4  m: σ=1.0→3e-4   σ=1.3→0.008
        sigma_side = 0.5  # 1.3 → 1.0 → 0.7 → 0.5. Tighter side cost
                          # window so MPC barely detours unless very close
                          # to obstacle. Proximity at d=1m: 0.14, at
                          # d=0.7m: 0.37 — engagement starts at ~1 m.
        proximity_side = ca.exp(-d2 / (2.0 * sigma_side * sigma_side))
        # 0.25 m (was 0.4 → 0.25 to pair with R_safe 0.3 and corridor
        # h-constraint). With the soft corridor active, a 0.4 m detour put
        # the cost minimum at the corridor edge → lateral push fighting the
        # corridor push, and MPC braked to release the tension. 0.25 m
        # places the detour line at e_c=±0.25 well inside the corridor
        # (±0.35 usable), so side cost and corridor cost don't compete.
        # Required clearance from obstacle still met: detour 0.25 + R_safe
        # 0.3 (with car edge buffer) = 0.55 m closest approach center-to-center.
        # D_DETOUR_SIDE now read from p_sym[4] (rqt-tunable). Default value
        # in self.D_detour_live = 0.15. Live tunable per-cycle.
        # |side_pref| ∈ {0, 1}; smooth approx avoids non-differentiable |·|
        # 1e-3 (was 1e-9): keeps sqrt's gradient bounded near 0 — with
        # 1e-9, ∂sqrt/∂x at x=0 is ~16k, which makes the GN Hessian
        # spike when proximity → 0 and overshoots IPM step.
        # B: side_term off (obstacle avoidance, VPMPCC 없음). residual 자리는
        # 유지 (codegen W 호환).
        side_term = ca.SX(0.0)
        # Adaptive lane-tracking attenuation (mirrors IPOPT MPCC.py technique).
        # Multiplies the e_c/e_l cost by 1 - 0.95·exp(-d²/2σ²): centerline
        # tracking is normal far away, fades to ~5% at obstacle center.
        # Without this, q_cte (pull to centerline) and q_side (push to detour
        # line) fight each other near the obstacle → MPC brakes hard to
        # release the tension and outputs jerky control. With attenuation,
        # the centerline pull naturally relaxes when needed, so detour is
        # "free" near obstacle and tracking resumes once past. NONLINEAR_LS
        # form: cost = q · residual², so multiplying residual by sqrt(att)
        # multiplies cost by att.
        # σ_atten = 1.0 (Gaussian width):
        #   d=∞    → att=1.00 (normal)
        #   d=2 m  → att≈0.87
        #   d=1 m  → att≈0.42 (cost halved)
        #   d=0.5  → att≈0.16
        #   d=0    → att≈0.05 (tracking off)
        # B (VPMPCC simplify): obstacle attenuation + κ-attenuation off.
        # VPMPCC 는 corner 에서도 centerline tracking 유지.
        # Restored (avoidance Option 2): lane-tracking attenuation near the
        # selected obstacle, mirrors ifac_mpcc ipopt_kinematic.py:202. Fades
        # the e_c/e_l contouring cost to ~5% at the obstacle center so the
        # detour is "free" and not fighting the centerline pull. Gated by d²
        # to the obstacle (line 837): no obstacle → sentinel → d²≈1e12 →
        # proximity_atten=0 → attenuation=1.0 (Phase-B-identical).
        sigma_atten     = 1.0
        proximity_atten = ca.exp(-d2 / (2.0 * sigma_atten * sigma_atten))
        attenuation     = 1.0 - 0.95 * proximity_atten
        att_kappa   = ca.SX(1.0)
        sqrt_att    = ca.sqrt(attenuation * att_kappa + 1e-6)
        # 8th residual: δ − δ_prev → steer-rate cost. Penalises stage-to-
        # stage δ change directly, which kills within-prediction zigzag.
        # Cost weights q_cte_def, q_lag_def in W are baked at codegen,
        # but multiplied by q_cte_scale_p / q_lag_scale_p at the residual
        # level so the rqt slider can change the EFFECTIVE weight live:
        #   cost = W_baked · (sqrt(scale_p) · sqrt(att) · e_c)²
        #        = W_baked · scale_p · att · e_c²
        # default scale=1.0 → effective = baked.
        sqrt_q_cte_scale   = ca.sqrt(q_cte_scale_p   + 1e-9)
        sqrt_q_lag_scale   = ca.sqrt(q_lag_scale_p   + 1e-9)
        sqrt_q_psi_scale   = ca.sqrt(q_psi_scale_p   + 1e-9)
        sqrt_q_v_scale     = ca.sqrt(q_v_scale_p     + 1e-9)
        sqrt_q_dd_scale    = ca.sqrt(q_dd_scale_p    + 1e-9)
        sqrt_q_p_scale     = ca.sqrt(q_p_scale_p     + 1e-9)
        sqrt_q_drate_scale = ca.sqrt(q_drate_scale_p + 1e-9)
        sqrt_q_dv_scale    = ca.sqrt(q_dv_scale_p    + 1e-9)

        # ── Apex-biased lateral reference (2026-06-01, 해결3 복원) ──
        # e_c_ref=0 (pure centerline) made the cost BLIND to the racing line:
        # progress = centerline-arc-rate (capped at v_max) gives zero reward
        # for apex-cutting, so low q_cte / wide corridor (tested) did NOT make
        # the car cut apexes. The fix is to bias the lateral REFERENCE toward
        # the inside of each corner (κ-dependent) so the contouring cost itself
        # pulls the car onto an in-out-in apex line.
        #   e_c sign: e_c<0 = left, >0 = right of centerline.
        #   left turn (κ>0) → apex on the left → bias e_c_ref<0 = -sign(κ)·D.
        # signed_kappa_lut is the apex-kernel-smoothed signed κ (bias kicks in
        # BEFORE the geometric apex, decays on exit — see the kernel comments).
        # tanh(·/κ_ref) saturates so |e_c_ref| ≤ D_apex on real corners and ≈0
        # on straights. D_apex_p (=D_apex_live param) tunes depth; 0 = off.
        _signed_k = self.signed_kappa_lut(s_periodic)
        e_c_ref = -D_apex_p * ca.tanh(_signed_k / 0.20)

        # 9th residual: q_dv · a_x (longitudinal accel penalty).
        # 2026-05-27 review #8 — VPMPCC q_Δv 와 동일 정신. 이전엔 baked weight
        # self.q_dv=15 만 정의되고 cost 에 미연결 (ghost weight) → 연결.
        _spd_tgt = float(self.speed_target) if self.speed_target else float(self.v_max)  # R3: progress target (decoupled from hard cap)
        y_expr   = ca.vertcat(sqrt_q_cte_scale   * sqrt_att * (e_c - e_c_ref),
                              sqrt_q_lag_scale   * sqrt_att * e_l,
                              sqrt_q_psi_scale   * yaw_err,
                              sqrt_q_v_scale     * (vx_for_cost - ref_v_expr),
                              # curvature feedforward (2026-06-19, ref-confirmed):
                              # pull delta toward atan(L*kappa) — the steer the
                              # curvature needs — not toward 0. Fixes apex under-
                              # steer (all ref MPCCs lack this). δ_ff=0 on straights.
                              sqrt_q_dd_scale    * (delta - 1.15 * ca.atan(self.L * kappa_at_s * ca.sign(_signed_k))),
                              sqrt_q_p_scale     * (p_v - _spd_tgt),
                              side_term,
                              sqrt_q_drate_scale * (delta - delta_prev),
                              sqrt_q_dv_scale    * a_x_input)
        # ── LMPC terminal cost addition (Rosolia 2018 §IV.B, simplified) ──
        # 2026-05-28 #18: terminal value-function approximation via softmin
        # over Sampled Safe Set (K=10 nearest historical states).
        #
        # softmin_β(g_i) = -1/β · log Σ_i exp(-β · g_i)
        #   where g_i = Q_i + α · d_i² ,  d_i² = weighted_L2(x_N - x*_i)²
        #
        # Reviewer 2026-05-28: NONLINEAR_LS 의 squared(=cost = w·softmin²) 가
        # 의미상 어긋나지만 EXTERNAL 로 가면 acados 구조 변경 큼. 절충:
        # softmin 자체가 항상 양수가 아니라도 squared 형태 OK (위치 = SS min
        # 으로 동일 수렴). w_lmpc=0 (default) → 영향 0.
        #
        # x_N 의 weighted L2 to ss_states[k] (4-dim subset: px, py, ψ, vx).
        # vx 는 weight 0.3 (vx mismatch 영향 작게 — warm_transfer #4-B 와 짝).
        W_lmpc_diag = ca.diag(ca.SX([1.0, 1.0, 1.5, 0.3]))   # px, py, ψ, vx
        # x_N 의 해당 4 dim: state = [x, y, psi, vx, vy, r, s, delta_prev]
        # 변수명은 우리 acados 스코프 기준 (x_, y_, psi, vx_for_cost).
        # 2026-06-10 unified layout: kinematic 도 8-state (vx = state[3]) →
        # terminal cost 에 vx 사용 가능. 양 모드 동일.
        x_N_4 = ca.vertcat(x_, y_, psi, vx_for_cost)
        # ── min-time terminal: proximity-weighted cost-to-go (NLS-compatible) ──
        # 2026-05-30: the single nearest-point attractor (ss[:,0], CONSTANT
        # Q_best) only TRACKED the nearest stored state (its corner speed
        # included) → never minimized time-to-go → lap time DRIFTED up after
        # ~10 laps (not Rosolia-monotonic; measured). Replace with a smooth
        # softmax over the K safe-set points: terminal cost = proximity-weighted
        # cost-to-go  cog(x_N) = Σ_j w_j·Q_j ,  w_j = softmax_j(-β·d_j²).
        # Minimizing √(lmpc_w·cog) pulls x_N toward LOW cost-to-go (far-along)
        # states that are also REACHABLE (near) → forward progress / min-time.
        # Padding points (Q=1e6, far) get w≈0. Single smooth scalar residual →
        # stays in NONLINEAR_LS (no EXTERNAL_COST); a small reg toward the
        # nearest point keeps the Gauss-Newton Hessian positive-definite (the
        # 2026-05-28 log-sum-exp softmin's −1/β·log instability is avoided).
        _d2_cols = []
        for _j in range(K_LMPC):
            _dj = x_N_4 - lmpc_ss_states[:, _j]
            _d2_cols.append(ca.sum1(_dj * (W_lmpc_diag @ _dj)))
        d2_vec = ca.vertcat(*_d2_cols)                      # (K,1)
        _neg = -lmpc_beta_p * d2_vec
        _w_un = ca.exp(_neg - ca.mmax(_neg))                # numerically stable softmax
        _w_sm = _w_un / (ca.sum1(_w_un) + 1e-12)
        cog = ca.sum1(_w_sm * lmpc_ss_Q)                    # proximity-weighted cost-to-go
        d2_best = d2_vec[0]                                 # nearest-point reg (conditioning)
        lmpc_residual = ca.sqrt(lmpc_w_p + 1e-12) * ca.sqrt(
            cog + (lmpc_alpha_p + lmpc_reg_w_p) * d2_best + 1e-6
        )

        # analytic terminal cost-to-go: land horizon-end vx on the global optimal
        # speed profile ref_v_expr(s_N). Removes N-dependency (brakes for corners
        # beyond the horizon). Reuses BO-tunable q_v scale; baked terminal emphasis x3.
        # 2026-06-10 unified layout: vx is a state in BOTH modes → vterm everywhere.
        vterm_res = sqrt_q_v_scale * (vx_for_cost - ref_v_expr)
        if self._lmpc_joint and alpha_sym is not None:
            # ── B3 Step 2: convex-α terminal (Racing-LMPC racing_mpc.cpp:479-504) ──
            # x_N soft-anchored to the SS convex hull Σαⱼ·SSⱼ, and the cost-to-go
            # Qᵀα minimized LINEARLY (CONL ψ_e) → the solver picks the convex combo
            # of safe-set points with lowest time-to-go that x_N can reach. Gated
            # by √(lmpc_w_p): inert (= baseline) when LMPC off, engages once the
            # safe set is populated (lmpc_w>0).
            _wl   = ca.sqrt(lmpc_w_p + 1e-12)
            _ssa  = ca.mtimes(lmpc_ss_states, alpha_sym)          # Σαⱼ·SSⱼ  (4: px,py,ψ,vx)
            _wanc = ca.sqrt(ca.SX([20.0, 20.0, 2.0, 2.0]))        # per-state hull-slack weight
            _anc  = _wl * (_wanc * (x_N_4 - _ssa))                # 4 soft-anchor residuals
            _cog  = _wl * ca.dot(lmpc_ss_Q, alpha_sym)            # cost-to-go (linear in ψ_e)
            y_expr_e = ca.vertcat(e_c, e_l, yaw_err, vterm_res,
                                  _anc[0], _anc[1], _anc[2], _anc[3], _cog)
        else:
            y_expr_e = ca.vertcat(e_c, e_l, yaw_err, vterm_res, lmpc_residual)

        # ---- Constraints (h) — minimal set for stable SQP_RTI ----
        # 1) obstacle half-plane (replaces 2026-05-06 the non-convex annulus
        #    d²−R²≥0). The annulus form is bistable: slack absorbs the
        #    "diving through obstacle" path as a local optimum, so the
        #    predicted trajectory ends *inside* the keepout (observed:
        #    pred_endpoint within R_safe of obs in 14% of obstacle-active
        #    cycles → "경로 라인이 장애물에 막힘" symptom + last-moment
        #    brake/burst). Half-plane: project (car − obs) onto the
        #    right-perpendicular of the centerline tangent at the predicted
        #    stage, then require side_pref · projection ≥ R_safe + R_CAR.
        #    Convex, one-sided, prediction physically cannot cross to the
        #    other side. side_pref ∈ {-1, 0, +1} from decide_side_pref;
        #    when side_pref ≈ 0 (no obstacle / sentinel), a |side_pref|-
        #    gated big-M term keeps the constraint trivially satisfied.
        # 2) corridor lateral bound — added 2026-05-06 to prevent sim-wall
        #    stuck. Project corridor boundary points onto the same right-
        #    perpendicular as e_c, then bound e_c. Smooth max/min so it
        #    works for CW/CCW orientation and avoids the kink that caused
        #    HPIPM S_MINSTEP at narrow stages.
        # 3) lateral accel limit (kinematic).
        # R_CAR now read from p_sym[5] (rqt-tunable). Default value in
        # self.R_car_live = 0.02. Live tunable per-cycle. Used by both
        # h_obs (obstacle margin) and h_corridor (wall margin).
        R_CAR = R_car_p
        # ---- (1) obstacle half-plane (Frenet + proximity-gated) ----
        # Frenet form: side_pref · (e_c − e_c_obs) ≥ R + R_CAR. Naturally
        # curve-aware because e_c at each predicted stage is measured
        # against the centerline at THAT stage's s. No tangent rotation
        # issue.
        # Proximity gate: gate = |side_pref| · exp(−d²/2σ²). Disables the
        # constraint smoothly when far from the obstacle (d>3 m → gate≈0)
        # or when no obstacle is selected (side_pref≈0). Big-M on the
        # complementary side keeps the constraint trivially satisfied
        # outside the active zone.
        SIGMA_OBS = 1.0   # 2.0 → 1.4 → 1.0. Half-plane gate fires in
                          # a tighter window (~0.7 m vs ~1 m). Less
                          # detour anticipation, smaller swerve.
        prox_obs  = ca.exp(-d2 / (2.0 * SIGMA_OBS * SIGMA_OBS))
        abs_side  = ca.sqrt(side_pref * side_pref + 1e-3)
        gate      = abs_side * prox_obs           # ≈ 1 only when active
        BIG_OBS   = 50.0
        h_obs = (gate * (side_pref * (e_c - e_c_obs_p) - (R_safe_p + R_CAR))
                 + (1.0 - gate) * BIG_OBS)
        # ---- (2) corridor ----
        w_left  = sin_t * (left_x - ref_x) - cos_t * (left_y - ref_y)
        w_right = sin_t * (right_x - ref_x) - cos_t * (right_y - ref_y)
        EPS_MM    = 1e-4
        diff_lr   = w_left - w_right
        sqrt_diff = ca.sqrt(diff_lr * diff_lr + EPS_MM)
        upper_lat = 0.5 * (w_left + w_right + sqrt_diff) - R_CAR
        lower_lat = 0.5 * (w_left + w_right - sqrt_diff) + R_CAR
        h_corridor_top = upper_lat - e_c
        h_corridor_bot = e_c - lower_lat
        # ---- (3) a_lat ----
        # Kinematic: a_lat = v² tan(δ)/L (geometric centripetal acc).
        # Dynamic: a_lat = vx · r (true lateral acceleration). Both
        # are computed in the model branch above into `a_lat_expr`.
        a_lat = a_lat_expr
        # ---- (4) friction ellipse (2026-06-10 spec) ----
        # (a_x/a_lim)² + (a_lat/a_lim)² ≤ 1, a_lim = μ·g·η. u[0]=a_x 는 unified
        # 입력 레이아웃에서 양 모드 공통. 제동·가속 중 가용 a_lat 이 자동 감소
        # (combined-slip). 고그립(μ=1.0489)에선 a_lim≈9.8 → 거의 안 물림 = 기존
        # 동작 보존; 저그립(μ=0.6)에선 a_lim≈5.6 = 실질 한계.
        _a_lim = max(1e-3, grip_a_lat_limit(self.dyn_mu, self.ellipse_frac))
        h_ellipse = (u[0] / _a_lim) ** 2 + (a_lat / _a_lim) ** 2
        # C-STEER (2026-06-19): hard steering-rate row |(δ−δ_prev)/dT| ≤ sv_max.
        # δ_prev is the augmented state (= last-applied steer); bounded as a
        # path h-constraint (NOT slacked) it forces servo-feasible steering and
        # kills the cycle-to-cycle hunting that the soft q_drate cost could not.
        steer_rate = (u[1] - delta_prev) / self.dT
        if self._lmpc_joint and alpha_sym is not None:
            # B3: α simplex Σα=1 — hard eq row, then steer-rate row (both NOT slacked).
            model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat, h_ellipse, ca.sum1(alpha_sym), steer_rate)
        else:
            model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat, h_ellipse, steer_rate)
        # C-STEER node-0: path con_h above covers stages 1..N-1; the PUBLISHED
        # control is u_0 (stage 0). A stage-0 h-constraint with ONLY the steer-
        # rate (state-only rows would be constant at the fixed x0 → drop them)
        # rate-limits the published δ_0 → kills the full-lock launch swerve.
        model_ac.con_h_expr_0 = ca.vertcat(steer_rate)

        # ---- Compose model ----
        # B4' error-dynamics regression: add affine velocity correction to the
        # blended dynamic f_expl. nx-wide via explicit rows [3,4,5]=[vx,vy,r] so
        # it composes with joint-α (nx=18) and plain (nx=8) alike. Gated: when
        # _err_regr is False the slots stay 0 → exact baseline f_expl.
        if self._err_regr:
            f_expl = f_expl + ca.vertcat(
                ca.SX.zeros(3), e_corr_sym[0], e_corr_sym[1], e_corr_sym[2],
                ca.SX.zeros(f_expl.shape[0] - 6))
        model_ac.f_impl_expr = xdot - f_expl
        model_ac.f_expl_expr = f_expl
        model_ac.x = x
        model_ac.xdot = xdot
        model_ac.u = u
        model_ac.z = ca.vertcat([])
        model_ac.p = p_sym
        ocp.model = model_ac
        ocp.dims.np = n_p_total

        # ---- Cost: NONLINEAR_LS ----
        # W is fixed at codegen; LIVE tuning of q_cte/q_lag/q_d_delta done by
        # multiplying inside y_expr if needed later. For now, use defaults.
        # Tuned for kinematic limits: too large q_cte/q_psi forces tighter
        # tracking than dynamics can follow at corners → car stalls or
        # bounces off the line.
        # 2026-05-27: baked weights 강화 — 박힘 ↓ 위해 centerline 추종 강.
        # BO 가 scale 0.3~5 학습 → effective q_cte=4.5~75, q_lag=24~400, etc.
        # local minima 회피 위해 baked 는 paper VPMPCC 의 중간 값.
        q_cte_def     = 15.0    # 9→15 (centerline lateral 추종 강)
        q_lag_def     = 80.0    # 45→80 (along-track 강)
        q_psi_def     = 10.0
        q_v_def       = 12.0
        q_dd_def      = 5.0
                                # × 20 stages now totals ~48 vs q_lag 60,
                                # so solver feels significant cost for
                                # large |δ|. Without this the solver was
                                # cycle-to-cycle picking +0.4 / −0.1
                                # alternating (65% sign-flip rate observed)
                                # because both gave similar predicted cost
                                # — flat surface. Higher q_d_delta forces
                                # the solver to commit to moderate δ
                                # spread across the horizon, killing the
                                # prediction-line trembling.
        q_p_def       = 1.0     # progress pull 약화 (4→1) — 보수적 시작, BO 가 scale 학습
        q_side_def    = 3.0     # very soft hint (was 12 → 3, mirrors
                                # IPOPT MPCC.py W_SIDE=3). With the new
                                # attenuation that fades q_cte/q_lag near
                                # the obstacle, side cost no longer needs
                                # to fight q_cte for the detour — it's
                                # just a gentle nudge. q_side=12 was too
                                # much: it produced cost spikes when the
                                # car was forced to choose between the
                                # detour line and the half-plane edge.
        q_d_rate_def  = 80.0    # 원복 (S2 100 으로 안 효과)
                                # 30 → 50 → 80. Strong rate cost minimises
                                # corner→straight transition wobble.
        q_dv_def      = 15.0    # 2026-05-27 #8: longitudinal accel a_x penalty.
                                # 이전엔 self.q_dv = 15 만 정의되고 cost 연결
                                # 안 됨 → 9th residual 로 연결.
        ny   = 9   # 9 residuals (cte, lag, psi, v, dd, p, side, drate, dv)
        # terminal: vx cost-to-go residual (vterm) before lmpc_residual — BOTH modes
        # (2026-06-10 unified layout). B3 joint: lmpc_residual(1) → anchor(4)+cog(1) = +4.
        ny_e = 9 if self._lmpc_joint else 5
        # ---- Cost: CONVEX_OVER_NONLINEAR (Phase B0, 2026-06-04) ----
        # Migrated from NONLINEAR_LS. With ψ(r) = ½·rᵀWr and r = y_expr this
        # reproduces the LS cost EXACTLY (same Gauss-Newton Hessian), but the
        # CONL form lets later phases add convex non-LS terms — e.g. a linear
        # progress reward −γ·p_v (B1) that pure NLS cannot express.
        W_mat = np.diag([q_cte_def, q_lag_def, q_psi_def,
                         q_v_def, q_dd_def, q_p_def, q_side_def,
                         q_d_rate_def, q_dv_def])
        # W_e: vx terminal cost-to-go (q_v_def*3) before the LMPC residual
        # (last entry 1.0, since residual already carries sqrt(lmpc_w)).
        # 2026-06-10 unified layout: same in both modes.
        if self._lmpc_joint:
            # [ec×5, el×5, yaw×4, vterm×3, anchor×4 (√wt in residual→1), cog (linear→0)]
            W_e_diag = [q_cte_def * 5.0, q_lag_def * 5.0, q_psi_def * 4.0, q_v_def * 3.0,
                        1.0, 1.0, 1.0, 1.0, 0.0]
        else:
            W_e_diag = [q_cte_def * 5.0, q_lag_def * 5.0,
                        q_psi_def * 4.0, q_v_def * 3.0, 1.0]
        W_e_mat = np.diag(W_e_diag)
        ocp.cost.cost_type   = 'CONVEX_OVER_NONLINEAR'
        ocp.cost.cost_type_0 = 'CONVEX_OVER_NONLINEAR'
        ocp.cost.cost_type_e = 'CONVEX_OVER_NONLINEAR'
        # residual r(x,u) — identical to the NLS y_expr (live q_*_scale already
        # baked inside via sqrt(q_*_scale) factors → effective w = W·q_scale).
        ocp.model.cost_y_expr   = y_expr
        ocp.model.cost_y_expr_0 = y_expr
        ocp.model.cost_y_expr_e = y_expr_e
        # outer convex ψ(r) = ½ rᵀ W r, written in a fresh symbolic r.
        _r_conl   = ca.SX.sym('r_conl',   ny)
        _r_conl_e = ca.SX.sym('r_conl_e', ny_e)
        ocp.model.cost_r_in_psi_expr   = _r_conl
        ocp.model.cost_r_in_psi_expr_0 = _r_conl
        ocp.model.cost_r_in_psi_expr_e = _r_conl_e
        ocp.model.cost_psi_expr   = 0.5 * ca.mtimes([_r_conl.T,   ca.DM(W_mat),   _r_conl])
        ocp.model.cost_psi_expr_0 = 0.5 * ca.mtimes([_r_conl.T,   ca.DM(W_mat),   _r_conl])
        if self._lmpc_joint:
            # B3 Step2: cost-to-go Qᵀα enters ψ_e LINEARLY (last residual, W_e=0).
            ocp.model.cost_psi_expr_e = (0.5 * ca.mtimes([_r_conl_e.T, ca.DM(W_e_mat), _r_conl_e])
                                         + self._lmpc_cog_w * _r_conl_e[ny_e - 1])
        else:
            ocp.model.cost_psi_expr_e = 0.5 * ca.mtimes([_r_conl_e.T, ca.DM(W_e_mat), _r_conl_e])
        ocp.cost.yref   = np.zeros(ny)
        ocp.cost.yref_0 = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # ---- Dimensions / horizon ----
        ocp.solver_options.N_horizon = self.N
        # multi-dt time_steps (pyramidal grid 면 가까이 sharp, 멀리 long planning).
        # tf = sum(time_steps), acados 가 stage 별 dt 자동 적용.
        ocp.solver_options.time_steps = np.array(self.time_steps, dtype=float)
        ocp.solver_options.tf = float(np.sum(self.time_steps))
        Tf = ocp.solver_options.tf

        # ---- Initial state placeholder ----
        if self._lmpc_joint:
            # B3: fix only the 8 physical states at t=0; α(K) left FREE so the
            # solver optimizes the convex combination jointly with x/u.
            ocp.constraints.idxbx_0 = np.arange(8)
            ocp.constraints.lbx_0 = np.zeros(8)
            ocp.constraints.ubx_0 = np.zeros(8)
        else:
            ocp.constraints.x0 = np.zeros(nx)

        # ---- Input bounds ----
        # 2026-06-10 unified layout: u = [a_x, δ, p_v] in BOTH modes — bounds on
        # accel/decel from sim (max_accel=7.51, max_decel=8.26). v_max enforced
        # as a STATE bound on vx instead.
        ocp.constraints.idxbu = np.array([0, 1, 2])
        # a_min 줄임: -8.26 → -3.0. 강한 brake (8m/s²) 는 실제로 거의
        # 발생 안 하는데 솔버가 vx<1 singular 영역에서 "어떻게든 멈춰"
        # 라며 -8.26 출력 → 다음 cycle vx 음수 예측 → Pacejka 망가짐
        # → ACADOS_MINSTEP 캐스케이드. -3.0 이면 충분히 감속 가능하고
        # 솔버가 미친 brake 못 함.
        a_min_dyn = A_MIN_DYN   # 솔버 제동한계 — model_policy 단일소스 (ref_v a_long 과 동기)
        a_max_dyn =  4.0   # 7.51 → 4.0 (Agent R-round3 Fix 2). F1TENTH RWD @ vx≈0
                           # grip ~4 m/s². 7.51 (sim max_accel) 은 saturated 되어
                           # solver 가 "instant fix" 권장 → traj 예측 unrealistic →
                           # zigzag + over-shoot. 4 가 realistic, 차도 추종 가능.
        ocp.constraints.lbu = np.array([a_min_dyn, self.theta_min, 0.0])
        ocp.constraints.ubu = np.array([a_max_dyn, self.theta_max, self.p_max])
        # Kept for the per-stage kinematic p_v grip cap in solve() — per-stage
        # solver.set(k,"ubu",…) must pass the FULL bound vector, so remember
        # the codegen-time values to copy-and-modify slot 2 only.
        self._ubu_codegen = np.array([a_max_dyn, self.theta_max, self.p_max])
        # vx lower bound 0: 절대 후진 안 함. -1.0 허용 시 tire model
        # 의 vx<0 영역 (실제로는 vx²=0.01 이라도 sign(vx)·F 가 발산)
        # 에서 솔버 폭주 가능. 0 으로 강제하면 reverse 자체 차단.
        ocp.constraints.idxbx = np.array([3, 4, 5])
        ocp.constraints.lbx   = np.array([0.0, -10.0, -20.0])
        ocp.constraints.ubx   = np.array([self.v_max + 0.5, 10.0, 20.0])   # 작은 margin (0.5 m/s). v_max 까지 빡빡하게 강제하면 solver IPM 발산 → 12m teleport. 원래 +2 로 풀어줌.
        if self._lmpc_joint:
            # B3: α ∈ [0,1] state bound on the K augmented states (all stages).
            _Ka = int(dyn['K_aug'])
            self._lmpc_K_aug = _Ka   # needed for per-stage ubx padding below
            ocp.constraints.idxbx = np.concatenate([ocp.constraints.idxbx, np.arange(8, 8 + _Ka)])
            ocp.constraints.lbx   = np.concatenate([ocp.constraints.lbx, np.zeros(_Ka)])
            ocp.constraints.ubx   = np.concatenate([ocp.constraints.ubx, np.ones(_Ka)])

        # ---- h bounds (only h_obs + a_lat now) ----
        # uh[0] HUGE: obstacle absence is encoded by setting obs_x/y to a
        # sentinel ~1e6 m, which makes h_obs = d² ≈ 1e12. With BIG=1e3
        # that's a 10⁹-magnitude constraint violation every cycle, slack
        # cost explodes, and HPIPM produces NaN step directions →
        # ACADOS_MINSTEP. With uh=1e15 the trivial case is well within
        # bounds.
        # a_lat_max = HARD backstop on actual lateral accel (vx·r). It must
        # sit ABOVE the soft κ-cap a_lat_safe, else it binds every corner:
        # tracking ref_v=√(a_lat_safe/κ) gives steady-corner a_lat = vx²κ =
        # a_lat_safe, so a_lat_max < a_lat_safe (the old 8 vs deploy-9 case)
        # silently slacks the constraint on every apex. Derive it from the
        # codegen-time a_lat_safe_live (pushed from yaml before setup_MPC)
        # with a +1.0 backstop margin, floored at the historical 8.0.
        #   NOTE: a full 12 was tried and reverted (hairpin trembling — cost
        #   surface too flat at high κ → IPM zigzag). +1.0 keeps the cap a
        #   true backstop (just above the soft cap) without re-entering that
        #   flat region; the steer-output EMA filter further damps any wobble.
        a_lat_max = max(8.0, float(self.a_lat_safe_eff()) + 1.0)
        self._log.info(f"[MPC-acados] a_lat hard cap = {a_lat_max:.2f} (a_lat_safe={a_lat_max - 1.0:.2f} + 1.0 backstop)")
        # h order: [h_obs, corr_top, corr_bot, a_lat, ellipse, (Σα if joint)]
        # ellipse 행: 0 ≤ h ≤ 1 (h 는 제곱합이라 하한 0 은 자연 충족 — 무해).
        if self._lmpc_joint:
            ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max, 0.0, 1.0, -self.sv_max])  # Σα=1 eq, steer-rate
            ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max, 1.0, 1.0, self.sv_max])
        else:
            ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max, 0.0, -self.sv_max])  # +steer-rate (hard)
            ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max, 1.0, self.sv_max])
        # C-STEER stage-0 bounds: steer-rate only, hard (not slacked).
        ocp.constraints.lh_0 = np.array([-self.sv_max])
        ocp.constraints.uh_0 = np.array([self.sv_max])
        # Slack on first five — corridor, obstacle, a_lat, ellipse can be
        # transiently violated. Slack absorbs without triggering cascade.
        ocp.constraints.idxsh = np.array([0, 1, 2, 3, 4])   # ellipse(4) 포함, Σα 비slack
        ns = 5
        ocp.constraints.lsh = np.zeros(ns)
        ocp.constraints.ush = np.zeros(ns)
        # Per-constraint slack tuning — second reduction. Quadratic Zl was
        # the dominant source of cost spikes (slack=2m × Zl=500 = 2000
        # per stage); reducing 500→80 capped spikes at ~6k. Reducing
        # further to 30 caps them at ~2-3k while keeping a bounded but
        # still-meaningful push back to the constraint via the linear zl
        # term. Linear zl is bumped slightly so the gradient at small
        # violations stays informative (push = zl + 2·Zl·s; with smaller
        # Zl, the linear term is what the optimizer feels for s < 0.5m).
        #   idx 0 (h_obs):       zl=40,  Zl=30   (was 30/80)
        #   idx 1,2 (corridor):  zl=20,  Zl=15   (was 15/30)
        #   idx 3 (a_lat):       zl=50,  Zl=15   (was 50/30)
        #   idx 4 (ellipse):     zl=50,  Zl=15   (a_lat 행과 동일 — spec §2)
        # (2026-06-02 corridor 6배 강화 시도 → 역효과 5.1접촉/랩: 강한 corridor
        #  push 가 반대편 클립 유발. 원복.)
        ocp.cost.zl = np.array([40.0, 20.0, 20.0, 50.0, 50.0])
        ocp.cost.zu = np.array([40.0, 20.0, 20.0, 50.0, 50.0])
        ocp.cost.Zl = np.array([30.0, 15.0, 15.0, 15.0, 15.0])
        ocp.cost.Zu = np.array([30.0, 15.0, 15.0, 15.0, 15.0])

        # ---- Initial parameter values (overridden every cycle) ----
        ocp.parameter_values = np.zeros(n_p_total)
        ocp.parameter_values[4]  = self.D_detour_live       # D_DETOUR
        ocp.parameter_values[5]  = self.R_car_live          # R_CAR
        ocp.parameter_values[6]  = self.q_cte_scale_live    # q_cte scale
        ocp.parameter_values[7]  = self.R_safe_live         # R_safe
        ocp.parameter_values[8]  = self.q_lag_scale_live    # q_lag scale
        ocp.parameter_values[10] = self.a_lat_safe_eff()    # A_LAT_SAFE (μ-clamped)
        ocp.parameter_values[11] = self.D_apex_live         # D_apex
        ocp.parameter_values[12] = self.q_psi_scale_live    # q_psi scale
        ocp.parameter_values[13] = self.q_v_scale_live      # q_v scale
        ocp.parameter_values[14] = self.q_dd_scale_live     # q_dd scale (steer reg)
        ocp.parameter_values[15] = self.q_p_scale_live      # q_p scale (progress)
        ocp.parameter_values[16] = self.q_drate_scale_live  # q_d_rate scale
        ocp.parameter_values[17] = self.q_dv_scale_live     # q_dv scale (a_x)

        # ---- Solver options ----
        # NONLINEAR_LS form makes the Gauss-Newton Hessian = J^T·W·J
        # automatically PSD. SQP_RTI works again. PROJECT regularize is
        # an extra safety net (acados forum recommendation).
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # SQP_RTI single-iter for BOTH kinematic and dynamic.
        # Multi-iter SQP was diverging at startup (~15 iters too much
        # freedom when warm-start far from optimum → IPM step grows
        # unbounded). RTI = 1 SQP iter per cycle, naturally stable.
        # Pacejka's higher-order derivatives are tamed by strong LM (3.0)
        # below — same approach as kinematic, just more regularization.
        #
        # nlp_solver_iters>1 (2026-06-19): use AS-RTI (advanced-step RTI), NOT
        # full SQP. Full SQP+globalization hit Maximum_Iterations every cycle →
        # the node's RTI-oriented logic reset the warm-start each time → 0.5 m/s
        # crawl + STUCK (tested 06-19). AS-RTI keeps SQP_RTI semantics (SUCCESS
        # status, real-time, warm-start preserved) while doing _nlp_iters
        # STANDARD_RTI iterations per cycle — so it escapes the 1-iter active-set
        # trap without the divergence (uncapped SQP) OR the reset-thrash (capped
        # SQP). as_rti_level: 4 = STANDARD_RTI (each AS iter = one full RTI).
        _nlp_iters = int(getattr(self, 'nlp_solver_iters', 1))
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        if _nlp_iters > 1:
            ocp.solver_options.as_rti_level = 4          # STANDARD_RTI
            ocp.solver_options.as_rti_iter  = _nlp_iters
        self._log.info("[MPC-acados] nlp_solver=SQP_RTI as_rti_iter=%d (nlp_solver_iters=%d)"
                       % (int(getattr(ocp.solver_options, 'as_rti_iter', 1)), _nlp_iters))
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # PROJECT (CONVEXIFY caused hairpin trembling — Hessian rebuilt
        # slightly differently each cycle, IPM step direction wobbled
        # across multiple near-equal optima in tight corners).
        # PROJECT just clips negative eigenvalues to small positive,
        # leaving block structure stable cycle-to-cycle → deterministic
        # step direction.
        ocp.solver_options.regularize_method = 'PROJECT'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.print_level = 0
        # Back to 100 — 200 was an exploration knob that combined with
        # CONVEXIFY/loose a_lat caused hairpin instability (more iters
        # = more chance to wander between near-equal optima). 100 is
        # sufficient when cost surface is well-conditioned.
        ocp.solver_options.qp_solver_iter_max = 100
        # LM for dynamic linear tire. Balanced: strong enough for IPM
        # stability with slip derivatives, loose enough that cost weights can
        # actually steer the solution (not over-damped). Overridable via the
        # `lm_dynamic` ROS param (set onto self before setup_MPC) so it can be
        # swept / BO-tuned without code edits.
        _lm_dyn = float(getattr(self, 'lm_dynamic', 1.0))
        # NOTE(2026-06-10 unified layout): kept mode-dependent ON PURPOSE — this
        # is a NUMERIC conditioning knob, not an algorithmic difference. f_kin has
        # no Pacejka slip derivatives, so it needs far less damping (0.2).
        ocp.solver_options.levenberg_marquardt = _lm_dyn if self.use_dynamic else 0.2
        self._log.info("[MPC-acados] levenberg_marquardt = %.3f (use_dynamic=%s)"
                       % (ocp.solver_options.levenberg_marquardt, self.use_dynamic))

        # Codegen + build — dir/json keyed to (model, nx) so kinematic<->dynamic
        # <->LMPC toggles never reuse a stale codegen (see codegen_paths()).
        ocp.code_export_directory, json_path = codegen_paths(
            self.use_dynamic, self._lmpc_joint, self._nx_solver,
            dyn_mu=float(self.dyn_mu))
        self._log.info("[MPC-acados] generating solver (~30 s first time)...")
        self.solver = AcadosOcpSolver(ocp, json_file=json_path)
        # GP residual wrap (gp_residual_wrapper.wrap_solver_with_gp) needs the
        # AcadosOcp object — setup_MPC built it as a local, so stash it.
        self.ocp = ocp
        self._log.info("[MPC-acados] solver ready")

        # Stash dim info for solve()
        self._n_p_const = n_p_const
        self._n_p_total = n_p_total

        # Storage
        self.X0 = np.zeros((self.N + 1, getattr(self, '_nx_solver', self.n_states)))
        self.u0 = np.zeros((self.N, self.n_controls))

    # ------------------------------------------------------------------
    # Bound construction helpers (mirror IPOPT solve())
    # ------------------------------------------------------------------
    def _obstacle_frenet(self, ox, oy):
        """Obstacle's Frenet coords (s_obs, e_c_obs) at its nearest centerline
        point. Returns (0.0, 0.0) for sentinel. Computed ONCE when an
        obstacle is committed (see solve()'s commit-once logic), not every
        cycle, so the brute-force search is one-shot per obstacle pass.
        """
        if ox > 1e5 or oy > 1e5 or self.kappa_grid is None:
            return 0.0, 0.0
        n = len(self.kappa_grid)
        best_d2 = float("inf"); best_s = 0.0
        for i in range(n):
            s = i * self.kappa_ds
            cxi = float(self.center_lut_x(s))
            cyi = float(self.center_lut_y(s))
            d2 = (cxi - ox) ** 2 + (cyi - oy) ** 2
            if d2 < best_d2:
                best_d2 = d2; best_s = s
        rxc = float(self.center_lut_x(best_s))
        ryc = float(self.center_lut_y(best_s))
        dxt = float(self.center_lut_dx(best_s))
        dyt = float(self.center_lut_dy(best_s))
        nrm = math.sqrt(dxt * dxt + dyt * dyt) + 1e-9
        sin_t_obs = dyt / nrm
        cos_t_obs = dxt / nrm
        e_c_obs = sin_t_obs * (ox - rxc) - cos_t_obs * (oy - ryc)
        return best_s, e_c_obs

    def get_path_constraints_points(self, prev_soln):
        """Sample left/right boundary at each predicted stage's s.
        2026-06-10 unified layout: BOTH modes are 8-state → s = col 6
        (vx/vy/r 가 3/4/5 자리).
        """
        right_points = np.zeros((self.N, 2))
        left_points = np.zeros((self.N, 2))
        idx_s = 6
        for k in range(1, self.N + 1):
            sk = float(prev_soln[k, idx_s]) % self.path_length
            right_points[k - 1, :] = np.array([self.right_lut_x(sk),
                                               self.right_lut_y(sk)],
                                              dtype=object).squeeze()
            left_points[k - 1, :] = np.array([self.left_lut_x(sk),
                                              self.left_lut_y(sk)],
                                             dtype=object).squeeze()
        return right_points, left_points

    def _kappa_at(self, s):
        """O(1) curvature lookup at arclength s (modulo path_length).
        Kept for diagnostics — not used to tighten constraints anymore."""
        if self.kappa_grid is None:
            return 0.0
        sw = float(s) % self.path_length
        idx = int(sw / self.kappa_ds)
        if idx < 0:
            idx = 0
        elif idx >= len(self.kappa_grid):
            idx = len(self.kappa_grid) - 1
        return abs(float(self.kappa_grid[idx]))

    def select_front_obstacle(self, curr_x, curr_y, curr_yaw,
                              obstacles, D_max=20.0, D_min_trig=10.0,
                              ang_max=2.0 * math.pi / 3.0):
        """Pick the closest obstacle in front of the car, or sentinel
        [1e6, 1e6, 1e6] when none qualify.

        D_max bumped 12→20 so curving tracks where obstacle is on next
        segment (Euclidean farther than path-distance) still trigger.
        ang_max bumped π/2→2π/3 (90°→120°) so obstacles slightly to the
        side on tight curves aren't filtered out before the car has
        rotated to face them.
        """
        if obstacles is None or len(obstacles) == 0:
            return [1e6, 1e6, 1e6]
        best = [1e6, 1e6, 1e6]
        cx, cy = curr_x + math.cos(curr_yaw) * 0.05, curr_y + math.sin(curr_yaw) * 0.05
        for ob in obstacles:
            if hasattr(ob, '__len__') and len(ob) >= 2:
                ox, oy = float(ob[0]), float(ob[1])
            else:
                continue
            dx, dy = ox - cx, oy - cy
            d = math.hypot(dx, dy)
            if d > D_max:
                continue
            # in front (with looser cone)?
            ang = math.atan2(dy, dx) - curr_yaw
            while ang > math.pi: ang -= 2 * math.pi
            while ang < -math.pi: ang += 2 * math.pi
            if abs(ang) > ang_max:
                continue
            if d < best[0]:
                best = [d, ox, oy]
        return best

    def decide_side_pref(self, s_obs, e_c_obs,
                         win_back=1.0, win_fwd=2.0, ds=0.25, margin=0.1):
        """Window-aware side decision (2026-06-11). Replaces the
        single-point top-2 boundary-distance compare whose centerline tie
        always returned -1 ("always avoids down" bug). Samples the labeled
        left/right corridor room along s ∈ [s_obs−win_back, s_obs+win_fwd]
        — the detour tube the car actually drives — and lets
        decide_side_window pick the side whose bottleneck (then mean room)
        is larger. Failure-driven flip (_side_history) stays as the rare
        fallback for what the map can't predict. One-shot per obstacle
        commit, so the ~13×6 LUT evals are negligible.
        """
        L = float(self.path_length)
        n = max(2, int(round((win_back + win_fwd) / ds)) + 1)
        w_l = np.empty(n)
        w_r = np.empty(n)
        for i in range(n):
            sk = (float(s_obs) - win_back + i * ds) % L
            cx = float(self.center_lut_x(sk)); cy = float(self.center_lut_y(sk))
            dxt = float(self.center_lut_dx(sk)); dyt = float(self.center_lut_dy(sk))
            nrm = math.hypot(dxt, dyt) + 1e-9
            sin_t, cos_t = dyt / nrm, dxt / nrm
            w_l[i] = (sin_t * (float(self.left_lut_x(sk)) - cx)
                      - cos_t * (float(self.left_lut_y(sk)) - cy))
            w_r[i] = (sin_t * (float(self.right_lut_x(sk)) - cx)
                      - cos_t * (float(self.right_lut_y(sk)) - cy))
        return decide_side_window(e_c_obs, w_l, w_r, margin=margin)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, initial_state, obstacles):
        """initial_state: numpy length 7 from ROS node = [x, y, psi, vx, vy, r, s].
        We extend internally to 8: append delta_prev = the steer command we
        sent last cycle (initialised to 0). 2026-06-10 unified layout: SAME
        in both kinematic and dynamic modes.
        """
        L = float(self.path_length)

        # Augment state with the previous applied δ for steer-rate cost.
        # Node sends 7 → expand to 8 (append δ_prev). target = self.n_states.
        if len(initial_state) == self.n_states - 1:
            last_delta = float(getattr(self, '_last_delta_applied', 0.0))
            initial_state = np.append(np.asarray(initial_state, dtype=float),
                                       last_delta)

        # State-vector index helpers — unified 8-state layout in BOTH modes:
        # [x, y, ψ, vx, vy, r, s, δ_prev].
        IDX_S = 6
        IDX_DELTA_PREV = 7

        # ---- Unwrap sensor s into solver-internal monotonic coordinate ----
        # Node passes current_s ∈ [0, L). We track lap_count internally and
        # produce solver_s = sensor_s + lap_count·L which grows unboundedly
        # across laps. Dynamics ṡ = p_v then stays smooth — no wrap
        # discontinuity ever reaches the QP, so no lap-rollover MINSTEP and
        # no cost spike at lap boundaries.
        sensor_s = float(initial_state[IDX_S])
        if self._last_sensor_s is not None:
            d = sensor_s - self._last_sensor_s
            if d < -L / 2.0:
                self.lap_count += 1   # forward wrap (typical: 86 → 0)
            elif d > L / 2.0:
                self.lap_count -= 1   # reverse (rare: backward sensor jump)
        self._last_sensor_s = sensor_s
        initial_state[IDX_S] = sensor_s + self.lap_count * L
        s0 = float(initial_state[IDX_S])

        # ---- ψ unwrap via persistent offset (same trick as lap_count for s) ----
        # Sensor yaw ∈ [−π, π]; we accumulate ±2π each time the sensor
        # wraps and pass `sensor_yaw + yaw_offset` to the solver. The
        # solver's internal ψ then grows monotonically — no wrap event
        # ever reaches the QP, no multi-iter SQP fixup needed, no
        # repeated wrap-detection at every cycle in the wrap zone (which
        # was the underlying cause of the 974-cost spike at s≈56 every
        # lap). Cost (yaw_err via sin/cos identities) and dynamics (cos/
        # sin of ψ) are 2π-periodic, so unbounded ψ is harmless.
        sensor_yaw = float(initial_state[2])
        if self._last_sensor_yaw is not None:
            d_yaw = sensor_yaw - self._last_sensor_yaw
            if d_yaw < -math.pi:
                self._yaw_offset += 2.0 * math.pi
            elif d_yaw > math.pi:
                self._yaw_offset -= 2.0 * math.pi
        self._last_sensor_yaw = sensor_yaw
        initial_state[2] = sensor_yaw + self._yaw_offset
        psi_unwrapped = False  # legacy flag — nothing to do, kept for the
                               # multi-iter trigger below (always False now)

        # First-cycle warm start — dynamics-feasible forward rollout from
        # the actual car state. Previously we placed X0[1..N] on the
        # centerline regardless of where the car was, so the X0[0]→X0[1]
        # jump violated dynamics by up to ~0.5 m laterally. acados then
        # could not make any progress and reported ACADOS_MINSTEP at QP
        # iter 1. Rolling forward with the same controls used in u0 keeps
        # X0[k+1] = X0[k] + dt·f(X0[k], u_k), satisfying dynamics by
        # construction.
        if not self.WARM_START:
            # Cold-start path fires after stuck/spike fallback or obstacle
            # commit/release. solver.reset() here is critical for the
            # stuck case: stuck override sets u_seq=zeros, the solver's
            # dual variables then carry a bias "u=0 minimizes the QP",
            # and the next cycle's 1-iter SQP_RTI can't escape — the car
            # stays at v=0 forever even though cold-rollout pushes a
            # reasonable u0=seed_v warm-start. Reset zeros the duals so
            # the rollout actually drives the solution. Note: ψ-wrap
            # doesn't go through this path (it sets X0 directly without
            # WARM_START=False), so the Plan-B "no reset on wrap" stays.
            try:
                self.solver.reset()
            except Exception:
                pass
            try:
                seed_v = max(float(self.ref_v(s0 % L)) * 0.5, 1.0)
            except Exception:
                seed_v = max(self.v_max * 0.4, 1.0)
            self.u0 = np.zeros((self.N, self.n_controls))
            # Warm-start (unified 8-state, both modes): u = [a_x, δ, p_v].
            # Hold a_x = 0 (no acceleration during warm-start) so vx stays
            # at the car's current velocity. No ramp = no input bound risk.
            # Solver's first iter then finds the correct a_x. p_v set to
            # seed_v so s grows during warm-start. Skip slip dynamics —
            # vy, r held constant so X0 stays well-conditioned.
            self.u0[:, 0] = 0.0      # a_x (no acceleration in seed)
            self.u0[:, 1] = 0.0      # delta
            self.u0[:, 2] = seed_v   # p_v (input, can be set free)
            self.X0[0, :self.n_states] = initial_state
            if self._lmpc_joint:
                self.X0[0, self.n_states:] = 1.0 / max(1, self._nx_solver - self.n_states)  # uniform α seed
            for k in range(self.N):
                xk = self.X0[k, :]
                uk = self.u0[k, :]
                delta_, p_ = uk[1], uk[2]
                psi_ = xk[2]
                vx_  = xk[3]            # vx stays constant in seed
                delta_prev_ = xk[7]
                dx_dt   = vx_ * np.cos(psi_)
                dy_dt   = vx_ * np.sin(psi_)
                dpsi_dt = (vx_ / self.L) * np.tan(delta_)
                ds_dt   = p_
                ddprev  = (delta_ - delta_prev_) / self.dT
                self.X0[k + 1, :self.n_states] = xk[:self.n_states] + self.dT * np.array(
                    [dx_dt, dy_dt, dpsi_dt, 0.0, 0.0, 0.0,
                     ds_dt, ddprev])
                self.X0[k + 1, self.n_states:] = self.X0[k, self.n_states:]  # α constant over horizon
            for k in range(self.N + 1):
                self.solver.set(k, "x", self.X0[k, :])
            for k in range(self.N):
                self.solver.set(k, "u", self.u0[k, :])
            self.WARM_START = True
            # Flag for multi-iter SQP — solver.reset() above zeroed dual
            # variables; without this flag the main solve below runs only
            # 1 SQP_RTI iteration which can't rebuild the dual active set
            # → returns near-zero (or otherwise corrupt) controls →
            # stuck-at-zero loop after every stuck recovery / fallback.
            self._just_cold_started = True

        # (No s-wrap fixup needed — solver-internal s is monotonic.)

        # ---- Per-cycle parameters ----
        sel = self.select_front_obstacle(initial_state[0], initial_state[1],
                                         initial_state[2], obstacles)
        self.dbg_n_obs_input = len(obstacles) if obstacles is not None else 0
        self.dbg_sel_dmin, self.dbg_sel_x, self.dbg_sel_y = sel

        # ---- Commit-once obstacle / side decision ----
        # Once we engage an obstacle, FREEZE the chosen avoidance side and
        # the obstacle Frenet coords until the car has driven past it
        # longitudinally. Without this, decide_side_pref / select_front_obs
        # may flip mid-pass (e.g., as the car turns into the obstacle the
        # corridor sample distances change), and the half-plane direction
        # would suddenly reverse → prediction "bursts" the other way.
        # Stored: (ox, oy, s_obs, side_pref, e_c_obs).
        sel_is_real = sel[1] < 1e5
        if not hasattr(self, '_committed_obs'):
            self._committed_obs = None

        # Release current commitment if (a) car drove past it in s, (b) it
        # is no longer in the obstacle list, or (c) we drifted very far.
        if self._committed_obs is not None:
            cox, coy, cs_obs, cside, ce_c = self._committed_obs
            s_car = float(initial_state[IDX_S]) % L
            delta_s = (s_car - cs_obs) % L
            car_to_obs = math.hypot(initial_state[0] - cox,
                                     initial_state[1] - coy)
            obs_still_present = False
            if obstacles is not None and len(obstacles) > 0:
                obs_still_present = any(
                    (op[0] - cox) ** 2 + (op[1] - coy) ** 2 < 0.09  # 0.3 m
                    for op in obstacles)
            if 1.5 < delta_s < 0.5 * L:
                self._log.info("[MPC] passed committed obs (Δs=%.2f m) — release", delta_s)
                self._committed_obs = None
                self.WARM_START = False
            elif not obs_still_present:
                self._log.info("[MPC] committed obs removed — release")
                self._committed_obs = None
                self.WARM_START = False
            elif car_to_obs > 12.0:
                # Drifted way past; safety release.
                self._committed_obs = None

        # Track NEW commit this cycle so we can:
        #   (a) push a stronger detour prior into X0 (option 2)
        #   (b) run extra SQP iters before the main solve (option 3)
        just_committed = False

        # Side-decision cache (hysteresis). Maps obstacle position →
        # last committed side. Re-using the same side for the same
        # obstacle prevents the "sometimes goes up, sometimes down" flip
        # symptom — once a side is chosen for an obstacle, stick with it
        # for all subsequent passes.
        if not hasattr(self, '_side_history'):
            self._side_history = {}

        # Commit on first engagement. Trigger distance is rqt-tunable
        # via self.commit_dist_live (default 10 m). Larger = engage
        # detour earlier (smoother but more conservative); smaller =
        # late commit (snappy but riskier).
        if (self._committed_obs is None and sel_is_real
                and float(sel[0]) < self.commit_dist_live):
            s_obs_new, e_c_obs_new = self._obstacle_frenet(sel[1], sel[2])
            s_car_now = float(initial_state[IDX_S]) % L
            delta_s_now = (s_car_now - s_obs_new) % L
            # In-front zone (see prior comment block).
            obs_is_ahead = (delta_s_now < 1.5) or (delta_s_now > 0.5 * L)
            if obs_is_ahead:
                # Cache key: obstacle position to 0.1 m precision so
                # near-identical obstacles map to same side.
                key = (round(float(sel[1]), 1), round(float(sel[2]), 1))
                if key in self._side_history:
                    sp_new = self._side_history[key]
                    self._log.info("[MPC] reusing cached side=%+d for obs at %s", sp_new, key)
                else:
                    sp_new = self.decide_side_pref(s_obs_new, e_c_obs_new)
                    if sp_new != 0:
                        self._side_history[key] = sp_new
                if sp_new != 0:
                    self._committed_obs = (
                        float(sel[1]), float(sel[2]),
                        s_obs_new, int(sp_new), e_c_obs_new)
                    self._log.info(
                        "[MPC] committed obs=(%.2f,%.2f) s_obs=%.2f e_c_obs=%+.2f side=%+d",
                        sel[1], sel[2], s_obs_new, e_c_obs_new, sp_new)
                    self.WARM_START = False  # rebuild with frozen side
                    just_committed = True

        # Use committed values; if no commitment, fall back to sentinel.
        if self._committed_obs is not None:
            cox, coy, cs_obs, cside, ce_c = self._committed_obs
            sel = [math.hypot(initial_state[0] - cox,
                              initial_state[1] - coy), cox, coy]
            side_pref = cside
            e_c_obs_val = ce_c
        else:
            sel = [1e6, 1e6, 1e6]
            side_pref = 0
            e_c_obs_val = 0.0
        self.dbg_side_pref = float(side_pref)
        self.dbg_sel_dmin, self.dbg_sel_x, self.dbg_sel_y = sel

        # Avoidance prior on warm-start X0 — Option 2: at the just-committed
        # cycle, push HARDER and FARTHER. Without this strong push, X0 is
        # the previous (no-obstacle) prediction; the half-plane constraint
        # is suddenly active and slack absorbs a 1-2 m violation → cost
        # spike. With the strong commit-time push, X0 already curves around
        # the obstacle, so the half-plane is near-satisfied from cycle one.
        # After commit (subsequent cycles), the regular gentler push
        # maintains the detour shape as the car approaches.
        if side_pref != 0:
            if just_committed:
                D_PRIOR     = 0.40   # bigger lateral kick at commit
                SIGMA_SQ    = 4.0    # σ=2.0 — wider Gaussian, more stages
                TRIGGER_D   = 8.0    # always push at commit
            else:
                D_PRIOR     = 0.30
                SIGMA_SQ    = 1.0
                TRIGGER_D   = 4.0
            if float(sel[0]) < TRIGGER_D:
                ox, oy = sel[1], sel[2]
                for k in range(1, self.N + 1):
                    xk, yk, psik = self.X0[k, 0], self.X0[k, 1], self.X0[k, 2]
                    d2k = (xk - ox) ** 2 + (yk - oy) ** 2
                    w = math.exp(-d2k / (2.0 * SIGMA_SQ))
                    nx_, ny_ = -math.sin(psik), math.cos(psik)
                    self.X0[k, 0] += side_pref * D_PRIOR * w * nx_
                    self.X0[k, 1] += side_pref * D_PRIOR * w * ny_
                for k in range(self.N + 1):
                    self.solver.set(k, "x", self.X0[k, :])

        # Per-stage parameter array
        # X0[k, IDX_S] is unbounded (solver-internal monotonic s), so always
        # wrap with % L for spline lookups (corridor / track boundary).
        _alat_eff = float(self.a_lat_safe_eff())   # 사이클당 1회 — stage 루프 불변
        for k in range(self.N + 1):
            sk = float(self.X0[k, IDX_S]) % L
            try:
                lx = float(self.left_lut_x(sk));  ly = float(self.left_lut_y(sk))
                rx = float(self.right_lut_x(sk)); ry = float(self.right_lut_y(sk))
            except Exception:
                lx = ly = rx = ry = 0.0
            # ── ★ 2026-06-02 per-stage HARD corner-speed cap (kinematic) ──
            # 진단: 곡률 soft-cost(ref_v) 와 slack 된 a_lat 제약은 q_p(progress)
            # 에 밀려 코너 진입속도를 강제 못 함 → R=0.87m 코너에 v=3~5 진입 →
            # 필요 a_lat=v²/κ⁻¹ ≫ 한계 → understeer 로 벽 클립/충돌(맵은 안 좁음).
            # → 각 예측 stage 의 속도 컨트롤 u[0] 상한을 v ≤ √(a_lat_safe/|κ(s_k)|)
            # 으로 HARD 제약. 직선(|κ|→0)은 v_max 그대로(빠름), 코너만 물리적
            # grip 속도로 강제(깨끗). PP 가 암묵적으로 하는 grip-limited racing.
            # |κ| 는 forward-max LUT(lookahead 6m) → 코너 6m 전부터 감속 시작.
            # s_k 는 이전 solve 의 warm-start 궤적(self.X0)에서 — 약간의 오차는
            # 보수적 floor 로 흡수. 후진(stuck-recover)은 mpc_node 가 cmd.speed
            # 로 직접 처리 → u[0] lbu=0 이므로 이 cap 과 무간섭.
            try:
                _absk = float(self.abs_kappa_lut(sk))
                _vcap = math.sqrt(_alat_eff / (_absk + 1e-3))
                # floor 1.0: κ-스파이크나 cold warm-start 에서 v→0 stall 방지
                # (가장 타이트한 코너도 √(6/0.84)=2.67 라 floor 는 평소 불활성).
                _vcap = min(float(self.v_max), max(_vcap, 1.0))
                # unified 8-state (both modes): vx 는 STATE idx 3. HARD ubx[vx]
                # cap 은 vx 가 못 따라잡으면(a_min=-3 제동한계) QP infeasible →
                # MINSTEP 28→95 폭증(2026-06-03 측정). 대신 GENEROUS margin(×1.6)
                # 으로 극단 과속만 막고, 코너속도는 soft ref_v+a_lat 가 담당.
                # 이래야 slip-aware 추종을 살리면서 MINSTEP 안 늘림.
                # k<N only: the terminal node has no state box (nbx_e=0), so
                # setting ubx there throws (was 61×/run log spam at k=N).
                if 1 <= k < self.N:
                    _vcap_dyn = min(float(self.v_max) + 0.5, _vcap * 1.6)
                    # joint-α: idxbx was extended to [3,4,5, 8..8+Ka], so the
                    # per-stage ubx MUST match that length or acados throws a
                    # dimension error (previously swallowed by the bare except →
                    # the per-stage corner speed cap was silently dead the entire
                    # time joint-α LMPC was on). Pad with the α upper bounds (1.0).
                    _ub = [_vcap_dyn, 10.0, 20.0]
                    if self._lmpc_joint:
                        _ub = _ub + [1.0] * int(getattr(self, '_lmpc_K_aug', 10))
                    self.solver.set(k, "ubx", np.array(_ub))
                # ── kinematic-only p_v grip cap (mode-dependent NUMERIC knob,
                # same family as the LM damping — NOT an algorithmic difference;
                # constraint STRUCTURE stays identical in both modes). f_kin has
                # no tire physics, so soft ref_v + slacked a_lat lose to the
                # progress reward at corner entry (2026-06-02 diagnosis; 2026-06-10
                # final2 bring-up: deterministic crash at the same 3 corners every
                # lap). The old 5-state kin capped the speed INPUT hard at ×1.0;
                # in the unified layout the speed-intent input is p_v (u[2]) —
                # cap it to the κ-grip speed. INPUT bound → always feasible
                # (a hard ubx[vx] ×1.0 measurably spiked MINSTEP 28→95), and the
                # ×1.6 ubx stays untouched in both modes. Dynamic mode skips
                # this: the tire model self-limits, validated clean at ×1.6.
                if (not self.use_dynamic) and k < self.N:
                    _ubu = self._ubu_codegen.copy()
                    _ubu[2] = min(_ubu[2], _vcap)
                    self.solver.set(k, "ubu", _ubu)
            except Exception as _e:
                # Was `pass` — that hid the joint-α ubx dimension bug for weeks.
                # Log (throttled) so a per-stage constraint failure is visible.
                self._log.warn_throttle(2.0,
                    "[MPC-acados] per-stage bound set failed (k=%d): %s", k, _e)
            # 2026-05-28 #18 LMPC: p_arr 길이 22 → 76. Reviewer #★1 confirmed —
            # codegen 길이가 76 이면 22 길이 set 은 dimension throw. p_arr 전체
            # 76 으로 짜되, LMPC slots (18..71) 은 attr 가 있으면 사용, 없으면 0
            # (default → lmpc_w=0 → LMPC term 0 → 기존 동작 보존).
            p_arr = np.zeros(self._n_p_total, dtype=float)
            # 0..17 — 기존 constants
            p_arr[0]  = float(sel[0])               # dmin (debug)
            p_arr[1]  = float(sel[1])               # obs_x
            p_arr[2]  = float(sel[2])               # obs_y
            p_arr[3]  = float(side_pref)
            p_arr[4]  = float(self.D_detour_live)
            p_arr[5]  = float(self.R_car_live)
            p_arr[6]  = float(self.q_cte_scale_live)
            p_arr[7]  = float(self.R_safe_live)
            p_arr[8]  = float(self.q_lag_scale_live)
            p_arr[9]  = e_c_obs_val
            p_arr[10] = _alat_eff
            p_arr[11] = float(self.D_apex_live)
            p_arr[12] = float(self.q_psi_scale_live)
            p_arr[13] = float(self.q_v_scale_live)
            p_arr[14] = float(self.q_dd_scale_live)
            p_arr[15] = float(self.q_p_scale_live)
            p_arr[16] = float(self.q_drate_scale_live)
            p_arr[17] = float(self.q_dv_scale_live)
            # 18..67 — LMPC SS slots (caller fills via self._lmpc_ss_states / _lmpc_ss_Q
            # 외부에서 set_lmpc_params() 로 갱신). 미설정 시 zeros + lmpc_w=0 으로 무시.
            ss_states = getattr(self, '_lmpc_ss_states', None)
            ss_Q      = getattr(self, '_lmpc_ss_Q', None)
            if ss_states is not None and ss_Q is not None:
                # ss_states (4, 10) F-order pack, ss_Q (10,)
                p_arr[18:58] = ss_states.flatten(order='F')  # column-major: [s0_x, s0_y, s0_psi, s0_vx, s1_x, ...]
                p_arr[58:68] = ss_Q
            else:
                # padding Q = 1e6 (reviewer: exp(-β·1e6) ≈ 0 자연 무시)
                p_arr[58:68] = 1e6
            # 68..71 — LMPC scalars (default OFF)
            # 2026-06-11: LMPC term cost gated OFF while committed to an
            # obstacle. Safe set 랩들은 장애물 없이 기록된 라인 — cost-to-go 가
            # 터미널 상태를 장애물 관통 라인으로 끌어당겨 half-plane 회피와
            # 싸운다 (관측: LMPC on + 회피 중 wedge/stuck). side_pref!=0 인
            # 동안만 0; 통과 후 release 되면 원래 weight 복귀.
            _lmpc_w = float(getattr(self, 'lmpc_w_live', 0.0))
            if side_pref != 0:
                _lmpc_w = 0.0
            p_arr[68] = _lmpc_w
            p_arr[69] = float(getattr(self, 'lmpc_alpha_live', 1.0))
            p_arr[70] = float(getattr(self, 'lmpc_beta_live', 0.05))  # reviewer 권장
            p_arr[71] = float(getattr(self, 'lmpc_reg_w_live', 0.001))
            # 72..75 — corridor (per-stage)
            p_arr[72] = lx; p_arr[73] = ly
            p_arr[74] = rx; p_arr[75] = ry
            # 76..78 — B4' e_corr (const across horizon)
            p_arr[76] = float(self._e_corr[0])
            p_arr[77] = float(self._e_corr[1])
            p_arr[78] = float(self._e_corr[2])
            self.solver.set(k, "p", p_arr)

        # ---- Stage 0 init state via tightened bounds ----
        self.solver.set(0, "lbx", initial_state)
        self.solver.set(0, "ubx", initial_state)

        # Option 3: at "transient" cycles (new commit, ψ-wrap, just-cold-
        # started after a stuck/spike fallback), the cost surface and/or
        # dual active set change abruptly. SQP_RTI's single iter cannot
        # converge duals from cold (or near-cold) state on those cycles
        # → corrupt control output (near-zero, saturated, or oscillating).
        # Run 2 extra solve() calls before the main solve so SQP has
        # converged. Pure cycles (~99%) skip and do single-iter as before.
        # Cost: ~6 ms extra on transient cycles, well within 25 ms budget.
        # Critical for breaking the stuck-at-zero loop: stuck override
        # produces u=0; next cycle's cold-start reset() zeros duals; if
        # we don't multi-iter here, 1-iter from cold dual returns ~0
        # again → infinite stuck.
        cold_started = getattr(self, '_just_cold_started', False)
        # Multi-iter pre-solve ONLY at transient events (commit, ψ-wrap,
        # cold-start). Per-cycle multi-iter caused oscillating predictions
        # because the second iter found a slightly different optimum
        # than the first → trajectory zigzag visible in RViz.
        if just_committed or psi_unwrapped or cold_started:
            for _ in range(2):
                try:
                    self.solver.solve()
                except Exception:
                    break
        self._just_cold_started = False

        # ---- Solve ----
        try:
            status = self.solver.solve()
            self.dbg_solver_status = self._status_to_string(status)
            traj = np.array([self.solver.get(k, "x") for k in range(self.N + 1)])
            u_seq = np.array([self.solver.get(k, "u") for k in range(self.N)])
            # status: 0=OK, 1=NaN/Failure, 2=MaxIter, 3=MinStep, 4=QP_Failure.
            # 3/4 at the hairpin produces a corrupt warm-start that locks
            # subsequent QPs into a 36-iter limit cycle (observed s≈48m
            # cascade). Forcing WARM_START=False rebuilds X0/u0 from the
            # current car state via Euler rollout next cycle.
            # 2026-06-08: include 2 (MaxIter) — a QP that hit the iteration limit
            # without converging was being accepted as a good solve, applying a
            # half-converged control and seeding a bad warm-start.
            bad_status = status in (1, 2, 3, 4)
            has_nan = np.isnan(traj).any() or np.isnan(u_seq).any()
            if bad_status or has_nan:
                self._log.warn_throttle(1.0,
                    "[MPC-acados] status=%s nan=%s — reset warm-start",
                    self.dbg_solver_status, has_nan)
                self.WARM_START = False
                # Safe fallback control: gentle slow-down, hold heading.
                # Avoids feeding the ROS node a corrupt v_cmd while next
                # cycle re-seeds.
                try:
                    seed_v = max(float(self.ref_v(s0 % L)) * 0.3, 0.5)
                except Exception:
                    seed_v = 1.0
                # joint-α: solver expects _nx_solver-wide x (8 phys + K α). A bare
                # 8-wide tile here makes self.X0 8-wide → next cycle's set(k,"x")
                # raises "dimension 18 vs 8". Build at solver width, α uniform.
                _nxs = getattr(self, '_nx_solver', self.n_states)
                traj = np.zeros((self.N + 1, _nxs))
                traj[:, :self.n_states] = initial_state
                if _nxs > self.n_states:
                    traj[:, self.n_states:] = 1.0 / max(1, _nxs - self.n_states)
                u_seq = np.zeros((self.N, self.n_controls))
                # u[0] = a_x (unified, both modes) — small positive accel to
                # keep the car coasting forward. p_v = seed_v.
                u_seq[:, 0] = 0.5
                # Also seed predicted vx so the output speed is sensible.
                traj[:, 3] = seed_v
                u_seq[:, 2] = seed_v
        except Exception as e:
            self._log.warn_throttle(2.0, "[MPC-acados] solver exception %s — reset", str(e))
            self.WARM_START = False
            _nxs = getattr(self, '_nx_solver', self.n_states)
            traj = np.zeros((self.N + 1, _nxs))
            traj[:, :self.n_states] = initial_state
            if _nxs > self.n_states:
                traj[:, self.n_states:] = 1.0 / max(1, _nxs - self.n_states)
            u_seq = np.zeros((self.N, self.n_controls))
            self.dbg_solver_status = "exception"

        try:
            opti_value = float(self.solver.get_cost())
        except Exception:
            opti_value = float('nan')

        # Cost-spike fallback — when the QP "succeeded" but cost is far
        # above the steady-state level, the solution is untrustworthy
        # (typically saturated v_max + steer=±0.4 from a corrupt warm-
        # start at obstacle commit / ψ-wrap). Output gentle ref_v·0.3
        # straight forward + warm-start rebuild.
        cost_spike = ((not np.isnan(opti_value))
                      and opti_value > self.cost_spike_thr_live)

        # Stuck detection (option D) — cost-spike fallback alone misses
        # the case where the QP returns saturated controls with cost
        # below threshold (observed: cost ≈ 270, vcmd=6, steer=+0.4 at
        # s=60 lap 4, car wedged into wall, fallback never engages).
        # Estimate v_actual from sensed position diff; if it stays below
        # 0.1 m/s while we keep commanding > 2 m/s for several cycles,
        # the car is physically stuck even though the optimizer thinks
        # everything is fine. Override with v=0/steer=0 to release the
        # wall contact, then rebuild warm-start next cycle.
        now_t = monotonic_now()
        v_est = 0.0
        # disp = raw per-cycle displacement [m]. v_est(=disp/dtm) 는 wall-clock dtm
        # 노이즈(timing hiccup)로 작은 disp 인데도 작게 나와 stuck 오발동(2026-05-29
        # forensic: 2/23 false-fire). disp 자체로도 게이트 → 진짜 위치동결만 stuck.
        # prior pos 없으면 1e9 (첫 cycle 미발동).
        disp = 1e9
        if hasattr(self, '_pos_for_v') and self._pos_for_v is not None:
            px, py, pt = self._pos_for_v
            dtm = max(now_t - pt, 1e-3)
            disp = math.hypot(initial_state[0] - px, initial_state[1] - py)
            v_est = disp / dtm
        self._pos_for_v = (float(initial_state[0]),
                            float(initial_state[1]), now_t)
        # Agent R-round3 Fix 1: persistent STUCK release for ~0.5s (20 cycles)
        # with reverse cmd. 이전 1-cycle release 는 cmd=0 후 다음 cycle 의
        # _v_cmd_for_stuck=0 → count reset → release 종료 → 다시 forward cmd →
        # wall contact 안 풀려 infinite loop. release_remaining 카운터로 강제 지속.
        v_cmd_prev = getattr(self, '_v_cmd_for_stuck', 0.0)
        self._stuck_release_remaining = getattr(self, '_stuck_release_remaining', 0)
        # 2026-05-27: strict `> 2.0` 가 fallback 의 seed_v=2.0 과 정확히 일치 → 미발동.
        # 1.5 로 완화: fallback v_floor (=2.0) 보다 낮으면 발동.
        # 2026-05-28 #15: v=6 sim 박힘 시 MPC "자포자기 모드" 진입 → vcmd=0.02 같이 작은 양수 →
        # `> 1.5` 미발동. STUCK 감지 못함 → 영원히 박힘 (1시간 누적 lap_count=144 false).
        # → `> 0.0` 으로 완화 (의도 cmd=0 정확과 자포자기 vcmd≠0 구분).
        # 2026-05-29 startup/respawn grace: stuck-release 의 후진(-0.5)이 출발·리스폰
        # 직후 오발동 (정지→가속 빈 구간을 박힘으로 착각 → "살짝 뒤로 감"). 차가 한 번
        # 이라도 실제로 움직인 뒤(_has_moved)에만 stuck 판정. 리스폰 점프(v_est 가
        # 차 최대속도보다 비현실적으로 큼)는 위치 텔레포트 → latch 리셋 + 그 cycle 무시.
        teleport = v_est > 20.0
        if teleport:
            self._has_moved = False
            # respawn/teleport mid-release: abort the reverse so the car does
            # not creep backward right after being moved to a fresh pose.
            self._stuck_release_remaining = 0
            self._stuck_origin = None
        elif v_est > 0.5:
            self._has_moved = True
        if (v_est < 0.1 and disp < 0.02 and v_cmd_prev > 0.0
                and getattr(self, '_has_moved', False)
                and self._stuck_release_remaining == 0):
            self._stuck_count = getattr(self, '_stuck_count', 0) + 1
        else:
            self._stuck_count = 0
        if self._stuck_count > 10:                  # ~0.25 s detect
            self._stuck_release_remaining = 20      # ~0.5 s forced release (ceiling)
            self._stuck_count = 0
            # Remember where we wedged so the reverse can stop as soon as the
            # car has backed away STUCK_REVERSE_DIST (early-exit below).
            self._stuck_origin = (float(initial_state[0]), float(initial_state[1]))
            # 2026-06-11 failure-driven side flip: STUCK while committed to an
            # obstacle (and near it) = the cached side choice FAILED. Without
            # this, _side_history replays the same losing side every lap
            # (관측: 같은 장애물에서 매 랩 100% stuck→reverse 반복). Flip the
            # cached side and release — the commit logic re-engages next cycle
            # with the opposite side, so the car retries the other way within
            # the same pass.
            if getattr(self, '_committed_obs', None) is not None:
                cox, coy, cs_obs, cside, ce_c = self._committed_obs
                d_obs = math.hypot(initial_state[0] - cox,
                                   initial_state[1] - coy)
                if d_obs < 5.0:
                    key = (round(cox, 1), round(coy, 1))
                    self._side_history[key] = -cside
                    self._committed_obs = None
                    self.WARM_START = False
                    self._log.warn(
                        "[MPC] STUCK during avoidance (d_obs=%.1f m) — "
                        "flip side %+d→%+d for obs %s", d_obs, cside, -cside, key)
        # Early-exit the forced reverse: once the car has separated from the
        # wedge point by STUCK_REVERSE_DIST, end the release immediately rather
        # than running the full 20-cycle countdown. Without this the car kept
        # creeping backward after it had already freed itself ("박힌 뒤 다시
        # 나오고 그 뒤에 살짝 더 뒤로 갔다가 출발"). The countdown stays as a
        # ceiling for the rare case the car is still jammed at that distance.
        if self._stuck_release_remaining > 0 and getattr(self, '_stuck_origin', None):
            ox, oy = self._stuck_origin
            back_dist = math.hypot(initial_state[0] - ox, initial_state[1] - oy)
            if back_dist > self.STUCK_REVERSE_DIST:
                self._stuck_release_remaining = 0
                self._stuck_origin = None
        is_stuck = self._stuck_release_remaining > 0
        if is_stuck:
            self._stuck_release_remaining -= 1
        self._stuck_release_active = False          # set True below if stuck

        if cost_spike or is_stuck:
            if is_stuck:
                self._log.warn_throttle(1.0,
                    "[MPC] STUCK release n=%d (v_est=%.2f, last_vcmd=%.2f) — reversing",
                    self._stuck_release_remaining, v_est, v_cmd_prev)
                u_seq = np.zeros((self.N, self.n_controls))
                u_seq[:, 0] = -1.5     # reverse a_x — back away from wall (unified)
                traj[:, 3] = -0.5      # reverse vx in trajectory
                self._stuck_release_active = True
                # Steer EMA reset so old zigzag steer doesn't persist into release
                if hasattr(self, '_steer_filt'):
                    self._steer_filt = 0.0
            else:
                self._log.warn_throttle(1.0,
                    "[MPC] cost %.0f > %.0f — safe fallback",
                    opti_value, self.cost_spike_thr_live)
                # 2026-05-28 fix: v_floor 가 항상 ~2 → v=6 운전 중 fallback 후
                # 차가 2 로 강제 감속 → 다음 cycle 또 cost spike → 무한 cycle.
                # v_max 의 절반 사용 → fallback 후 적당 속도 유지 → 곧 회복 가능.
                # v=6 → v_floor=3.0, v=8 → v_floor=4.0.
                v_floor = max(0.5 * self.v_max, v_est + 1.0)
                try:
                    seed_v = max(float(self.ref_v(s0 % L)) * 0.5, v_floor)
                except Exception:
                    seed_v = v_floor
                u_seq = np.zeros((self.N, self.n_controls))
                u_seq[:, 0] = 1.5          # a_x — stronger accel (unified)
                traj[:, 3] = seed_v        # set predicted vx for output
                u_seq[:, 1] = 0.0          # delta
                u_seq[:, 2] = seed_v       # p_v
            self.WARM_START = False
        # Stash this cycle's commanded v for next-cycle stuck-check.
        # Unified: ROS node sends a traj-derived vx as cmd.drive.speed, so the
        # stuck detector must compare actual v against THAT value, not
        # traj[1, 3] (which at vx≈1.7 + 0.04·0.5 ≈ 1.73 is always < 2.0 so the
        # detector never fires — Agent R2 fix 2026-05-26).
        self._v_cmd_for_stuck = float(traj[-1, 3]) if traj.shape[0] > 1 else 0.0

        # Steer-output EMA filter — corrects cycle-to-cycle steer trembling
        # observed at curves/hairpin (65% of cycles had steer sign flips:
        # +0.40 → −0.10 → +0.40 oscillating). Cost surface is ~flat over
        # multiple δ values that all achieve a similar predicted ψ change,
        # so the IPM picks different corners each cycle. Smoothing the
        # output via EMA absorbs the cycle-to-cycle jitter without
        # changing what the solver sees (warm-start is unaffected). Only
        # δ is filtered — v is left alone so braking and accel stay snappy.
        # α = 0.6 — moderate smoothing (60% new sample, 40% previous);
        # racing-friendly response while killing the +/− oscillation.
        # Agent R-round3 Fix 3: adaptive α — vx<1.5 (Pacejka linear singularity 영역)
        # 에서 cost surface 가 δ 에 대해 거의 flat → solver 가 ±0.15 zigzag pick.
        # 그 영역만 α 줄여 heavy smoothing. vx 충분 (≥1.5) 이면 alpha_steer_live 그대로.
        vx_now = float(initial_state[3])   # unified: vx = state[3] in both modes
        scale = min(1.0, max(0.2, vx_now / 1.5))    # vx=0 → 0.2, vx≥1.5 → 1.0
        alpha = self.alpha_steer_live * scale
        new_steer = float(u_seq[0, 1])
        prev_filt = getattr(self, '_steer_filt', new_steer)
        filt_steer = alpha * new_steer + (1.0 - alpha) * prev_filt
        # Defensive clip: EMA of two in-bound values is in-bound by induction,
        # but a numerical overshoot or a prev_filt seeded from a different
        # bound regime must never push the actuator past the physical limit.
        filt_steer = min(self.theta_max, max(self.theta_min, filt_steer))
        self._steer_filt = filt_steer
        u_seq[0, 1] = filt_steer

        # Track applied δ for next cycle's state-augmentation (delta_prev).
        # We use the FILTERED δ (what actually goes to the actuator) so the
        # rate-cost residual is consistent with the physical command stream.
        self._last_delta_applied = filt_steer

        self.X0 = traj
        self.u0 = u_seq

        # Visualize boundary along predicted s
        try:
            self._publish_boundary(traj)
        except Exception:
            pass

        # Return shapes that match IPOPT MPC: (first_control DM, traj, u_seq, opti)
        return ca.DM(u_seq[0, :]), traj, u_seq, opti_value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _status_to_string(status):
        return {
            0: 'Solve_Succeeded',
            1: 'Failure',
            2: 'Maximum_Iterations_Exceeded',
            3: 'Minimum_Step_Size_Reached',
            4: 'QP_Failure',
        }.get(status, f'status_{status}')

    def _publish_boundary(self, traj):
        """Visualize the EFFECTIVE drivable corridor (track boundary minus
        R_CAR margin), so the user sees what region the MPC actually
        permits the car to occupy. Lowering R_CAR via rqt → dots move
        outward (closer to wall); raising → dots move inward.
        """
        if self.boundary_hook is None:
            return
        right_pts, left_pts = self.get_path_constraints_points(traj)
        # Shift each boundary point toward the centerline by R_CAR. We
        # approximate the inward normal at boundary_i as (center_i - boundary_i)
        # normalized, where center_i is the centerline point at the same s.
        L = float(self.path_length)
        margin = float(self.R_car_live)
        IDX_S_LOCAL = 6   # unified 8-state layout (both modes)
        shifted = []
        for k in range(traj.shape[0] - 1):
            sk = float(traj[k + 1, IDX_S_LOCAL]) % L
            try:
                cx_s = float(self.center_lut_x(sk))
                cy_s = float(self.center_lut_y(sk))
            except Exception:
                cx_s = 0.0; cy_s = 0.0
            for px, py in (right_pts[k], left_pts[k]):
                vx = cx_s - px; vy = cy_s - py
                d  = math.hypot(vx, vy) + 1e-9
                shifted.append((px + margin * vx / d, py + margin * vy / d))
        self.boundary_hook(shifted)

    def heading(self, yaw):
        """yaw → (x, y, z, w) tuple. Caller wraps into geometry_msgs/Quaternion."""
        return yaw_to_quat(yaw)

    def init_mpc_start_conditions(self):
        self.X0 = np.zeros((self.N + 1, getattr(self, '_nx_solver', self.n_states)))
        self.u0 = np.zeros((self.N, self.n_controls))
