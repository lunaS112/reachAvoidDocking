"""
BRT-Based Optimal Controller for Docking13D

Two-phase control strategy using the learned DeepReach value function
for the 13-state spacecraft docking problem:
  - Phase 1 (Convergence): Outside BRT, use V(x, tMax) gradient to steer toward BRT.
  - Phase 2 (Precision):   Inside BRT, use time-varying V(x, t_remaining) for
                            optimal control with countdown timer.

The controller supports reversible phase transitions: if the chaser leaves
the BRT during Phase 2, an expanding-window time search attempts to find a
tighter BRT shell.  If that fails, it reverts to Phase 1.

Usage:
    controller = BRTController13D(
        checkpoint_path='./runs/Docking13D_RA/training/checkpoints/model_final.pth',
        tMax=14.0,
    )
    result = controller.simulate_docking(initial_state, max_sim_time=60.0)
"""

import time as _time

import torch
import numpy as np
import pickle
import os
import sys
import inspect

# Add project root to path (3 levels up: controllers -> utils -> project_root)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))

from utils import modules
from utils import diff_operators  # noqa: F401 — used transitively by io_to_dv
from dynamics import dynamics as dynamics_module
from utils.controllers.docking13d_mixin import Docking13DControllerMixin, _quat_mul_np, _quat_error_angle_np
from utils.controllers.safety_filter import SafetyFilter
from utils.controllers.min_time_search import find_min_brat_time_single, STATUS_HOLD


class BRTController13D(Docking13DControllerMixin):
    """Two-phase BRT-based optimal controller for 13D docking.

    Phase 1 (Convergence): V(x, tMax) > 0  — use gradient to approach BRT.
    Phase 2 (Precision):   V(x, tMax) <= 0 — use V(x, t_remaining) countdown.
    """

    def __init__(self, checkpoint_path, tMax=14.0, dt=0.1, device='cuda',
                 search_resolution=0.1, safety_filter=None,
                 safety_margin_phase1=0.1, safety_margin_phase2=0.02,
                 debug_phase2=False,
                 gradient_fallback=True, grad_threshold=0.01,
                 avoid_proximity_margin=1.0,
                 pd_torque_proximity=None):
        """
        Args:
            checkpoint_path:       Path to the trained model checkpoint.
            tMax:                  BRT time-horizon cap (seconds).
            dt:                    Control / integration timestep (seconds).
            device:                Torch device ('cuda' or 'cpu').
            search_resolution:     Time step for BRT time search (seconds).
            safety_margin_phase1:  Safety filter margin outside BRAT (Phase 1).
                                   Higher value triggers filter earlier.
            safety_margin_phase2:  Safety filter margin inside BRAT (Phase 2).
                                   Lower value for minimal intervention.
            debug_phase2:          Enable detailed Phase 2 search diagnostics.
            gradient_fallback:     Enable L2 goal-directed fallback when
                                   Phase 1 gradient stagnates (default True).
            grad_threshold:        Norm of dvds on control-coupled dims below
                                   which the gradient is considered stagnant.
            avoid_proximity_margin: Obstacle SDF distance (m) below which
                                   the fallback is hard-suppressed.
            pd_torque_proximity:   Multiplier for goal tolerances that defines
                                   the proximity region where PD torque
                                   replaces bang-bang torque.  None = disabled
                                   (pure bang-bang).  Float > 0 = enabled.
        """
        self.checkpoint_path = checkpoint_path
        self.tMax = tMax
        self.dt = dt
        self.device = device
        self.search_resolution = search_resolution
        self.debug_phase2 = debug_phase2
        self.safety_filter = safety_filter or SafetyFilter(mode=0)
        self.safety_margin_phase1 = safety_margin_phase1
        self.safety_margin_phase2 = safety_margin_phase2

        # Gradient fallback parameters (Phase 1 only)
        self.gradient_fallback = gradient_fallback
        self.grad_threshold = grad_threshold
        self.avoid_proximity_margin = avoid_proximity_margin

        # Proximity PD torque parameters
        self.pd_torque_proximity = pd_torque_proximity

        # Derive experiment directory from checkpoint path
        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        self._load_dynamics_and_model()

        # Derived fallback constants (PD gains for virtual gradient)
        self.fallback_kp_trans = 0.5 * self.dynamics.F_bar
        self.fallback_kd_trans = 1.0 * self.dynamics.F_bar
        self.fallback_kp_rot = 0.5 * self.dynamics.tau_bar
        self.fallback_kd_rot = 1.0 * self.dynamics.tau_bar
        self.goal_state_np = self.dynamics.goal_state.cpu().numpy()
        self.q_goal_np = self.dynamics.q_goal.detach().cpu().numpy()

        # Proportional torque scaling: when |omega_i| < _omega_scale,
        # BRAT torque magnitude is proportional to |omega_i| / _omega_scale.
        # Above _omega_scale, full bang-bang.  Threshold set at 10x the
        # tightest omega tolerance for a smooth transition zone.
        if self.pd_torque_proximity is not None:
            self._omega_scale = 10.0 * self.dynamics.eps_omega_pitchyaw  # ~0.0262 rad/s
            print(f"  [PD Torque] Enabled  proximity={self.pd_torque_proximity:.1f}x  "
                  f"omega_scale={self._omega_scale:.4f} rad/s")

        self.reset()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_dynamics_and_model(self):
        """Load dynamics class and SIREN model weights from checkpoint."""
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)

        # Dynamics
        dynamics_class = getattr(dynamics_module, self.orig_opt.dynamics_class)
        sig = inspect.signature(dynamics_class)
        kwargs = {}
        for param_name in sig.parameters:
            if param_name != 'self' and hasattr(self.orig_opt, param_name):
                kwargs[param_name] = getattr(self.orig_opt, param_name)
        self.dynamics = dynamics_class(**kwargs)
        self.dynamics.set_model(self.orig_opt.deepReach_model)

        if hasattr(self.orig_opt, 'state_range') and self.orig_opt.state_range is not None:
            self.dynamics.override_state_range(self.orig_opt.state_range)

        # Model
        self.model = modules.SingleBVPNet(
            in_features=self.dynamics.input_dim,
            out_features=1,
            type=self.orig_opt.model,
            mode=self.orig_opt.model_mode,
            final_layer_factor=1.,
            hidden_features=self.orig_opt.num_nl,
            num_hidden_layers=self.orig_opt.num_hl,
            periodic_transform_fn=self.dynamics.periodic_transform_fn,
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()

        # Store dynamics parameters used for control
        self.state_dim = self.dynamics.state_dim  # 13

        print(f"BRTController13D loaded | {self.dynamics.name} | "
              f"tMax={self.tMax}s  dt={self.dt}s")

    def reset(self):
        """Reset controller state for a new simulation."""
        self.in_brt_phase = False
        self.t_remaining = self.tMax
        self.phase_transition_time = None

        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.phase_history = []
        self.t_remaining_history = []
        self.sim_time_history = []

        self.brt_reacquisition_count = 0
        self.brt_time_adjustments = []
        self.phase2_debug_log = []

        # Gradient fallback tracking (Phase 1)
        self.fallback_weight_history = []

        # --- Diagnostic tracking ---
        self.diagnostic_history = []
        self._consecutive_v_increases = 0

        # Proportional torque tracking
        self.pd_torque_active_history = []
        self._pd_latched = False  # hysteresis latch

        self.safety_filter.reset()

    # ------------------------------------------------------------------
    # Value-function queries (dynamics-agnostic)
    # ------------------------------------------------------------------

    def get_value(self, state, time_query):
        """Query the scalar value V(x, t)."""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        time_tensor = torch.tensor([[time_query]], dtype=torch.float32,
                                   device=self.device)
        coord = torch.cat([time_tensor, state], dim=-1)
        model_input = self.dynamics.coord_to_input(coord)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze()

        return self.dynamics.io_to_value(model_input, output).item()

    def get_gradient(self, state, time_query):
        """Query spatial gradient dV/ds (length state_dim)."""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        time_tensor = torch.tensor([[time_query]], dtype=torch.float32,
                                   device=self.device)
        coord = torch.cat([time_tensor, state], dim=-1)
        model_input = self.dynamics.coord_to_input(coord)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze()
        model_in = result['model_in']

        dv = self.dynamics.io_to_dv(model_in, output)
        # dv shape: (1, 1+state_dim) -> [dvdt, dvds_0, ..., dvds_12]
        dvds = dv[0, 1:].detach().cpu().numpy()
        return dvds

    def get_values_batch(self, state, times):
        """Query V(x, t_i) for a single state at multiple times."""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 2:
            state = state.squeeze(0)

        n = len(times)
        time_tensor = torch.tensor(
            times, dtype=torch.float32, device=self.device).unsqueeze(-1)
        state_batch = state.unsqueeze(0).expand(n, -1)
        coords = torch.cat([time_tensor, state_batch], dim=-1)

        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        values = self.dynamics.io_to_value(model_input, output)
        return values.cpu().numpy()

    def get_values_batch_states(self, states, time):
        """
        Query V(x, t) for multiple states at a single fixed time in one forward pass.

        Args:
            states: (N, state_dim) numpy array or torch tensor of state vectors
            time: scalar time value to query

        Returns:
            numpy array of shape (N,): V(x_i, t) for each state
        """
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        states = states.to(self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        n = states.shape[0]
        time_col = torch.full((n, 1), time, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, states], dim=-1)          # (N, 1+state_dim)

        model_input = self.dynamics.coord_to_input(coords)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)            # (N,)

        values = self.dynamics.io_to_value(model_input, output)
        return values.cpu().numpy()

    # ------------------------------------------------------------------
    # Optimal control (13D-specific)
    # ------------------------------------------------------------------

    def get_optimal_control(self, state, time_query):
        """Compute bang-bang optimal control from value-function gradient.

        Uses the rotation-aware allocation from the mixin.

        Returns
        -------
        control : numpy (6,)
        dvds    : numpy (state_dim,) — raw spatial gradient (for diagnostics).
        """
        dvds = self.get_gradient(state, time_query)
        control = self._compute_brt_control_13d(dvds, state)
        return control, dvds

    def is_in_brt(self, state):
        """True if V(state, tMax) <= 0."""
        return self.get_value(state, self.tMax) <= 0

    # ------------------------------------------------------------------
    # Gradient-fallback helpers (Phase 1 only)
    # ------------------------------------------------------------------

    def _compute_l2_virtual_gradient(self, state):
        """PD-like virtual gradient pointing toward the goal set.

        Only populates control-coupled indices:
          [3:6]   — velocities (LVLH frame, matching real gradient convention)
          [10:13] — angular velocities (body frame)

        Translation uses position/velocity error in LVLH.
        Rotation uses quaternion error converted to a rotation vector.
        """
        goal = self.goal_state_np

        # --- Translation (LVLH frame) ---
        pos_err = state[0:3] - goal[0:3]
        vel_err = state[3:6] - goal[3:6]

        # --- Rotation (quaternion error → rotation vector) ---
        q = np.asarray(state[6:10], dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        # Error quaternion: q_err = q_goal_conj ⊗ q
        q_goal_conj = np.array([self.q_goal_np[0],
                                -self.q_goal_np[1],
                                -self.q_goal_np[2],
                                -self.q_goal_np[3]])
        q_err = _quat_mul_np(q_goal_conj, q)

        # Shortest path: ensure scalar part >= 0
        if q_err[0] < 0:
            q_err = -q_err

        # Rotation vector (body frame): 2 * q_err_vec / |q_err_scalar|
        qv_norm = np.linalg.norm(q_err[1:4])
        if abs(q_err[0]) > 1e-12:
            err_vec = 2.0 * q_err[1:4] * np.arctan2(qv_norm, q_err[0]) / max(qv_norm, 1e-12)
        else:
            err_vec = 2.0 * q_err[1:4]

        omega_err = state[10:13] - goal[10:13]

        # --- Assemble virtual gradient (same shape as dvds) ---
        vg = np.zeros(self.state_dim)
        vg[3:6] = self.fallback_kp_trans * pos_err + self.fallback_kd_trans * vel_err
        vg[10:13] = self.fallback_kp_rot * err_vec + self.fallback_kd_rot * omega_err
        return vg

    def _avoid_proximity_check(self, state):
        """Return True if state is too close to the obstacle for fallback."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            return float(self.dynamics.avoid_fn(s)) < self.avoid_proximity_margin

    # ------------------------------------------------------------------
    # Proximity PD torque
    # ------------------------------------------------------------------

    def _in_proximity_zone(self, state, mult):
        """Check if state is within mult * tolerances for pos/vel/quat."""
        d = self.dynamics
        pos = state[:3]
        vel = state[3:6]
        q = state[6:10]

        # Position: |x|, |z| < mult * eps_p, y in expanded goal band
        if abs(pos[0]) > mult * d.eps_p:
            return False
        if abs(pos[2]) > mult * d.eps_p:
            return False
        y_margin = mult * (d.goal_y_max - d.goal_y_min) / 2.0
        if not (d.goal_y_min - y_margin <= pos[1] <= d.goal_y_max + y_margin):
            return False

        # Velocity: lateral and axial within mult * tolerance
        if np.sqrt(vel[0]**2 + vel[2]**2) > mult * d.eps_v_lateral:
            return False
        vy_center = (d.eps_v_axial_lo + d.eps_v_axial_hi) / 2.0
        vy_half = (d.eps_v_axial_hi - d.eps_v_axial_lo) / 2.0
        if abs(vel[1] - vy_center) > mult * vy_half:
            return False

        # Quaternion: angle error < mult * eps_q
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm
        q_err = _quat_error_angle_np(q, self.q_goal_np)
        if q_err > mult * d.eps_q:
            return False

        return True

    def _check_pd_proximity(self, state):
        """Check if PD torque should be active, with hysteresis.

        Activates at ``pd_torque_proximity`` * tolerances.
        Once latched on, stays active until state leaves a wider zone
        (``2 * pd_torque_proximity``) to prevent flickering.
        """
        mult = self.pd_torque_proximity
        if mult is None:
            return False

        if not self._pd_latched:
            # Not yet latched — check activation threshold
            if self._in_proximity_zone(state, mult):
                self._pd_latched = True
                return True
            return False
        else:
            # Already latched — only deactivate if far outside (2x the zone)
            if not self._in_proximity_zone(state, 2.0 * mult):
                self._pd_latched = False
                return False
            return True

    def _compute_proportional_torque(self, state, dvds):
        """Scale BRAT bang-bang torque proportionally near zero omega.

        Preserves the BRAT gradient direction (which accounts for coupled
        attitude-translation dynamics) but reduces magnitude as omega
        approaches zero, preventing chatter.

        When |omega_i| > omega_scale: full bang-bang (±tau_bar)
        When |omega_i| < omega_scale: tau = sign(coeff) * tau_bar * |omega_i| / omega_scale

        Returns numpy (3,) torque vector.
        """
        omega = state[10:13]

        # Get BRAT gradient direction (same as _compute_brt_control_13d)
        I_np = self.dynamics.I.detach().cpu().numpy()
        p_omega = np.asarray(dvds[10:13], dtype=np.float64)
        coeff_tau = np.linalg.solve(I_np.T, p_omega)

        # Per-axis proportional scaling
        tau = np.zeros(3)
        for i in range(3):
            bang_dir = -1.0 if coeff_tau[i] > 0 else 1.0
            scale = min(abs(omega[i]) / self._omega_scale, 1.0)
            tau[i] = bang_dir * self.dynamics.tau_bar * scale

        return tau

    # ------------------------------------------------------------------
    # BRAT min-time search (delegates to shared utility)
    # ------------------------------------------------------------------

    def _search_brat_time(self, state):
        """Find the minimum time t* where V(x, t*) <= 0.

        Delegates to the shared find_min_brat_time_single utility which
        always returns a valid (t_star, status) — no Phase 1 fallback needed.
        """
        value_fn = lambda times: self.get_values_batch(state, times)
        if self.debug_phase2:
            return find_min_brat_time_single(
                value_fn, self.tMax,
                resolution=self.search_resolution,
                return_details=True,
                t_remaining=self.t_remaining)
        return find_min_brat_time_single(
            value_fn, self.tMax,
            resolution=self.search_resolution,
            t_remaining=self.t_remaining)

    def _record_phase2_debug(self, sim_time, state, search_details,
                             t_star, status, query_time, value, dvds,
                             raw_control, applied_control):
        """Persist high-signal Phase 2 diagnostics for postmortem analysis."""
        entry = {
            'sim_time': float(sim_time),
            'state': np.asarray(state, dtype=np.float64).tolist(),
            'value_tmax': float(self.get_value(state, self.tMax)),
            'selected_t_star': float(t_star),
            'selected_status': status,
            'query_time': float(query_time),
            'value_at_query': float(value),
            'gradient_dvds': np.asarray(dvds, dtype=np.float64).tolist(),
            'raw_control': np.asarray(raw_control, dtype=np.float64).tolist(),
            'applied_control': np.asarray(applied_control, dtype=np.float64).tolist(),
            'safety_filter_modified_control': bool(
                not np.allclose(raw_control, applied_control)),
            'search': search_details,
        }
        self.phase2_debug_log.append(entry)

        print(
            f"[BRAT Phase2] t={sim_time:6.2f}s  t*={t_star:5.2f}s ({status})  "
            f"V(t*)={value: .4f}  V(tMax)={entry['value_tmax']: .4f}  "
            f"neg={search_details['n_nonpositive']:3d}  "
            f"u_raw={np.array2string(raw_control, precision=3)}"
            f"  u_applied={np.array2string(applied_control, precision=3)}"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _check_state_bounds(self, state):
        """Compare each state dim against dynamics.state_range_.

        Returns
        -------
        dict with:
            out_of_domain : bool — True if any dimension is outside range.
            violations    : list of (dim_idx, value, lo, hi) tuples.
            max_violation_ratio : float — worst |overshoot| / half_range.
        """
        sr = self.dynamics.state_range_.cpu().numpy()  # (13, 2)
        violations = []
        max_ratio = 0.0
        dim_names = ['x', 'y', 'z', 'vx', 'vy', 'vz',
                     'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz']
        for i in range(len(state)):
            lo, hi = float(sr[i, 0]), float(sr[i, 1])
            val = float(state[i])
            half = (hi - lo) / 2.0
            if val < lo or val > hi:
                overshoot = max(lo - val, val - hi)
                ratio = overshoot / (half + 1e-12)
                violations.append({
                    'dim': i,
                    'name': dim_names[i] if i < len(dim_names) else f'd{i}',
                    'value': val,
                    'lo': lo,
                    'hi': hi,
                    'overshoot': overshoot,
                    'ratio': ratio,
                })
                max_ratio = max(max_ratio, ratio)
        return {
            'out_of_domain': len(violations) > 0,
            'violations': violations,
            'max_violation_ratio': max_ratio,
        }

    # ------------------------------------------------------------------
    # Two-phase control law
    # ------------------------------------------------------------------

    def u_fn(self, state, sim_time):
        """Compute one control step with two-phase logic + history tracking."""
        # Phase transition detection
        if not self.in_brt_phase:
            if self.is_in_brt(state):
                self.in_brt_phase = True
                self.phase_transition_time = sim_time
                print(f"  [BRT13D] Entered BRAT at t={sim_time:.2f}s -> Phase 2")

        if self.in_brt_phase:
            # Phase 2: per-step min-time query (always returns a valid t*)
            search_result = self._search_brat_time(state)
            if self.debug_phase2:
                t_star, status, search_details = search_result
            else:
                t_star, status = search_result
                search_details = None
            # STATUS_HOLD: count down; otherwise accept the searched t*
            if status == STATUS_HOLD:
                self.t_remaining = max(t_star - self.dt, 0.01)
            else:
                self.t_remaining = t_star
            query_time = max(t_star, 0.01)
            if self.debug_phase2:
                control, dvds = self.get_optimal_control(state, query_time)
            else:
                control, dvds = self.get_optimal_control(state, query_time)
            value = self.get_value(state, query_time)
            phase = 2

            if status != 'strict':
                self.brt_reacquisition_count += 1
                self.brt_time_adjustments.append({
                    'sim_time': sim_time,
                    't_star': t_star,
                    'status': status,
                    'value': value,
                })
        else:
            # Phase 1: Use fixed tMax value function
            control, dvds = self.get_optimal_control(state, self.tMax)
            value = self.get_value(state, self.tMax)
            phase = 1
            fallback_weight = 0.0

            if self.gradient_fallback:
                ctrl_dims = np.concatenate([dvds[3:6], dvds[10:13]])
                grad_mag = np.linalg.norm(ctrl_dims)
                if (grad_mag < self.grad_threshold
                        and not self._avoid_proximity_check(state)):
                    fallback_weight = 1.0 - (grad_mag / self.grad_threshold)
                    virtual_dvds = self._compute_l2_virtual_gradient(state)
                    blended = dvds + fallback_weight * virtual_dvds
                    control = self._compute_brt_control_13d(blended, state)

            self.fallback_weight_history.append(fallback_weight)

        # --- Per-step diagnostics ---
        grad_mag = float(np.linalg.norm(dvds))
        bounds_check = self._check_state_bounds(state)

        # Value delta (vs previous step)
        v_delta = 0.0
        if len(self.value_history) > 0:
            v_delta = value - self.value_history[-1]

        # Track consecutive V increases
        if v_delta > 0 and len(self.value_history) > 0:
            self._consecutive_v_increases += 1
        else:
            self._consecutive_v_increases = 0

        # Print warnings on concerning conditions (throttled to avoid spam)
        if bounds_check['out_of_domain']:
            if not hasattr(self, '_ood_warned'):
                self._ood_warned = 0
            self._ood_warned += 1
            if self._ood_warned <= 3:  # only first 3 warnings
                viol_strs = [f"{v['name']}={v['value']:.4f} "
                             f"[{v['lo']:.2f},{v['hi']:.2f}]" 
                             for v in bounds_check['violations']]
                joined = "; ".join(viol_strs)
                print(f"    WARN t={sim_time:.2f}s out-of-domain: {joined}")

        if grad_mag < 1e-6:
            if not hasattr(self, '_dead_grad_warned'):
                self._dead_grad_warned = 0
            self._dead_grad_warned += 1
            if self._dead_grad_warned <= 2:
                print(f"    WARN t={sim_time:.2f}s dead gradient "
                      f"|dV/ds|={grad_mag:.2e}")

        if self._consecutive_v_increases >= 5:
            if self._consecutive_v_increases == 5:  # print once at threshold
                print(f"    WARN t={sim_time:.2f}s V rising "
                      f"({self._consecutive_v_increases}+ consecutive, "
                      f"V={value:.4f})")

        self.diagnostic_history.append({
            'sim_time': sim_time,
            'phase': phase,
            'query_time': query_time if self.in_brt_phase or phase == 2 else self.tMax,
            'value': value,
            'v_delta': v_delta,
            'grad_magnitude': grad_mag,
            'grad_pos': float(np.linalg.norm(dvds[:3])),
            'grad_vel': float(np.linalg.norm(dvds[3:6])),
            'grad_quat': float(np.linalg.norm(dvds[6:10])),
            'grad_omega': float(np.linalg.norm(dvds[10:13])),
            'out_of_domain': bounds_check['out_of_domain'],
            'n_violations': len(bounds_check['violations']),
            'max_violation_ratio': bounds_check['max_violation_ratio'],
            'consecutive_v_increases': self._consecutive_v_increases,
            'velocity_norm': float(np.linalg.norm(state[3:6])),
            'omega_norm': float(np.linalg.norm(state[10:13])),
        })

        # --- Proportional torque substitution (if enabled and in proximity) ---
        pd_active = False
        if self.pd_torque_proximity is not None and self._check_pd_proximity(state):
            pd_active = True
            control[3:6] = self._compute_proportional_torque(state, dvds)
        self.pd_torque_active_history.append(pd_active)

        # Safety filter post-processing (phase-dependent margin)
        raw_control = control.copy()
        self.safety_filter.set_margin(
            self.safety_margin_phase2 if self.in_brt_phase
            else self.safety_margin_phase1)
        control = self.safety_filter.apply(state, control)

        if self.in_brt_phase and self.debug_phase2:
            self._record_phase2_debug(
                sim_time, state, search_details, t_star, status,
                query_time, value, dvds, raw_control, control)

        # History
        self.state_history.append(
            state.copy() if isinstance(state, np.ndarray) else state)
        self.control_history.append(control)
        self.value_history.append(value)
        self.phase_history.append(phase)
        self.t_remaining_history.append(
            self.t_remaining if self.in_brt_phase else self.tMax)
        self.sim_time_history.append(sim_time)

        return control

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run full docking simulation with two-phase BRT control.

        Returns:
            dict with trajectory, controls, values, phases, etc.
        """
        self.reset()
        t_wall_start = _time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn_13d

        print(f"  [BRT13D] Starting  V(x,tMax)={self.get_value(state, self.tMax):.4f}")

        docked = False
        collided = False
        dock_time = None
        post_dock_duration = 1.0  # seconds to continue after docking

        # Per-component reach_fn tracking
        reach_fn_comp_history = {
            'position': [], 'vlat': [], 'vax': [],
            'attitude': [], 'omega_py': [], 'omega_roll': [],
        }

        for step in range(num_steps):
            sim_time = step * self.dt

            # Stop after post-dock coast period
            if docked and (sim_time - dock_time) >= post_dock_duration:
                break

            # Normal control — keep BRT active even after docking so
            # the chaser can converge closer to the goal point.
            control = self.u_fn(state, sim_time)

            # Record per-component reach_fn values
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32,
                                   device=self.device)
                comps = self.dynamics.reach_fn_components(s_t)
                for k in reach_fn_comp_history:
                    reach_fn_comp_history[k].append(float(comps[k]))

            # Termination checks (only before docking)
            if not docked and self._check_docked_13d(state):
                docked = True
                dock_time = sim_time
                print(f"  [BRT13D] Docking at t={sim_time:.2f}s")

            if not docked and self._check_collision_13d(state):
                collided = True
                print(f"  [BRT13D] Collision at t={sim_time:.2f}s")
                break

            # Euler integration
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot

            # Quaternion normalization (replaces theta wrap)
            state = self._wrap_state_13d(state)

        wall_time = _time.perf_counter() - t_wall_start

        controls_arr = np.array(self.control_history)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        result = {
            'trajectory': np.array(self.state_history),
            'controls': controls_arr,
            'values': np.array(self.value_history),
            'phases': np.array(self.phase_history),
            't_remaining': np.array(self.t_remaining_history),
            'times': np.array(self.sim_time_history),
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'brt_13d',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'phase_transition_time': self.phase_transition_time,
            'brt_reacquisition_count': self.brt_reacquisition_count,
            'brt_time_adjustments': self.brt_time_adjustments,
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            'phase2_debug_log': self.phase2_debug_log,
            # --- gradient fallback (Phase 1) ---
            'fallback_weights': self.fallback_weight_history,
            'n_fallback_steps': sum(
                1 for w in self.fallback_weight_history if w > 0),
            # --- per-component reach_fn breakdown ---
            'reach_fn_components': {
                k: np.array(v) for k, v in reach_fn_comp_history.items()
            },
            # --- PD torque tracking ---
            'pd_torque_active': np.array(self.pd_torque_active_history, dtype=bool),
            'n_pd_torque_steps': int(sum(self.pd_torque_active_history)),
        }

        # --- Diagnostic summary --- #
        if self.diagnostic_history:
            diag = self.diagnostic_history
            ood_steps = sum(1 for d in diag if d['out_of_domain'])
            grad_mags = [d['grad_magnitude'] for d in diag]
            v_deltas = [d['v_delta'] for d in diag]
            max_vel = max(d['velocity_norm'] for d in diag)
            max_omega = max(d['omega_norm'] for d in diag)
            max_consec = max(d['consecutive_v_increases'] for d in diag)
            v_increase_steps = sum(1 for d in v_deltas if d > 0)

            # State range for reference
            sr = self.dynamics.state_range_.cpu().numpy()
            vel_range = f"[{sr[3,0]:.2f}, {sr[3,1]:.2f}]"

            summary = {
                'total_steps': len(diag),
                'out_of_domain_steps': ood_steps,
                'out_of_domain_pct': 100.0 * ood_steps / max(len(diag), 1),
                'max_velocity_norm': max_vel,
                'max_omega_norm': max_omega,
                'velocity_training_range': vel_range,
                'grad_magnitude_min': min(grad_mags),
                'grad_magnitude_max': max(grad_mags),
                'grad_magnitude_mean': np.mean(grad_mags),
                'v_increase_steps': v_increase_steps,
                'v_increase_pct': 100.0 * v_increase_steps / max(len(diag) - 1, 1),
                'max_consecutive_v_increases': max_consec,
                'final_value': diag[-1]['value'],
            }

            print("\n" + "-"*60)
            print("  BRT Diagnostic Summary")
            print("-"*60)
            print(f"  Steps         : {summary['total_steps']}")
            print(f"  Out-of-domain : {summary['out_of_domain_steps']} "
                  f"({summary['out_of_domain_pct']:.1f}%)")
            print(f"  Max |vel|     : {summary['max_velocity_norm']:.4f} m/s  "
                  f"(range: {vel_range})")
            print(f"  Max |omega|   : {summary['max_omega_norm']:.4f} rad/s  "
                  f"(range: [{sr[10,0]:.2f}, {sr[10,1]:.2f}])")
            print(f"  |dV/ds|       : {summary['grad_magnitude_min']:.1e} / "
                  f"{summary['grad_magnitude_max']:.1e} / "
                  f"{summary['grad_magnitude_mean']:.1e}  (min/max/mean)")
            print(f"  V increasing  : {summary['v_increase_steps']} steps "
                  f"({summary['v_increase_pct']:.1f}%), "
                  f"max run={summary['max_consecutive_v_increases']}")
            print(f"  Final V       : {summary['final_value']:.4f}")
            n_pd = result.get('n_pd_torque_steps', 0)
            total = summary['total_steps']
            if self.pd_torque_proximity is not None:
                print(f"  PD torque     : {n_pd}/{total} steps "
                      f"({100.0 * n_pd / max(total, 1):.1f}%)")
            print("-"*60)

            result['diagnostics'] = {
                'history': diag,
                'summary': summary,
            }

        return result

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def get_value_grid_3d(self, time_query, current_state,
                          grid_bounds=None, resolution=40):
        """Compute V on a 3D (x,y,z) grid, slicing other states from *current_state*.

        Args:
            time_query:    Scalar time for V query.
            current_state: (13,) numpy array — vel/quat/omega taken from here.
            grid_bounds:   [(xmin,xmax), (ymin,ymax), (zmin,zmax)] or None.
            resolution:    Grid points per axis.

        Returns:
            (X, Y, Z, V) — meshgrid arrays and value field.
        """
        if grid_bounds is None:
            sr = self.dynamics.state_test_range()
            grid_bounds = [
                (sr[0][0], sr[0][1]),
                (sr[1][0], sr[1][1]),
                (sr[2][0], sr[2][1]),
            ]

        xs = np.linspace(grid_bounds[0][0], grid_bounds[0][1], resolution)
        ys = np.linspace(grid_bounds[1][0], grid_bounds[1][1], resolution)
        zs = np.linspace(grid_bounds[2][0], grid_bounds[2][1], resolution)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        N = resolution ** 3
        states = np.tile(current_state, (N, 1)).astype(np.float32)
        states[:, 0] = X.ravel()
        states[:, 1] = Y.ravel()
        states[:, 2] = Z.ravel()

        # Batch forward pass
        states_t = torch.tensor(states, dtype=torch.float32,
                                device=self.device)
        time_col = torch.full((N, 1), time_query, dtype=torch.float32,
                              device=self.device)
        coords = torch.cat([time_col, states_t], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        V_flat = self.dynamics.io_to_value(model_input, output).cpu().numpy()
        V = V_flat.reshape(resolution, resolution, resolution)

        return X, Y, Z, V

    def data_to_visualize(self):
        """Provide data dict for diagnostics panels."""
        s = np.array(self.state_history)
        u = np.array(self.control_history)
        return {
            'x (m)': [1, s[:, 0], {'color': 'b'}],
            'y (m)': [1, s[:, 1], {'color': 'g'}],
            'z (m)': [1, s[:, 2], {'color': 'r'}],
            'vx (m/s)': [2, s[:, 3], {'color': 'b'}],
            'vy (m/s)': [2, s[:, 4], {'color': 'g'}],
            'vz (m/s)': [2, s[:, 5], {'color': 'r'}],
            'q0': [3, s[:, 6], {'color': 'purple'}],
            'q1': [3, s[:, 7], {'color': 'orange'}],
            'q2': [3, s[:, 8], {'color': 'cyan'}],
            'q3': [3, s[:, 9], {'color': 'brown'}],
            'ωx (rad/s)': [4, s[:, 10], {'color': 'b'}],
            'ωy (rad/s)': [4, s[:, 11], {'color': 'g'}],
            'ωz (rad/s)': [4, s[:, 12], {'color': 'r'}],
            'Fx (N)': [5, u[:, 0], {'color': 'r'}],
            'Fy (N)': [5, u[:, 1], {'color': 'g'}],
            'Fz (N)': [5, u[:, 2], {'color': 'b'}],
            'τx (N·m)': [6, u[:, 3], {'color': 'r'}],
            'τy (N·m)': [6, u[:, 4], {'color': 'g'}],
            'τz (N·m)': [6, u[:, 5], {'color': 'b'}],
            'Distance (m)': [7, np.sqrt(s[:, 0]**2 + (s[:, 1] - self.dynamics.goal_y_center)**2 + s[:, 2]**2),
                             {'color': 'orange'}],
        }
