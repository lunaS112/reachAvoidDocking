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
from utils.controllers.docking13d_mixin import Docking13DControllerMixin
from utils.controllers.safety_filter import SafetyFilter


class BRTController13D(Docking13DControllerMixin):
    """Two-phase BRT-based optimal controller for 13D docking.

    Phase 1 (Convergence): V(x, tMax) > 0  — use gradient to approach BRT.
    Phase 2 (Precision):   V(x, tMax) <= 0 — use V(x, t_remaining) countdown.
    """

    def __init__(self, checkpoint_path, tMax=14.0, dt=0.1, device='cuda',
                 search_window=0.5, search_resolution=0.1,
                 max_search_expansions=1, safety_filter=None):
        """
        Args:
            checkpoint_path:       Path to the trained model checkpoint.
            tMax:                  BRT time-horizon cap (seconds).
            dt:                    Control / integration timestep (seconds).
            device:                Torch device ('cuda' or 'cpu').
            search_window:         Initial half-width of BRT reacquisition
                                   time search (seconds).
            search_resolution:     Time step for BRT time search (seconds).
            max_search_expansions: Max window-doubling attempts before
                                   falling back to Phase 1.
        """
        self.checkpoint_path = checkpoint_path
        self.tMax = tMax
        self.dt = dt
        self.device = device
        self.search_window = search_window
        self.search_resolution = search_resolution
        self.max_search_expansions = max_search_expansions
        self.safety_filter = safety_filter or SafetyFilter(mode=0)

        # Derive experiment directory from checkpoint path
        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        self._load_dynamics_and_model()
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

        # --- Diagnostic tracking ---
        self.diagnostic_history = []
        self._consecutive_v_increases = 0

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
    # BRT reacquisition (dynamics-agnostic)
    # ------------------------------------------------------------------

    def _search_brt_time(self, state, current_time):
        """Expanding-window search for the tightest BRT containing *state*."""
        window = self.search_window

        for _ in range(1 + self.max_search_expansions):
            t_low = max(current_time - window, 0.01)
            t_high = min(current_time + window, self.tMax)
            times = np.arange(t_low, t_high + self.search_resolution * 0.5,
                              self.search_resolution)
            if len(times) == 0:
                window *= 2
                continue

            values = self.get_values_batch(state, times)
            valid = values <= 0
            if np.any(valid):
                return float(np.min(times[valid]))
            window *= 2

        return None

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
                self.t_remaining = self.tMax
                self.phase_transition_time = sim_time
                print(f"  [BRT13D] Entered BRT at t={sim_time:.2f}s -> Phase 2")

        if self.in_brt_phase:
            query_time = max(self.t_remaining, 0.01)
            value = self.get_value(state, query_time)

            if value > 0:
                # Left the BRT — try to reacquire
                new_time = self._search_brt_time(state, query_time)
                if new_time is not None:
                    self.brt_reacquisition_count += 1
                    self.brt_time_adjustments.append({
                        'sim_time': sim_time,
                        'old_time': query_time,
                        'new_time': new_time,
                        'value_before': value,
                    })
                    self.t_remaining = new_time
                else:
                    self.in_brt_phase = False

        if self.in_brt_phase:
            query_time = max(self.t_remaining, 0.01)
            control, dvds = self.get_optimal_control(state, query_time)
            value = self.get_value(state, query_time)
            self.t_remaining -= self.dt
            phase = 2
        else:
            control, dvds = self.get_optimal_control(state, self.tMax)
            value = self.get_value(state, self.tMax)
            phase = 1

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

        # Safety filter post-processing
        control = self.safety_filter.apply(state, control)

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

        for step in range(num_steps):
            sim_time = step * self.dt

            # Stop after post-dock coast period
            if docked and (sim_time - dock_time) >= post_dock_duration:
                break

            # Normal control — keep BRT active even after docking so
            # the chaser can converge closer to the goal point.
            control = self.u_fn(state, sim_time)

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
