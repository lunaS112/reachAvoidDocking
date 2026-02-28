"""
MPC + Learned Terminal Cost Controller for Docking13D

Receding-horizon MPC with a short effective planning horizon, augmented by
the learned DeepReach value function as terminal cost.

Two-phase terminal cost strategy:
  Phase 1 (Approach): V(x, tMax) — static terminal cost while outside BRT.
  Phase 2 (Tracking): V(x, t_remaining) — time-varying cost once inside BRT.
                       One-way transition: timer never resets.

Graduated stagnation-escape:
  NORMAL -> EXPLORING (larger eps_var) -> BRT_FALLBACK (pure gradient control).
  If progress resumes, reverts to NORMAL.

Usage:
    controller = MPCTerminalController13D(
        checkpoint_path='./runs/Docking13D_RA/training/checkpoints/model_final.pth',
        effective_horizon_sec=2.0,
        tMax=14.0,
    )
    result = controller.simulate_docking(initial_state, max_sim_time=60.0)
"""

import time
import torch
import numpy as np
import pickle
import os
import sys
import inspect

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))

from utils import modules
from utils import diff_operators  # noqa: F401
from utils.MPC import MPC
from dynamics import dynamics as dynamics_module
from utils.controllers.docking13d_mixin import Docking13DControllerMixin


class MPCTerminalController13D(Docking13DControllerMixin):
    """Short-horizon MPC + learned terminal cost for 13D docking."""

    def __init__(self, checkpoint_path, effective_horizon_sec=2.0, tMax=14.0,
                 dt=0.1, num_samples=500, num_refinement=10, device='cuda',
                 cost_type='reachability', effort_weight=0.0,
                 exploration_factor=3.0, exploration_patience=2,
                 escape_thresh=0.5):
        """
        Args:
            checkpoint_path:       Path to trained model checkpoint.
            effective_horizon_sec: Short MPC planning horizon (seconds).
            tMax:                  Time at which V(x, tMax) is queried.
            dt:                    Control / integration timestep (seconds).
            num_samples:           Random-shooting samples per refinement.
            num_refinement:        Iterative refinement passes per step.
            device:                Torch device.
            cost_type:             Base cost for the short horizon.
            effort_weight:         Weight for control effort penalty (0 = off).
            exploration_factor:    eps_var multiplier in EXPLORING mode.
            exploration_patience:  Stagnation windows before BRT fallback.
            escape_thresh:         Distance improvement (m) to escape local min.
        """
        self.checkpoint_path = checkpoint_path
        self.effective_horizon_sec = effective_horizon_sec
        self.effective_horizon = int(effective_horizon_sec / dt)
        self.tMax = tMax
        self.dt = dt
        self.num_samples = num_samples
        self.num_refinement = num_refinement
        self.device = device
        self.cost_type = cost_type
        self.effort_weight = effort_weight
        self.exploration_factor_setting = exploration_factor
        self.exploration_patience = exploration_patience
        self.escape_thresh = escape_thresh

        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        self._load_dynamics_and_model()

        self.mpc = MPC(
            dT=self.dt,
            horizon=self.effective_horizon,
            receding_horizon=1,
            num_samples=self.num_samples,
            dynamics_=self.dynamics,
            device=self.device,
            mode='MPC',
            sample_mode='gaussian',
            style='direct',
            num_iterative_refinement=self.num_refinement,
            cost_type=self.cost_type,
        )

        self.reset()

        effort_str = (f"  effort_weight={self.effort_weight}"
                      if self.effort_weight > 0 else "")
        print(f"MPCTerminalController13D initialised | "
              f"horizon={self.effective_horizon_sec}s  tMax={self.tMax}s  "
              f"samples={self.num_samples}  refinements={self.num_refinement}"
              f"{effort_str}")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_dynamics_and_model(self):
        """Load dynamics and trained SIREN model."""
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
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {self.checkpoint_path}")

    def reset(self):
        """Reset controller state for a new simulation."""
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.sim_time_history = []
        self.phase_history = []
        self.t_remaining_history = []
        self._warm_started = False

        # Phase tracking (one-way)
        self.in_brt = False
        self.t_remaining = self.tMax
        self.brt_entry_time = None

        # Diagnostics
        self._last_reach_avoid = 0.0
        self._last_terminal = 0.0
        self._last_t_query = self.tMax

        # Graduated stagnation-escape
        self._stagnation_count = 0
        self._control_mode = 'normal'
        self._exploration_factor = 1.0
        self._mode_entry_dist = None

    # ------------------------------------------------------------------
    # Value-function queries
    # ------------------------------------------------------------------

    def _is_in_brt(self, state_np):
        """True if V(state, tMax) <= 0."""
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), self.tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze()
        value = self.dynamics.io_to_value(model_input, output)
        return value.item() <= 0

    def get_value(self, state_np, time_query):
        """Query V(state, time_query) for a single state."""
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), time_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze()
        return self.dynamics.io_to_value(model_input, output).item()

    # ------------------------------------------------------------------
    # Terminal cost evaluation
    # ------------------------------------------------------------------

    def _evaluate_terminal_values(self, terminal_states, t_query=None):
        """Batch-evaluate V(x, t_query) for terminal states.

        Args:
            terminal_states: (A, N, state_dim) tensor.
            t_query:         Scalar time (defaults to tMax).

        Returns:
            (A, N) tensor of terminal values.
        """
        if t_query is None:
            t_query = self.tMax

        A, N, D = terminal_states.shape
        flat = terminal_states.reshape(A * N, D)

        test_range = torch.tensor(
            self.dynamics.state_test_range(), device=self.device)
        flat_clamped = torch.clamp(flat, test_range[..., 0], test_range[..., 1])

        time_col = torch.full(
            (A * N, 1), t_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_clamped], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        values = self.dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

    # ------------------------------------------------------------------
    # Phase tracking
    # ------------------------------------------------------------------

    def _update_phase(self, state, sim_time):
        """One-way phase transition: Phase 1 -> Phase 2 when entering BRT."""
        if not self.in_brt:
            if self._is_in_brt(state):
                self.in_brt = True
                self.t_remaining = self.tMax
                self.brt_entry_time = sim_time
                print(f"[MPC+Terminal13D] Entered BRT at t={sim_time:.2f}s "
                      f"-> Phase 2 (t_remaining={self.tMax:.1f}s)")

        if self.in_brt:
            t_query = max(self.t_remaining, 0.01)
            phase = 2
        else:
            t_query = self.tMax
            phase = 1

        self.phase_history.append(phase)
        self.t_remaining_history.append(
            self.t_remaining if self.in_brt else self.tMax)
        self._last_t_query = t_query

        return phase, t_query

    # ------------------------------------------------------------------
    # BRT optimal-control fallback (13D-specific)
    # ------------------------------------------------------------------

    def _compute_brt_control(self, state):
        """Bang-bang optimal control from value-function gradient."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), self._last_t_query, dtype=torch.float32,
            device=self.device)
        coord = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coord)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze()
        model_in = result['model_in']

        dv = self.dynamics.io_to_dv(model_in, output)
        dvds = dv[0, 1:].detach().cpu().numpy()

        return self._compute_brt_control_13d(dvds, state)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run full docking simulation with MPC + terminal cost.

        Returns:
            dict with the shared result format.
        """
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn_13d

        print(f"[MPC+Terminal13D] Starting from state: {state}")

        docked = False
        collided = False
        dock_time = None
        post_dock_duration = 1.0  # seconds to continue after docking

        # Stagnation detection
        log_interval = 50
        stagnation_thresh = 0.1
        prev_log_dist = None
        prev_log_V = None

        for step in range(num_steps):
            sim_time = step * self.dt

            # Stop after post-dock coast period
            if docked and (sim_time - dock_time) >= post_dock_duration:
                break

            # --- Control selection (mode-aware) ---
            # Keep control active even after docking so chaser can
            # converge closer to the goal point.
            if self._control_mode == 'brt_fallback':
                phase, t_query = self._update_phase(state, sim_time)
                control = self._compute_brt_control(state)
                cost_val = self.get_value(state, self._last_t_query)
                if self.in_brt:
                    self.t_remaining -= self.dt
            else:
                control, cost_val = self._mpc_step(state, sim_time)

            # Record
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)


            # --- Periodic logging + graduated stagnation ---
            if step % log_interval == 0:
                dist = float(np.linalg.norm(state[:3]))
                V_now = self.get_value(state, self._last_t_query)
                phase = self.phase_history[-1]
                ra = self._last_reach_avoid
                tv = self._last_terminal
                dominates = 'terminal' if tv <= ra else 'reach_avoid'

                delta_str = ''
                stag_flag = ''
                if prev_log_dist is not None:
                    d_dist = prev_log_dist - dist
                    d_V = prev_log_V - V_now
                    delta_str = (f'  delta_dist={d_dist:+.4f}m  '
                                 f'delta_V={d_V:+.4f}')
                    if d_dist < stagnation_thresh:
                        stag_flag = '  ** STAGNATING **'

                    if d_dist >= stagnation_thresh:
                        if (self._control_mode != 'normal'
                                and self._mode_entry_dist is not None):
                            if self._mode_entry_dist - dist >= self.escape_thresh:
                                self._control_mode = 'normal'
                                self._exploration_factor = 1.0
                                self._stagnation_count = 0
                                self._warm_started = False
                                print('  -> Escaped local min, returning to NORMAL')
                    else:
                        self._stagnation_count += 1
                        if self._control_mode == 'normal':
                            self._control_mode = 'exploring'
                            self._exploration_factor = self.exploration_factor_setting
                            self._mode_entry_dist = dist
                            print(f'  -> Switching to EXPLORING '
                                  f'(eps_var x{self._exploration_factor})')
                        elif (self._control_mode == 'exploring'
                              and self._stagnation_count >= self.exploration_patience):
                            self._control_mode = 'brt_fallback'
                            print('  -> Exploration failed, switching to BRT_FALLBACK')

                mode_tag = self._control_mode.upper()
                print(
                    f'[MPC+Terminal13D] Step {step:>4d} t={sim_time:5.1f}s '
                    f'| Phase {phase} | {mode_tag} | dist={dist:.3f}m '
                    f'| V(x,t)={V_now:.4f}\n'
                    f'  best_combined={cost_val:.4f}  '
                    f'reach_avoid={ra:.4f}  terminal={tv:.4f} '
                    f'({dominates}){delta_str}{stag_flag}')

                prev_log_dist = dist
                prev_log_V = V_now

            # Termination (only before docking)
            if not docked and self._check_docked_13d(state):
                docked = True
                dock_time = sim_time
                print(f"[MPC+Terminal13D] Docking successful at t={sim_time:.2f}s")

            if not docked and self._check_collision_13d(state):
                collided = True
                print(f"[MPC+Terminal13D] Collision at t={sim_time:.2f}s")
                break

            # Euler integration
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot

            # Quaternion normalization
            state = self._wrap_state_13d(state)

        wall_time = time.perf_counter() - t_wall_start

        controls_arr = np.array(self.control_history)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        return {
            'trajectory': np.array(self.state_history),
            'controls': controls_arr,
            'values': np.array(self.value_history),
            'times': np.array(self.sim_time_history),
            'phases': np.array(self.phase_history),
            't_remaining': np.array(self.t_remaining_history),
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'mpc_terminal_13d',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'brt_entry_time': self.brt_entry_time,
        }

    # ------------------------------------------------------------------
    # Core MPC + terminal cost
    # ------------------------------------------------------------------

    def _mpc_step(self, state, sim_time):
        """One MPC step with phase-aware terminal cost.

        Returns:
            (first_control, best_combined_cost)
        """
        phase, t_query = self._update_phase(state, sim_time)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.mpc.batch_size = 1
        self.mpc.horizon = self.effective_horizon

        if not self._warm_started:
            self.mpc.init_control_tensors()
            self._warm_started = True
        else:
            shifted = self.mpc.control_tensors[:, 1:, :].clone()
            pad = torch.zeros(
                1, 1, self.dynamics.control_dim, device=self.device)
            self.mpc.control_tensors = torch.cat([shifted, pad], dim=1)

        best_cost_val = float('inf')
        for _ in range(self.num_refinement):
            state_trajs, permuted_controls = self.mpc.rollout_dynamics(
                state_tensor, start_iter=0,
                rollout_horizon=self.effective_horizon,
                eps_var_factor=self._exploration_factor)

            # Short-horizon reachability cost
            reach_avoid_cost = self.dynamics.cost_fn(state_trajs)

            # Terminal cost from learned VF
            terminal_states = state_trajs[:, :, -1, :]
            terminal_values = self._evaluate_terminal_values(
                terminal_states, t_query=t_query)

            # Combine: max( min(reach_avoid, terminal), cummax(-avoid) )
            combined = torch.minimum(reach_avoid_cost, terminal_values)
            avoid_max = torch.max(
                -self.dynamics.avoid_fn(state_trajs), dim=-1).values
            combined = torch.maximum(combined, avoid_max)

            # Control effort penalty
            if self.effort_weight > 0:
                control_norms = torch.norm(permuted_controls, dim=-1)
                effort_per_sample = torch.sum(control_norms, dim=-1) * self.dt
                on_track = (combined <= 0).float()
                combined = combined + self.effort_weight * effort_per_sample * on_track

            # Select best
            best_costs, best_idx = combined.min(dim=1)
            best_cost_val = best_costs.item()

            bi = best_idx[0]
            self._last_reach_avoid = reach_avoid_cost[0, bi].item()
            self._last_terminal = terminal_values[0, bi].item()

            idx_ctrl = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx_ctrl = idx_ctrl.expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3))
            best_controls = torch.gather(
                permuted_controls, dim=1, index=idx_ctrl).squeeze(1)
            self.mpc.control_tensors = best_controls.clone()

        # Decrement timer (Phase 2 only)
        if self.in_brt:
            self.t_remaining -= self.dt

        first_control = self.mpc.control_tensors[0, 0, :].detach().cpu().numpy()
        return first_control, best_cost_val
