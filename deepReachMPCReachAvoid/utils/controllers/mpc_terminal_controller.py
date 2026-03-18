"""
MPC + Learned Terminal Cost Controller for Docking6D

Gradient-based receding-horizon MPC with a short effective planning horizon,
augmented by the learned DeepReach value function as a differentiable terminal
cost.  The MPC handles short-term trajectory optimisation via differentiable
shooting + multi-start Adam, while the value function provides long-term
guidance toward the docking goal.

Two-phase terminal cost strategy:
  Phase 1 (Approach): V(x, tMax) -- static terminal cost while outside the BRAT
  Phase 2 (Tracking): V(x, t*) -- time-varying terminal cost once inside the
                       BRAT. Each step, t* = min t s.t. V(x, t) <= 0 (with
                       argmin fallback if no exact zero-crossing). One-way
                       transition: once in Phase 2, stays in Phase 2.

Usage:
    controller = MPCTerminalController(
        checkpoint_path='./runs/Docking6D_RA/training/checkpoints/model_final.pth',
        effective_horizon_sec=2.0,
        tMax=14.0,
    )
    result = controller.simulate_docking(initial_state, max_sim_time=30.0)
"""

import time
import torch
import numpy as np
import pickle
import os
import sys
import inspect
from tqdm import tqdm

# Add project root to path (3 levels up: controllers -> utils -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils import modules
from utils.gradient_mpc import GradientMPC, DifferentiableValueFunction
from dynamics import dynamics as dynamics_module
from utils.controllers import clip_state_for_execution
from utils.controllers.safety_filter import SafetyFilter
from utils.controllers.min_time_search import find_min_brat_time_single, STATUS_HOLD


class MPCTerminalController:
    """
    Gradient-based short-horizon MPC controller augmented with the learned value
    function as a differentiable terminal cost.

    At each simulation step the controller:
      1. Optimises control sequences via differentiable shooting + Adam
         with ``num_restarts`` parallel random restarts.
      2. Evaluates a combined cost:
         min(reach_avoid_cost, V_terminal) clamped by max(-avoid).
      3. Applies the first control and warm-starts the next step.
    """

    def __init__(self, checkpoint_path, effective_horizon_sec=2.0, tMax=14.0,
                 dt=0.1, device='cuda',
                 effort_weight=0.0,
                 exploration_factor=3.0, exploration_patience=2,
                 escape_thresh=0.5, safety_filter=None,
                 safety_margin_phase1=0.1, safety_margin_phase2=0.02,
                 search_resolution=0.1, debug_phase2=False,
                 gradient_lr=1.0, gradient_iters=50, num_restarts=8):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint.
            effective_horizon_sec: Short MPC planning horizon in seconds.
            tMax: Time at which V(x, tMax) is queried for terminal cost.
            dt: Control / integration timestep in seconds.
            device: Torch device ('cuda' or 'cpu').
            effort_weight: Weight for control effort penalty (0.0 = disabled).
            exploration_factor: Multiplier for goal_weight escalation when in
                                EXPLORING mode (default 3.0).
            exploration_patience: Number of stagnation windows (each 5 s) in
                                  EXPLORING mode before switching to BRAT
                                  fallback (default 2).
            escape_thresh: Distance improvement (m) from stagnation entry
                           required to consider the local min escaped and
                           return to NORMAL mode (default 0.5).
            gradient_lr: Adam learning rate (default 1.0).
            gradient_iters: Adam iterations per MPC step (default 50).
            num_restarts: Parallel random restarts (default 8).
        """
        self.checkpoint_path = checkpoint_path
        self.effective_horizon_sec = effective_horizon_sec
        self.effective_horizon = int(effective_horizon_sec / dt)
        self.tMax = tMax
        self.dt = dt
        self.device = device
        self.effort_weight = effort_weight
        self.exploration_factor_setting = exploration_factor
        self.exploration_patience = exploration_patience
        self.escape_thresh = escape_thresh
        self.search_resolution = search_resolution
        self.debug_phase2 = debug_phase2
        self.avoid_proximity_margin = 1.0

        # Derive experiment directory from checkpoint path
        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        # Load dynamics AND learned model
        self._load_dynamics_and_model()

        # Instantiate the gradient MPC solver
        self.gradient_mpc = GradientMPC(
            dt=self.dt,
            horizon=self.effective_horizon,
            dynamics=self.dynamics,
            device=self.device,
            num_iters=gradient_iters,
            lr=gradient_lr,
            num_restarts=num_restarts,
        )

        # Differentiable value function (bypasses SingleBVPNet's .detach())
        self.diff_value_fn = DifferentiableValueFunction(
            self.model, self.dynamics, self.device)

        # Safety filter (no-op when mode=0 or None)
        self.safety_filter = safety_filter or SafetyFilter(mode=0)
        self.safety_margin_phase1 = safety_margin_phase1
        self.safety_margin_phase2 = safety_margin_phase2

        self.reset()

        effort_str = (f"  effort_weight={self.effort_weight}"
                      if self.effort_weight > 0 else "")
        print(f"MPCTerminalController initialised  |  "
              f"horizon={self.effective_horizon_sec}s  tMax={self.tMax}s  "
              f"iters={gradient_iters}  restarts={num_restarts}  "
              f"lr={gradient_lr}{effort_str}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_dynamics_and_model(self):
        """Load dynamics and the trained SIREN model from checkpoint."""
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)

        # Dynamics
        dynamics_class = getattr(dynamics_module, self.orig_opt.dynamics_class)
        sig = inspect.signature(dynamics_class)
        kwargs = {}
        for param_name in sig.parameters.keys():
            if param_name != 'self' and hasattr(self.orig_opt, param_name):
                kwargs[param_name] = getattr(self.orig_opt, param_name)
        self.dynamics = dynamics_class(**kwargs)
        self.dynamics.set_model(self.orig_opt.deepReach_model)

        # Fallback: ensure normalization matches training even if
        # state_range was not passed through the constructor
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
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
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
        self._warm_controls = None
        self.safety_filter.reset()

        # Phase tracking (one-way: once in_brat is True it stays True)
        self.in_brat = False
        self.t_remaining = self.tMax
        self.brat_entry_time = None

        # Diagnostics: cost breakdown from last MPC step
        self._last_reach_avoid = 0.0
        self._last_terminal = 0.0
        self._last_t_query = self.tMax

        # Graduated stagnation-escape state
        self._stagnation_count = 0
        self._control_mode = 'normal'   # 'normal' | 'exploring' | 'brat_fallback'
        self._goal_weight = 0.0
        self._near_obstacle = False
        self._mode_entry_dist = None
        self.phase2_debug_log = []
        self._last_phase2_state_search = None

    # ------------------------------------------------------------------
    # Value function queries
    # ------------------------------------------------------------------

    def _is_in_brat(self, state_np):
        """Check if V(state, tMax) <= 0 (state is inside the BRAT)."""
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
        """
        Query V(state, time_query) for a single state.

        Useful for animation / diagnostics.

        Args:
            state_np: numpy array of shape (state_dim,)
            time_query: scalar time to query

        Returns:
            float: V(state, time_query)
        """
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
    # Phase tracking (extracted so BRAT fallback path can reuse it)
    # ------------------------------------------------------------------

    def _update_phase(self, state, sim_time):
        """
        Phase tracking and t_query computation.

        Phase 1 (Approach): V(x, tMax)  -- static terminal cost
        Phase 2 (Tracking): V(x, t*) -- per-step min-time query

        Transition is one-way: once inside the BRAT, min-time queries
        determine t* each step (no countdown timer).

        Returns:
            (phase, t_query)
        """
        if not self.in_brat:
            if self._is_in_brat(state):
                self.in_brat = True
                self.brat_entry_time = sim_time
                print(f"[MPC+Terminal] Entered BRAT at t={sim_time:.2f}s "
                      f"-> Phase 2")

        if self.in_brat:
            # Per-step min-time query for the current state
            state_np = state if isinstance(state, np.ndarray) else np.array(state)
            value_fn = lambda times: self._value_at_times_single(state_np, times)
            if self.debug_phase2:
                t_star, _status, search_details = find_min_brat_time_single(
                    value_fn, self.tMax,
                    resolution=self.search_resolution,
                    return_details=True,
                    t_remaining=self.t_remaining)
                self._last_phase2_state_search = search_details
            else:
                t_star, _status = find_min_brat_time_single(
                    value_fn, self.tMax,
                    resolution=self.search_resolution,
                    t_remaining=self.t_remaining)
                self._last_phase2_state_search = None
            # STATUS_HOLD: count down; otherwise accept the searched t*
            if _status == STATUS_HOLD:
                self.t_remaining = max(t_star - self.dt, 0.01)
            else:
                self.t_remaining = t_star
            t_query = max(t_star, 0.01)
            phase = 2
        else:
            t_query = self.tMax
            phase = 1

        self.phase_history.append(phase)
        self.t_remaining_history.append(
            self.t_remaining if self.in_brat else self.tMax)
        self._last_t_query = t_query

        return phase, t_query

    def _value_at_times_single(self, state_np, times):
        """Query V(state, t_i) for a single state at multiple times."""
        n = len(times)
        s = torch.tensor(state_np, dtype=torch.float32, device=self.device)
        state_batch = s.unsqueeze(0).expand(n, -1)
        time_col = torch.tensor(
            times, dtype=torch.float32, device=self.device).unsqueeze(-1)
        coords = torch.cat([time_col, state_batch], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)
        values = self.dynamics.io_to_value(model_input, output)
        return values.cpu().numpy()

    # ------------------------------------------------------------------
    # BRAT optimal-control fallback
    # ------------------------------------------------------------------

    def _compute_brat_control(self, state):
        """
        Bang-bang optimal control from value function gradient (BRAT fallback).

        Replicates the gradient-based control strategy used by the pure BRAT
        controller.  The spatial gradient dV/ds is computed via a forward pass
        with grad tracking, and control is set to +-u_max opposing the gradient
        direction.
        """
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

        u_x = -self.dynamics.u_bar if dvds[2] > 0 else self.dynamics.u_bar
        u_y = -self.dynamics.u_bar if dvds[3] > 0 else self.dynamics.u_bar
        u_theta = (-self.dynamics.u_theta_bar if dvds[5] > 0
                   else self.dynamics.u_theta_bar)
        return np.array([u_x, u_y, u_theta])

    # ------------------------------------------------------------------
    # Cost function construction
    # ------------------------------------------------------------------

    def _build_cost_fn(self, phase, t_query):
        """Build the cost function for gradient MPC.

        Combines:
          1. Short-horizon reach-avoid cost (analytical)
          2. Terminal cost from learned value function (differentiable through SIREN)
          3. Control effort penalty (optional)
          4. Goal-directed regularisation (stagnation escape)
        """
        diff_value_fn = self.diff_value_fn
        dynamics = self.dynamics
        effort_weight = self.effort_weight
        dt = self.dt
        goal_weight = self._goal_weight
        near_obstacle = self._near_obstacle
        device = self.device

        def cost_fn(trajectory, controls):
            # 1. Short-horizon reach-avoid cost
            reach_avoid = dynamics.cost_fn(trajectory)  # (K,)

            # 2. Terminal cost (differentiable through SIREN)
            terminal_states = trajectory[:, -1, :]  # (K, 6)
            terminal_values = diff_value_fn(terminal_states, t_query)  # (K,)

            # 3. Combine (reach-avoid formulation)
            combined = torch.minimum(reach_avoid, terminal_values)
            avoid_max = torch.max(
                -dynamics.avoid_fn(trajectory), dim=-1).values  # (K,)
            combined = torch.maximum(combined, avoid_max)

            # 4. Control effort penalty
            if effort_weight > 0:
                control_norms = torch.norm(controls, dim=-1)  # (K, H)
                effort = torch.sum(control_norms, dim=-1) * dt  # (K,)
                on_track = (combined <= 0).float()
                combined = combined + effort_weight * effort * on_track

            # 5. Goal-directed regularisation (stagnation escape)
            if goal_weight > 0 and not near_obstacle:
                combined = combined + goal_weight * _goal_directed_cost(
                    trajectory, dynamics, device)

            return combined

        return cost_fn

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """
        Run a full docking simulation.

        Returns:
            dict with the shared result format.
        """
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn

        print(f"[MPC+Terminal] Starting from state: {state}")

        docked = False
        collided = False
        n_clipped = 0

        # Stagnation detection state
        log_interval = 50          # log every 50 steps (5 s at dt=0.1)
        stagnation_thresh = 0.1    # metres improvement required per window
        prev_log_dist = None
        prev_log_V = None

        pbar = tqdm(range(num_steps), desc="[MPC+Terminal] Simulating",
                   unit="step", leave=True)
        for step in pbar:
            sim_time = step * self.dt

            # --- Control selection (mode-aware) ---
            if self._control_mode == 'brat_fallback':
                phase, t_query = self._update_phase(state, sim_time)
                control = self._compute_brat_control(state)
                cost_val = self.get_value(state, self._last_t_query)
            else:
                control, cost_val = self._mpc_step(state, sim_time)

            # Post-process through safety filter (no-op when mode=0)
            self.safety_filter.set_margin(
                self.safety_margin_phase2 if self.in_brat else self.safety_margin_phase1)
            control = self.safety_filter.apply(state, control)

            # Record history
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)

            # Update progress bar
            dist = float(np.sqrt(state[0]**2 + state[1]**2))
            phase_now = self.phase_history[-1] if self.phase_history else 1
            pbar.set_postfix(
                t=f"{sim_time:.1f}s", phase=phase_now,
                mode=self._control_mode, dist=f"{dist:.3f}m",
                cost=f"{cost_val:.4f}")

            # --- Periodic diagnostic logging + graduated stagnation ---
            if step % log_interval == 0:
                dist = float(np.sqrt(state[0]**2 + state[1]**2))
                V_now = self.get_value(state, self._last_t_query)
                phase = self.phase_history[-1]
                ra = self._last_reach_avoid
                tv = self._last_terminal
                dominates = 'terminal' if tv <= ra else 'reach_avoid'

                delta_str = ''
                stag_flag = ''
                if prev_log_dist is not None:
                    d_dist = prev_log_dist - dist    # positive = closer
                    d_V = prev_log_V - V_now         # positive = improving
                    delta_str = (f'  delta_dist={d_dist:+.4f}m  '
                                 f'delta_V={d_V:+.4f}')
                    if d_dist < stagnation_thresh:
                        stag_flag = '  ** STAGNATING **'

                    # --- Graduated stagnation response ---
                    if d_dist >= stagnation_thresh:
                        # Making progress -- check if we've escaped
                        if (self._control_mode != 'normal'
                                and self._mode_entry_dist is not None):
                            if (self._mode_entry_dist - dist
                                    >= self.escape_thresh):
                                self._control_mode = 'normal'
                                self._goal_weight = 0.0
                                self._stagnation_count = 0
                                self._warm_controls = None
                                print(f'  -> Escaped local min, '
                                      f'returning to NORMAL')
                    else:
                        self._stagnation_count += 1
                        if self._control_mode == 'normal':
                            self._control_mode = 'exploring'
                            self._goal_weight = 0.1
                            self._mode_entry_dist = dist
                            print(f'  -> Switching to EXPLORING '
                                  f'(goal_weight={self._goal_weight})')
                        elif self._control_mode == 'exploring':
                            # Escalate goal weight
                            self._goal_weight = min(
                                self._goal_weight * self.exploration_factor_setting,
                                1.0)
                            if (self._stagnation_count
                                    >= self.exploration_patience):
                                self._control_mode = 'brat_fallback'
                                print(f'  -> Exploration failed, switching '
                                      f'to BRAT_FALLBACK')
                            else:
                                print(f'  -> Escalating goal_weight='
                                      f'{self._goal_weight:.3f}')

                mode_tag = self._control_mode.upper()
                print(
                    f'[MPC+Terminal] Step {step:>4d} t={sim_time:5.1f}s '
                    f'| Phase {phase} | {mode_tag} | dist={dist:.3f}m '
                    f'| V(x,t)={V_now:.4f}\n'
                    f'  best_combined={cost_val:.4f}  '
                    f'reach_avoid={ra:.4f}  terminal={tv:.4f} '
                    f'({dominates}){delta_str}{stag_flag}')

                prev_log_dist = dist
                prev_log_V = V_now

            # Termination
            if self._check_docked(state):
                docked = True
                pbar.set_postfix(t=f"{sim_time:.1f}s", status="DOCKED")
                print(f"\n[MPC+Terminal] Docking successful at t={sim_time:.2f}s")
                break

            if self._check_collision(state):
                collided = True
                pbar.set_postfix(t=f"{sim_time:.1f}s", status="COLLISION")
                print(f"\n[MPC+Terminal] Collision at t={sim_time:.2f}s")
                break

            # Integrate (Euler)
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot
            state, clipped = clip_state_for_execution(state, self.dynamics)
            if clipped:
                n_clipped += 1
        pbar.close()

        wall_time = time.perf_counter() - t_wall_start

        controls_arr = np.array(self.control_history)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        result = {
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
            'controller_type': 'mpc_terminal',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'brat_entry_time': self.brat_entry_time,
            # --- safety filter ---
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            # --- execution clamping ---
            'n_clipped_steps': n_clipped,
            'phase2_debug_log': self.phase2_debug_log,
        }
        return result

    # ------------------------------------------------------------------
    # Core MPC + terminal cost logic (gradient-based)
    # ------------------------------------------------------------------

    def _mpc_step(self, state, sim_time):
        """
        Run one gradient-based MPC step with phase-aware terminal cost.

        Phase 1 (Approach): terminal cost = V(x, tMax)
        Phase 2 (Tracking): terminal cost = V(x, t*) using current state's t*

        Returns:
            (first_control, best_combined_cost)
        """
        phase, t_query = self._update_phase(state, sim_time)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device)

        # Near-obstacle check (suppresses goal-directed cost near obstacles)
        with torch.no_grad():
            avoid_val = float(self.dynamics.avoid_fn(state_tensor.unsqueeze(0)).item())
        self._near_obstacle = avoid_val < self.avoid_proximity_margin

        cost_fn = self._build_cost_fn(phase, t_query)

        # Warm-start: shift previous solution
        warm = None
        if self._warm_controls is not None:
            warm = torch.cat([
                self._warm_controls[1:],
                torch.zeros(1, 3, device=self.device)
            ], dim=0)

        best_controls, best_cost, best_traj = self.gradient_mpc.optimize(
            state_tensor, cost_fn, warm_start=warm)

        self._warm_controls = best_controls.detach()

        # Diagnostics (matching existing interface)
        with torch.no_grad():
            self._last_reach_avoid = self.dynamics.cost_fn(
                best_traj.unsqueeze(0)).item()
            self._last_terminal = self.diff_value_fn(
                best_traj[-1].unsqueeze(0), t_query).item()

        first_control = best_controls[0].cpu().numpy()
        return first_control, best_cost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_dynamics_fn(self, state, control):
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()

    def _check_docked(self, state):
        d = self.dynamics
        px, py, vx, vy, theta, omega = state
        x_ok = np.abs(px) <= d.eps_p
        y_ok = d.goal_y_min <= py <= d.goal_y_max
        pos_ok = x_ok and y_ok
        vel_ok = np.sqrt(vx**2 + vy**2) <= d.eps_v
        theta_goal = d.goal_state[4].item()
        omega_goal = d.goal_state[5].item()
        theta_diff = np.abs(
            np.arctan2(np.sin(theta - theta_goal),
                       np.cos(theta - theta_goal)))
        theta_ok = theta_diff <= d.eps_theta
        omega_ok = np.abs(omega - omega_goal) <= d.eps_omega
        return pos_ok and vel_ok and theta_ok and omega_ok

    def _check_collision(self, state):
        """Orientation-aware collision check (actual chaser corners)."""
        return self.dynamics.check_collision_oriented(state)


def _goal_directed_cost(trajectory, dynamics, device):
    """Quadratic goal-directed cost for stagnation escape.

    Provides useful gradients everywhere, unlike the tanh-saturated reach_fn.
    Weights match dynamics.Q = diag([3,3,10,10,5,5]).
    """
    terminal = trajectory[:, -1, :]  # (K, 6)
    goal = dynamics.goal_state.to(device).float()

    pos_err = terminal[:, :2] - goal[:2]
    vel_err = terminal[:, 2:4] - goal[2:4]
    theta_err = torch.atan2(
        torch.sin(terminal[:, 4] - goal[4]),
        torch.cos(terminal[:, 4] - goal[4]))
    omega_err = terminal[:, 5] - goal[5]

    return (3.0 * (pos_err ** 2).sum(-1)
            + 10.0 * (vel_err ** 2).sum(-1)
            + 5.0 * theta_err ** 2
            + 5.0 * omega_err ** 2)
