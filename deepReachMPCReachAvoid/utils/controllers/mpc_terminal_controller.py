"""
MPC + Learned Terminal Cost Controller for Docking6D

Receding-horizon MPC with a short effective planning horizon, augmented by the
learned DeepReach value function as terminal cost.  The MPC handles short-term
trajectory optimisation while the value function provides long-term guidance
toward the docking goal.

Two-phase terminal cost strategy:
  Phase 1 (Approach): V(x, tMax) -- static terminal cost while outside the BRT
  Phase 2 (Tracking): V(x, t_remaining) -- time-varying terminal cost once
                       inside the BRT. t_remaining counts down from tMax and
                       is never reset (permanent one-way transition).

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

# Add project root to path (3 levels up: controllers -> utils -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils import modules
from utils.MPC import MPC
from dynamics import dynamics as dynamics_module
from utils.controllers import clip_state_for_execution
from utils.controllers.safety_filter import SafetyFilter


class MPCTerminalController:
    """
    Short-horizon MPC controller augmented with the learned value function as
    terminal cost.

    At each simulation step the controller:
      1. Rolls out `num_samples` perturbed control sequences over a short
         `effective_horizon`.
      2. Evaluates the reachability cost over the short horizon.
      3. Evaluates V(x_terminal, tMax) for each sample's terminal state using
         the learned DeepReach model.
      4. Combines costs:  combined = max( min(reach_avoid, V_terminal),
                                          cummax(-avoid) )
      5. Selects the best sample and applies its first control.
      6. Warm-starts the next step by shifting the selected control sequence.
    """

    def __init__(self, checkpoint_path, effective_horizon_sec=2.0, tMax=14.0,
                 dt=0.1, num_samples=500, num_refinement=10, device='cuda',
                 cost_type='reachability', effort_weight=0.0,
                 exploration_factor=3.0, exploration_patience=2,
                 escape_thresh=0.5, safety_filter=None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint.
            effective_horizon_sec: Short MPC planning horizon in seconds.
            tMax: Time at which V(x, tMax) is queried for terminal cost.
            dt: Control / integration timestep in seconds.
            num_samples: Number of random-shooting samples per refinement.
            num_refinement: Number of iterative refinement passes per step.
            device: Torch device ('cuda' or 'cpu').
            cost_type: Base cost for the short-horizon trajectories.
            effort_weight: Weight for control effort penalty (0.0 = disabled).
                           Adds effort_weight * sum_k(||u_k||_2 * dt) to the
                           combined cost. Recommended range: 0.001 - 0.05.
            exploration_factor: Multiplier for eps_var when in EXPLORING mode
                                (default 3.0).
            exploration_patience: Number of stagnation windows (each 5 s) in
                                  EXPLORING mode before switching to BRT
                                  fallback (default 2).
            escape_thresh: Distance improvement (m) from stagnation entry
                           required to consider the local min escaped and
                           return to NORMAL mode (default 0.5).
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

        # Derive experiment directory from checkpoint path
        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        # Load dynamics AND learned model
        self._load_dynamics_and_model()

        # Instantiate the MPC engine (used only for rollout_dynamics &
        # control-tensor bookkeeping; cost computation is done manually)
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

        # Safety filter (no-op when mode=0 or None)
        self.safety_filter = safety_filter or SafetyFilter(mode=0)

        self.reset()

        effort_str = (f"  effort_weight={self.effort_weight}"
                      if self.effort_weight > 0 else "")
        print(f"MPCTerminalController initialised  |  "
              f"horizon={self.effective_horizon_sec}s  tMax={self.tMax}s  "
              f"samples={self.num_samples}  refinements={self.num_refinement}"
              f"{effort_str}")

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
        self._warm_started = False
        self.safety_filter.reset()

        # Phase tracking (one-way: once in_brt is True it stays True)
        self.in_brt = False
        self.t_remaining = self.tMax
        self.brt_entry_time = None

        # Diagnostics: cost breakdown from last MPC step
        self._last_reach_avoid = 0.0
        self._last_terminal = 0.0
        self._last_t_query = self.tMax

        # Graduated stagnation-escape state
        self._stagnation_count = 0
        self._control_mode = 'normal'   # 'normal' | 'exploring' | 'brt_fallback'
        self._exploration_factor = 1.0
        self._mode_entry_dist = None

    # ------------------------------------------------------------------
    # Value function queries
    # ------------------------------------------------------------------

    def _is_in_brt(self, state_np):
        """Check if V(state, tMax) <= 0 (state is inside the BRT)."""
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
    # Terminal cost evaluation
    # ------------------------------------------------------------------

    def _evaluate_terminal_values(self, terminal_states, t_query=None):
        """
        Evaluate V(x, t_query) for a batch of terminal states.

        Args:
            terminal_states: (A, N, state_dim) tensor of terminal states
                             where A=batch_size (1) and N=num_samples.
            t_query: Time at which to query the value function.
                     Defaults to self.tMax if not provided.

        Returns:
            (A, N) tensor of terminal values.
        """
        if t_query is None:
            t_query = self.tMax

        A, N, D = terminal_states.shape

        # Flatten to (A*N, D)
        flat_states = terminal_states.reshape(A * N, D)

        # Clamp to state test range to avoid out-of-distribution queries
        test_range = torch.tensor(
            self.dynamics.state_test_range(), device=self.device)
        flat_states_clamped = torch.clamp(
            flat_states, test_range[..., 0], test_range[..., 1])

        # Build coordinate tensor [t_query, state]
        time_col = torch.full(
            (A * N, 1), t_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_states_clamped], dim=-1)

        # Normalise and forward pass
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        values = self.dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

    # ------------------------------------------------------------------
    # Phase tracking (extracted so BRT fallback path can reuse it)
    # ------------------------------------------------------------------

    def _update_phase(self, state, sim_time):
        """
        Phase tracking and t_query computation.

        Phase 1 (Approach): V(x, tMax)  -- static terminal cost
        Phase 2 (Tracking): V(x, t_remaining) -- countdown from tMax

        Transition is one-way: once inside the BRT, the timer counts down
        permanently even if the chaser temporarily exits.

        Returns:
            (phase, t_query)
        """
        if not self.in_brt:
            if self._is_in_brt(state):
                self.in_brt = True
                self.t_remaining = self.tMax
                self.brt_entry_time = sim_time
                print(f"[MPC+Terminal] Entered BRT at t={sim_time:.2f}s "
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
    # BRT optimal-control fallback
    # ------------------------------------------------------------------

    def _compute_brt_control(self, state):
        """
        Bang-bang optimal control from value function gradient (BRT fallback).

        Replicates the gradient-based control strategy used by the pure BRT
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
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """
        Run a full docking simulation.

        Returns:
            dict with the shared result format (see plan).
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

        for step in range(num_steps):
            sim_time = step * self.dt

            # --- Control selection (mode-aware) ---
            if self._control_mode == 'brt_fallback':
                phase, t_query = self._update_phase(state, sim_time)
                control = self._compute_brt_control(state)
                cost_val = self.get_value(state, self._last_t_query)
                if self.in_brt:
                    self.t_remaining -= self.dt
            else:
                control, cost_val = self._mpc_step(state, sim_time)

            # Post-process through safety filter (no-op when mode=0)
            control = self.safety_filter.apply(state, control)

            # Record history
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)

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
                                self._exploration_factor = 1.0
                                self._stagnation_count = 0
                                self._warm_started = False
                                print(f'  -> Escaped local min, '
                                      f'returning to NORMAL')
                    else:
                        self._stagnation_count += 1
                        if self._control_mode == 'normal':
                            self._control_mode = 'exploring'
                            self._exploration_factor = \
                                self.exploration_factor_setting
                            self._mode_entry_dist = dist
                            print(f'  -> Switching to EXPLORING '
                                  f'(eps_var x{self._exploration_factor})')
                        elif (self._control_mode == 'exploring'
                              and self._stagnation_count
                              >= self.exploration_patience):
                            self._control_mode = 'brt_fallback'
                            print(f'  -> Exploration failed, switching '
                                  f'to BRT_FALLBACK')

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
                print(f"[MPC+Terminal] Docking successful at t={sim_time:.2f}s")
                break

            if self._check_collision(state):
                collided = True
                print(f"[MPC+Terminal] Collision at t={sim_time:.2f}s")
                break

            # Integrate (Euler)
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot
            state, clipped = clip_state_for_execution(state, self.dynamics)
            if clipped:
                n_clipped += 1

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
            'brt_entry_time': self.brt_entry_time,
            # --- safety filter ---
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            # --- execution clamping ---
            'n_clipped_steps': n_clipped,
        }
        return result

    # ------------------------------------------------------------------
    # Core MPC + terminal cost logic
    # ------------------------------------------------------------------

    def _mpc_step(self, state, sim_time):
        """
        Run one MPC optimisation step with phase-aware terminal cost.

        Phase 1 (Approach): terminal cost = V(x, tMax)  -- static
        Phase 2 (Tracking): terminal cost = V(x, t_remaining) -- countdown

        Returns:
            (first_control, best_combined_cost)
        """
        phase, t_query = self._update_phase(state, sim_time)

        # --- MPC optimisation ---
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

        # Iterative refinement with custom cost (MPC reachability + terminal)
        best_cost_val = float('inf')
        for _ in range(self.num_refinement):
            # Roll out num_samples perturbed trajectories
            # state_trajs: (1, N, H+1, D),  permuted_controls: (1, N, H, D_u)
            state_trajs, permuted_controls = self.mpc.rollout_dynamics(
                state_tensor, start_iter=0,
                rollout_horizon=self.effective_horizon,
                eps_var_factor=self._exploration_factor)

            # --- Reachability cost over short horizon ---
            reach_avoid_cost = self.dynamics.cost_fn(state_trajs)  # (1, N)

            # --- Terminal cost from learned value function ---
            terminal_states = state_trajs[:, :, -1, :]        # (1, N, D)
            terminal_values = self._evaluate_terminal_values(
                terminal_states, t_query=t_query)              # (1, N)

            # --- Combine costs (reach-avoid formulation) ---
            combined = torch.minimum(reach_avoid_cost, terminal_values)
            avoid_max = torch.max(
                -self.dynamics.avoid_fn(state_trajs), dim=-1).values  # (1, N)
            combined = torch.maximum(combined, avoid_max)

            # --- Control effort penalty (fuel minimization) ---
            if self.effort_weight > 0:
                # permuted_controls: (1, N, H, D_u)
                control_norms = torch.norm(
                    permuted_controls, dim=-1)           # (1, N, H)
                effort_per_sample = torch.sum(
                    control_norms, dim=-1) * self.dt     # (1, N)
                # Only penalize effort when the trajectory is on track
                # (cost <= 0 means inside the BRT).  When cost > 0 (outside
                # BRT), the controller needs full effort to reach the goal --
                # penalizing effort here causes the optimizer to collapse to
                # zero controls.
                on_track = (combined <= 0).float()
                combined = combined + self.effort_weight * effort_per_sample * on_track

            # --- Select best sample (argmin for reach / reach_avoid) ---
            best_costs, best_idx = combined.min(dim=1)        # (1,)
            best_cost_val = best_costs.item()

            # Store cost breakdown for the best sample (diagnostics)
            bi = best_idx[0]
            self._last_reach_avoid = reach_avoid_cost[0, bi].item()
            self._last_terminal = terminal_values[0, bi].item()

            # Update control tensors with the best sample's controls
            idx_ctrl = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx_ctrl = idx_ctrl.expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3))
            best_controls = torch.gather(
                permuted_controls, dim=1, index=idx_ctrl).squeeze(1)  # (1,H,Du)
            self.mpc.control_tensors = best_controls.clone()

        # --- Decrement timer after using it (Phase 2 only) ---
        if self.in_brt:
            self.t_remaining -= self.dt

        first_control = self.mpc.control_tensors[0, 0, :].detach().cpu().numpy()
        return first_control, best_cost_val

    # ------------------------------------------------------------------
    # Internal helpers (shared with BRT / MPC controllers)
    # ------------------------------------------------------------------

    def _default_dynamics_fn(self, state, control):
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()

    def _check_docked(self, state):
        px, py, vx, vy, theta, omega = state
        pos_ok = np.sqrt(px**2 + py**2) <= self.dynamics.eps_p
        vel_ok = np.sqrt(vx**2 + vy**2) <= self.dynamics.eps_v
        theta_diff = np.abs(
            np.arctan2(np.sin(theta - np.pi / 2),
                       np.cos(theta - np.pi / 2)))
        theta_ok = theta_diff <= self.dynamics.eps_theta
        omega_ok = np.abs(omega) <= self.dynamics.eps_omega
        return pos_ok and vel_ok and theta_ok and omega_ok

    def _check_collision(self, state):
        """Orientation-aware collision check (actual chaser corners)."""
        return self.dynamics.check_collision_oriented(state)
