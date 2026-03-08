"""
Cascaded MPC + Terminal Cost Controller for Docking6D

Short-horizon MPC with a 3-phase terminal cost strategy using outer and inner
learned value functions:

  Phase 1 (Approach):  V_outer(x, outer_tMax)  -- static, far from outer BRT
  Phase 2 (Transit):   V_outer(x, outer_t_remaining) -- outer BRT entered,
                        countdown from outer_tMax (permanent, one-way)
  Phase 3 (Precision): V_inner(x, inner_t_remaining) -- inner BRT entered,
                        countdown from inner_tMax (permanent, one-way)

Phase transitions are irreversible: once a BRT is entered, t_remaining counts
down monotonically and is never reset, even if the chaser temporarily exits
the BRT.

Usage:
    controller = CascadedMPCTerminalController(
        outer_checkpoint='./runs/Docking6D_RA/training/checkpoints/model_final.pth',
        inner_checkpoint='./runs/Docking6D_Inner/training/checkpoints/model_final.pth',
        outer_tMax=14.0,
        inner_tMax=3.0,
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils import modules
from utils.MPC import MPC
from dynamics import dynamics as dynamics_module


class CascadedMPCTerminalController:
    """
    MPC + terminal cost controller with 3-phase cascaded strategy.

    Phase transitions are one-way (irreversible):
      Phase 1 -> Phase 2:  first entry into outer BRT (V_outer <= 0)
      Phase 2 -> Phase 3:  first entry into inner BRT (V_inner <= 0)

    Once a phase is entered, its t_remaining timer counts down permanently
    from tMax, even if the chaser temporarily exits the BRT.

    The MPC rollout engine and dynamics come from the outer model.
    The inner model is loaded separately for terminal cost evaluation.
    """

    def __init__(self, outer_checkpoint, inner_checkpoint,
                 effective_horizon_sec=2.0, outer_tMax=14.0, inner_tMax=3.0,
                 dt=0.1, num_samples=500, num_refinement=10, device='cuda',
                 cost_type='reachability', effort_weight=0.0,
                 exploration_factor=3.0, exploration_patience=2,
                 escape_thresh=0.5):
        """
        Args:
            outer_checkpoint: Path to trained outer (long-horizon) model.
            inner_checkpoint: Path to trained inner (short-horizon) model.
            effective_horizon_sec: Short MPC planning horizon in seconds.
            outer_tMax: Time horizon for outer BRT terminal cost queries.
            inner_tMax: Time horizon for inner BRT terminal cost queries.
            dt: Control / integration timestep in seconds.
            num_samples: Number of random-shooting samples per refinement.
            num_refinement: Number of iterative refinement passes per step.
            device: Torch device.
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
        self.effective_horizon_sec = effective_horizon_sec
        self.effective_horizon = int(effective_horizon_sec / dt)
        self.outer_tMax = outer_tMax
        self.inner_tMax = inner_tMax
        self.dt = dt
        self.num_samples = num_samples
        self.num_refinement = num_refinement
        self.device = device
        self.cost_type = cost_type
        self.effort_weight = effort_weight
        self.exploration_factor_setting = exploration_factor
        self.exploration_patience = exploration_patience
        self.escape_thresh = escape_thresh

        # --- Load outer model (provides dynamics + MPC engine + outer terminal cost) ---
        self.outer_experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(outer_checkpoint)))
        self._load_outer_model(outer_checkpoint)

        # --- Load inner model (provides inner terminal cost) ---
        self.inner_experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(inner_checkpoint)))
        self._load_inner_model(inner_checkpoint)

        # --- MPC engine (uses outer dynamics for rollout) ---
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
        print(f"CascadedMPCTerminalController initialised  |  "
              f"horizon={effective_horizon_sec}s  outer_tMax={outer_tMax}s  "
              f"inner_tMax={inner_tMax}s  samples={num_samples}"
              f"{effort_str}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_outer_model(self, checkpoint_path):
        """Load outer dynamics and model."""
        opt_path = os.path.join(self.outer_experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.outer_opt = pickle.load(f)

        dynamics_class = getattr(dynamics_module, self.outer_opt.dynamics_class)
        sig = inspect.signature(dynamics_class)
        kwargs = {}
        for pn in sig.parameters.keys():
            if pn != 'self' and hasattr(self.outer_opt, pn):
                kwargs[pn] = getattr(self.outer_opt, pn)
        self.dynamics = dynamics_class(**kwargs)
        self.dynamics.set_model(self.outer_opt.deepReach_model)
        if hasattr(self.outer_opt, 'state_range') and self.outer_opt.state_range is not None:
            self.dynamics.override_state_range(self.outer_opt.state_range)

        self.outer_model = modules.SingleBVPNet(
            in_features=self.dynamics.input_dim, out_features=1,
            type=self.outer_opt.model, mode=self.outer_opt.model_mode,
            final_layer_factor=1.,
            hidden_features=self.outer_opt.num_nl,
            num_hidden_layers=self.outer_opt.num_hl,
            periodic_transform_fn=self.dynamics.periodic_transform_fn,
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.outer_model.load_state_dict(ckpt['model'])
        self.outer_model.to(self.device)
        self.outer_model.eval()
        print(f"Loaded outer model from {checkpoint_path}")

    def _load_inner_model(self, checkpoint_path):
        """Load inner dynamics and model (separate normalization)."""
        opt_path = os.path.join(self.inner_experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.inner_opt = pickle.load(f)

        dynamics_class = getattr(dynamics_module, self.inner_opt.dynamics_class)
        sig = inspect.signature(dynamics_class)
        kwargs = {}
        for pn in sig.parameters.keys():
            if pn != 'self' and hasattr(self.inner_opt, pn):
                kwargs[pn] = getattr(self.inner_opt, pn)
        self.inner_dynamics = dynamics_class(**kwargs)
        self.inner_dynamics.set_model(self.inner_opt.deepReach_model)
        if hasattr(self.inner_opt, 'state_range') and self.inner_opt.state_range is not None:
            self.inner_dynamics.override_state_range(self.inner_opt.state_range)

        self.inner_model = modules.SingleBVPNet(
            in_features=self.inner_dynamics.input_dim, out_features=1,
            type=self.inner_opt.model, mode=self.inner_opt.model_mode,
            final_layer_factor=1.,
            hidden_features=self.inner_opt.num_nl,
            num_hidden_layers=self.inner_opt.num_hl,
            periodic_transform_fn=self.inner_dynamics.periodic_transform_fn,
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.inner_model.load_state_dict(ckpt['model'])
        self.inner_model.to(self.device)
        self.inner_model.eval()
        print(f"Loaded inner model from {checkpoint_path}")

    def reset(self):
        """Reset controller state for a new simulation."""
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.sim_time_history = []
        self.phase_history = []
        self.t_remaining_history = []
        self.terminal_mode_history = []  # 'outer' or 'inner' (kept for compat)
        self._warm_started = False

        # 3-phase tracking (one-way transitions)
        self.phase = 1  # 1=approach, 2=outer transit, 3=inner precision
        self.outer_t_remaining = self.outer_tMax
        self.inner_t_remaining = self.inner_tMax
        self.outer_entry_time = None
        self.inner_entry_time = None

        # Diagnostics: cost breakdown from last MPC step
        self._last_reach_avoid = 0.0
        self._last_terminal = 0.0
        self._last_t_query = self.outer_tMax
        self._last_use_inner = False

        # Graduated stagnation-escape state
        self._stagnation_count = 0
        self._control_mode = 'normal'   # 'normal' | 'exploring' | 'brt_fallback'
        self._exploration_factor = 1.0
        self._mode_entry_dist = None

    # ------------------------------------------------------------------
    # Value function queries
    # ------------------------------------------------------------------

    def _is_in_outer_brt(self, state_np):
        """Check if V_outer(state, outer_tMax) <= 0."""
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), self.outer_tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.outer_model({'coords': model_input})
            output = result['model_out'].squeeze()
        value = self.dynamics.io_to_value(model_input, output)
        return value.item() <= 0

    def _is_in_inner_brt(self, state_np):
        """Check if V_inner(state, inner_tMax) <= 0."""
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), self.inner_tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.inner_dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.inner_model({'coords': model_input})
            output = result['model_out'].squeeze()
        value = self.inner_dynamics.io_to_value(model_input, output)
        return value.item() <= 0

    def get_value_outer(self, state_np, time_query):
        """
        Query V_outer(state, time_query) for a single state.

        Useful for animation / diagnostics.
        """
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), time_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.outer_model({'coords': model_input})
            output = result['model_out'].squeeze()
        return self.dynamics.io_to_value(model_input, output).item()

    def get_value_inner(self, state_np, time_query):
        """
        Query V_inner(state, time_query) for a single state.

        Useful for animation / diagnostics.
        """
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), time_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.inner_dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.inner_model({'coords': model_input})
            output = result['model_out'].squeeze()
        return self.inner_dynamics.io_to_value(model_input, output).item()

    # ------------------------------------------------------------------
    # Terminal cost evaluation
    # ------------------------------------------------------------------

    def _evaluate_terminal_values_outer(self, terminal_states, t_query=None):
        """
        Evaluate V_outer(x, t_query) for a batch of terminal states.

        Args:
            terminal_states: (A, N, state_dim)
            t_query: Time to query. Defaults to self.outer_tMax.

        Returns:
            (A, N) tensor of terminal values.
        """
        if t_query is None:
            t_query = self.outer_tMax

        A, N, D = terminal_states.shape
        flat_states = terminal_states.reshape(A * N, D)

        test_range = torch.tensor(
            self.dynamics.state_test_range(), device=self.device)
        flat_states_clamped = torch.clamp(
            flat_states, test_range[..., 0], test_range[..., 1])

        time_col = torch.full(
            (A * N, 1), t_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_states_clamped], dim=-1)

        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.outer_model({'coords': model_input})
            output = result['model_out'].squeeze(-1)
        values = self.dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

    def _evaluate_terminal_values_inner(self, terminal_states, t_query=None):
        """
        Evaluate V_inner(x, t_query) for a batch of terminal states.

        Args:
            terminal_states: (A, N, state_dim)
            t_query: Time to query. Defaults to self.inner_tMax.

        Returns:
            (A, N) tensor of terminal values.
        """
        if t_query is None:
            t_query = self.inner_tMax

        A, N, D = terminal_states.shape
        flat_states = terminal_states.reshape(A * N, D)

        test_range = torch.tensor(
            self.inner_dynamics.state_test_range(), device=self.device)
        flat_states_clamped = torch.clamp(
            flat_states, test_range[..., 0], test_range[..., 1])

        time_col = torch.full(
            (A * N, 1), t_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_states_clamped], dim=-1)

        model_input = self.inner_dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.inner_model({'coords': model_input})
            output = result['model_out'].squeeze(-1)
        values = self.inner_dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

    # ------------------------------------------------------------------
    # Phase tracking (extracted so BRT fallback path can reuse it)
    # ------------------------------------------------------------------

    def _update_phase(self, state, sim_time):
        """
        Three-phase tracking and t_query computation.

        Phase 1 (Approach):  V_outer(x, outer_tMax)  -- static
        Phase 2 (Transit):   V_outer(x, outer_t_remaining) -- countdown
        Phase 3 (Precision): V_inner(x, inner_t_remaining) -- countdown

        Transitions are one-way (irreversible).

        Returns:
            (phase, t_query, use_inner, mode_str)
        """
        if self.phase == 1:
            if self._is_in_outer_brt(state):
                self.phase = 2
                self.outer_t_remaining = self.outer_tMax
                self.outer_entry_time = sim_time
                print(f"[Cascaded MPC+Terminal] Entered outer BRT at "
                      f"t={sim_time:.2f}s -> Phase 2 "
                      f"(outer_t_remaining={self.outer_tMax:.1f}s)")

        if self.phase == 2:
            if self._is_in_inner_brt(state):
                self.phase = 3
                self.inner_t_remaining = self.inner_tMax
                self.inner_entry_time = sim_time
                print(f"[Cascaded MPC+Terminal] Entered inner BRT at "
                      f"t={sim_time:.2f}s -> Phase 3 "
                      f"(inner_t_remaining={self.inner_tMax:.1f}s)")

        if self.phase == 1:
            t_query = self.outer_tMax
            use_inner = False
            mode = 'outer'
        elif self.phase == 2:
            t_query = max(self.outer_t_remaining, 0.01)
            use_inner = False
            mode = 'outer'
        else:  # phase == 3
            t_query = max(self.inner_t_remaining, 0.01)
            use_inner = True
            mode = 'inner'

        self.phase_history.append(self.phase)
        if self.phase == 3:
            self.t_remaining_history.append(self.inner_t_remaining)
        elif self.phase == 2:
            self.t_remaining_history.append(self.outer_t_remaining)
        else:
            self.t_remaining_history.append(self.outer_tMax)
        self._last_t_query = t_query
        self._last_use_inner = use_inner

        return self.phase, t_query, use_inner, mode

    # ------------------------------------------------------------------
    # BRT optimal-control fallback
    # ------------------------------------------------------------------

    def _compute_brt_control(self, state):
        """
        Bang-bang optimal control from value function gradient (BRT fallback).

        Uses the outer model + dynamics for phases 1-2, or the inner model +
        inner_dynamics for phase 3.
        """
        if self._last_use_inner:
            model = self.inner_model
            dyn = self.inner_dynamics
        else:
            model = self.outer_model
            dyn = self.dynamics

        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), self._last_t_query, dtype=torch.float32,
            device=self.device)
        coord = torch.cat([time_col, s], dim=-1)
        model_input = dyn.coord_to_input(coord)

        result = model({'coords': model_input})
        output = result['model_out'].squeeze()
        model_in = result['model_in']

        dv = dyn.io_to_dv(model_in, output)
        dvds = dv[0, 1:].detach().cpu().numpy()

        u_x = -dyn.u_bar if dvds[2] > 0 else dyn.u_bar
        u_y = -dyn.u_bar if dvds[3] > 0 else dyn.u_bar
        u_theta = -dyn.u_theta_bar if dvds[5] > 0 else dyn.u_theta_bar
        return np.array([u_x, u_y, u_theta])

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run a full docking simulation."""
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn

        print(f"[Cascaded MPC+Terminal] Starting from state: {state}")

        docked = False
        collided = False

        # Stagnation detection state
        log_interval = 50          # log every 50 steps (5 s at dt=0.1)
        stagnation_thresh = 0.1    # metres improvement required per window
        prev_log_dist = None
        prev_log_V = None

        for step in range(num_steps):
            sim_time = step * self.dt

            # --- Control selection (mode-aware) ---
            if self._control_mode == 'brt_fallback':
                phase, t_query, use_inner, mode = \
                    self._update_phase(state, sim_time)
                control = self._compute_brt_control(state)
                if self._last_use_inner:
                    cost_val = self.get_value_inner(
                        state, self._last_t_query)
                else:
                    cost_val = self.get_value_outer(
                        state, self._last_t_query)
                if self.phase >= 2:
                    self.outer_t_remaining -= self.dt
                if self.phase == 3:
                    self.inner_t_remaining -= self.dt
            else:
                control, cost_val, mode = self._mpc_step(state, sim_time)

            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)
            self.terminal_mode_history.append(mode)

            # --- Periodic diagnostic logging + graduated stagnation ---
            if step % log_interval == 0:
                dist = float(np.sqrt(state[0]**2 + state[1]**2))
                if self._last_use_inner:
                    V_now = self.get_value_inner(state, self._last_t_query)
                    model_tag = 'inner'
                else:
                    V_now = self.get_value_outer(state, self._last_t_query)
                    model_tag = 'outer'
                cur_phase = self.phase_history[-1]
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

                    # --- Graduated stagnation response ---
                    if d_dist >= stagnation_thresh:
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

                ctrl_mode_tag = self._control_mode.upper()
                print(
                    f'[Cascaded MPC+Terminal] Step {step:>4d} t={sim_time:5.1f}s '
                    f'| Phase {cur_phase} ({model_tag}) '
                    f'| {ctrl_mode_tag} | dist={dist:.3f}m '
                    f'| V_{model_tag}(x,t)={V_now:.4f}\n'
                    f'  best_combined={cost_val:.4f}  '
                    f'reach_avoid={ra:.4f}  terminal={tv:.4f} '
                    f'({dominates}){delta_str}{stag_flag}')

                prev_log_dist = dist
                prev_log_V = V_now

            if self._check_docked(state):
                docked = True
                print(f"[Cascaded MPC+Terminal] Docking successful at t={sim_time:.2f}s")
                break

            if self._check_collision(state):
                collided = True
                print(f"[Cascaded MPC+Terminal] Collision at t={sim_time:.2f}s")
                break

            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot
            state[4] = np.arctan2(np.sin(state[4]), np.cos(state[4]))

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
            'controller_type': 'cascaded_mpc_terminal',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'terminal_modes': self.terminal_mode_history,
            'outer_entry_time': self.outer_entry_time,
            'inner_entry_time': self.inner_entry_time,
        }
        return result

    # ------------------------------------------------------------------
    # Core MPC + cascaded terminal cost
    # ------------------------------------------------------------------

    def _mpc_step(self, state, sim_time):
        """
        One MPC step with 3-phase cascaded terminal cost.

        Phase 1 (Approach):  V_outer(x, outer_tMax)  -- static
        Phase 2 (Transit):   V_outer(x, outer_t_remaining) -- countdown
        Phase 3 (Precision): V_inner(x, inner_t_remaining) -- countdown

        Returns:
            (first_control, best_cost, terminal_mode)
        """
        phase, t_query, use_inner, mode = self._update_phase(state, sim_time)

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

        best_cost_val = float('inf')
        for _ in range(self.num_refinement):
            state_trajs, permuted_controls = self.mpc.rollout_dynamics(
                state_tensor, start_iter=0,
                rollout_horizon=self.effective_horizon,
                eps_var_factor=self._exploration_factor)

            reach_avoid_cost = self.dynamics.cost_fn(state_trajs)

            terminal_states = state_trajs[:, :, -1, :]
            if use_inner:
                terminal_values = self._evaluate_terminal_values_inner(
                    terminal_states, t_query=t_query)
            else:
                terminal_values = self._evaluate_terminal_values_outer(
                    terminal_states, t_query=t_query)

            combined = torch.minimum(reach_avoid_cost, terminal_values)
            avoid_max = torch.max(
                -self.dynamics.avoid_fn(state_trajs), dim=-1).values
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

            best_costs, best_idx = combined.min(dim=1)
            best_cost_val = best_costs.item()

            # Store cost breakdown for the best sample (diagnostics)
            bi = best_idx[0]
            self._last_reach_avoid = reach_avoid_cost[0, bi].item()
            self._last_terminal = terminal_values[0, bi].item()

            idx_ctrl = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx_ctrl = idx_ctrl.expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3))
            best_controls = torch.gather(
                permuted_controls, dim=1, index=idx_ctrl).squeeze(1)
            self.mpc.control_tensors = best_controls.clone()

        # --- Decrement timers after using them ---
        if self.phase >= 2:
            self.outer_t_remaining -= self.dt
        if self.phase == 3:
            self.inner_t_remaining -= self.dt

        first_control = self.mpc.control_tensors[0, 0, :].detach().cpu().numpy()
        return first_control, best_cost_val, mode

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
