"""
Cascaded MPC + Terminal Cost Controller for Docking6D

Short-horizon MPC that switches the terminal cost value function based on
proximity to the goal:
- Far from goal: V_outer(x, tMax_outer) as terminal cost
- Inside inner BRT: V_inner(x, tMax_inner) as terminal cost (finer resolution)

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
    MPC + terminal cost controller that switches between outer and inner
    value functions based on whether the current state is inside the inner BRT.

    At each step:
      1. Check if current state satisfies V_inner(x, tMax_inner) <= 0
      2. If yes: use V_inner as terminal cost (precision mode)
      3. If no:  use V_outer as terminal cost (approach mode)

    The MPC rollout engine and dynamics come from the outer model.
    The inner model is loaded separately for terminal cost evaluation.
    """

    def __init__(self, outer_checkpoint, inner_checkpoint,
                 effective_horizon_sec=2.0, outer_tMax=14.0, inner_tMax=3.0,
                 dt=0.1, num_samples=500, num_refinement=10, device='cuda',
                 cost_type='reachability'):
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

        print(f"CascadedMPCTerminalController initialised  |  "
              f"horizon={effective_horizon_sec}s  outer_tMax={outer_tMax}s  "
              f"inner_tMax={inner_tMax}s  samples={num_samples}")

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
        self.terminal_mode_history = []  # 'outer' or 'inner'
        self._warm_started = False

    # ------------------------------------------------------------------
    # Terminal cost evaluation
    # ------------------------------------------------------------------

    def _is_in_inner_brt(self, state_np):
        """Check if a single state (numpy) is inside the inner BRT."""
        s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full((1, 1), self.inner_tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.inner_dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.inner_model({'coords': model_input})
            output = result['model_out'].squeeze()
        value = self.inner_dynamics.io_to_value(model_input, output)
        return value.item() <= 0

    def _evaluate_terminal_values_outer(self, terminal_states):
        """Evaluate V_outer(x, tMax_outer) for a batch of terminal states."""
        A, N, D = terminal_states.shape
        flat_states = terminal_states.reshape(A * N, D)

        test_range = torch.tensor(self.dynamics.state_test_range(), device=self.device)
        flat_states_clamped = torch.clamp(flat_states, test_range[..., 0], test_range[..., 1])

        time_col = torch.full((A * N, 1), self.outer_tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_states_clamped], dim=-1)

        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.outer_model({'coords': model_input})
            output = result['model_out'].squeeze(-1)
        values = self.dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

    def _evaluate_terminal_values_inner(self, terminal_states):
        """Evaluate V_inner(x, tMax_inner) for a batch of terminal states."""
        A, N, D = terminal_states.shape
        flat_states = terminal_states.reshape(A * N, D)

        test_range = torch.tensor(self.inner_dynamics.state_test_range(), device=self.device)
        flat_states_clamped = torch.clamp(flat_states, test_range[..., 0], test_range[..., 1])

        time_col = torch.full((A * N, 1), self.inner_tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_states_clamped], dim=-1)

        model_input = self.inner_dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.inner_model({'coords': model_input})
            output = result['model_out'].squeeze(-1)
        values = self.inner_dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

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

        for step in range(num_steps):
            sim_time = step * self.dt

            control, cost_val, mode = self._mpc_step(state)

            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)
            self.terminal_mode_history.append(mode)

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
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'cascaded_mpc_terminal',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'terminal_modes': self.terminal_mode_history,
        }
        return result

    # ------------------------------------------------------------------
    # Core MPC + cascaded terminal cost
    # ------------------------------------------------------------------

    def _mpc_step(self, state):
        """
        One MPC step with cascaded terminal cost.

        Returns:
            (first_control, best_cost, terminal_mode)
        """
        # Decide which model to use for terminal cost
        use_inner = self._is_in_inner_brt(state)
        mode = 'inner' if use_inner else 'outer'

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
                rollout_horizon=self.effective_horizon)

            reach_avoid_cost = self.dynamics.cost_fn(state_trajs)

            terminal_states = state_trajs[:, :, -1, :]
            if use_inner:
                terminal_values = self._evaluate_terminal_values_inner(terminal_states)
            else:
                terminal_values = self._evaluate_terminal_values_outer(terminal_states)

            combined = torch.minimum(reach_avoid_cost, terminal_values)
            avoid_max = torch.max(
                -self.dynamics.avoid_fn(state_trajs), dim=-1).values
            combined = torch.maximum(combined, avoid_max)

            best_costs, best_idx = combined.min(dim=1)
            best_cost_val = best_costs.item()

            idx_ctrl = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx_ctrl = idx_ctrl.expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3))
            best_controls = torch.gather(
                permuted_controls, dim=1, index=idx_ctrl).squeeze(1)
            self.mpc.control_tensors = best_controls.clone()

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
