"""
MPC + Learned Terminal Cost Controller for Docking6D

Receding-horizon MPC with a short effective planning horizon, augmented by the
learned DeepReach value function V(x, tMax) as terminal cost.  The MPC handles
short-term trajectory optimisation while V(x, tMax) provides long-term
guidance toward the docking goal.

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
                 cost_type='reachability'):
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

        self.reset()

        print(f"MPCTerminalController initialised  |  "
              f"horizon={self.effective_horizon_sec}s  tMax={self.tMax}s  "
              f"samples={self.num_samples}  refinements={self.num_refinement}")

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
        self._warm_started = False

    # ------------------------------------------------------------------
    # Terminal cost evaluation
    # ------------------------------------------------------------------

    def _evaluate_terminal_values(self, terminal_states):
        """
        Evaluate V(x, tMax) for a batch of terminal states.

        Args:
            terminal_states: (A, N, state_dim) tensor of terminal states
                             where A=batch_size (1) and N=num_samples.

        Returns:
            (A, N) tensor of terminal values.
        """
        A, N, D = terminal_states.shape

        # Flatten to (A*N, D)
        flat_states = terminal_states.reshape(A * N, D)

        # Clamp to state test range to avoid out-of-distribution queries
        test_range = torch.tensor(
            self.dynamics.state_test_range(), device=self.device)
        flat_states_clamped = torch.clamp(
            flat_states, test_range[..., 0], test_range[..., 1])

        # Build coordinate tensor [tMax, state]
        time_col = torch.full(
            (A * N, 1), self.tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_states_clamped], dim=-1)

        # Normalise and forward pass
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        values = self.dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

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

        for step in range(num_steps):
            sim_time = step * self.dt

            # --- MPC + terminal cost step ---
            control, cost_val = self._mpc_step(state)

            # Record history
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)

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
            'controller_type': 'mpc_terminal',
            'control_effort': control_effort,
            'wall_time': wall_time,
        }
        return result

    # ------------------------------------------------------------------
    # Core MPC + terminal cost logic
    # ------------------------------------------------------------------

    def _mpc_step(self, state):
        """
        Run one MPC optimisation step with terminal cost and return
        (first_control, best_combined_cost).
        """
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
                rollout_horizon=self.effective_horizon)

            # --- Reachability cost over short horizon ---
            # cost_fn expects (..., H+1, D) -> (...) so we pass (1, N, H+1, D)
            reach_avoid_cost = self.dynamics.cost_fn(state_trajs)  # (1, N)

            # --- Terminal cost from learned value function ---
            terminal_states = state_trajs[:, :, -1, :]        # (1, N, D)
            terminal_values = self._evaluate_terminal_values(
                terminal_states)                               # (1, N)

            # --- Combine costs (reach-avoid formulation) ---
            # Pattern from MPC.py warm_start_with_policy (lines 206-223)
            combined = torch.minimum(reach_avoid_cost, terminal_values)
            avoid_max = torch.max(
                -self.dynamics.avoid_fn(state_trajs), dim=-1).values  # (1, N)
            combined = torch.maximum(combined, avoid_max)

            # --- Select best sample (argmin for reach / reach_avoid) ---
            best_costs, best_idx = combined.min(dim=1)        # (1,)
            best_cost_val = best_costs.item()

            # Update control tensors with the best sample's controls
            idx_ctrl = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx_ctrl = idx_ctrl.expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3))
            best_controls = torch.gather(
                permuted_controls, dim=1, index=idx_ctrl).squeeze(1)  # (1,H,Du)
            self.mpc.control_tensors = best_controls.clone()

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
