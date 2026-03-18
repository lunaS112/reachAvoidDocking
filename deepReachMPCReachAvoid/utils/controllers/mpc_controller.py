"""
MPC-Only Controller for Docking6D

Direct-style MPC controller using random-shooting optimization with the
reachability cost function. No learned value function is used -- this serves
as a pure-MPC baseline for comparison against the BRAT controller.

At each simulation step the controller:
  1. Rolls out `num_samples` perturbed control sequences over the full
     planning horizon using `rollout_dynamics`.
  2. Evaluates the analytical reachability cost for each sample trajectory.
  3. Selects the best sample and updates the control plan.
  4. Warm-starts the next step by shifting the control sequence.

Usage:
    controller = MPCController(
        checkpoint_path='./runs/Docking6D_RA/training/checkpoints/model_final.pth',
        planning_horizon_sec=20.0,
        mpc_dt=0.5,
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

from utils.MPC import MPC
from dynamics import dynamics as dynamics_module
from utils.controllers.safety_filter import SafetyFilter
from utils.controllers import clip_state_for_execution


class MPCController:
    """
    Direct-style MPC controller using random-shooting optimization.

    At each simulation step the controller:
      1. Rolls out ``num_samples`` perturbed control sequences over the full
         ``planning_horizon_steps`` horizon via ``rollout_dynamics``.
      2. Evaluates the analytical reachability cost for each trajectory.
      3. Selects the best sample and updates the control plan.
      4. Applies the first control and warm-starts the next step.

    No learned value function is used; cost evaluation relies solely on the
    analytical reach_fn / avoid_fn defined in the dynamics class.
    """

    def __init__(self, checkpoint_path, planning_horizon_sec=20.0,
                 mpc_dt=0.5, dt=0.1,
                 num_samples=100, num_refinement=10, device='cuda',
                 cost_type='reachability', safety_filter=None):
        """
        Args:
            checkpoint_path:      Path to a trained model checkpoint (used only
                                  to load the dynamics config from orig_opt.pickle).
            planning_horizon_sec: MPC planning horizon in seconds (default 20).
            mpc_dt:               MPC planning timestep in seconds (default 0.5).
            dt:                   Simulation integration timestep (default 0.1).
            num_samples:          Random-shooting samples per rollout (default 100).
            num_refinement:       Iterative refinement passes (default 10).
            device:               Torch device ('cuda' or 'cpu').
            cost_type:            Cost for trajectory selection ('reachability').
        """
        self.checkpoint_path = checkpoint_path
        self.planning_horizon_sec = planning_horizon_sec
        self.mpc_dt = mpc_dt
        self.dt = dt
        self.planning_horizon_steps = int(planning_horizon_sec / mpc_dt)
        self.num_samples = num_samples
        self.num_refinement = num_refinement
        self.device = device
        self.cost_type = cost_type

        # Derive experiment directory from checkpoint path
        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        # Load dynamics
        self._load_dynamics()

        # Instantiate the MPC engine (direct style: rollout full horizon)
        self.mpc = MPC(
            dT=self.mpc_dt,
            horizon=self.planning_horizon_steps,
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

        # Control state
        self.reset()

        print(f"MPCController initialised  |  horizon={self.planning_horizon_sec}s  "
              f"mpc_dt={self.mpc_dt}s  sim_dt={self.dt}s  "
              f"horizon_steps={self.planning_horizon_steps}  "
              f"samples={self.num_samples}  refinements={self.num_refinement}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_dynamics(self):
        """Load the dynamics class from the experiment's saved options."""
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)

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

    def reset(self):
        """Reset controller state for a new simulation."""
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.sim_time_history = []
        self._warm_started = False
        self.safety_filter.reset()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """
        Run a full docking simulation using direct-style MPC.

        The MPC replans at every simulation step (dt).

        Args:
            initial_state: Initial state [px, py, vx, vy, theta, omega].
            max_sim_time:  Maximum simulation duration in seconds.
            dynamics_fn:   Optional f(state, control) -> state_dot.
                           Defaults to Docking6D CW equations.

        Returns:
            dict with the shared result format.
        """
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn

        print(f"[MPC] Starting from state: {state}")

        docked = False
        collided = False
        n_clipped = 0

        for step in range(num_steps):
            sim_time = step * self.dt

            # --- MPC replan every step ---
            control, cost_val = self._mpc_step(state)

            # Post-process through safety filter (no-op when mode=0)
            control = self.safety_filter.apply(state, control)

            # Record history
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)

            # Termination checks
            if self._check_docked(state):
                docked = True
                print(f"[MPC] Docking successful at t={sim_time:.2f}s")
                break

            if self._check_collision(state):
                collided = True
                print(f"[MPC] Collision detected at t={sim_time:.2f}s")
                break

            # Integrate dynamics at fine dt (Euler)
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot
            state, clipped = clip_state_for_execution(state, self.dynamics)
            if clipped:
                n_clipped += 1

        wall_time = time.perf_counter() - t_wall_start

        # Compute control effort: sum(||u_k||_2 * dt)
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
            'controller_type': 'mpc',
            'control_effort': control_effort,
            'wall_time': wall_time,
            # --- safety filter ---
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            # --- execution clamping ---
            'n_clipped_steps': n_clipped,
        }
        return result

    # ------------------------------------------------------------------
    # Core MPC logic (direct style)
    # ------------------------------------------------------------------

    def _mpc_step(self, state):
        """
        Run one direct-style MPC optimisation step and return
        (first_control, best_cost).

        Rolls out the full planning horizon, selects the best sample,
        and warm-starts the next step.
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.mpc.batch_size = 1
        self.mpc.horizon = self.planning_horizon_steps

        if not self._warm_started:
            self.mpc.init_control_tensors()
            self._warm_started = True
        else:
            # Shift and pad (warm-start)
            shifted = self.mpc.control_tensors[:, 1:, :].clone()
            pad = torch.zeros(
                1, 1, self.dynamics.control_dim, device=self.device)
            self.mpc.control_tensors = torch.cat([shifted, pad], dim=1)

        best_cost_val = float('inf')
        for _ in range(self.num_refinement):
            # Roll out num_samples perturbed trajectories over full horizon
            # state_trajs: (1, N, H+1, D),  permuted_controls: (1, N, H, D_u)
            state_trajs, permuted_controls = self.mpc.rollout_dynamics(
                state_tensor, start_iter=0,
                rollout_horizon=self.planning_horizon_steps)

            # Analytical reachability cost (no learned terminal cost)
            costs = self.dynamics.cost_fn(state_trajs)  # (1, N)

            # Select best sample (argmin for reach / reach_avoid)
            best_costs, best_idx = costs.min(dim=1)     # (1,)
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

    def _default_dynamics_fn(self, state, control):
        """Evaluate Docking6D CW equations."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()

    def _check_docked(self, state):
        """Check if all docking tolerances are met."""
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
