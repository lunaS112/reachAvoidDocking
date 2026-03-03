"""
MPC-Only Controller for Docking13D

Direct-style MPC controller using random-shooting optimisation with the
analytical reachability cost function.  No learned value function is used —
this serves as a pure-MPC baseline for comparison.

At each simulation step the controller:
  1. Rolls out ``num_samples`` perturbed control sequences over the full
     planning horizon via ``rollout_dynamics``.
  2. Evaluates the analytical reachability cost for each trajectory.
  3. Selects the best sample and updates the control plan.
  4. Applies the first control and warm-starts the next step.

Usage:
    controller = MPCController13D(
        checkpoint_path='./runs/Docking13D_RA/training/checkpoints/model_final.pth',
        planning_horizon_sec=20.0,
        mpc_dt=0.5,
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

from utils.MPC import MPC
from dynamics import dynamics as dynamics_module
from utils.controllers.docking13d_mixin import Docking13DControllerMixin


class MPCController13D(Docking13DControllerMixin):
    """Direct-style random-shooting MPC controller for 13D docking (no learned VF)."""

    def __init__(self, checkpoint_path, planning_horizon_sec=20.0,
                 mpc_dt=0.5, dt=0.1,
                 num_samples=300, num_refinement=15, device='cuda',
                 cost_type='reachability'):
        """
        Args:
            checkpoint_path:      Path to trained checkpoint (used only to
                                  load dynamics config from orig_opt.pickle).
            planning_horizon_sec: MPC planning horizon (seconds).
            mpc_dt:               MPC planning timestep (seconds).
            dt:                   Simulation integration timestep (seconds).
            num_samples:          Random-shooting samples per rollout.
            num_refinement:       Iterative refinement passes.
            device:               Torch device.
            cost_type:            Cost type for trajectory selection.
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

        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        self._load_dynamics()

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

        self.reset()

        print(f"MPCController13D initialised | horizon={self.planning_horizon_sec}s "
              f"mpc_dt={self.mpc_dt}s  sim_dt={self.dt}s  "
              f"horizon_steps={self.planning_horizon_steps}  "
              f"samples={self.num_samples}  refinements={self.num_refinement}")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_dynamics(self):
        """Load dynamics class from saved experiment options."""
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)

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

    def reset(self):
        """Reset controller state for a new simulation."""
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.sim_time_history = []
        self._warm_started = False

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run a full docking simulation using direct-style MPC.

        Returns:
            dict with the shared result format.
        """
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn_13d

        print(f"  [MPC13D] Starting  dist={np.linalg.norm(state[:3]):.3f}m")

        docked = False
        collided = False
        dock_time = None
        post_dock_duration = 1.0  # seconds to continue after docking

        for step in range(num_steps):
            sim_time = step * self.dt

            # Stop after post-dock coast period
            if docked and (sim_time - dock_time) >= post_dock_duration:
                break

            # Keep MPC active even after docking so chaser can
            # converge closer to the goal point.
            control, cost_val = self._mpc_step(state)

            # Record
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)

            # Termination (only before docking)
            if not docked and self._check_docked_13d(state):
                docked = True
                dock_time = sim_time
                print(f"  [MPC13D] Docking at t={sim_time:.2f}s")

            if not docked and self._check_collision_13d(state):
                collided = True
                print(f"  [MPC13D] Collision at t={sim_time:.2f}s")
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
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'mpc_13d',
            'control_effort': control_effort,
            'wall_time': wall_time,
        }

    # ------------------------------------------------------------------
    # Core MPC logic
    # ------------------------------------------------------------------

    def _mpc_step(self, state):
        """One direct-style MPC optimisation step.

        Returns:
            (first_control, best_cost)
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.mpc.batch_size = 1
        self.mpc.horizon = self.planning_horizon_steps

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
                rollout_horizon=self.planning_horizon_steps)

            costs = self.dynamics.cost_fn(state_trajs)  # (1, N)
            best_costs, best_idx = costs.min(dim=1)
            best_cost_val = best_costs.item()

            idx_ctrl = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx_ctrl = idx_ctrl.expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3))
            best_controls = torch.gather(
                permuted_controls, dim=1, index=idx_ctrl).squeeze(1)
            self.mpc.control_tensors = best_controls.clone()

        first_control = self.mpc.control_tensors[0, 0, :].detach().cpu().numpy()
        return first_control, best_cost_val
