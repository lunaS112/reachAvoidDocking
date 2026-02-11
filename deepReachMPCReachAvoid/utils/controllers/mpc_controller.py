"""
MPC-Only Controller for Docking6D

Receding-horizon MPC controller using random-shooting optimization with the
reachability cost function. No learned value function is used -- this serves
as a pure-MPC baseline for comparison against the BRT controller.

The MPC operates at a coarser planning timestep (mpc_dt, default 0.5 s) while
the simulation integrates at a finer timestep (dt, default 0.1 s).  Between
MPC replanning steps the control is held constant.

Configuration matches the working MPC setup in MPC_values_viz.py:
  style='receding', receding_horizon=1, dT=0.5, T=20.

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


class MPCController:
    """
    Receding-horizon MPC controller using random-shooting optimization.

    At each MPC planning step the controller calls ``get_opt_trajs`` which
    builds up an optimised trajectory step-by-step (receding style):
      1. At each planning step, roll out ``num_samples`` perturbed control
         sequences over the remaining horizon.
      2. Select the best sample and commit its first control.
      3. Advance the internal trajectory and repeat for the full horizon.

    Only the *first* control from the resulting trajectory is applied to the
    simulation.  The simulation integrates at the finer ``dt`` while the MPC
    replans every ``mpc_dt`` seconds.

    No learned value function is used; cost evaluation relies solely on the
    analytical reach_fn / avoid_fn defined in the dynamics class.
    """

    def __init__(self, checkpoint_path, planning_horizon_sec=20.0,
                 mpc_dt=0.5, dt=0.1,
                 num_samples=100, num_refinement=10, device='cuda',
                 cost_type='reachability'):
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
        self.num_samples = num_samples
        self.num_refinement = num_refinement
        self.device = device
        self.cost_type = cost_type

        # How many simulation steps between MPC replans
        self.mpc_steps_per_sim = max(1, round(mpc_dt / dt))

        # Derive experiment directory from checkpoint path
        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        # Load dynamics
        self._load_dynamics()

        # Instantiate the MPC engine (matching MPC_values_viz.py configuration)
        self.mpc = MPC(
            dT=self.mpc_dt,
            horizon=None,              # set internally by get_opt_trajs
            receding_horizon=1,        # commit one planning step at a time
            num_samples=self.num_samples,
            dynamics_=self.dynamics,
            device=self.device,
            mode='MPC',
            sample_mode='gaussian',
            style='receding',          # receding, not direct
            num_iterative_refinement=self.num_refinement,
            cost_type=self.cost_type,
        )

        # Control state
        self.reset()

        print(f"MPCController initialised  |  horizon={self.planning_horizon_sec}s  "
              f"mpc_dt={self.mpc_dt}s  sim_dt={self.dt}s  "
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

    def reset(self):
        """Reset controller state for a new simulation."""
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.sim_time_history = []

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """
        Run a full docking simulation using receding-horizon MPC.

        The MPC replans every ``mpc_dt`` seconds.  Between replans the control
        is held constant while the simulation integrates at the finer ``dt``.

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
        current_control = np.zeros(self.dynamics.control_dim)
        current_cost = 0.0

        for step in range(num_steps):
            sim_time = step * self.dt

            # --- MPC replan at the coarser mpc_dt cadence ---
            if step % self.mpc_steps_per_sim == 0:
                current_control, current_cost = self._mpc_step(state)

            # Record history
            self.state_history.append(state.copy())
            self.control_history.append(current_control.copy())
            self.value_history.append(current_cost)
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
            state_dot = dynamics_fn(state, current_control)
            state = state + self.dt * state_dot
            state[4] = np.arctan2(np.sin(state[4]), np.cos(state[4]))

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
        }
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mpc_step(self, state):
        """
        Run a full receding-horizon trajectory optimisation from *state* and
        return the first control along with the reach-avoid cost.

        This mirrors how ``MPC_values_viz.py`` calls ``get_opt_trajs``.
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Configure MPC for this call (get_opt_trajs reads these attributes)
        self.mpc.T = self.planning_horizon_sec
        self.mpc.batch_size = 1

        # Run receding-style trajectory optimisation
        with torch.no_grad():
            state_trajs, avoid_values, reach_values, num_iters = \
                self.mpc.get_opt_trajs(state_tensor, policy=None, t=0.0)

        # Extract the first control (committed at the first planning step)
        first_control = self.mpc.control_tensors[0, 0, :].detach().cpu().numpy()

        # Compute reach-avoid cost (same formula as get_batch_data line 61)
        cost = torch.min(
            torch.maximum(
                reach_values,
                torch.cummax(-avoid_values, dim=-1).values),
            dim=-1).values[0].item()

        return first_control, cost

    def _default_dynamics_fn(self, state, control):
        """Evaluate Docking6D CW equations."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()

    def _check_docked(self, state):
        """Check if all docking tolerances are met."""
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
        """Check if the chaser is inside the failure set."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.avoid_fn(s).item() < 0
