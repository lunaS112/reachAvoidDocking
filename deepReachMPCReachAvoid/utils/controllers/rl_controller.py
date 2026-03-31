"""
RL-Based Controller for Docking6D

Loads a trained DDQNSingle Q-network and implements the standard
simulate_docking() interface for the controller comparison framework.
Optionally applies a BRT least-restrictive safety filter.

Usage:
    controller = RLController(
        rl_checkpoint_path='../RLBaseline/experiments/.../model/Q-400000.pth',
        architecture=[256, 256],
    )
    result = controller.simulate_docking(initial_state, max_sim_time=30.0)
"""

import itertools
import os
import sys
import time as _time

import numpy as np
import torch

# Project root for dynamics import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from dynamics.dynamics import Docking6D
from utils.controllers.safety_filter import SafetyFilter
from utils.controllers import clip_state_for_execution

# RL model
_RL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'RLBaseline'))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

from RARL.model import Model


class RLController:
    """6D docking controller using a trained DDQN reach-avoid Q-network.

    Implements simulate_docking() for drop-in use with run_controller.py.
    """

    def __init__(self, rl_checkpoint_path, dt=0.1, device='cuda',
                 safety_filter=None, architecture=None, activation='Tanh'):
        """
        Args:
            rl_checkpoint_path: Path to Q-network .pth file.
            dt: Control timestep (must match training).
            device: Torch device.
            safety_filter: SafetyFilter instance or None.
            architecture: Hidden layer dims (must match training).
            activation: Activation function (must match training).
        """
        if architecture is None:
            architecture = [256, 256]

        self.dt = dt
        self.device = device
        self.safety_filter = safety_filter or SafetyFilter(mode=0)

        # Dynamics (read-only)
        self.dynamics = Docking6D(set_mode='reach_avoid')

        # Build discrete action set (must match training env exactly)
        u_bar = float(self.dynamics.u_bar)
        u_theta_bar = float(self.dynamics.u_theta_bar)
        force_levels = [-u_bar, 0.0, u_bar]
        torque_levels = [-u_theta_bar, 0.0, u_theta_bar]
        self.discrete_controls = np.array(
            list(itertools.product(force_levels, force_levels, torque_levels)),
            dtype=np.float64)

        # Build and load Q-network
        state_dim = 6
        action_num = len(self.discrete_controls)  # 27
        dim_list = [state_dim] + list(architecture) + [action_num]
        self.Q_network = Model(dim_list, actType=activation)
        self.Q_network.load_state_dict(
            torch.load(rl_checkpoint_path, map_location=device, weights_only=True))
        self.Q_network.to(device)
        self.Q_network.eval()

        self._last_q_value = 0.0

        print(f"[RLController] Loaded {rl_checkpoint_path}")
        print(f"  arch={dim_list}, actions={action_num}, dt={dt}")

    def reset(self):
        """Reset controller state for a new simulation."""
        self.safety_filter.reset()
        self._last_q_value = 0.0

    def u_fn(self, state, sim_time):
        """Compute control action from state using the Q-network.

        Caches the min Q-value in ``self._last_q_value`` for diagnostic
        logging without requiring a second forward pass.
        """
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.Q_network(s_t)
            self._last_q_value = q_vals.min(dim=1)[0].item()
            action_idx = q_vals.min(dim=1)[1].item()
        return self.discrete_controls[action_idx].copy()

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run docking simulation using the RL policy.

        Args:
            initial_state: (6,) numpy array [px, py, vx, vy, theta, omega].
            max_sim_time: Maximum simulation time (seconds).
            dynamics_fn: Optional f(state, control) -> state_dot. If None,
                uses Docking6D.dsdt via Euler integration.

        Returns:
            dict with standardized comparison fields.
        """
        self.safety_filter.reset()
        t_wall_start = _time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn

        states, controls, times, values = [], [], [], []
        docked = False
        collided = False
        n_clipped = 0

        for step in range(num_steps):
            sim_time = step * self.dt

            control = self.u_fn(state, sim_time)
            value = self._last_q_value

            # Apply safety filter
            control = self.safety_filter.apply(state, control)

            # Record
            states.append(state.copy())
            controls.append(control.copy())
            times.append(sim_time)
            values.append(value)

            # Termination checks
            if self._check_docked(state):
                docked = True
                break

            if self._check_collision(state):
                collided = True
                break

            # Integrate
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot

            # Wrap/clamp
            state, clipped = clip_state_for_execution(state, self.dynamics)
            if clipped:
                n_clipped += 1

        wall_time = _time.perf_counter() - t_wall_start

        controls_arr = np.array(controls)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        return {
            'trajectory': np.array(states),
            'controls': controls_arr,
            'times': np.array(times),
            'values': np.array(values),
            'success': docked and not collided,
            'docked': docked,
            'collision': collided,
            'final_state': state,
            'control_effort': control_effort,
            'wall_time': wall_time,
            'controller_type': 'rl',
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            'n_clipped_steps': n_clipped,
        }

    def _default_dynamics_fn(self, state, control):
        """Euler dynamics using Docking6D.dsdt."""
        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        dsdt = self.dynamics.dsdt(s_t, c_t, None)
        return dsdt.squeeze().cpu().numpy()

    def _check_docked(self, state):
        """Check if all docking tolerances are met (same as BRATController)."""
        d = self.dynamics
        px, py, vx, vy, theta, omega = state
        x_ok = np.abs(px) <= d.eps_p
        y_ok = d.goal_y_min <= py <= d.goal_y_max
        vel_ok = np.sqrt(vx**2 + vy**2) <= d.eps_v
        theta_goal = d.goal_state[4].item()
        omega_goal = d.goal_state[5].item()
        theta_diff = np.abs(np.arctan2(
            np.sin(theta - theta_goal), np.cos(theta - theta_goal)))
        theta_ok = theta_diff <= d.eps_theta
        omega_ok = np.abs(omega - omega_goal) <= d.eps_omega
        return (x_ok and y_ok) and vel_ok and theta_ok and omega_ok

    def _check_collision(self, state):
        """Orientation-aware collision check."""
        return self.dynamics.check_collision_oriented(state)
