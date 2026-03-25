"""
RL-Based Controller for Docking13D

Loads a trained DDQNSingle Q-network and implements the standard
simulate_docking() interface for the 13D controller comparison framework.
Optionally applies a BRT least-restrictive safety filter.

Usage:
    controller = RLController13D(
        rl_checkpoint_path='../RLBaseline/experiments/.../model/Q-800000.pth',
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

# Project root for dynamics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from dynamics.dynamics import Docking13D
from utils.controllers.docking13d_mixin import Docking13DControllerMixin
from utils.controllers.safety_filter import SafetyFilter

# RL model
_RL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'RLBaseline'))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

from RARL.model import Model


class RLController13D(Docking13DControllerMixin):
    """13D docking controller using a trained DDQN reach-avoid Q-network.

    Inherits docking/collision checks from Docking13DControllerMixin.
    Implements simulate_docking() for drop-in use with run_controller_13d.py.
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

        # Dynamics (read-only, required by Docking13DControllerMixin)
        self.dynamics = Docking13D(set_mode='reach_avoid')

        # Build discrete action set — 6 axes × {-max, 0, +max} = 729
        F_bar = float(self.dynamics.F_bar)
        tau_bar = float(self.dynamics.tau_bar)
        force_levels = [-F_bar, 0.0, F_bar]
        torque_levels = [-tau_bar, 0.0, tau_bar]
        self.discrete_controls = np.array(
            list(itertools.product(
                force_levels, force_levels, force_levels,
                torque_levels, torque_levels, torque_levels)),
            dtype=np.float64)

        # Build and load Q-network
        state_dim = 13
        action_num = len(self.discrete_controls)  # 729
        dim_list = [state_dim] + list(architecture) + [action_num]
        self.Q_network = Model(dim_list, actType=activation)
        self.Q_network.load_state_dict(
            torch.load(rl_checkpoint_path, map_location=device, weights_only=True))
        self.Q_network.to(device)
        self.Q_network.eval()

        print(f"[RLController13D] Loaded {rl_checkpoint_path}")
        print(f"  arch={dim_list}, actions={action_num}, dt={dt}")

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run docking simulation using the RL policy.

        Args:
            initial_state: (13,) numpy array.
            max_sim_time: Maximum simulation time (seconds).
            dynamics_fn: Optional f(state, control) -> state_dot.

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

        for step in range(num_steps):
            sim_time = step * self.dt

            # Query Q-network
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.Q_network(s_t)
                action_idx = q_vals.min(dim=1)[1].item()
                value = q_vals.min(dim=1)[0].item()

            control = self.discrete_controls[action_idx].copy()

            # Apply safety filter
            control = self.safety_filter.apply(state, control)

            # Record
            states.append(state.copy())
            controls.append(control.copy())
            times.append(sim_time)
            values.append(value)

            # Termination checks (using mixin methods)
            if self._check_docked_13d(state):
                docked = True
                break

            if self._check_collision_13d(state):
                collided = True
                break

            # Integrate
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot

            # Normalize quaternion and clamp
            self._wrap_state_13d(state)

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
            'controller_type': 'rl_13d',
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            'n_clipped_steps': 0,
        }

    def _default_dynamics_fn(self, state, control):
        """Euler dynamics using Docking13D.dsdt."""
        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        dsdt = self.dynamics.dsdt(s_t, c_t, None)
        return dsdt.squeeze().cpu().numpy()
