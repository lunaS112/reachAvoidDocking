"""
Cascaded BRT Controller for Docking6D

Uses two learned value functions in a 3-phase control strategy:
- Phase 1 (Approach): Outside outer BRT, use V_outer(x, tMax) gradient to steer toward outer BRT
- Phase 2 (Transit):  Inside outer BRT, use time-varying V_outer(x, t_remaining) toward inner BRT
- Phase 3 (Precision): Inside inner BRT, use time-varying V_inner(x, t_remaining) for docking

The outer model (e.g., tMax=15s) handles long-range approach.
The inner model (e.g., tMax=3s) handles final precision docking.

Usage:
    controller = CascadedBRTController(
        outer_checkpoint='./runs/Docking6D_RA/training/checkpoints/model_final.pth',
        inner_checkpoint='./runs/Docking6D_Inner/training/checkpoints/model_final.pth',
        outer_tMax=14.0,
        inner_tMax=3.0,
    )
    result = controller.simulate_docking(initial_state, max_sim_time=30.0)
"""

import time as _time

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils.controllers.brt_controller import BRTController


class CascadedBRTController:
    """
    Cascaded two-BRT controller for precision docking.

    Three phases:
      Phase 1 (Approach):  V_outer(x, tMax_outer) > 0 -- use outer gradient at fixed tMax
      Phase 2 (Transit):   Inside outer BRT -- use V_outer(x, t_remaining) with decrementing time
      Phase 3 (Precision): Inside inner BRT -- use V_inner(x, t_remaining) with decrementing time

    Phase transitions:
      1 -> 2: V_outer(x, tMax_outer) <= 0
      2 -> 3: V_inner(x, tMax_inner) <= 0
      3 -> 2: Exit inner BRT (reacquisition fails) AND still inside outer BRT
      3 -> 1: Exit inner BRT AND outer BRT
      2 -> 1: Exit outer BRT (reacquisition fails)
    """

    def __init__(self, outer_checkpoint, inner_checkpoint,
                 outer_tMax=14.0, inner_tMax=3.0, dt=0.1, device='cuda',
                 search_window=0.5, search_resolution=0.1, max_search_expansions=1):
        """
        Args:
            outer_checkpoint: Path to trained outer (long-horizon) model checkpoint.
            inner_checkpoint: Path to trained inner (short-horizon) model checkpoint.
            outer_tMax: Time horizon for outer BRT queries.
            inner_tMax: Time horizon for inner BRT queries.
            dt: Control update frequency in seconds.
            device: Torch device.
            search_window: BRT reacquisition search half-width (seconds).
            search_resolution: BRT reacquisition search step (seconds).
            max_search_expansions: Max reacquisition window expansions.
        """
        self.outer = BRTController(
            outer_checkpoint, tMax=outer_tMax, dt=dt, device=device,
            search_window=search_window, search_resolution=search_resolution,
            max_search_expansions=max_search_expansions,
        )
        self.inner = BRTController(
            inner_checkpoint, tMax=inner_tMax, dt=dt, device=device,
            search_window=search_window, search_resolution=search_resolution,
            max_search_expansions=max_search_expansions,
        )
        self.dt = dt
        self.device = device
        self.dynamics = self.outer.dynamics
        self.reset()

        print(f"CascadedBRTController initialised  |  "
              f"outer_tMax={outer_tMax}s  inner_tMax={inner_tMax}s  dt={dt}s")

    def reset(self):
        """Reset controller state for a new simulation."""
        self.phase = 1  # 1=approach, 2=outer tracking, 3=inner precision
        self.outer_t_remaining = self.outer.tMax
        self.inner_t_remaining = self.inner.tMax
        self.outer_entry_time = None
        self.inner_entry_time = None

        # History tracking
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.phase_history = []
        self.t_remaining_history = []
        self.sim_time_history = []

        # Reacquisition tracking
        self.outer_reacquisition_count = 0
        self.inner_reacquisition_count = 0

    def is_in_outer_brt(self, state):
        """Check if state is inside the outer BRT."""
        return self.outer.get_value(state, self.outer.tMax) <= 0

    def is_in_inner_brt(self, state):
        """Check if state is inside the inner BRT."""
        return self.inner.get_value(state, self.inner.tMax) <= 0

    def u_fn(self, state, sim_time):
        """
        Control function implementing the 3-phase cascaded strategy.

        Args:
            state: Current state [px, py, vx, vy, theta, omega]
            sim_time: Current simulation time

        Returns:
            numpy array: Control [u_x, u_y, u_theta]
        """
        # --- Phase transition checks (bottom-up: check inner first) ---
        if self.phase <= 2:
            if self.is_in_inner_brt(state):
                if self.phase < 3:
                    self.phase = 3
                    self.inner_t_remaining = self.inner.tMax
                    self.inner_entry_time = sim_time
                    print(f"  [Cascaded] Entered inner BRT at t={sim_time:.2f}s -> Phase 3")

        if self.phase == 1:
            if self.is_in_outer_brt(state):
                self.phase = 2
                self.outer_t_remaining = self.outer.tMax
                self.outer_entry_time = sim_time
                print(f"  [Cascaded] Entered outer BRT at t={sim_time:.2f}s -> Phase 2")

        # --- Compute control based on current phase ---
        if self.phase == 3:
            query_time = max(self.inner_t_remaining, 0.01)
            value = self.inner.get_value(state, query_time)

            if value > 0:
                # Exited inner BRT -- try reacquisition
                new_time = self.inner._search_brt_time(state, query_time)
                if new_time is not None:
                    self.inner_t_remaining = new_time
                    query_time = new_time
                    self.inner_reacquisition_count += 1
                else:
                    # Fall back: check if still inside outer BRT
                    if self.is_in_outer_brt(state):
                        self.phase = 2
                        # Search for tightest outer BRT time slice
                        outer_time = self.outer._search_brt_time(state, self.outer_t_remaining)
                        if outer_time is not None:
                            self.outer_t_remaining = outer_time
                        print(f"  [Cascaded] Exited inner BRT at t={sim_time:.2f}s -> Phase 2")
                    else:
                        self.phase = 1
                        print(f"  [Cascaded] Exited both BRTs at t={sim_time:.2f}s -> Phase 1")

        if self.phase == 3:
            query_time = max(self.inner_t_remaining, 0.01)
            control = self.inner.get_optimal_control(state, query_time)
            value = self.inner.get_value(state, query_time)
            self.inner_t_remaining -= self.dt
            t_rem = self.inner_t_remaining

        elif self.phase == 2:
            query_time = max(self.outer_t_remaining, 0.01)
            value = self.outer.get_value(state, query_time)

            if value > 0:
                # Exited outer BRT -- try reacquisition
                new_time = self.outer._search_brt_time(state, query_time)
                if new_time is not None:
                    self.outer_t_remaining = new_time
                    query_time = new_time
                    self.outer_reacquisition_count += 1
                else:
                    self.phase = 1
                    print(f"  [Cascaded] Exited outer BRT at t={sim_time:.2f}s -> Phase 1")

            if self.phase == 2:
                control = self.outer.get_optimal_control(state, query_time)
                value = self.outer.get_value(state, query_time)
                self.outer_t_remaining -= self.dt
                t_rem = self.outer_t_remaining
            else:
                # Fell back to phase 1, compute below
                pass

        if self.phase == 1:
            control = self.outer.get_optimal_control(state, self.outer.tMax)
            value = self.outer.get_value(state, self.outer.tMax)
            t_rem = self.outer.tMax

        # Record history
        self.state_history.append(state.copy() if isinstance(state, np.ndarray) else state)
        self.control_history.append(control)
        self.value_history.append(value)
        self.phase_history.append(self.phase)
        self.t_remaining_history.append(t_rem)
        self.sim_time_history.append(sim_time)

        return control

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """
        Run full docking simulation with cascaded BRT control.

        Args:
            initial_state: Initial state vector.
            max_sim_time: Maximum simulation time in seconds.
            dynamics_fn: Optional custom dynamics function.

        Returns:
            dict: Simulation results.
        """
        self.reset()
        t_wall_start = _time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn

        print(f"[Cascaded BRT] Starting from state: {state}")
        print(f"  V_outer(x, tMax): {self.outer.get_value(state, self.outer.tMax):.4f}")
        print(f"  V_inner(x, tMax): {self.inner.get_value(state, self.inner.tMax):.4f}")

        docked = False
        collided = False

        for step in range(num_steps):
            sim_time = step * self.dt

            control = self.u_fn(state, sim_time)

            if self._check_docked(state):
                docked = True
                print(f"[Cascaded BRT] Docking successful at t={sim_time:.2f}s")
                break

            if self._check_collision(state):
                collided = True
                print(f"[Cascaded BRT] Collision at t={sim_time:.2f}s")
                break

            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot
            state[4] = np.arctan2(np.sin(state[4]), np.cos(state[4]))

        wall_time = _time.perf_counter() - t_wall_start

        controls_arr = np.array(self.control_history)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        result = {
            'trajectory': np.array(self.state_history),
            'controls': controls_arr,
            'values': np.array(self.value_history),
            'phases': np.array(self.phase_history),
            't_remaining': np.array(self.t_remaining_history),
            'times': np.array(self.sim_time_history),
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'cascaded_brt',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'outer_entry_time': self.outer_entry_time,
            'inner_entry_time': self.inner_entry_time,
            'outer_reacquisition_count': self.outer_reacquisition_count,
            'inner_reacquisition_count': self.inner_reacquisition_count,
        }
        return result

    def _default_dynamics_fn(self, state, control):
        """Default dynamics using the inner controller's dynamics (same physical system)."""
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.inner.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()

    def _check_docked(self, state):
        """Check if state is within docking tolerance."""
        dynamics = self.outer.dynamics
        px, py, vx, vy, theta, omega = state
        pos_ok = np.sqrt(px**2 + py**2) <= dynamics.eps_p
        vel_ok = np.sqrt(vx**2 + vy**2) <= dynamics.eps_v
        theta_diff = np.abs(np.arctan2(np.sin(theta - np.pi/2), np.cos(theta - np.pi/2)))
        theta_ok = theta_diff <= dynamics.eps_theta
        omega_ok = np.abs(omega) <= dynamics.eps_omega
        return pos_ok and vel_ok and theta_ok and omega_ok

    def _check_collision(self, state):
        """Orientation-aware collision check (actual chaser corners)."""
        return self.inner.dynamics.check_collision_oriented(state)

    def data_to_visualize(self):
        """Provides data for visualization."""
        s_history = np.array(self.state_history)
        u_history = np.array(self.control_history)

        data = {
            'x (m)': [1, s_history[:, 0], {'color': 'b'}],
            'y (m)': [1, s_history[:, 1], {'color': 'g'}],
            'vx (m/s)': [2, s_history[:, 2], {'color': 'b'}],
            'vy (m/s)': [2, s_history[:, 3], {'color': 'g'}],
            'theta (rad)': [3, s_history[:, 4], {'color': 'r'}],
            'omega (rad/s)': [3, s_history[:, 5], {'color': 'b'}],
            'u_x (N)': [4, u_history[:, 0], {'color': 'r'}],
            'u_y (N)': [4, u_history[:, 1], {'color': 'g'}],
            'u_theta (N*m)': [4, u_history[:, 2], {'color': 'purple'}],
            'Distance (m)': [5, np.sqrt(s_history[:, 0]**2 + s_history[:, 1]**2), {'color': 'orange'}],
        }
        return data
