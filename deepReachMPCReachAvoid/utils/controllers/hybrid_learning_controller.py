"""
Hybrid Learning Controller wrapper for cmpt720_hybrid_hj checkpoints
"""
import numpy as np
import torch
import time as _time
from pathlib import Path


class HybridLearningController:
    """
    Wraps a pre-trained hybrid learning SIREN network checkpoint.
    Uses supervised MSE + PDE residual training.
    """

    def __init__(self, checkpoint_path, dt=0.1, max_sim_time=60.0,
                 device='cuda', cache_dir=None, tMax=None):
        self.checkpoint_path = checkpoint_path
        self.dt = dt
        self.max_sim_time = max_sim_time
        self.device = device
        self.cache_dir = cache_dir

        # Load checkpoint
        print(f"Loading hybrid learning checkpoint: {Path(checkpoint_path).name}")
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Infer tMax from checkpoint or parameter
        if isinstance(self.checkpoint, dict) and 'tMax' in self.checkpoint:
            self.tMax = self.checkpoint['tMax']
        elif tMax is not None:
            self.tMax = tMax
        else:
            self.tMax = 10.0
            print(f"Warning: tMax not found, defaulting to {self.tMax}")

        print(f"✓ Hybrid Learning Controller initialized (tMax={self.tMax}s)")

        # Load dynamics
        from utils.dynamics import Docking6D
        self.dynamics = Docking6D()
        
        self.reset()

    def reset(self):
        self.state_history = []
        self.control_history = []
        self.sim_time_history = []

    def simulate_docking(self, initial_state, max_sim_time):
        """Run docking simulation."""
        self.reset()
        t_wall_start = _time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        docked = False
        collided = False

        for step in range(num_steps):
            sim_time = step * self.dt

            # Check termination
            if self._check_docked(state):
                docked = True
                break

            if self._check_collision(state):
                collided = True
                break

            # Get control (fallback for now)
            control = self._get_control(state, sim_time)

            # Integrate
            state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
            control_tensor = torch.tensor([control], dtype=torch.float32, device=self.device)
            state_dot = self.dynamics.dsdt(state_tensor, control_tensor, None)
            state = state + self.dt * state_dot.cpu().numpy()[0]

            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.sim_time_history.append(sim_time)

        wall_time = _time.perf_counter() - t_wall_start

        # Compute control effort
        if self.control_history:
            controls_arr = np.array(self.control_history)
            control_effort = float(np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)
        else:
            control_effort = 0.0

        return {
            'docked': docked,
            'collision': collided,
            'success': docked and not collided,
            'final_state': state,
            'trajectory': np.array(self.state_history) if self.state_history else np.array([initial_state]),
            'controls': np.array(self.control_history) if self.control_history else np.array([[0, 0, 0]]),
            'times': np.array(self.sim_time_history) if self.sim_time_history else np.array([0]),
            'control_effort': control_effort,
            'wall_time': wall_time,
            'n_clipped_steps': 0,
        }

    def _check_docked(self, state):
        px, py, vx, vy, theta, omega = state
        x_ok = np.abs(px) <= self.dynamics.eps_p
        y_ok = self.dynamics.goal_y_min <= py <= self.dynamics.goal_y_max
        pos_ok = x_ok and y_ok
        vel_ok = np.sqrt(vx**2 + vy**2) <= self.dynamics.eps_v
        theta_goal = self.dynamics.goal_state[4].item()
        omega_goal = self.dynamics.goal_state[5].item()
        theta_diff = np.abs(np.arctan2(np.sin(theta - theta_goal), np.cos(theta - theta_goal)))
        theta_ok = theta_diff <= self.dynamics.eps_theta
        omega_ok = np.abs(omega - omega_goal) <= self.dynamics.eps_omega
        return pos_ok and vel_ok and theta_ok and omega_ok

    def _check_collision(self, state):
        return self.dynamics.check_collision_oriented(state)

    def _get_control(self, state, t):
        """Simple proportional control (placeholder)."""
        px, py, vx, vy, theta, omega = state
        u_px = np.clip(-0.5 * px, -1.0, 1.0)
        u_py = np.clip(-0.5 * py, -1.0, 1.0)
        u_theta = 0.0
        return np.array([u_px, u_py, u_theta])
