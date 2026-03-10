"""
BRAT-Based Optimal Controller for Docking6D

This module implements a two-phase control strategy using the learned DeepReach 
value function:
- Phase 1 (Convergence): Outside BRAT, use V(x, tMax) gradient to steer toward BRAT
- Phase 2 (Precision): Inside BRAT, use time-varying V(x, t_remaining) for optimal control

Usage:
    controller = BRATController(
        checkpoint_path='./runs/Docking6D_RA/training/checkpoints/model_final.pth',
        tMax=14.0  # Tunable to avoid training artifacts
    )
    result = controller.simulate_docking(initial_state, max_sim_time=30.0)
"""

import time as _time

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys

# Add project root to path (3 levels up: controllers -> utils -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils import modules
from utils import diff_operators
from dynamics import dynamics as dynamics_module
from utils.controllers.safety_filter import SafetyFilter
from utils.controllers import clip_state_for_execution
from utils.controllers.min_time_search import find_min_brat_time_single


class BRATController:
    """
    Two-phase BRAT-based optimal controller using learned DeepReach value function.
    
    Phase 1 (Convergence): When V(x, tMax) > 0 (outside BRAT)
        - Use gradient of V(x, tMax) to steer toward the BRAT
        - Value function visualization is STATIC at tMax
        
    Phase 2 (Precision): When V(x, tMax) <= 0 (inside BRAT)
        - Query min-time t* where V(x, t*) <= 0 each step
        - Use time-varying V(x, t*) for optimal control
        - Value function visualization SHRINKS as t* decreases
    """
    
    def __init__(self, checkpoint_path, tMax=14.0, dt=0.1, device='cuda',
                 search_resolution=0.1,
                 safety_filter=None,
                 safety_margin_phase1=0.1, safety_margin_phase2=0.02):
        """
        Initialize the BRAT controller.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (model_final.pth)
            tMax: Maximum time horizon for BRAT queries. Set lower than trained 
                  horizon to avoid training artifacts (e.g., 14.0 instead of 15.0)
            dt: Control update frequency in seconds
            device: Torch device ('cuda' or 'cpu')
            search_resolution: Time step resolution for BRAT time search (seconds)
        """
        self.checkpoint_path = checkpoint_path
        self.tMax = tMax
        self.dt = dt
        self.device = device
        self.search_resolution = search_resolution
        
        # Derive experiment directory from checkpoint path
        # checkpoint_path: ./runs/EXPNAME/training/checkpoints/model_final.pth
        self.experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
        
        # Load dynamics and model
        self._load_dynamics_and_model()
        
        # Safety filter (no-op when mode=0 or None)
        self.safety_filter = safety_filter or SafetyFilter(mode=0)
        self.safety_margin_phase1 = safety_margin_phase1
        self.safety_margin_phase2 = safety_margin_phase2
        
        # Control state
        self.reset()
        
    def _load_dynamics_and_model(self):
        """Load the dynamics class and trained model from checkpoint."""
        # Load original experiment options
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)
        
        # Instantiate dynamics class
        dynamics_class = getattr(dynamics_module, self.orig_opt.dynamics_class)
        # Get constructor parameters that exist in orig_opt
        import inspect
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
        
        # Create model with same architecture as training
        self.model = modules.SingleBVPNet(
            in_features=self.dynamics.input_dim,
            out_features=1,
            type=self.orig_opt.model,
            mode=self.orig_opt.model_mode,
            final_layer_factor=1.,
            hidden_features=self.orig_opt.num_nl,
            num_hidden_layers=self.orig_opt.num_hl,
            periodic_transform_fn=self.dynamics.periodic_transform_fn
        )
        
        # Load checkpoint weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store useful dynamics parameters
        self.u_bar = self.dynamics.u_bar
        self.u_theta_bar = self.dynamics.u_theta_bar
        self.state_dim = self.dynamics.state_dim
        
        print(f"Loaded model from {self.checkpoint_path}")
        print(f"Dynamics: {self.dynamics.name}, tMax: {self.tMax}s, dt: {self.dt}s")
        
    def reset(self):
        """Reset controller state for a new simulation."""
        self.in_brat_phase = False
        self.t_remaining = self.tMax
        self.phase_transition_time = None
        self.safety_filter.reset()
        
        # History tracking
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.phase_history = []
        self.t_remaining_history = []
        self.sim_time_history = []
        
        # BRAT reacquisition tracking
        self.brat_reacquisition_count = 0
        self.brat_time_adjustments = []
        
    def get_value(self, state, time):
        """
        Query the value function V(x, t).
        
        Args:
            state: State vector [px, py, vx, vy, theta, omega] as numpy array or torch tensor
            time: Time horizon for value query
            
        Returns:
            float: Value V(x, t)
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        # Ensure state is 2D: (batch, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Create coordinate tensor [t, state]
        time_tensor = torch.tensor([[time]], dtype=torch.float32, device=self.device)
        coord = torch.cat([time_tensor, state], dim=-1)
        
        # Normalize coordinates to model input
        model_input = self.dynamics.coord_to_input(coord)
        
        # Forward pass
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze()
            
        # Convert output to real value using dynamics' io_to_value
        value = self.dynamics.io_to_value(model_input, output)
        
        return value.item()
    
    def get_gradient(self, state, time):
        """
        Query the value function gradient dV/ds.
        
        Args:
            state: State vector [px, py, vx, vy, theta, omega]
            time: Time horizon for gradient query
            
        Returns:
            numpy array: Gradient [dV/dpx, dV/dpy, dV/dvx, dV/dvy, dV/dtheta, dV/domega]
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        # Ensure state is 2D: (batch, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Create coordinate tensor [t, state]
        time_tensor = torch.tensor([[time]], dtype=torch.float32, device=self.device)
        coord = torch.cat([time_tensor, state], dim=-1)
        
        # Normalize coordinates to model input
        model_input = self.dynamics.coord_to_input(coord)
        
        # Forward pass - the model internally creates coords_org with requires_grad=True
        # and returns it as 'model_in'. The output is connected to 'model_in', not to
        # the original model_input, so we MUST use result['model_in'] for gradients.
        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze()
        model_in = result['model_in']  # This is the tensor connected to output
        
        # Compute gradient using dynamics' io_to_dv
        # CRITICAL: Use model_in (not model_input) as it's connected to output in the graph
        dv = self.dynamics.io_to_dv(model_in, output)
        
        # dv is [dvdt, dvds1, dvds2, ..., dvds6]
        # Return only spatial gradients (dvds)
        dvds = dv[0, 1:].detach().cpu().numpy()
        
        return dvds
    
    def get_optimal_control(self, state, time):
        """
        Compute optimal bang-bang control from value function gradient.
        
        Args:
            state: State vector [px, py, vx, vy, theta, omega]
            time: Time horizon for control computation
            
        Returns:
            numpy array: Control [u_x, u_y, u_theta]
        """
        dvds = self.get_gradient(state, time)
        
        # Bang-bang control: u = -u_max * sign(dV/dv)
        # dvds indices: [0:px, 1:py, 2:vx, 3:vy, 4:theta, 5:omega]
        u_x = -self.u_bar if dvds[2] > 0 else self.u_bar
        u_y = -self.u_bar if dvds[3] > 0 else self.u_bar
        u_theta = -self.u_theta_bar if dvds[5] > 0 else self.u_theta_bar
        
        return np.array([u_x, u_y, u_theta])
    
    def is_in_brat(self, state):
        """
        Check if state is inside the BRAT (V(x, tMax) <= 0).
        
        Args:
            state: State vector [px, py, vx, vy, theta, omega]
            
        Returns:
            bool: True if inside BRAT
        """
        value = self.get_value(state, self.tMax)
        return value <= 0
    
    def get_values_batch(self, state, times):
        """
        Query value function V(x, t) for a single state at multiple times in one forward pass.
        
        Replicates the exact normalization pipeline of get_value:
            coord = [t, state] -> coord_to_input -> forward pass -> io_to_value
        
        Args:
            state: State vector [px, py, vx, vy, theta, omega] as numpy array or torch tensor
            times: 1D numpy array of time values to query
            
        Returns:
            numpy array: V(x, t_i) for each time in times
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        # Ensure state is 1D
        if state.dim() == 2:
            state = state.squeeze(0)
        
        n = len(times)
        
        # Create batch of [t_i, state] coordinates
        time_tensor = torch.tensor(times, dtype=torch.float32, device=self.device).unsqueeze(-1)  # (n, 1)
        state_batch = state.unsqueeze(0).expand(n, -1)  # (n, state_dim)
        coords = torch.cat([time_tensor, state_batch], dim=-1)  # (n, 1 + state_dim)
        
        # Normalize coordinates to model input (same pipeline as get_value)
        model_input = self.dynamics.coord_to_input(coords)
        
        # Single batched forward pass
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)  # (n,)
        
        # Convert output to real values
        values = self.dynamics.io_to_value(model_input, output)
        
        return values.cpu().numpy()

    def get_values_batch_states(self, states, time):
        """
        Query V(x, t) for multiple states at a single fixed time in one forward pass.

        Args:
            states: (N, state_dim) numpy array or torch tensor of state vectors
            time: scalar time value to query

        Returns:
            numpy array of shape (N,): V(x_i, t) for each state
        """
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        states = states.to(self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        n = states.shape[0]
        time_col = torch.full((n, 1), time, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, states], dim=-1)          # (N, 1+state_dim)

        model_input = self.dynamics.coord_to_input(coords)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)            # (N,)

        values = self.dynamics.io_to_value(model_input, output)
        return values.cpu().numpy()

    def _search_brat_time(self, state):
        """
        Find the minimum time t* where V(x, t*) <= 0 (strict + argmin fallback).
        
        Delegates to the shared find_min_brat_time_single utility which always
        returns a valid (t_star, status) — no Phase 1 fallback needed.
        
        Args:
            state: Current state vector
            
        Returns:
            (t_star, status): t_star is the selected time (float),
            status is 'strict' or 'argmin'.
        """
        value_fn = lambda times: self.get_values_batch(state, times)
        return find_min_brat_time_single(
            value_fn, self.tMax, resolution=self.search_resolution)
    
    def u_fn(self, state, sim_time):
        """
        Control function compatible with simulation interface.
        Implements the two-phase control strategy.
        
        Args:
            state: Current state [px, py, vx, vy, theta, omega]
            sim_time: Current simulation time
            
        Returns:
            numpy array: Control [u_x, u_y, u_theta]
        """
        # Determine phase and compute control
        if not self.in_brat_phase:
            # Check if we've entered the BRAT
            if self.is_in_brat(state):
                self.in_brat_phase = True
                self.phase_transition_time = sim_time
                print(f"Entered BRAT at t={sim_time:.2f}s, starting Phase 2")
            
        if self.in_brat_phase:
            # Phase 2: per-step min-time query (always returns a valid t*)
            t_star, status = self._search_brat_time(state)
            self.t_remaining = t_star
            query_time = max(t_star, 0.01)
            control = self.get_optimal_control(state, query_time)
            value = self.get_value(state, query_time)
            phase = 2

            if status != 'strict':
                self.brat_reacquisition_count += 1
                self.brat_time_adjustments.append({
                    'sim_time': sim_time,
                    't_star': t_star,
                    'status': status,
                    'value': value,
                })
        else:
            # Phase 1: Use fixed tMax value function
            control = self.get_optimal_control(state, self.tMax)
            value = self.get_value(state, self.tMax)
            phase = 1
            
        # Post-process through safety filter (no-op when mode=0)
        self.safety_filter.set_margin(
            self.safety_margin_phase2 if self.in_brat_phase else self.safety_margin_phase1)
        control = self.safety_filter.apply(state, control)

        # Record history
        self.state_history.append(state.copy() if isinstance(state, np.ndarray) else state)
        self.control_history.append(control)
        self.value_history.append(value)
        self.phase_history.append(phase)
        self.t_remaining_history.append(self.t_remaining if self.in_brat_phase else self.tMax)
        self.sim_time_history.append(sim_time)
        
        return control
    
    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """
        Run full docking simulation with two-phase control.
        
        Args:
            initial_state: Initial state [px, py, vx, vy, theta, omega]
            max_sim_time: Maximum simulation time in seconds
            dynamics_fn: Optional custom dynamics function f(state, control) -> state_dot
                        If None, uses self.dynamics.dsdt
                        
        Returns:
            dict: Simulation results containing:
                - trajectory: (N, 6) state trajectory
                - controls: (N, 3) control inputs
                - values: (N,) value function along trajectory
                - phases: (N,) phase indicator (1 or 2)
                - t_remaining: (N,) time remaining in BRAT phase
                - times: (N,) simulation times
                - success: bool, whether docking was successful
                - phase_transition_time: time when entered BRAT (or None)
        """
        self.reset()
        t_wall_start = _time.perf_counter()
        
        # Initialize
        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1
        
        # Use provided dynamics or default
        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn
            
        print(f"Starting simulation from state: {state}")
        print(f"Initial V(x, tMax): {self.get_value(state, self.tMax):.4f}")
        
        docked = False
        collided = False
        n_clipped = 0
        
        # Simulation loop
        for step in range(num_steps):
            sim_time = step * self.dt
            
            # Get control (this also records history)
            control = self.u_fn(state, sim_time)
            
            # Check termination conditions
            if self._check_docked(state):
                docked = True
                print(f"Docking successful at t={sim_time:.2f}s")
                break
                
            if self._check_collision(state):
                collided = True
                print(f"Collision detected at t={sim_time:.2f}s")
                break
            
            # Integrate dynamics
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot
            
            # Wrap theta and clamp velocities/omega to training domain
            state, clipped = clip_state_for_execution(state, self.dynamics)
            if clipped:
                n_clipped += 1
        
        wall_time = _time.perf_counter() - t_wall_start
        
        # Compute control effort: sum(||u_k||_2 * dt)
        controls_arr = np.array(self.control_history)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)
        
        # Package results (original fields + shared comparison fields)
        result = {
            # --- original BRAT fields ---
            'trajectory': np.array(self.state_history),
            'controls': controls_arr,
            'values': np.array(self.value_history),
            'phases': np.array(self.phase_history),
            't_remaining': np.array(self.t_remaining_history),
            'times': np.array(self.sim_time_history),
            'success': docked and not collided,
            'phase_transition_time': self.phase_transition_time,
            'final_state': state,
            'brat_reacquisition_count': self.brat_reacquisition_count,
            'brat_time_adjustments': self.brat_time_adjustments,
            # --- shared comparison fields ---
            'collision': collided,
            'docked': docked,
            'controller_type': 'brat',
            'control_effort': control_effort,
            'wall_time': wall_time,
            # --- safety filter ---
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            # --- execution clamping ---
            'n_clipped_steps': n_clipped,
        }
        
        return result
    
    def _default_dynamics_fn(self, state, control):
        """Default dynamics function using Docking6D CW equations."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        control_tensor = torch.tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        dsdt = self.dynamics.dsdt(state_tensor, control_tensor, None)
        
        return dsdt.squeeze().cpu().numpy()
    
    def _check_docked(self, state):
        """Check if state is within docking tolerance."""
        d = self.dynamics
        px, py, vx, vy, theta, omega = state
        x_ok = np.abs(px) <= d.eps_p
        y_ok = d.goal_y_min <= py <= d.goal_y_max
        pos_ok = x_ok and y_ok
        vel_ok = np.sqrt(vx**2 + vy**2) <= d.eps_v
        theta_goal = d.goal_state[4].item()
        omega_goal = d.goal_state[5].item()
        theta_diff = np.abs(np.arctan2(np.sin(theta - theta_goal), np.cos(theta - theta_goal)))
        theta_ok = theta_diff <= d.eps_theta
        omega_ok = np.abs(omega - omega_goal) <= d.eps_omega
        return pos_ok and vel_ok and theta_ok and omega_ok
    
    def _check_collision(self, state):
        """Check if state is in collision with target spacecraft.

        Uses the orientation-aware check (actual chaser corners) rather
        than the conservative circular-buffer ``avoid_fn``.
        """
        return self.dynamics.check_collision_oriented(state)
    
    def get_value_grid(self, time, x_range=(-15, 15), y_range=(-15, 15), 
                       resolution=50, fixed_state=None):
        """
        Compute value function on a 2D grid for visualization.
        
        Args:
            time: Time for value function query
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            resolution: Grid resolution
            fixed_state: Fixed values for [vx, vy, theta, omega], default [0, 0, pi/2, 0]
            
        Returns:
            X, Y, V: Meshgrid arrays and value function
        """
        if fixed_state is None:
            fixed_state = [0.0, 0.0, np.pi/2, 0.0]
            
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        V = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([X[i, j], Y[i, j]] + list(fixed_state))
                V[i, j] = self.get_value(state, time)
                
        return X, Y, V
    
    def data_to_visualize(self):
        """
        Provides data for visualization (compatible with existing animation interface).
        
        Returns:
            dict: Data arrays for plotting
        """
        s_history = np.array(self.state_history)
        u_history = np.array(self.control_history)
        
        data = {
            # Position data
            'x (m)': [1, s_history[:, 0], {'color': 'b'}],
            'y (m)': [1, s_history[:, 1], {'color': 'g'}],
            
            # Velocity data
            'vx (m/s)': [2, s_history[:, 2], {'color': 'b'}],
            'vy (m/s)': [2, s_history[:, 3], {'color': 'g'}],
            
            # Orientation data
            'θ (rad)': [3, s_history[:, 4], {'color': 'r'}],
            'ω (rad/s)': [3, s_history[:, 5], {'color': 'b'}],
            
            # Control inputs
            'u_x (N)': [4, u_history[:, 0], {'color': 'r'}],
            'u_y (N)': [4, u_history[:, 1], {'color': 'g'}],
            'u_θ (N·m)': [4, u_history[:, 2], {'color': 'purple'}],
            
            # Distance to origin
            'Distance (m)': [5, np.sqrt(s_history[:, 0]**2 + s_history[:, 1]**2), {'color': 'orange'}],
        }
        
        return data


if __name__ == '__main__':
    # Quick test
    checkpoint_path = './runs/Docking6D_RA/training/checkpoints/model_final.pth'
    
    if os.path.exists(checkpoint_path):
        controller = BRATController(checkpoint_path, tMax=14.0, dt=0.1)
        
        # Test value query
        test_state = np.array([5.0, -5.0, 0.0, 0.0, np.pi/2, 0.0])
        value = controller.get_value(test_state, 14.0)
        print(f"V({test_state}, t=14) = {value:.4f}")
        
        # Test gradient
        grad = controller.get_gradient(test_state, 14.0)
        print(f"∇V = {grad}")
        
        # Test optimal control
        control = controller.get_optimal_control(test_state, 14.0)
        print(f"u* = {control}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
