#!/usr/bin/env python3
"""
trajectory_animation.py

Animate a 6D system trajectory with side control plots and a moving time cursor.

State (per timestep): [x, y, u, v, theta, Omega]
Controls (per timestep): [u1, u2, u3]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Rectangle
from matplotlib.gridspec import GridSpec
from utils import MPC_viz_helper as viz
from pathlib import Path as pathlib



class TrajectoryAnim:
    def __init__(self, state_rollouts, control_rollouts, dynamics_, dt=0.1):
        self.marker_size = 1.0
        self.window = 10.0  # Changed to 10s for ±5s window
        self.pad = 0.1
        self.blit = False  # Set to False if having issues, True for better performance
        self.dpi = 120
        self.figsize = (14.0, 8.0)
        self.state_rollouts = state_rollouts
        self.control_rollouts = control_rollouts
        self.dynamics_ = dynamics_
        self.dt = dt
        self.fps = 1/dt # Frames per second for animation
        self.t = np.arange(len(state_rollouts)) * dt
        
        # Create separate time array for controls (one element shorter)
        self.t_control = np.arange(len(control_rollouts)) * dt
        

    def chaser_vertices(self, frame_idx) -> np.ndarray:
        """Get chaser satellite vertices for given frame"""
        chaser_center = self.state_rollouts[frame_idx, :2]
        chaser_theta = self.state_rollouts[frame_idx, 4]
        chaser_width = self.dynamics_.w_c
        chaser_height = self.dynamics_.h_c
        
        # Get the four corners of the rectangle
        half_width = chaser_width / 2.0
        half_height = chaser_height / 2.0
        corners = np.array([
            [-half_width, -half_height],
            [ half_width, -half_height],
            [ half_width,  half_height],
            [-half_width,  half_height]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(chaser_theta), -np.sin(chaser_theta)],
            [np.sin(chaser_theta),  np.cos(chaser_theta)]
        ])
        
        # Rotate and translate corners
        rotated_corners = (R @ corners.T).T + chaser_center
        return rotated_corners

    def setup_figure(self):
        """Setup the figure with trajectory plot and control plots"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create grid layout
        gs = GridSpec(6, 3, figure=fig, width_ratios=[1, 2, 1], height_ratios=[1, 1, 1, 1, 1, 1])
        
        # Main trajectory plot (spans all rows of middle column)
        self.ax_traj = fig.add_subplot(gs[:,1])
        
        # Control plots (right column)
        self.ax_u1 = fig.add_subplot(gs[0:2, 2])
        self.ax_u2 = fig.add_subplot(gs[2:4, 2])
        self.ax_u3 = fig.add_subplot(gs[4:, 2])

        # State plots (left column)
        self.ax_s1 = fig.add_subplot(gs[0, 0])
        self.ax_s2 = fig.add_subplot(gs[1, 0])
        self.ax_s3 = fig.add_subplot(gs[2, 0])
        self.ax_s4 = fig.add_subplot(gs[3, 0])
        self.ax_s5 = fig.add_subplot(gs[4, 0])
        self.ax_s6 = fig.add_subplot(gs[5, 0])
        
        return fig

    def setup_trajectory_plot(self):
        """Setup the main trajectory visualization"""
        # Plot full trajectory as faded line
        self.ax_traj.plot(self.state_rollouts[:, 0], self.state_rollouts[:, 1], 
                         'b-', alpha=0.3, linewidth=1, label='Trajectory')
        
        # Create chaser satellite polygon
        chaser_verts = self.chaser_vertices(0)
        self.chaser_poly = Polygon(chaser_verts, closed=True, 
                                  facecolor='blue', edgecolor='darkblue', 
                                  alpha=0.7, label='Chaser')
        self.ax_traj.add_patch(self.chaser_poly)
        
        # Create target satellite polygon (stationary)
        w_t = self.dynamics_.w_t
        h_t = self.dynamics_.h_t
        post_hw_x = self.dynamics_.post_hw_x
        post_length = self.dynamics_.post_length
        viz.draw_target_body(self.ax_traj, w_t, h_t, post_hw_x, post_length, color='red', linewidth=2.0, alpha=0.7)
        
        # Add recent trajectory trace
        self.trace_line, = self.ax_traj.plot([], [], 'b-', linewidth=2, alpha=0.8)
        
        # Add current position marker
        self.pos_marker, = self.ax_traj.plot([], [], 'bo', markersize=8)
        
        # Set equal aspect ratio and labels
        self.ax_traj.set_aspect('equal')
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Y Position (m)')
        self.ax_traj.set_title('Satellite Docking Trajectory')
        self.ax_traj.legend()
        self.ax_traj.grid(True, alpha=0.3)
        
        # Set axis limits
        margin = 7.5
        self.ax_traj.set_xlim(0 - margin, 0 + margin)
        self.ax_traj.set_ylim(0 - margin, 0 + margin)

    def setup_control_plots(self):
        """Setup the control visualization plots"""
        control_labels = ['u1 (Thrust X)', 'u2 (Thrust Y)', 'u3 (Torque)']
        self.control_axes = [self.ax_u1, self.ax_u2, self.ax_u3]
        self.control_lines = []
        self.control_markers = []
        self.control_time_cursors = []
        
        for i, (ax, label) in enumerate(zip(self.control_axes, control_labels)):
            # Plot full control history as faded line using control time array
            ax.plot(self.t_control, self.control_rollouts[:, i], 'g-', alpha=0.3, linewidth=1)
            
            # Create sliding window line
            line, = ax.plot([], [], 'g-', linewidth=2)
            self.control_lines.append(line)
            
            # Create current value marker
            marker, = ax.plot([], [], 'ro', markersize=6)
            self.control_markers.append(marker)
            
            # Create time cursor (vertical line)
            cursor = ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            self.control_time_cursors.append(cursor)
            
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            
            # Set y-limits based on control range
            control_range = self.control_rollouts[:, i]
            margin = 0.1 * (np.max(control_range) - np.min(control_range))
            if margin == 0:  # Handle case where all controls are the same
                margin = 0.1
            ax.set_ylim(np.min(control_range) - margin, np.max(control_range) + margin)
        
        # Only show x-label on bottom plot
        self.ax_u3.set_xlabel('Time (s)')


    def setup_state_plots(self):
        """Setup the state visualization plots with reach/avoid background colors"""
        state_labels = ['x', 'y', 'Vx', 'Vy', 'theta', 'omega']
        self.state_axes = [self.ax_s1, self.ax_s2, self.ax_s3, self.ax_s4, self.ax_s5, self.ax_s6]
        self.state_lines = []
        self.state_markers = []
        self.state_time_cursors = []
        
        for i, (ax, label) in enumerate(zip(self.state_axes, state_labels)):
            # Add background colors based on reach/avoid sets
            self.add_state_background_colors(ax, i)
            
            # Plot full state history as faded line using state time array
            ax.plot(self.t, self.state_rollouts[:, i], 'g-', alpha=0.3, linewidth=1)
            
            # Create sliding window line
            line, = ax.plot([], [], 'g-', linewidth=2)
            self.state_lines.append(line)

            # Create current value marker
            marker, = ax.plot([], [], 'ro', markersize=6)
            self.state_markers.append(marker)

            # Create time cursor (vertical line)
            cursor = ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            self.state_time_cursors.append(cursor)
            
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

            # Set y-limits based on state range
            state_range = self.state_rollouts[:, i]
            margin = 0.1 * (np.max(state_range) - np.min(state_range))
            if margin == 0:  # Handle case where all states are the same
                margin = 0.1
            ax.set_ylim(np.min(state_range) - margin, np.max(state_range) + margin)

        # Only show x-label on bottom plot
        self.ax_s6.set_xlabel('Time (s)')

    def add_state_background_colors(self, ax, state_idx):
        """Add background colors to state plots based on reach/avoid sets"""
        # Get the y-limits for this state dimension
        state_values = self.state_rollouts[:, state_idx]
        y_min = np.min(state_values) - 0.1 * (np.max(state_values) - np.min(state_values))
        y_max = np.max(state_values) + 0.1 * (np.max(state_values) - np.min(state_values))
        
        # Get time limits
        t_min, t_max = self.t[0], self.t[-1]
        
        # Create a simpler grid - fewer points for efficiency
        n_time_points = 50
        n_state_points = 50
        time_points = np.linspace(t_min, t_max, n_time_points)
        state_points = np.linspace(y_min, y_max, n_state_points)
        
        # Use the goal state as reference for other dimensions
        ref_state = self.dynamics_.goal_state.cpu().numpy()
        
        # Create background color array
        background_colors = np.zeros((n_state_points, n_time_points, 4))  # RGBA
        
        #print(f"Computing background for state {state_idx}...")  # Debug
        
        # Vectorized evaluation - create all test states at once
        test_states = np.zeros((n_state_points, self.dynamics_.state_dim))
        test_states[:, :] = ref_state  # Start with reference state
        
        for j, state_val in enumerate(state_points):
            test_states[j, state_idx] = state_val
        
        # Convert to tensor and evaluate
        test_states_tensor = torch.tensor(test_states, dtype=torch.float32).to(
            self.dynamics_.goal_state.device if hasattr(self.dynamics_.goal_state, 'device') else 'cpu')
        
        try:
            with torch.no_grad():
                reach_vals = self.dynamics_.reach_fn(test_states_tensor).cpu().numpy()
                avoid_vals = self.dynamics_.avoid_fn(test_states_tensor).cpu().numpy()
            
            #print(f"Reach values range: [{np.min(reach_vals):.3f}, {np.max(reach_vals):.3f}]")
            #print(f"Avoid values range: [{np.min(avoid_vals):.3f}, {np.max(avoid_vals):.3f}]")
            
            for j in range(n_state_points):
                for i in range(n_time_points):
                    # Check if in reach set (≤ 0) and not in avoid set (> 0)
                    in_reach = reach_vals[j] <= 0
                    in_avoid = avoid_vals[j] <= 0
                    
                    if in_avoid:
                        background_colors[j, i] = [1, 0, 0, 0.3]  # Red for avoid
                    elif in_reach:
                        background_colors[j, i] = [0, 1, 0, 0.3]  # Green for reach
                    # else: transparent (no color)
            
            # Plot the background
            extent = [t_min, t_max, y_min, y_max]
            ax.imshow(background_colors, extent=extent, aspect='auto', origin='lower', alpha=0.5)
            
            #print(f"Background colors applied for state {state_idx}")
            
        except Exception as e:
            print(f"Error computing background colors for state {state_idx}: {e}")
            # Skip background colors if there's an error
            pass
        
        # Add legend only for the first subplot
        if state_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.3, label='Reach Set'),
                Patch(facecolor='red', alpha=0.3, label='Avoid Set')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def get_window_indices(self, frame_idx):
        """Get indices for sliding window centered around current frame"""
        current_time = self.t[frame_idx]
        
        # Find indices for ±window/2 seconds around current time
        left_time = current_time - self.window / 2
        right_time = current_time + self.window / 2
        
        # Convert to indices
        start_idx = np.searchsorted(self.t, left_time, side='left')
        end_idx = np.searchsorted(self.t, right_time, side='right')
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(self.state_rollouts), end_idx)
        
        return start_idx, end_idx

    def update_animation(self, frame_idx):
        """Update function for animation"""
        # Update chaser satellite position and orientation
        chaser_verts = self.chaser_vertices(frame_idx)
        self.chaser_poly.set_xy(chaser_verts)
        
        # Update trajectory trace (recent path)
        trace_start = max(0, frame_idx - 50)  # Show last 50 points
        trace_x = self.state_rollouts[trace_start:frame_idx+1, 0]
        trace_y = self.state_rollouts[trace_start:frame_idx+1, 1]
        self.trace_line.set_data(trace_x, trace_y)
        
        # Update current position marker
        current_x = self.state_rollouts[frame_idx, 0]
        current_y = self.state_rollouts[frame_idx, 1]
        self.pos_marker.set_data([current_x], [current_y])
        
        # Update control plots
        current_time = self.t[frame_idx]
        
        # Make sure we don't go beyond control array bounds
        control_frame_idx = min(frame_idx, len(self.control_rollouts) - 1)
        start_idx, end_idx = self.get_window_indices(frame_idx)
        
        for i, (line, marker, cursor, ax) in enumerate(zip(self.control_lines, 
                                                      self.control_markers, 
                                                      self.control_time_cursors,
                                                      self.control_axes)):
            # Update sliding window - clamp indices for control array
            control_start_idx = max(0, min(start_idx, len(self.control_rollouts) - 1))
            control_end_idx = max(1, min(end_idx, len(self.control_rollouts)))
            
            window_t = self.t_control[control_start_idx:control_end_idx]
            window_u = self.control_rollouts[control_start_idx:control_end_idx, i]
            line.set_data(window_t, window_u)
            
            # Update current value marker
            current_u = self.control_rollouts[control_frame_idx, i]
            current_control_time = self.t_control[control_frame_idx]
            marker.set_data([current_control_time], [current_u])
            
            # Update time cursor
            cursor.set_xdata([current_time])
            
            # Set x-axis limits centered on current time with ±window/2 window
            left_time = current_time - self.window / 2
            right_time = current_time + self.window / 2
            ax.set_xlim(left_time, right_time)
        
        # Update state subplots (sliding window and markers)
        # clamp indices for state arrays
        state_start_idx = max(0, start_idx)
        state_end_idx = min(len(self.state_rollouts), end_idx)
        window_t_state = self.t[state_start_idx:state_end_idx]
        
        for i, (line, marker, cursor, ax) in enumerate(zip(self.state_lines,
                                                           self.state_markers,
                                                           self.state_time_cursors,
                                                           self.state_axes)):
            window_s = self.state_rollouts[state_start_idx:state_end_idx, i]
            line.set_data(window_t_state, window_s)
            
            # Current state value (clamp index)
            current_state_val = self.state_rollouts[min(frame_idx, len(self.state_rollouts)-1), i]
            marker.set_data([current_time], [current_state_val])
            
            # Update time cursor
            cursor.set_xdata([current_time])
            
            # Set x-axis limits centered on current time
            left_time = current_time - self.window / 2
            right_time = current_time + self.window / 2
            ax.set_xlim(left_time, right_time)

            # Dynamic y-axis based on current window data
            if len(window_s) > 0:
                y_margin = 0.1 * (np.max(window_s) - np.min(window_s))
                if y_margin == 0:
                    y_margin = 0.1
                ax.set_ylim(np.min(window_s) - y_margin, np.max(window_s) + y_margin)
        
        # Return all artists that need to be redrawn
        artists = [self.chaser_poly, self.trace_line, self.pos_marker]
        artists.extend(self.control_lines)
        artists.extend(self.control_markers)
        artists.extend(self.control_time_cursors)
        artists.extend(self.state_lines)
        artists.extend(self.state_markers)
        artists.extend(self.state_time_cursors)
        
        return artists

    def animation(self, states, controls,):
        """Create and return the animation object"""
        # Setup figure and plots
        fig = self.setup_figure()
        self.setup_trajectory_plot()
        self.setup_state_plots()
        self.setup_control_plots()
        
        plt.tight_layout()
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, 
            self.update_animation,
            frames=len(states),
            interval=self.dt * 1000,  # Convert to milliseconds
            blit=self.blit,
            repeat=True
        )
        
        return anim

    def trajectory_animation(self):
        """Main function to create and display the animation"""
        states = self.state_rollouts
        controls = self.control_rollouts
        anim = self.animation(states, controls)
        plt.show()
        return anim

    def save_animation(self, filename, writer='pillow'):
        """Save animation to file"""
        anim = self.animation(self.state_rollouts, self.control_rollouts)
        anim.save(filename, writer=writer, fps=self.fps, dpi=self.dpi)
        print(f"Animation saved to {filename}")


def animate_trajectory(mpc, initial_conditions, T, dt, 
                      save_def, window=5.0,):
    """
    Convenience function to create trajectory animation
    
    Args:
        mpc: MPC object with dynamics
        initial_conditions: initial state array
        T: total timesteps
        dt: timestep duration
        window: sliding window size for control plots (seconds)
        save_def: optional path to save animation
    
    Returns:
        Animation object
    """

    for i in range(initial_conditions.shape[0]):
        _, state_rollouts, _, _ = mpc.get_batch_data(initial_conditions[i].unsqueeze(0), T)
        state_rollouts = state_rollouts[0].detach().cpu().numpy()  # Extract single trajectory
        actual_T = state_rollouts.shape[0] - 1  # Because states include initial state
        # Extract control inputs by computing them from state dynamics
        control_rollouts = np.zeros((actual_T, 3))  # [ux, uy, tau] for each trajectory

        for t in range(actual_T):
            # Current state
            state_curr = torch.tensor(state_rollouts[t, :]).to(mpc.device)
            state_next = torch.tensor(state_rollouts[t+1, :]).to(mpc.device)

            # Compute control that would produce this state transition
            control = viz.compute_control_from_transition(state_curr, state_next, mpc.dT, mpc.dynamics_)
            control_rollouts[t, :] = control.detach().cpu().numpy()

        animator = TrajectoryAnim(state_rollouts, control_rollouts, mpc.dynamics_, dt)
        animator.window = window
        
        if save_def:
            save_def.parent.mkdir(parents=True, exist_ok=True)
            save_def_str = f"{save_def}__gif_traj_{i}.gif"
            animator.save_animation(save_def_str)
        
        #return animator.trajectory_animation()
