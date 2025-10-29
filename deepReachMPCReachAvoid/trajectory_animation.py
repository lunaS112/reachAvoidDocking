#!/usr/bin/env python3
"""
trajectory_animation.py

Animate a 6D system trajectory with side control plots and a moving time cursor.

State (per timestep): [x, y, u, v, theta, Omega]
Controls (per timestep): [u1, u2, u3]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Rectangle
from matplotlib.gridspec import GridSpec


class TrajectoryAnim:
    def __init__(self, state_rollouts, control_rollouts, dynamics_, dt=0.1):
        self.fps = 30
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
        self.t = np.arange(len(state_rollouts)) * dt
        
        # Create separate time array for controls (one element shorter)
        self.t_control = np.arange(len(control_rollouts)) * dt
        
        # Much faster animation interval - ignore dt relationship for visualization
        # Use 50ms (20 FPS) or 33ms (30 FPS) regardless of actual dt
        self.interval = 33  # Fast animation

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

    def target_vertices(self) -> np.ndarray:
        """Get target satellite vertices (assumed stationary at origin)"""
        target_width = getattr(self.dynamics_, 'w_t', 1.0)  # Default width
        target_height = getattr(self.dynamics_, 'h_t', 1.0)  # Default height
        
        half_width = target_width / 2.0
        half_height = target_height / 2.0
        corners = np.array([
            [-half_width, -half_height],
            [ half_width, -half_height],
            [ half_width,  half_height],
            [-half_width,  half_height]
        ])
        return corners

    def setup_figure(self):
        """Setup the figure with trajectory plot and control plots"""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create grid layout
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1, 1])
        
        # Main trajectory plot (spans all rows of left column)
        self.ax_traj = fig.add_subplot(gs[:, 0])
        
        # Control plots (right column)
        self.ax_u1 = fig.add_subplot(gs[0, 1])
        self.ax_u2 = fig.add_subplot(gs[1, 1])
        self.ax_u3 = fig.add_subplot(gs[2, 1])
        
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
        target_verts = self.target_vertices()
        self.target_poly = Polygon(target_verts, closed=True, 
                                  facecolor='red', edgecolor='darkred', 
                                  alpha=0.7, label='Target')
        self.ax_traj.add_patch(self.target_poly)
        
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
        
        # Set initial axis limits
        all_x = self.state_rollouts[:, 0]
        all_y = self.state_rollouts[:, 1]
        margin = 0.5
        self.ax_traj.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        self.ax_traj.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)

    def setup_control_plots(self):
        """Setup the control visualization plots"""
        control_labels = ['u1 (Thrust X)', 'u2 (Thrust Y)', 'u3 (Torque)']
        self.control_axes = [self.ax_u1, self.ax_u2, self.ax_u3]
        self.control_lines = []
        self.control_markers = []
        self.time_cursors = []
        
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
            self.time_cursors.append(cursor)
            
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
        
        for i, (line, marker, cursor) in enumerate(zip(self.control_lines, 
                                                      self.control_markers, 
                                                      self.time_cursors)):
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
            
            # Set x-axis limits centered on current time with ±5s window
            left_time = current_time - self.window / 2
            right_time = current_time + self.window / 2
            self.control_axes[i].set_xlim(left_time, right_time)
        
        # Return all artists that need to be redrawn
        artists = [self.chaser_poly, self.trace_line, self.pos_marker]
        artists.extend(self.control_lines)
        artists.extend(self.control_markers)
        artists.extend(self.time_cursors)
        
        return artists

    def animation(self, states, controls):
        """Create and return the animation object"""
        # Setup figure and plots
        fig = self.setup_figure()
        self.setup_trajectory_plot()
        self.setup_control_plots()
        
        plt.tight_layout()
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, 
            self.update_animation,
            frames=len(states),
            interval=self.interval,
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


def animate_trajectory(state_rollouts, control_rollouts, dynamics_, dt=0.1, 
                      window=5.0, save_path=None):
    """
    Convenience function to create trajectory animation
    
    Args:
        state_rollouts: numpy array of shape (T, 6) with states [x, y, u, v, theta, Omega]
        control_rollouts: numpy array of shape (T, 3) with controls [u1, u2, u3]
        dynamics_: dynamics object with satellite dimensions
        dt: time step
        window: sliding window size for control plots (seconds)
        save_path: optional path to save animation
    
    Returns:
        Animation object
    """
    animator = TrajectoryAnim(state_rollouts, control_rollouts, dynamics_, dt)
    animator.window = window
    
    if save_path:
        animator.save_animation(save_path)
    
    return animator.trajectory_animation()


# Demo function
def demo():
    """Create demo animation with synthetic data"""
    # Create synthetic trajectory data
    T = 200
    t = np.linspace(0, 20, T)
    
    # Spiral trajectory toward origin
    r = 5 * np.exp(-t/10)
    theta_traj = t
    x = r * np.cos(theta_traj)
    y = r * np.sin(theta_traj)
    
    # Velocities
    u = np.gradient(x, t)
    v = np.gradient(y, t)
    
    # Orientation (facing velocity direction)
    theta = np.arctan2(v, u)
    Omega = np.gradient(theta, t)
    
    states = np.column_stack([x, y, u, v, theta, Omega])
    
    # Synthetic controls
    controls = np.random.normal(0, 0.1, (T, 3))
    
    # Create mock dynamics object
    class MockDynamics:
        def __init__(self):
            self.w_c = 0.5  # chaser width
            self.h_c = 0.3  # chaser height
            self.w_t = 0.4  # target width  
            self.h_t = 0.4  # target height
    
    dynamics = MockDynamics()
    
    # Create and show animation
    animator = TrajectoryAnim(states, controls, dynamics, dt=0.1)
    return animator.trajectory_animation()


if __name__ == "__main__":
    demo()
