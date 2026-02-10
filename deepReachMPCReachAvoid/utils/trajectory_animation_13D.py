#!/usr/bin/env python3
"""
trajectory_animation_13D.py

Animate a 13D docking system trajectory in 3D space.

State (per timestep): [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
Controls (per timestep): [Fx, Fy, Fz, tx, ty, tz]

This is a simplified 3D animation showing only the chaser and target spacecraft
without the side control/state plots.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path as pathlib
from utils import MPC_viz_helper_13D as viz


class TrajectoryAnim13D:
    def __init__(self, state_rollouts, control_rollouts, dynamics_, dt=0.1):
        """
        Initialize the 13D trajectory animator.
        
        Args:
            state_rollouts: np.ndarray of shape (T+1, 13) - state trajectory
            control_rollouts: np.ndarray of shape (T, 6) - control trajectory
            dynamics_: Docking13D dynamics object
            dt: timestep duration
        """
        self.dpi = 120
        self.figsize = (10.0, 8.0)
        self.state_rollouts = state_rollouts
        self.control_rollouts = control_rollouts
        self.dynamics_ = dynamics_
        self.dt = dt
        self.fps = 1/dt  # Frames per second for animation
        self.t = np.arange(len(state_rollouts)) * dt
        
        # Chaser spacecraft dimensions
        self.w_c = dynamics_.w_c  # x-dim
        self.h_c = dynamics_.h_c  # y-dim
        self.d_c = dynamics_.d_c  # z-dim
        
        # Target spacecraft dimensions
        self.w_t = dynamics_.w_t
        self.h_t = dynamics_.h_t
        self.d_t = dynamics_.d_t
        self.dock_rad = dynamics_.dock_rad

    def quat_to_rotation_matrix(self, q):
        """
        Convert quaternion (scalar-first) to rotation matrix.
        
        Args:
            q: quaternion [q0, q1, q2, q3] where q0 is scalar
            
        Returns:
            3x3 rotation matrix
        """
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        
        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        return R

    def get_box_vertices(self, center, dimensions, quaternion):
        """
        Get 8 vertices of a 3D box given center, dimensions, and orientation.
        
        Args:
            center: [x, y, z] position
            dimensions: [w, h, d] half-dimensions
            quaternion: [q0, q1, q2, q3] orientation (scalar-first)
            
        Returns:
            8x3 array of vertex positions
        """
        hw, hh, hd = dimensions[0]/2, dimensions[1]/2, dimensions[2]/2
        
        # Local vertices (centered at origin)
        local_verts = np.array([
            [-hw, -hh, -hd],
            [ hw, -hh, -hd],
            [ hw,  hh, -hd],
            [-hw,  hh, -hd],
            [-hw, -hh,  hd],
            [ hw, -hh,  hd],
            [ hw,  hh,  hd],
            [-hw,  hh,  hd]
        ])
        
        # Rotation matrix from quaternion
        R = self.quat_to_rotation_matrix(quaternion)
        
        # Rotate and translate
        world_verts = (R @ local_verts.T).T + center
        return world_verts

    def get_box_faces(self, vertices):
        """
        Get the 6 faces of a box from its 8 vertices.
        
        Args:
            vertices: 8x3 array of vertex positions
            
        Returns:
            List of 6 faces, each face is a list of 4 vertex coordinates
        """
        # Define faces by vertex indices
        face_indices = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5]   # right
        ]
        
        faces = [[vertices[i] for i in face] for face in face_indices]
        return faces

    def draw_target_spacecraft(self, ax):
        """
        Draw the stationary target spacecraft at origin.
        
        The target is represented as a box with a docking port (hemisphere opening).
        """
        # Target is at origin with identity orientation
        center = np.array([0.0, 0.0, 0.0])
        dimensions = np.array([self.w_t, self.h_t, self.d_t])
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        vertices = self.get_box_vertices(center, dimensions, identity_quat)
        faces = self.get_box_faces(vertices)
        
        # Draw the target body
        target_body = Poly3DCollection(faces, alpha=0.3, facecolor='red', 
                                        edgecolor='darkred', linewidth=1)
        ax.add_collection3d(target_body)
        
        # Draw docking port as a circle at the front face
        theta = np.linspace(0, 2*np.pi, 30)
        dock_x = np.zeros_like(theta)  # At x=0 (front face pointing toward chaser approach)
        dock_y = self.dock_rad * np.cos(theta)
        dock_z = self.dock_rad * np.sin(theta)
        ax.plot(dock_x, dock_y, dock_z, 'g-', linewidth=2, label='Docking Port')
        
        return target_body

    def draw_chaser_spacecraft(self, ax, frame_idx):
        """
        Draw the chaser spacecraft at its current position and orientation.
        
        Args:
            ax: matplotlib 3D axis
            frame_idx: current frame index
            
        Returns:
            Poly3DCollection object for the chaser
        """
        state = self.state_rollouts[frame_idx]
        center = state[:3]  # [x, y, z]
        quaternion = state[6:10]  # [q0, q1, q2, q3]
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(quaternion)
        if quat_norm > 0:
            quaternion = quaternion / quat_norm
        
        dimensions = np.array([self.w_c, self.h_c, self.d_c])
        vertices = self.get_box_vertices(center, dimensions, quaternion)
        faces = self.get_box_faces(vertices)
        
        chaser_body = Poly3DCollection(faces, alpha=0.7, facecolor='blue', 
                                        edgecolor='darkblue', linewidth=1)
        ax.add_collection3d(chaser_body)
        
        return chaser_body

    def setup_figure(self):
        """Setup the 3D figure."""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        self.fig = fig
        self.ax = ax
        
        return fig, ax

    def setup_plot(self):
        """Setup the 3D trajectory visualization."""
        ax = self.ax
        
        # Plot full trajectory as faded line
        ax.plot(self.state_rollouts[:, 0], 
                self.state_rollouts[:, 1], 
                self.state_rollouts[:, 2], 
                'b-', alpha=0.3, linewidth=1, label='Trajectory')
        
        # Draw stationary target
        self.target_body = self.draw_target_spacecraft(ax)
        
        # Initialize chaser (will be updated each frame)
        self.chaser_collection = None
        
        # Initialize trace line (recent trajectory)
        self.trace_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.8)
        
        # Initialize position marker
        self.pos_marker, = ax.plot([], [], [], 'bo', markersize=8)
        
        # Set labels
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title('13D Satellite Docking Trajectory', fontsize=12)
        
        # Set axis limits based on trajectory
        margin = 1.5
        x_min = min(np.min(self.state_rollouts[:, 0]), -self.w_t/2) - margin
        x_max = max(np.max(self.state_rollouts[:, 0]), self.w_t/2) + margin
        y_min = min(np.min(self.state_rollouts[:, 1]), -self.h_t/2) - margin
        y_max = max(np.max(self.state_rollouts[:, 1]), self.h_t/2) + margin
        z_min = min(np.min(self.state_rollouts[:, 2]), -self.d_t/2) - margin
        z_max = max(np.max(self.state_rollouts[:, 2]), self.d_t/2) + margin
        
        # Make axes roughly equal
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set initial view angle
        ax.view_init(elev=25, azim=45)

    def update_animation(self, frame_idx):
        """Update function for animation."""
        ax = self.ax
        
        # Remove previous chaser
        if self.chaser_collection is not None:
            self.chaser_collection.remove()
        
        # Draw chaser at new position
        self.chaser_collection = self.draw_chaser_spacecraft(ax, frame_idx)
        
        # Update trace line (last 20 frames or all if less)
        trace_start = max(0, frame_idx - 20)
        trace_x = self.state_rollouts[trace_start:frame_idx+1, 0]
        trace_y = self.state_rollouts[trace_start:frame_idx+1, 1]
        trace_z = self.state_rollouts[trace_start:frame_idx+1, 2]
        self.trace_line.set_data(trace_x, trace_y)
        self.trace_line.set_3d_properties(trace_z)
        
        # Update position marker
        pos = self.state_rollouts[frame_idx, :3]
        self.pos_marker.set_data([pos[0]], [pos[1]])
        self.pos_marker.set_3d_properties([pos[2]])
        
        # Update title with time
        current_time = frame_idx * self.dt
        ax.set_title(f'13D Satellite Docking Trajectory (t={current_time:.2f}s)', fontsize=12)
        
        # Slowly rotate view for better 3D perception
        ax.view_init(elev=25, azim=45 + frame_idx * 0.5)
        
        return [self.chaser_collection, self.trace_line, self.pos_marker]

    def create_animation(self):
        """Create and return the animation object."""
        fig, ax = self.setup_figure()
        self.setup_plot()
        
        plt.tight_layout()
        
        anim = animation.FuncAnimation(
            fig, 
            self.update_animation,
            frames=len(self.state_rollouts),
            interval=self.dt * 1000,  # Convert to milliseconds
            blit=False,  # 3D animations don't support blit well
            repeat=True
        )
        
        return anim

    def show_animation(self):
        """Create and display the animation."""
        anim = self.create_animation()
        plt.show()
        return anim

    def save_animation(self, filename, writer='pillow'):
        """Save animation to file."""
        anim = self.create_animation()
        anim.save(filename, writer=writer, fps=self.fps, dpi=self.dpi)
        plt.close(self.fig)
        print(f"Animation saved to {filename}")


def animate_trajectory_13D(mpc, initial_conditions, T, dt, save_def):
    """
    Convenience function to create 13D trajectory animation.
    
    Args:
        mpc: MPC object with Docking13D dynamics
        initial_conditions: tensor of initial states (batch_size, 13)
        T: total timesteps
        dt: timestep duration
        save_def: path prefix for saving animation
    
    Returns:
        None (saves animations to files)
    """
    for i in range(initial_conditions.shape[0]):
        print(f"Generating animation for trajectory {i+1}/{initial_conditions.shape[0]}...")
        
        # Get trajectory from MPC
        _, state_rollouts, _, _ = mpc.get_batch_data(initial_conditions[i].unsqueeze(0), T)
        state_rollouts = state_rollouts[0].detach().cpu().numpy()  # Extract single trajectory
        
        actual_T = state_rollouts.shape[0] - 1  # States include initial state
        
        # Extract control inputs by computing them from state dynamics
        control_rollouts = np.zeros((actual_T, 6))  # [Fx, Fy, Fz, tx, ty, tz]
        
        for t in range(actual_T):
            state_curr = torch.tensor(state_rollouts[t, :]).to(mpc.device)
            state_next = torch.tensor(state_rollouts[t+1, :]).to(mpc.device)
            
            # Compute control that would produce this state transition
            control = viz.compute_control_from_transition_13D(
                state_curr, state_next, mpc.dT, mpc.dynamics_
            )
            control_rollouts[t, :] = control.detach().cpu().numpy()
        
        # Create animator
        animator = TrajectoryAnim13D(state_rollouts, control_rollouts, mpc.dynamics_, dt)
        
        # Save animation
        if save_def:
            save_def.parent.mkdir(parents=True, exist_ok=True)
            save_def_str = f"{save_def}__gif_traj_{i}.gif"
            animator.save_animation(save_def_str)
