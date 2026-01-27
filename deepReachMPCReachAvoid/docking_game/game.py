#!/usr/bin/env python3
"""
Interactive Docking Game using 13D Spacecraft Dynamics (Pygame 2D version)

This game allows a user to control a chaser spacecraft and attempt to dock
with a target satellite. Uses pygame's 2D rendering with simple 3D projection.

Controls:
- WASD: Translational control (body X/Y)
- Q/E: Translational control (body Z - up/down)
- Arrow Keys: Rotational control (pitch/yaw)
- Z/C: Rotational control (roll)
- R: Reset game
- ESC: Quit
"""

import pygame
import numpy as np
import math
import sys

# ==============================================================================
# 13D Dynamics (CPU-based)
# ==============================================================================

class Docking13DDynamics:
    """
    13D docking dynamics for spacecraft control.
    
    State (13D): [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
    Control (6D): [Fx, Fy, Fz, tx, ty, tz]
    """
    
    def __init__(self):
        # Orbital parameters
        self.orbit_alt = 400  # km
        self.n = self.mean_motion()
        
        # Spacecraft parameters
        self.mc = 200.0  # mass (kg)
        self.w_c = 1.0   # width (m)
        self.h_c = 1.0   # height (m)
        self.d_c = 1.0   # depth (m)
        
        # Inertia tensor
        self.I = self.inertia_tensor()
        self.I_inv = np.linalg.inv(self.I)
        
        # Control limits
        self.F_bar = 20.0     # max force (N)
        self.tau_bar = 1.5    # max torque (N*m)
        self.v_max = 2.5      # max velocity (m/s)
        
    def mean_motion(self):
        mu = 3.986004418e14
        r_earth = 6371e3
        r = r_earth + (self.orbit_alt * 1e3)
        return math.sqrt(mu / (r ** 3))
    
    def inertia_tensor(self):
        m = self.mc
        wx, hy, dz = self.w_c, self.h_c, self.d_c
        Ixx = (m / 12.0) * (hy**2 + dz**2)
        Iyy = (m / 12.0) * (wx**2 + dz**2)
        Izz = (m / 12.0) * (wx**2 + hy**2)
        return np.diag([Ixx, Iyy, Izz])
    
    def quat_to_rotation_matrix(self, q):
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
    
    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    
    def clamp_control(self, control):
        control = np.array(control)
        control[0:3] = np.clip(control[0:3], -self.F_bar, self.F_bar)
        control[3:6] = np.clip(control[3:6], -self.tau_bar, self.tau_bar)
        return control
    
    def dsdt(self, state, control):
        dsdt = np.zeros(13)
        
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        q = self.normalize_quaternion(state[6:10])
        omega = state[10:13]
        
        Fb = control[0:3]
        taub = control[3:6]
        
        R_L2B = self.quat_to_rotation_matrix(q)
        R_B2L = R_L2B.T
        aL = R_B2L @ Fb / self.mc
        
        n = self.n
        
        # Position derivatives
        dsdt[0] = vx
        dsdt[1] = vy
        dsdt[2] = vz
        
        # Velocity derivatives (HCW + control)
        ax = 2*n*vy + 3*(n**2)*x + aL[0]
        ay = -2*n*vx + aL[1]
        az = -(n**2)*z + aL[2]
        
        # Velocity clamping
        if (vx >= self.v_max and ax > 0) or (vx <= -self.v_max and ax < 0):
            ax = 0
        if (vy >= self.v_max and ay > 0) or (vy <= -self.v_max and ay < 0):
            ay = 0
        if (vz >= self.v_max and az > 0) or (vz <= -self.v_max and az < 0):
            az = 0
            
        dsdt[3] = ax
        dsdt[4] = ay
        dsdt[5] = az
        
        # Quaternion kinematics
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        qdot = 0.5 * Omega @ q
        dsdt[6:10] = qdot
        
        # Angular velocity dynamics
        I_omega = self.I @ omega
        cross = np.cross(omega, I_omega)
        omega_dot = self.I_inv @ (taub - cross)
        dsdt[10:13] = omega_dot
        
        return dsdt
    
    def step(self, state, control, dt):
        control = self.clamp_control(control)
        
        # RK4 integration
        k1 = self.dsdt(state, control)
        k2 = self.dsdt(state + 0.5*dt*k1, control)
        k3 = self.dsdt(state + 0.5*dt*k2, control)
        k4 = self.dsdt(state + dt*k3, control)
        
        new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        new_state[6:10] = self.normalize_quaternion(new_state[6:10])
        
        return new_state


# ==============================================================================
# Simple 3D Renderer (Pygame 2D with projection)
# ==============================================================================

class Simple3DRenderer:
    """Simple 3D to 2D projection renderer."""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cx = screen_width // 2
        self.cy = screen_height // 2
        
        # Camera settings
        self.camera_distance = 15.0
        self.camera_elevation = 30.0  # degrees
        self.camera_azimuth = 45.0    # degrees
        self.fov = 500  # Field of view scale factor
        
    def get_camera_matrix(self):
        """Get rotation matrix for camera view."""
        elev = math.radians(self.camera_elevation)
        azim = math.radians(self.camera_azimuth)
        
        # Rotation around Z (azimuth)
        Rz = np.array([
            [math.cos(azim), -math.sin(azim), 0],
            [math.sin(azim), math.cos(azim), 0],
            [0, 0, 1]
        ])
        
        # Rotation around X (elevation)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(elev), -math.sin(elev)],
            [0, math.sin(elev), math.cos(elev)]
        ])
        
        return Rx @ Rz
    
    def project_point(self, point_3d):
        """Project 3D point to 2D screen coordinates."""
        R = self.get_camera_matrix()
        p = R @ np.array(point_3d)
        
        # Simple perspective projection
        z = p[2] + self.camera_distance
        if z < 0.1:
            z = 0.1
            
        scale = self.fov / z
        x_2d = int(self.cx + p[0] * scale)
        y_2d = int(self.cy - p[1] * scale)  # Flip Y for screen coords
        
        return (x_2d, y_2d), z
    
    def project_points(self, points_3d):
        """Project multiple 3D points to 2D."""
        results = []
        for p in points_3d:
            pt_2d, depth = self.project_point(p)
            results.append((pt_2d, depth))
        return results


# ==============================================================================
# Game Class
# ==============================================================================

class DockingGame:
    def __init__(self):
        pygame.init()
        
        self.width = 1280
        self.height = 720
        # Enable resizable window
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("13D Spacecraft Docking Game")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 16)
        self.font_large = pygame.font.SysFont('monospace', 24)
        
        # Colors
        self.BG_COLOR = (20, 25, 40)
        self.GRID_COLOR = (60, 70, 90)
        self.CHASER_COLOR = (80, 130, 220)
        self.CHASER_EDGE = (40, 70, 140)
        self.CHASER_DOCK_COLOR = (50, 255, 150)  # Green docking side on chaser
        self.TARGET_COLOR = (180, 180, 200)       # Gray target satellite
        self.TARGET_EDGE = (100, 100, 120)
        self.TARGET_DOCK_COLOR = (255, 200, 50)   # Yellow/orange lit docking side
        self.SUCCESS_COLOR = (50, 255, 100)
        self.TEXT_COLOR = (200, 210, 230)
        self.AXIS_X = (255, 80, 80)
        self.AXIS_Y = (80, 255, 80)
        self.AXIS_Z = (80, 80, 255)
        
        # Dynamics and renderer
        self.dynamics = Docking13DDynamics()
        self.renderer = Simple3DRenderer(self.width, self.height)
        
        # Physics timestep
        self.dt = 0.05
        
        # Control gains
        self.force_gain = 15.0
        self.torque_gain = 1.0
        
        # Cube vertices (unit cube centered at origin)
        s = 0.5
        self.cube_vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
        ])
        
        # Cube edges (pairs of vertex indices)
        self.cube_edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        # Cube faces (for filled rendering)
        # Face indices: 0=back(-X), 1=front(+X), 2=bottom(-Y), 3=top(+Y), 4=left(-Z), 5=right(+Z)
        self.cube_faces = [
            (0, 1, 2, 3),  # back  (-X face)
            (4, 5, 6, 7),  # front (+X face) - DOCKING SIDE
            (0, 1, 5, 4),  # bottom (-Y)
            (2, 3, 7, 6),  # top (+Y)
            (0, 3, 7, 4),  # left (-Z)
            (1, 2, 6, 5),  # right (+Z)
        ]
        
        # Docking face index (front face = +X direction)
        self.docking_face_idx = 1
        
        # Target satellite (static at origin)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        self.target_size = 1.5  # Slightly larger than chaser
        
        # Docking/collision tolerances
        self.collision_distance = 1.25  # Distance at which collision occurs (sum of half-sizes)
        self.dock_pos_epsilon = 0.3     # Position tolerance for successful dock (m)
        self.dock_vel_epsilon = 0.15    # Velocity tolerance (m/s)
        self.dock_omega_epsilon = 0.1   # Angular velocity tolerance (rad/s)
        self.dock_angle_epsilon = 15.0  # Orientation tolerance (degrees)
        
        self.reset()
        
    def reset(self):
        """Reset to initial state."""
        self.state = np.array([
            5.0, 3.0, 2.0,          # position
            0.0, 0.0, 0.0,          # velocity
            1.0, 0.0, 0.0, 0.0,     # quaternion (identity)
            0.0, 0.0, 0.0           # angular velocity
        ], dtype=np.float64)
        
        self.time_elapsed = 0.0
        self.game_over = False
        self.success = False
        self.failure = False
        self.failure_reason = ""
        self.last_trans_ctrl = np.zeros(3)
        self.last_rot_ctrl = np.zeros(3)
        
    def get_controls(self, keys):
        """Map keyboard to control inputs."""
        control = np.zeros(6)
        
        # Translation (WASD + Q/E) - body frame forces
        if keys[pygame.K_w]:
            control[0] += self.force_gain
        if keys[pygame.K_s]:
            control[0] -= self.force_gain
        if keys[pygame.K_d]:
            control[1] += self.force_gain
        if keys[pygame.K_a]:
            control[1] -= self.force_gain
        if keys[pygame.K_e]:
            control[2] += self.force_gain
        if keys[pygame.K_q]:
            control[2] -= self.force_gain
            
        # Rotation (arrows + Z/C) - body frame torques
        if keys[pygame.K_UP]:
            control[4] -= self.torque_gain
        if keys[pygame.K_DOWN]:
            control[4] += self.torque_gain
        if keys[pygame.K_RIGHT]:
            control[5] += self.torque_gain
        if keys[pygame.K_LEFT]:
            control[5] -= self.torque_gain
        if keys[pygame.K_z]:
            control[3] -= self.torque_gain
        if keys[pygame.K_c]:
            control[3] += self.torque_gain
            
        self.last_trans_ctrl = control[0:3].copy()
        self.last_rot_ctrl = control[3:6].copy()
        
        return control
    
    def quat_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees."""
        q0, q1, q2, q3 = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    def get_docking_alignment_angle(self):
        """
        Calculate the angle between chaser's +X axis and target's -X axis.
        For perfect docking, these should be anti-parallel (180 deg alignment).
        Returns the error angle in degrees (0 = perfect alignment).
        """
        R_chaser = self.dynamics.quat_to_rotation_matrix(self.state[6:10])
        R_target = self.dynamics.quat_to_rotation_matrix(self.target_quaternion)
        
        # Chaser docking direction: +X in body frame
        chaser_dock_dir = R_chaser @ np.array([1, 0, 0])
        # Target docking direction: -X in body frame (faces the chaser)
        target_dock_dir = R_target @ np.array([-1, 0, 0])
        
        # For proper docking, these vectors should be parallel (same direction)
        # since chaser approaches with +X facing target's -X
        dot = np.clip(np.dot(chaser_dock_dir, target_dock_dir), -1, 1)
        angle_error = np.degrees(np.arccos(dot))
        
        return angle_error
    
    def check_collision_and_docking(self):
        """
        Check for collision and docking conditions.
        Returns: (is_collision, is_success, failure_reason)
        """
        pos = self.state[0:3]
        vel = self.state[3:6]
        omega = self.state[10:13]
        
        pos_dist = np.linalg.norm(pos - self.target_position)
        vel_mag = np.linalg.norm(vel)
        omega_mag = np.linalg.norm(omega)
        angle_error = self.get_docking_alignment_angle()
        
        # Check if collision occurred (bodies touching)
        if pos_dist < self.collision_distance:
            # Check all docking conditions
            vel_ok = vel_mag < self.dock_vel_epsilon
            omega_ok = omega_mag < self.dock_omega_epsilon
            angle_ok = angle_error < self.dock_angle_epsilon
            
            if vel_ok and omega_ok and angle_ok:
                return True, True, ""  # Successful docking
            else:
                # Collision failure - determine reason
                reasons = []
                if not vel_ok:
                    reasons.append(f"velocity too high ({vel_mag:.2f} m/s)")
                if not omega_ok:
                    reasons.append(f"spinning too fast ({omega_mag:.2f} rad/s)")
                if not angle_ok:
                    reasons.append(f"misaligned ({angle_error:.1f}°)")
                return True, False, ", ".join(reasons)
        
        return False, False, ""
    
    def transform_vertices(self, vertices, position, quaternion):
        """Transform vertices by position and orientation."""
        R = self.dynamics.quat_to_rotation_matrix(quaternion)
        transformed = []
        for v in vertices:
            tv = R @ v + position
            transformed.append(tv)
        return np.array(transformed)
    
    def draw_grid(self):
        """Draw reference grid."""
        grid_size = 10
        for i in range(-grid_size, grid_size + 1):
            # Lines parallel to X
            p1, _ = self.renderer.project_point([i, -grid_size, 0])
            p2, _ = self.renderer.project_point([i, grid_size, 0])
            pygame.draw.line(self.screen, self.GRID_COLOR, p1, p2, 1)
            # Lines parallel to Y
            p1, _ = self.renderer.project_point([-grid_size, i, 0])
            p2, _ = self.renderer.project_point([grid_size, i, 0])
            pygame.draw.line(self.screen, self.GRID_COLOR, p1, p2, 1)
    
    def draw_docking_arrow(self, position, quaternion, face_normal_body, arrow_color, size, length=0.4):
        """
        Draw an arrow ON the docking face pointing in a direction on the face.
        The arrow lies flat on the docking face surface.
        Only draws if the face is visible (facing the camera).
        
        Args:
            position: center position of the spacecraft
            quaternion: orientation quaternion
            face_normal_body: normal direction of the face in body frame
            arrow_color: color of the arrow
            size: half-size of the spacecraft (to position arrow on face)
            length: arrow length on the face
        """
        R = self.dynamics.quat_to_rotation_matrix(quaternion)
        
        # Face normal in world/LVLH frame
        face_normal = R @ face_normal_body
        
        # Get camera direction (from target to camera)
        cam_matrix = self.renderer.get_camera_matrix()
        camera_pos = cam_matrix @ np.array([0, 0, self.renderer.camera_distance])
        face_center_world = position + face_normal * size
        view_dir = camera_pos - face_center_world
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
        
        # Check if face is visible (facing the camera)
        dot = np.dot(face_normal, view_dir)
        if dot < 0:
            # Face is pointing away from camera, don't draw arrow
            return None
        
        # Arrow points in +Y body direction ("up" on the -Z face in the X-Y plane)
        arrow_dir_body = np.array([0, 1, 0])
        arrow_dir = R @ arrow_dir_body
        
        # Arrow perpendicular direction (for arrowhead) - use X body axis
        arrow_perp_body = np.array([1, 0, 0])
        arrow_perp = R @ arrow_perp_body
        
        # Position arrow on the face (offset by face size in face_normal direction)
        face_center = position + face_normal * (size + 0.02)  # Slightly above face
        
        # Arrow geometry
        arrow_base = face_center - arrow_dir * (length * 0.5)
        arrow_tip = face_center + arrow_dir * (length * 0.5)
        
        # Arrowhead
        head_size = length * 0.35
        head_right = arrow_tip - arrow_dir * head_size + arrow_perp * (head_size * 0.5)
        head_left = arrow_tip - arrow_dir * head_size - arrow_perp * (head_size * 0.5)
        
        # Project and draw
        base_2d, depth_base = self.renderer.project_point(arrow_base)
        tip_2d, depth_tip = self.renderer.project_point(arrow_tip)
        hr_2d, _ = self.renderer.project_point(head_right)
        hl_2d, _ = self.renderer.project_point(head_left)
        
        # Return depth for sorting
        avg_depth = (depth_base + depth_tip) / 2
        
        pygame.draw.line(self.screen, arrow_color, base_2d, tip_2d, 4)
        pygame.draw.line(self.screen, arrow_color, tip_2d, hr_2d, 4)
        pygame.draw.line(self.screen, arrow_color, tip_2d, hl_2d, 4)
        
        return avg_depth
    
    def draw_target_satellite(self):
        """Draw the target satellite at origin with docking side highlighted."""
        # Create target cube vertices (scaled)
        s = self.target_size / 2
        target_vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
        ])
        
        vertices = self.transform_vertices(target_vertices, self.target_position, self.target_quaternion)
        
        # Project all vertices
        projected = self.renderer.project_points(vertices)
        pts_2d = [p[0] for p in projected]
        depths = [p[1] for p in projected]
        
        # Draw faces with painter's algorithm
        face_depths = []
        for i, face in enumerate(self.cube_faces):
            avg_depth = sum(depths[v] for v in face) / 4
            face_depths.append((avg_depth, i, face))
        
        face_depths.sort(reverse=True)
        
        # Target docking face is the BACK face (face 0, -X direction)
        target_docking_face_idx = 0
        
        for depth, idx, face in face_depths:
            points = [pts_2d[v] for v in face]
            shade_idx = [d[1] for d in face_depths].index(idx)
            shade = 0.5 + 0.5 * (shade_idx / len(face_depths))
            
            if idx == target_docking_face_idx:
                base_color = self.TARGET_DOCK_COLOR
            else:
                base_color = self.TARGET_COLOR
                
            shaded_color = tuple(int(c * shade) for c in base_color)
            pygame.draw.polygon(self.screen, shaded_color, points)
            pygame.draw.polygon(self.screen, self.TARGET_EDGE, points, 2)
        
        # Draw red arrow on target docking face (-Z direction, the yellow face in X-Y plane)
        # Arrow points in +Y body direction
        self.draw_docking_arrow(self.target_position, self.target_quaternion, 
                                np.array([0, 0, -1]), self.AXIS_X, size=s, length=0.5)
        
        # Return average depth for draw ordering
        return sum(depths) / len(depths)
    
    def draw_cube(self, position, quaternion, color, edge_color, dock_color=None, draw_dock_arrow=False):
        """Draw the spacecraft cube with optional docking face highlight."""
        vertices = self.transform_vertices(self.cube_vertices, position, quaternion)
        
        # Project all vertices
        projected = self.renderer.project_points(vertices)
        pts_2d = [p[0] for p in projected]
        depths = [p[1] for p in projected]
        
        # Draw faces (painter's algorithm - sort by depth)
        face_depths = []
        for i, face in enumerate(self.cube_faces):
            avg_depth = sum(depths[v] for v in face) / 4
            face_depths.append((avg_depth, i, face))
        
        face_depths.sort(reverse=True)  # Draw far faces first
        
        for depth, idx, face in face_depths:
            points = [pts_2d[v] for v in face]
            # Shade based on face order (simple lighting)
            shade_idx = [d[1] for d in face_depths].index(idx)
            shade = 0.5 + 0.5 * (shade_idx / len(face_depths))
            
            # Use docking color for the docking face (front face, +X)
            if dock_color is not None and idx == self.docking_face_idx:
                base_color = dock_color
            else:
                base_color = color
                
            shaded_color = tuple(int(c * shade) for c in base_color)
            pygame.draw.polygon(self.screen, shaded_color, points)
            pygame.draw.polygon(self.screen, edge_color, points, 2)
        
        # Draw red arrow on chaser docking face (+X direction)
        # Arrow points in +Z body direction so it aligns with target's arrow when docked
        if draw_dock_arrow:
            self.draw_docking_arrow(position, quaternion, np.array([1, 0, 0]), self.AXIS_X, size=0.5, length=0.4)
        
        # Return average depth for draw ordering
        return sum(depths) / len(depths)
    
    def draw_body_axes(self, position, quaternion, length=1.5):
        """Draw body frame axes."""
        R = self.dynamics.quat_to_rotation_matrix(quaternion)
        
        origin, _ = self.renderer.project_point(position)
        
        # X-axis (red)
        end_x = position + R @ np.array([length, 0, 0])
        end_x_2d, _ = self.renderer.project_point(end_x)
        pygame.draw.line(self.screen, self.AXIS_X, origin, end_x_2d, 3)
        
        # Y-axis (green)
        end_y = position + R @ np.array([0, length, 0])
        end_y_2d, _ = self.renderer.project_point(end_y)
        pygame.draw.line(self.screen, self.AXIS_Y, origin, end_y_2d, 3)
        
        # Z-axis (blue)
        end_z = position + R @ np.array([0, 0, length])
        end_z_2d, _ = self.renderer.project_point(end_z)
        pygame.draw.line(self.screen, self.AXIS_Z, origin, end_z_2d, 3)
    
    def get_docking_face_positions(self):
        """
        Get the center positions of the docking faces for both spacecraft.
        
        Returns:
            chaser_dock_pos: position of chaser's docking face center (+X face)
            target_dock_pos: position of target's docking face center (-X face)
        """
        # Chaser docking face is +X in body frame
        R_chaser = self.dynamics.quat_to_rotation_matrix(self.state[6:10])
        chaser_dock_normal = R_chaser @ np.array([1, 0, 0])
        chaser_dock_pos = self.state[0:3] + chaser_dock_normal * 0.5  # 0.5 = half chaser size
        
        # Target docking face is -X in body frame
        R_target = self.dynamics.quat_to_rotation_matrix(self.target_quaternion)
        target_dock_normal = R_target @ np.array([-1, 0, 0])
        target_dock_pos = self.target_position + target_dock_normal * (self.target_size / 2)
        
        return chaser_dock_pos, target_dock_pos
    
    def draw_hud(self):
        """Draw heads-up display."""
        pos = self.state[0:3]
        vel = self.state[3:6]
        omega = self.state[10:13]
        roll, pitch, yaw = self.quat_to_euler(self.state[6:10])
        
        # Calculate docking face distance
        chaser_dock_pos, target_dock_pos = self.get_docking_face_positions()
        dock_face_dist = chaser_dock_pos - target_dock_pos
        
        # Top-left: basic info
        lines = [
            f"Time: {self.time_elapsed:.1f}s",
            f"",
            f"Position:  [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}] m",
            f"Velocity:  [{vel[0]:7.2f}, {vel[1]:7.2f}, {vel[2]:7.2f}] m/s",
            f"Ang.Vel:   [{omega[0]:7.2f}, {omega[1]:7.2f}, {omega[2]:7.2f}] rad/s",
            f"",
            f"Speed: {np.linalg.norm(vel):.2f} m/s",
            f"Trans ctrl: {np.linalg.norm(self.last_trans_ctrl):.1f} N",
            f"Rot ctrl:   {np.linalg.norm(self.last_rot_ctrl):.2f} N·m",
        ]
        
        y = 20
        for line in lines:
            text = self.font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(text, (20, y))
            y += 20
            
        # Bottom-left: Distance to target panel (dock face to dock face)
        panel_y = self.height - 180
        panel_x = 20
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x - 5, panel_y - 5, 250, 170)
        pygame.draw.rect(self.screen, (30, 35, 50), panel_rect)
        pygame.draw.rect(self.screen, (60, 70, 90), panel_rect, 2)
        
        # Title
        title = self.font.render("DOCK FACE DISTANCE", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_x, panel_y))
        panel_y += 25
        
        # Docking face position errors with colored axis labels
        dist_items = [
            ("X:", dock_face_dist[0], "m", self.AXIS_X),
            ("Y:", dock_face_dist[1], "m", self.AXIS_Y),
            ("Z:", dock_face_dist[2], "m", self.AXIS_Z),
        ]
        
        for label, value, unit, color in dist_items:
            label_text = self.font.render(label, True, color)
            value_text = self.font.render(f"{value:7.2f} {unit}", True, self.TEXT_COLOR)
            self.screen.blit(label_text, (panel_x, panel_y))
            self.screen.blit(value_text, (panel_x + 25, panel_y))
            panel_y += 20
        
        panel_y += 5
        
        # Orientation errors
        angle_error = self.get_docking_alignment_angle()
        orient_items = [
            ("Roll:", roll, "°", self.AXIS_X),
            ("Pitch:", pitch, "°", self.AXIS_Y),
            ("Yaw:", yaw, "°", self.AXIS_Z),
            ("Align:", angle_error, "°", (255, 200, 50) if angle_error < self.dock_angle_epsilon else (255, 100, 100)),
        ]
        
        for label, value, unit, color in orient_items:
            label_text = self.font.render(label, True, color)
            value_text = self.font.render(f"{value:7.1f} {unit}", True, self.TEXT_COLOR)
            self.screen.blit(label_text, (panel_x, panel_y))
            self.screen.blit(value_text, (panel_x + 50, panel_y))
            panel_y += 20
            
        # Top-right: Controls help with color-coded axes
        help_x = self.width - 230
        help_y = 20
        
        # Title
        title = self.font.render("CONTROLS:", True, self.TEXT_COLOR)
        self.screen.blit(title, (help_x, help_y))
        help_y += 30
        
        # Translation controls with axis colors
        trans_label = self.font.render("Translation:", True, self.TEXT_COLOR)
        self.screen.blit(trans_label, (help_x, help_y))
        help_y += 20
        
        # W/S = X axis (red)
        ws_key = self.font.render("W/S", True, (255, 255, 255))
        ws_axis = self.font.render(" = X axis", True, self.AXIS_X)
        self.screen.blit(ws_key, (help_x + 10, help_y))
        self.screen.blit(ws_axis, (help_x + 45, help_y))
        help_y += 20
        
        # A/D = Y axis (green)
        ad_key = self.font.render("A/D", True, (255, 255, 255))
        ad_axis = self.font.render(" = Y axis", True, self.AXIS_Y)
        self.screen.blit(ad_key, (help_x + 10, help_y))
        self.screen.blit(ad_axis, (help_x + 45, help_y))
        help_y += 20
        
        # Q/E = Z axis (blue)
        qe_key = self.font.render("Q/E", True, (255, 255, 255))
        qe_axis = self.font.render(" = Z axis", True, self.AXIS_Z)
        self.screen.blit(qe_key, (help_x + 10, help_y))
        self.screen.blit(qe_axis, (help_x + 45, help_y))
        help_y += 30
        
        # Rotation controls
        rot_label = self.font.render("Rotation:", True, self.TEXT_COLOR)
        self.screen.blit(rot_label, (help_x, help_y))
        help_y += 20
        
        # Z/C = Roll (X axis, red)
        zc_key = self.font.render("Z/C", True, (255, 255, 255))
        zc_axis = self.font.render(" = Roll", True, self.AXIS_X)
        self.screen.blit(zc_key, (help_x + 10, help_y))
        self.screen.blit(zc_axis, (help_x + 45, help_y))
        help_y += 20
        
        # Up/Down = Pitch (Y axis, green)
        ud_key = self.font.render("↑/↓", True, (255, 255, 255))
        ud_axis = self.font.render(" = Pitch", True, self.AXIS_Y)
        self.screen.blit(ud_key, (help_x + 10, help_y))
        self.screen.blit(ud_axis, (help_x + 45, help_y))
        help_y += 20
        
        # Left/Right = Yaw (Z axis, blue)
        lr_key = self.font.render("←/→", True, (255, 255, 255))
        lr_axis = self.font.render(" = Yaw", True, self.AXIS_Z)
        self.screen.blit(lr_key, (help_x + 10, help_y))
        self.screen.blit(lr_axis, (help_x + 45, help_y))
        help_y += 30
        
        # Other controls
        other_lines = [
            "R - Reset",
            "ESC - Quit",
            "Mouse drag - Orbit",
        ]
        for line in other_lines:
            text = self.font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(text, (help_x, help_y))
            help_y += 20
            
        # Success/Failure messages
        if self.success:
            msg = "DOCKING SUCCESSFUL!"
            text = self.font_large.render(msg, True, self.SUCCESS_COLOR)
            rect = text.get_rect(center=(self.width//2, 50))
            self.screen.blit(text, rect)
        elif self.failure:
            msg = "COLLISION - DOCKING FAILED!"
            text = self.font_large.render(msg, True, (255, 80, 80))
            rect = text.get_rect(center=(self.width//2, 50))
            self.screen.blit(text, rect)
            
            reason_text = self.font.render(f"Reason: {self.failure_reason}", True, (255, 150, 150))
            reason_rect = reason_text.get_rect(center=(self.width//2, 80))
            self.screen.blit(reason_text, reason_rect)
    
    def run(self):
        """Main game loop."""
        print("\n" + "="*60)
        print("13D SPACECRAFT DOCKING GAME")
        print("="*60)
        print("\nCONTROLS:")
        print("  WASD:   Translate (body X/Y)")
        print("  Q/E:    Translate (body Z)")
        print("  Arrows: Rotate (pitch/yaw)")
        print("  Z/C:    Rotate (roll)")
        print("  R:      Reset")
        print("  ESC:    Quit")
        print("  Mouse drag: Orbit camera")
        print("\nGOAL: Dock at origin (yellow marker)")
        print("="*60 + "\n")
        
        running = True
        mouse_dragging = False
        last_mouse_pos = (0, 0)
        
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.width = event.w
                    self.height = event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self.renderer.screen_width = self.width
                    self.renderer.screen_height = self.height
                    self.renderer.cx = self.width // 2
                    self.renderer.cy = self.height // 2
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                        print("Game reset!")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 or event.button == 3:
                        mouse_dragging = True
                        last_mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if mouse_dragging:
                        current_pos = pygame.mouse.get_pos()
                        dx = current_pos[0] - last_mouse_pos[0]
                        dy = current_pos[1] - last_mouse_pos[1]
                        self.renderer.camera_azimuth += dx * 0.5
                        # Allow full 360 degree rotation (inverted vertical)
                        self.renderer.camera_elevation -= dy * 0.5
                        # Wrap elevation to stay in reasonable range
                        self.renderer.camera_elevation = self.renderer.camera_elevation % 360
                        if self.renderer.camera_elevation > 180:
                            self.renderer.camera_elevation -= 360
                        last_mouse_pos = current_pos
                elif event.type == pygame.MOUSEWHEEL:
                    self.renderer.camera_distance = np.clip(
                        self.renderer.camera_distance - event.y * 2, 5, 50
                    )
            
            # Update physics
            if not self.game_over:
                keys = pygame.key.get_pressed()
                control = self.get_controls(keys)
                self.state = self.dynamics.step(self.state, control, self.dt)
                self.time_elapsed += self.dt
                
                # Check collision and docking
                is_collision, is_success, failure_reason = self.check_collision_and_docking()
                if is_collision:
                    self.game_over = True
                    if is_success:
                        self.success = True
                        print(f"\n*** DOCKING SUCCESSFUL! Time: {self.time_elapsed:.1f}s ***\n")
                    else:
                        self.failure = True
                        self.failure_reason = failure_reason
                        print(f"\n*** COLLISION - DOCKING FAILED! {failure_reason} ***\n")
            
            # Render
            self.screen.fill(self.BG_COLOR)
            
            self.draw_grid()
            
            pos = self.state[0:3]
            quat = self.state[6:10]
            
            color = self.SUCCESS_COLOR if self.success else self.CHASER_COLOR
            edge_color = (30, 180, 70) if self.success else self.CHASER_EDGE
            dock_color = self.CHASER_DOCK_COLOR if not self.success else self.SUCCESS_COLOR
            
            # Calculate depths for proper draw ordering (painter's algorithm)
            # Get depth of each spacecraft center from camera
            _, target_depth = self.renderer.project_point(self.target_position)
            _, chaser_depth = self.renderer.project_point(pos)
            
            # Draw farther object first (higher depth value = farther from camera)
            if target_depth > chaser_depth:
                # Target is farther, draw it first
                self.draw_target_satellite()
                self.draw_cube(pos, quat, color, edge_color, dock_color, draw_dock_arrow=True)
                self.draw_body_axes(pos, quat)
            else:
                # Chaser is farther, draw it first
                self.draw_cube(pos, quat, color, edge_color, dock_color, draw_dock_arrow=True)
                self.draw_body_axes(pos, quat)
                self.draw_target_satellite()
            
            self.draw_hud()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    game = DockingGame()
    game.run()
