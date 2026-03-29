"""
Shared utilities for all 13D docking controllers.

Provides docking-check, collision-check, state-wrapping, and bang-bang
optimal control methods that are common to BRTController13D,
MPCController13D, and MPCTerminalController13D.

These are implemented as a mixin class so each controller can inherit
the functionality without code duplication while keeping the controller
hierarchies flat and easy to follow.
"""

import numpy as np
import torch


def _quat_mul_np(q, p):
    """Hamilton product of two scalar-first quaternions (numpy)."""
    q0, q1, q2, q3 = q
    p0, p1, p2, p3 = p
    return np.array([
        q0*p0 - q1*p1 - q2*p2 - q3*p3,
        q0*p1 + q1*p0 + q2*p3 - q3*p2,
        q0*p2 - q1*p3 + q2*p0 + q3*p1,
        q0*p3 + q1*p2 - q2*p1 + q3*p0,
    ])


def _quat_error_angle_np(q, q_goal):
    """Shortest rotation angle (radians) between two unit quaternions (numpy).

    Uses the atan2-based formula for numerical stability near zero angle.
    """
    q_goal_conj = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
    q_rel = _quat_mul_np(q_goal_conj, q)
    qv_norm = np.linalg.norm(q_rel[1:4])
    return 2.0 * np.arctan2(qv_norm, abs(q_rel[0]))


class Docking13DControllerMixin:
    """Mixin providing 13D-specific helpers.

    Expects the host class to have:
        self.dynamics   – a Docking13D instance
        self.device     – torch device string
    """

    def _check_docked_13d(self, state):
        """Check if all 13D docking tolerances are met.

        Tolerances checked (matching reach_fn in dynamics):
            - x,z lateral ≤ eps_p + y inside goal band
            - lateral velocity (xz) ≤ eps_v_lateral
            - axial velocity vy in [eps_v_axial_lo, eps_v_axial_hi]
            - Quaternion angle error ≤ eps_q
            - pitch/yaw rate (wy,wz) ≤ eps_omega_pitchyaw
            - roll rate |wx| ≤ eps_omega_roll
        """
        pos = np.asarray(state[:3], dtype=np.float64)
        vel = np.asarray(state[3:6], dtype=np.float64)
        q = np.asarray(state[6:10], dtype=np.float64)
        omega = np.asarray(state[10:13], dtype=np.float64)

        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        d = self.dynamics

        # Position: xz distance from post axis + y inside goal band
        xz_ok = np.sqrt(pos[0]**2 + pos[2]**2) <= d.eps_p
        y_ok = (d.goal_y_min <= pos[1] <= d.goal_y_max)
        pos_ok = xz_ok and y_ok

        # Velocity: lateral (xz) and axial (y) separately
        vlat_ok = np.sqrt(vel[0]**2 + vel[2]**2) <= d.eps_v_lateral
        vax_ok = (d.eps_v_axial_lo <= vel[1] <= d.eps_v_axial_hi)

        # Attitude
        q_goal_np = d.q_goal.detach().cpu().numpy()
        att_err = _quat_error_angle_np(q, q_goal_np)
        att_ok = att_err <= d.eps_q

        # Angular rate: pitch/yaw (wy,wz) and roll (wx) separately
        omg_py_ok = np.sqrt(omega[1]**2 + omega[2]**2) <= d.eps_omega_pitchyaw
        omg_r_ok = abs(omega[0]) <= d.eps_omega_roll

        return pos_ok and vlat_ok and vax_ok and att_ok and omg_py_ok and omg_r_ok

    def _check_collision_13d(self, state):
        """Orientation-aware 3D collision check (8-corner chaser box)."""
        return self.dynamics.check_collision_oriented_3d(state)

    @staticmethod
    def _wrap_state_13d(state):
        """Normalize the quaternion portion of a 13D state array in-place."""
        q_norm = np.linalg.norm(state[6:10])
        if q_norm > 1e-12:
            state[6:10] /= q_norm
        return state

    def _batch_check_collision_oriented(self, states):
        """Vectorised 8-corner oriented-box collision check.

        Args:
            states: (B, 13) torch tensor on device.

        Returns:
            (B,) bool tensor — True where any chaser corner is inside obstacle.
        """
        d = self.dynamics
        q = states[:, 6:10]
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)
        pos = states[:, :3]  # (B, 3)

        R = d.quat_to_R_LVLH_to_body(q)  # (B, 3, 3)  LVLH→body

        hw = d.w_c / 2.0
        hh = d.h_c / 2.0
        hd = d.d_c / 2.0
        # 8 body-frame corners (8, 3)
        signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], dtype=states.dtype, device=states.device)
        body_corners = signs * torch.tensor(
            [hw, hh, hd], dtype=states.dtype, device=states.device)

        # R^T maps body → LVLH:  corner @ R = (R^T @ corner^T)^T
        lvlh = torch.matmul(body_corners, R) + pos.unsqueeze(1)  # (B, 8, 3)

        cx, cy, cz = lvlh[..., 0], lvlh[..., 1], lvlh[..., 2]

        # Target body (y ∈ [0, h_t])
        half_w = d.w_t / 2.0
        half_d = d.d_t / 2.0
        in_body = ((torch.abs(cx) <= half_w)
                   & (cy >= 0) & (cy <= d.h_t)
                   & (torch.abs(cz) <= half_d))

        # Docking post (y ∈ [-post_length, 0])
        in_post = ((torch.abs(cx) <= d.post_hw_x)
                   & (torch.abs(cz) <= d.post_hw_z)
                   & (cy >= -d.post_length) & (cy <= 0))

        return (in_body | in_post).any(dim=-1)  # (B,)

    @staticmethod
    def _batch_wrap_quat(states):
        """Normalise quaternion for (B, 13) tensor (no velocity clamping).

        Matches _wrap_state_13d behaviour for fair comparison with
        sequential simulation.
        """
        q = states[:, 6:10]
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)
        return torch.cat([states[:, :6], q, states[:, 10:]], dim=-1)

    def _compute_brt_control_13d(self, dvds, state):
        """Bang-bang optimal control from value-function gradient for 13D.

        Force allocation:
            Body-frame coefficient = R(q) @ (dV/dv_LVLH) / m_c
            Sign of each body-frame coefficient determines bang direction.

        Torque allocation:
            Effective coefficient = I^{-T} @ (dV/d_omega)
            Sign determines bang direction.

        Args:
            dvds: numpy (state_dim,) spatial gradient of V.
            state: numpy (13,) current state (needs quaternion for R).

        Returns:
            numpy (6,) control [Fx, Fy, Fz, tx, ty, tz].
        """
        q = np.asarray(state[6:10], dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm
        R = self.dynamics._quat_to_R_np(q)  # LVLH-to-body (3x3)

        # Force: body coefficients = R @ p_v / mc
        p_v = np.asarray(dvds[3:6], dtype=np.float64)
        coeff_body = (R @ p_v) / self.dynamics.mc
        F = np.where(coeff_body > 0, -self.dynamics.F_bar, self.dynamics.F_bar)

        # Torque: effective coefficients = I^{-T} @ p_omega
        I_np = self.dynamics.I.detach().cpu().numpy()  # (3,3) diagonal
        p_omega = np.asarray(dvds[10:13], dtype=np.float64)
        coeff_tau = np.linalg.solve(I_np.T, p_omega)
        tau = np.where(coeff_tau > 0, -self.dynamics.tau_bar, self.dynamics.tau_bar)

        return np.concatenate([F, tau])

    def _compute_brt_control_batch(self, dvds_batch, states_batch):
        """Vectorised bang-bang optimal control for a batch of states.

        Args:
            dvds_batch:   (B, state_dim) torch tensor on self.device.
            states_batch: (B, 13) torch tensor on self.device.

        Returns:
            (B, 6) torch tensor — [Fx, Fy, Fz, tx, ty, tz] for each IC.
        """
        d = self.dynamics
        q = states_batch[:, 6:10]
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)
        R = d.quat_to_R_LVLH_to_body(q)  # (B, 3, 3) LVLH→body

        # Force: body coefficients = R @ p_v / mc
        p_v = dvds_batch[:, 3:6].float().unsqueeze(-1)       # (B, 3, 1)
        coeff_body = torch.bmm(R, p_v).squeeze(-1) / d.mc   # (B, 3)
        F = torch.where(coeff_body > 0,
                        torch.full_like(coeff_body, -d.F_bar),
                        torch.full_like(coeff_body,  d.F_bar))

        # Torque: I^{-T} @ p_omega
        I_np = d.I.detach().to(dvds_batch.device)
        I_inv_T = torch.inverse(I_np.T)                       # (3, 3)
        p_omega = dvds_batch[:, 10:13].float()                # (B, 3)
        coeff_tau = torch.mm(p_omega, I_inv_T.T)              # (B, 3)
        tau = torch.where(coeff_tau > 0,
                          torch.full_like(coeff_tau, -d.tau_bar),
                          torch.full_like(coeff_tau,  d.tau_bar))

        return torch.cat([F, tau], dim=-1)  # (B, 6)

    def _default_dynamics_fn_13d(self, state, control):
        """Evaluate Docking13D dynamics for Euler integration."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()
