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

        Tolerances checked:
            - 3D position L2 <= eps_p
            - 3D velocity L2 <= eps_v
            - Quaternion angle error <= eps_q
            - 3D angular velocity L2 <= eps_omega
        """
        pos = np.asarray(state[:3], dtype=np.float64)
        vel = np.asarray(state[3:6], dtype=np.float64)
        q = np.asarray(state[6:10], dtype=np.float64)
        omega = np.asarray(state[10:13], dtype=np.float64)

        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        pos_ok = np.linalg.norm(pos) <= self.dynamics.eps_p
        vel_ok = np.linalg.norm(vel) <= self.dynamics.eps_v

        q_goal_np = self.dynamics.q_goal.detach().cpu().numpy()
        att_err = _quat_error_angle_np(q, q_goal_np)
        att_ok = att_err <= self.dynamics.eps_q

        omg_ok = np.linalg.norm(omega) <= self.dynamics.eps_omega

        return pos_ok and vel_ok and att_ok and omg_ok

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

    def _default_dynamics_fn_13d(self, state, control):
        """Evaluate Docking13D dynamics for Euler integration."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()
