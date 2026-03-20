"""
Attitude & Angular-Velocity 6-Panel Animated GIF for Docking13D

Produces a single animated GIF with six synchronised panels (2x3):

  Top row (quaternion slices, other states at goal):
    (0,0)  q1 vs q2   (q0, q3 at goal)
    (0,1)  q1 vs q3   (q0, q2 at goal)
    (0,2)  q2 vs q3   (q0, q1 at goal)

  Bottom row (angular velocity slices, other states at goal):
    (1,0)  wx vs wy   (wz = 0)
    (1,1)  wx vs wz   (wy = 0)
    (1,2)  wy vs wz   (wx = 0)

Each frame corresponds to a different time query, animating the BRAT
boundary as it evolves over the countdown horizon.

Usage (from run_controller_13d.py, via --viz_gifs flag):

    from utils.controllers.attitude_slice_gif_13d import AttitudeSliceGIF13D
    gif = AttitudeSliceGIF13D(brt_viz, dynamics)
    gif.generate(result, output_path, max_frames=50, resolution=60)
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from utils.controllers.anim_utils import select_key_frames, get_t_queries
from utils.brt_visualization_13d import BRTVisualizer13D


# -- Plane definitions ---------------------------------------------------- #
# Quaternion planes: sweep vector-component pairs (q0 stays near-fixed)
_QUAT_PLANES = {
    'q1_q2': (7, 8),
    'q1_q3': (7, 9),
    'q2_q3': (8, 9),
}
_QUAT_LABELS = {
    'q1_q2': ('q1', 'q2'),
    'q1_q3': ('q1', 'q3'),
    'q2_q3': ('q2', 'q3'),
}
_QUAT_TITLES = {
    'q1_q2': 'q1–q2 Slice',
    'q1_q3': 'q1–q3 Slice',
    'q2_q3': 'q2–q3 Slice',
}

# Angular velocity planes
_OMEGA_PLANES = {
    'wx_wy': (10, 11),
    'wx_wz': (10, 12),
    'wy_wz': (11, 12),
}
_OMEGA_LABELS = {
    'wx_wy': ('\u03c9x (rad/s)', '\u03c9y (rad/s)'),
    'wx_wz': ('\u03c9x (rad/s)', '\u03c9z (rad/s)'),
    'wy_wz': ('\u03c9y (rad/s)', '\u03c9z (rad/s)'),
}
_OMEGA_TITLES = {
    'wx_wy': '\u03c9x–\u03c9y Slice',
    'wx_wz': '\u03c9x–\u03c9z Slice',
    'wy_wz': '\u03c9y–\u03c9z Slice',
}


def _q_goal_np(dynamics):
    qg = dynamics.q_goal
    if hasattr(qg, 'cpu'):
        qg = qg.detach().cpu().numpy()
    return np.asarray(qg, dtype=np.float64)


def _quat_error_angle_np(q, q_goal):
    """Vectorised quaternion error angle in numpy (matches dynamics method).

    Parameters
    ----------
    q      : (..., 4) array of quaternions [q0, q1, q2, q3]
    q_goal : (4,) goal quaternion

    Returns
    -------
    (...,) array of angles in radians.
    """
    # q_rel = conj(q_goal) * q   (Hamilton product)
    gw, gx, gy, gz = q_goal
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # conj(q_goal) = (gw, -gx, -gy, -gz)
    rw = gw * qw + gx * qx + gy * qy + gz * qz
    rx = gw * qx - gx * qw - gy * qz + gz * qy
    ry = gw * qy + gx * qz - gy * qw - gz * qx
    rz = gw * qz - gx * qy + gy * qx - gz * qw
    r0 = np.abs(rw)
    rv_norm = np.sqrt(rx**2 + ry**2 + rz**2 + 1e-12)
    return 2.0 * np.arctan2(rv_norm, r0)


class AttitudeSliceGIF13D:
    """6-panel animated GIF: quaternion + angular velocity value-function slices.

    Parameters
    ----------
    brt_viz  : BRTVisualizer13D
    dynamics : Docking13D
    """

    def __init__(self, brt_viz: BRTVisualizer13D, dynamics):
        self.brt_viz = brt_viz
        self.dynamics = dynamics

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _build_base_state(self) -> np.ndarray:
        """13D state at goal values for all dims."""
        d = self.dynamics
        q_goal = _q_goal_np(d)
        vy_mid = (d.eps_v_axial_lo + d.eps_v_axial_hi) / 2.0
        state = np.zeros(13, dtype=np.float64)
        state[0] = 0.0
        state[1] = d.goal_y_center
        state[2] = 0.0
        state[3] = 0.0
        state[4] = vy_mid
        state[5] = 0.0
        state[6:10] = q_goal
        state[10:13] = 0.0
        return state

    @staticmethod
    def _state_bounds(dynamics, idx: int):
        """Return axis bounds from state_range_."""
        sr = dynamics.state_range_
        if hasattr(sr, 'cpu'):
            sr = sr.detach().cpu().numpy()
        return (float(sr[idx, 0]), float(sr[idx, 1]))

    # ------------------------------------------------------------------ #
    #  Goal overlays
    # ------------------------------------------------------------------ #

    def _draw_quat_goal(self, ax, plane, grid_bounds_2d, resolution):
        """Draw the eps_q goal contour on a quaternion panel."""
        d = self.dynamics
        q_goal = _q_goal_np(d)
        idx_a, idx_b = _QUAT_PLANES[plane]

        a_vals = np.linspace(grid_bounds_2d[0][0], grid_bounds_2d[0][1],
                             resolution)
        b_vals = np.linspace(grid_bounds_2d[1][0], grid_bounds_2d[1][1],
                             resolution)
        Ag, Bg = np.meshgrid(a_vals, b_vals)

        # Build full quaternion at each grid point
        q_grid = np.broadcast_to(q_goal, Ag.shape + (4,)).copy()
        q_grid[..., idx_a - 6] = Ag
        q_grid[..., idx_b - 6] = Bg
        # Normalise
        q_grid /= (np.linalg.norm(q_grid, axis=-1, keepdims=True) + 1e-12)

        angle_err = _quat_error_angle_np(q_grid, q_goal)
        ax.contour(Ag, Bg, angle_err, levels=[d.eps_q],
                   colors='gold', linewidths=2.0, linestyles='--', zorder=8)

    def _draw_omega_goal(self, ax, plane):
        """Draw the goal region on an angular velocity panel."""
        d = self.dynamics
        if plane == 'wy_wz':
            # L2 norm constraint: circle
            circ = mpatches.Circle(
                (0, 0), d.eps_omega_pitchyaw,
                linewidth=2.0, edgecolor='gold', facecolor='gold',
                alpha=0.20, linestyle='--', zorder=8)
            ax.add_patch(circ)
        elif plane == 'wx_wy':
            rect = mpatches.Rectangle(
                (-d.eps_omega_roll, -d.eps_omega_pitchyaw),
                2 * d.eps_omega_roll, 2 * d.eps_omega_pitchyaw,
                linewidth=2.0, edgecolor='gold', facecolor='gold',
                alpha=0.20, linestyle='--', zorder=8)
            ax.add_patch(rect)
        elif plane == 'wx_wz':
            rect = mpatches.Rectangle(
                (-d.eps_omega_roll, -d.eps_omega_pitchyaw),
                2 * d.eps_omega_roll, 2 * d.eps_omega_pitchyaw,
                linewidth=2.0, edgecolor='gold', facecolor='gold',
                alpha=0.20, linestyle='--', zorder=8)
            ax.add_patch(rect)

    # ------------------------------------------------------------------ #
    #  Panel drawing
    # ------------------------------------------------------------------ #

    def _draw_panel(self, ax, A, B, V, dVda, dVdb,
                    plane, grid_bounds_2d, quiver_stride, vmax_abs,
                    resolution, state=None, traj_so_far=None):
        """Draw a single 2-D value-function slice on *ax*."""
        is_quat = plane in _QUAT_PLANES
        labels = _QUAT_LABELS if is_quat else _OMEGA_LABELS
        titles = _QUAT_TITLES if is_quat else _OMEGA_TITLES
        planes_map = _QUAT_PLANES if is_quat else _OMEGA_PLANES

        xlabel, ylabel = labels[plane]
        idx_a, idx_b = planes_map[plane]
        extent = [grid_bounds_2d[0][0], grid_bounds_2d[0][1],
                  grid_bounds_2d[1][0], grid_bounds_2d[1][1]]

        ax.imshow(V.T, origin='lower', extent=extent, aspect='auto',
                  cmap='RdYlBu', vmin=-vmax_abs, vmax=vmax_abs,
                  alpha=0.75, zorder=1)

        # V=0 contour
        a_vals = np.linspace(extent[0], extent[1], V.shape[0])
        b_vals = np.linspace(extent[2], extent[3], V.shape[1])
        Ag, Bg = np.meshgrid(a_vals, b_vals)
        ax.contour(Ag, Bg, V.T, levels=[0.0], colors='black',
                   linewidths=1.5, linestyles='-', zorder=6)

        # Quiver arrows: -nabla V
        s = quiver_stride
        u_q = -dVda[::s, ::s]
        v_q = -dVdb[::s, ::s]
        mag = np.sqrt(u_q**2 + v_q**2) + 1e-12
        ax.quiver(A[::s, ::s], B[::s, ::s], u_q / mag, v_q / mag,
                  color='k', alpha=0.25, scale=30, width=0.003,
                  headwidth=3.5, zorder=7)

        # Goal overlay
        if is_quat:
            self._draw_quat_goal(ax, plane, grid_bounds_2d, resolution)
        else:
            self._draw_omega_goal(ax, plane)

        # Trajectory trace and current-state marker
        if traj_so_far is not None and len(traj_so_far) > 1:
            ax.plot(traj_so_far[:, idx_a], traj_so_far[:, idx_b],
                    '-', color='cyan', linewidth=1.5, alpha=0.85, zorder=8)
        if state is not None:
            ax.plot(state[idx_a], state[idx_b], 'o', color='magenta',
                    markersize=6, markeredgecolor='white', markeredgewidth=1.0,
                    zorder=9)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(titles[plane], fontsize=10, pad=3)
        ax.grid(True, alpha=0.15, zorder=0)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def generate(self, result: dict, output_path: str, *,
                 max_frames: int = 50, resolution: int = 60,
                 quiver_stride: int = 6, fps: int = 5):
        """Generate the 6-panel attitude/omega GIF.

        Parameters
        ----------
        result        : simulation result dict.
        output_path   : path for the output .gif file.
        max_frames    : maximum number of animation frames.
        resolution    : grid resolution for 2-D slices.
        quiver_stride : stride for quiver arrows.
        fps           : frames per second.
        """
        d = self.dynamics
        traj = np.asarray(result['trajectory'])
        t_queries = get_t_queries(result)
        sim_times = np.asarray(result.get('sim_times',
                                          np.arange(len(traj)) * 0.1))
        key_idx = select_key_frames(len(traj), max_frames)

        base_state = self._build_base_state()

        quat_planes = ['q1_q2', 'q1_q3', 'q2_q3']
        omega_planes = ['wx_wy', 'wx_wz', 'wy_wz']
        all_planes = quat_planes + omega_planes

        # Grid bounds from state space range
        grid_bounds = {}
        for plane in quat_planes:
            idx_a, idx_b = _QUAT_PLANES[plane]
            grid_bounds[plane] = [
                self._state_bounds(d, idx_a),
                self._state_bounds(d, idx_b),
            ]
        for plane in omega_planes:
            idx_a, idx_b = _OMEGA_PLANES[plane]
            grid_bounds[plane] = [
                self._state_bounds(d, idx_a),
                self._state_bounds(d, idx_b),
            ]

        # ---- Pre-evaluate all frames -------------------------------- #
        n_frames = len(key_idx)
        print(f"\n[AttitudeSliceGIF] Evaluating {n_frames} frames "
              f"({resolution}\u00b2 per slice, 6 panels)...")

        frame_data = []
        for count, ki in enumerate(key_idx):
            t_q = float(t_queries[ki])
            fd = {'sim_time': sim_times[ki], 't_query': t_q,
                  'state': traj[ki], 'traj_so_far': traj[:ki + 1]}

            for plane in all_planes:
                is_quat = plane in _QUAT_PLANES
                planes_map = _QUAT_PLANES if is_quat else _OMEGA_PLANES
                idx_a, idx_b = planes_map[plane]
                ps = base_state.copy()
                # Fixed values are already at goal; no extra overrides needed

                A, B, V, dVda, dVdb = (
                    self.brt_viz.evaluate_brt_grid_2d_generic(
                        ps, t_q, idx_a, idx_b,
                        grid_bounds[plane], resolution))
                fd[plane] = {'A': A, 'B': B, 'V': V,
                             'dVda': dVda, 'dVdb': dVdb}

            frame_data.append(fd)
            if (count + 1) % 5 == 0 or count == n_frames - 1:
                print(f"  [{count + 1}/{n_frames}]")

        # ---- Build animation ----------------------------------------- #
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.30, top=0.90)

        plane_axes = {}
        for col, plane in enumerate(quat_planes):
            plane_axes[plane] = fig.add_subplot(gs[0, col])
        for col, plane in enumerate(omega_planes):
            plane_axes[plane] = fig.add_subplot(gs[1, col])

        vmax_abs = 1.0

        # Shared colorbar
        norm = mcolors.Normalize(vmin=-vmax_abs, vmax=vmax_abs)
        sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(plane_axes.values()),
                            shrink=0.85, pad=0.03, location='right')
        cbar.set_label('V(x, t)', fontsize=10)

        def _draw_frame(frame_idx):
            fd = frame_data[frame_idx]
            for plane in all_planes:
                ax = plane_axes[plane]
                ax.clear()
                pdata = fd[plane]
                self._draw_panel(
                    ax, pdata['A'], pdata['B'], pdata['V'],
                    pdata['dVda'], pdata['dVdb'],
                    plane, grid_bounds[plane], quiver_stride, vmax_abs,
                    resolution,
                    state=fd['state'], traj_so_far=fd['traj_so_far'])

            fig.suptitle(
                f"Attitude & \u03c9 BRAT  |  "
                f"t_query = {fd['t_query']:.2f} s  |  "
                f"sim_t = {fd['sim_time']:.2f} s  |  "
                f"pos=goal, v=goal",
                fontsize=11, y=0.96)

        _draw_frame(0)

        anim = FuncAnimation(fig, _draw_frame, frames=n_frames,
                             interval=1000 // fps, repeat=True)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"[AttitudeSliceGIF] Saved \u2192 {output_path}")
