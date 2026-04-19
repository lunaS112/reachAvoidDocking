"""
Velocity-Space 3-Panel Animated GIF for Docking13D

Produces a single animated GIF with three synchronised quadrants showing
2-D value-function slices in velocity space:

  Q1 (top-left):     Vx vs Vy  (vz fixed at 0)
  Q2 (top-right):    Vx vs Vz  (vy fixed at midpoint of goal band)
  Q3 (bottom-left):  Vy vs Vz  (vx fixed at 0)

Position, quaternion, and angular velocity are held at goal values so
the slices show the BRAT geometry at the target configuration.

Each frame corresponds to a different time query, animating the BRAT
boundary as it evolves over the countdown horizon.

Usage (from run_controller_13d.py, via --viz_gifs flag):

    from utils.controllers.velocity_slice_gif_13d import VelocitySliceGIF13D
    gif = VelocitySliceGIF13D(brat_viz, dynamics)
    gif.generate(result, output_path, max_frames=50, resolution=80)
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
from utils.brat_visualization_13d import BRATVisualizer13D


# Velocity plane definitions: (idx_a, idx_b, fixed_idx_in_vel_space)
# State indices: vx=3, vy=4, vz=5
_VEL_PLANES = {
    'vx_vy': (3, 4),
    'vx_vz': (3, 5),
    'vy_vz': (4, 5),
}

_VEL_LABELS = {
    'vx_vy': ('vx (m/s)', 'vy (m/s)'),
    'vx_vz': ('vx (m/s)', 'vz (m/s)'),
    'vy_vz': ('vy (m/s)', 'vz (m/s)'),
}

_VEL_TITLES = {
    'vx_vy': 'Vx–Vy Slice',
    'vx_vz': 'Vx–Vz Slice',
    'vy_vz': 'Vy–Vz Slice',
}


def _q_goal_np(dynamics):
    qg = dynamics.q_goal
    if hasattr(qg, 'cpu'):
        qg = qg.detach().cpu().numpy()
    return np.asarray(qg, dtype=np.float64)


def _goal_rect(dynamics, plane: str):
    """Return (x_lo, x_hi, y_lo, y_hi) for the goal set in this vel plane."""
    d = dynamics
    lat = d.eps_v_lateral
    lo = d.eps_v_axial_lo
    hi = d.eps_v_axial_hi
    if plane == 'vx_vy':
        return (-lat, lat, lo, hi)
    elif plane == 'vx_vz':
        return (-lat, lat, -lat, lat)
    elif plane == 'vy_vz':
        return (lo, hi, -lat, lat)


class VelocitySliceGIF13D:
    """3-panel animated GIF showing value-function slices in velocity space.

    Parameters
    ----------
    brat_viz  : BRATVisualizer13D
    dynamics : Docking13D
    """

    def __init__(self, brat_viz: BRATVisualizer13D, dynamics):
        self.brat_viz = brat_viz
        self.dynamics = dynamics

    def _build_base_state(self, vy_mid: float) -> np.ndarray:
        """Construct a 13D state at goal values for non-velocity dims."""
        d = self.dynamics
        q_goal = _q_goal_np(d)
        state = np.zeros(13, dtype=np.float64)
        # Position at goal center
        state[0] = 0.0
        state[1] = d.goal_y_center
        state[2] = 0.0
        # Velocity: vx=0, vy=midpoint, vz=0
        state[3] = 0.0
        state[4] = vy_mid
        state[5] = 0.0
        # Quaternion at goal
        state[6:10] = q_goal
        # Angular velocity = 0
        state[10:13] = 0.0
        return state

    @staticmethod
    def _vel_bounds(dynamics, idx: int):
        """Return velocity axis bounds from state space range."""
        sr = dynamics.state_range_
        if hasattr(sr, 'cpu'):
            sr = sr.detach().cpu().numpy()
        return (float(sr[idx, 0]), float(sr[idx, 1]))

    def _draw_vel_panel(self, ax, A, B, V, dVda, dVdb,
                        plane, grid_bounds_2d, quiver_stride, vmax_abs,
                        state=None, traj_so_far=None):
        """Draw a single velocity-space value-function slice on *ax*."""
        xlabel, ylabel = _VEL_LABELS[plane]
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

        # Quiver arrows: -∇V (steepest descent)
        s = quiver_stride
        u_q = -dVda[::s, ::s]
        v_q = -dVdb[::s, ::s]
        mag = np.sqrt(u_q**2 + v_q**2) + 1e-12
        ax.quiver(A[::s, ::s], B[::s, ::s], u_q / mag, v_q / mag,
                  color='k', alpha=0.25, scale=30, width=0.003,
                  headwidth=3.5, zorder=7)

        # Goal set overlay (circle for vx-vz L2 norm, rectangle otherwise)
        if plane == 'vx_vz':
            circ = mpatches.Circle(
                (0, 0), self.dynamics.eps_v_lateral,
                linewidth=2.0, edgecolor='gold', facecolor='gold',
                alpha=0.20, linestyle='--', zorder=8)
            ax.add_patch(circ)
        else:
            gx_lo, gx_hi, gy_lo, gy_hi = _goal_rect(self.dynamics, plane)
            rect = mpatches.Rectangle(
                (gx_lo, gy_lo), gx_hi - gx_lo, gy_hi - gy_lo,
                linewidth=2.0, edgecolor='gold', facecolor='gold',
                alpha=0.20, linestyle='--', zorder=8)
            ax.add_patch(rect)

        # Trajectory trace and current-state marker
        idx_a, idx_b = _VEL_PLANES[plane]
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
        ax.set_title(_VEL_TITLES[plane], fontsize=10, pad=3)
        ax.grid(True, alpha=0.15, zorder=0)

    def generate(self, result: dict, output_path: str, *,
                 max_frames: int = 50, resolution: int = 80,
                 quiver_stride: int = 6, fps: int = 5):
        """Generate the 3-panel velocity-space GIF.

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

        # Midpoint of vy goal band for the base state
        vy_mid = (d.eps_v_axial_lo + d.eps_v_axial_hi) / 2.0
        base_state = self._build_base_state(vy_mid)

        planes = ['vx_vy', 'vx_vz', 'vy_vz']

        # Grid bounds from state space range
        vel_bounds = {}
        for plane in planes:
            idx_a, idx_b = _VEL_PLANES[plane]
            vel_bounds[plane] = [
                self._vel_bounds(d, idx_a),
                self._vel_bounds(d, idx_b),
            ]

        # ---- Pre-evaluate all frames -------------------------------- #
        n_frames = len(key_idx)
        print(f"\n[VelocitySliceGIF] Evaluating {n_frames} frames "
              f"({resolution}² per slice)...")

        frame_data = []
        for count, ki in enumerate(key_idx):
            t_q = float(t_queries[ki])
            fd = {'sim_time': sim_times[ki], 't_query': t_q,
                  'state': traj[ki], 'traj_so_far': traj[:ki + 1]}

            for plane in planes:
                idx_a, idx_b = _VEL_PLANES[plane]
                # Build per-plane base state with correct fixed velocity
                ps = base_state.copy()
                if plane == 'vx_vy':
                    ps[5] = 0.0       # vz=0
                elif plane == 'vx_vz':
                    ps[4] = vy_mid    # vy=midpoint
                elif plane == 'vy_vz':
                    ps[3] = 0.0       # vx=0

                A, B, V, dVda, dVdb = (
                    self.brat_viz.evaluate_brat_grid_2d_generic(
                        ps, t_q, idx_a, idx_b,
                        vel_bounds[plane], resolution))
                fd[plane] = {'A': A, 'B': B, 'V': V,
                             'dVda': dVda, 'dVdb': dVdb}

            frame_data.append(fd)
            if (count + 1) % 5 == 0 or count == n_frames - 1:
                print(f"  [{count + 1}/{n_frames}]")

        # ---- Build animation ----------------------------------------- #
        fig = plt.figure(figsize=(18, 7))
        gs = fig.add_gridspec(1, 3, wspace=0.30, top=0.88)
        ax_vxvy = fig.add_subplot(gs[0, 0])
        ax_vxvz = fig.add_subplot(gs[0, 1])
        ax_vyvz = fig.add_subplot(gs[0, 2])
        plane_axes = {'vx_vy': ax_vxvy, 'vx_vz': ax_vxvz, 'vy_vz': ax_vyvz}

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
            for plane in planes:
                ax = plane_axes[plane]
                ax.clear()
                pdata = fd[plane]
                self._draw_vel_panel(
                    ax, pdata['A'], pdata['B'], pdata['V'],
                    pdata['dVda'], pdata['dVdb'],
                    plane, vel_bounds[plane], quiver_stride, vmax_abs,
                    state=fd['state'], traj_so_far=fd['traj_so_far'])

            fig.suptitle(
                f"Velocity-Space BRAT  |  "
                f"t_query = {fd['t_query']:.2f} s  |  "
                f"sim_t = {fd['sim_time']:.2f} s  |  "
                f"pos=goal, q=q_goal, ω=0",
                fontsize=11, y=0.96)

        _draw_frame(0)

        anim = FuncAnimation(fig, _draw_frame, frames=n_frames,
                             interval=1000 // fps, repeat=True)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"[VelocitySliceGIF] Saved → {output_path}")
