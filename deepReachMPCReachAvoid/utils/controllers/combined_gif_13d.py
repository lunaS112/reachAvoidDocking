"""
Combined 4-Panel Animated GIF for Docking13D

Produces a single animated GIF with four synchronised quadrants:

  Q1 (top-left):     XY value-function slice
  Q2 (top-right):    XZ value-function slice
  Q3 (bottom-left):  YZ value-function slice
  Q4 (bottom-right): 3-D isometric view (BRT blob + target + chaser)

Each frame corresponds to the same simulation time-step so the panels
animate in lockstep.

Usage (from run_controller_13d.py, via --viz_gifs flag):

    from utils.controllers.combined_gif_13d import CombinedGIF13D
    gif = CombinedGIF13D(brt_viz, dynamics)
    gif.generate(result, output_path, max_frames=50, resolution=80)
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

from utils.controllers.anim_utils import select_key_frames, get_t_queries
from utils.controllers.slice_gif_13d import (
    _draw_target_on_slice, _rotation_text, _IDX_MAP, _AXIS_LABELS,
)
from utils.brt_visualization_13d import BRTVisualizer13D, compute_scene_limits


class CombinedGIF13D:
    """4-panel animated GIF combining three 2-D slices and a 3-D view.

    Parameters
    ----------
    brt_viz  : BRTVisualizer13D
    dynamics : Docking13D
    """

    def __init__(self, brt_viz: BRTVisualizer13D, dynamics):
        self.brt_viz = brt_viz
        self.dynamics = dynamics

    # ------------------------------------------------------------------ #
    #  Internal: compute consistent grid bounds across trajectory
    # ------------------------------------------------------------------ #

    @staticmethod
    def _slice_extent(traj: np.ndarray, idx_a: int, idx_b: int,
                      margin: float = 2.0, clamp: float = 12.0,
                      context: float = 5.0):
        """Return ``[(a_min, a_max), (b_min, b_max)]`` for a given plane."""
        a_min = float(np.clip(traj[:, idx_a].min() - margin, -clamp, clamp))
        a_max = float(np.clip(traj[:, idx_a].max() + margin, -clamp, clamp))
        b_min = float(np.clip(traj[:, idx_b].min() - margin, -clamp, clamp))
        b_max = float(np.clip(traj[:, idx_b].max() + margin, -clamp, clamp))
        a_min = min(a_min, -context)
        a_max = max(a_max, context)
        b_min = min(b_min, -context)
        b_max = max(b_max, context)
        return [(a_min, a_max), (b_min, b_max)]

    # ------------------------------------------------------------------ #
    #  Internal: draw one 2-D slice panel
    # ------------------------------------------------------------------ #

    def _draw_slice_panel(self, ax, fd, plane, grid_bounds_2d,
                          quiver_stride, vmax_abs):
        """Draw a single 2-D value-function slice on *ax*."""
        idx_a, idx_b, idx_fixed = _IDX_MAP[plane]
        xlabel, ylabel, fixed_name = _AXIS_LABELS[plane]
        A, B, V = fd['A'], fd['B'], fd['V']
        dVda, dVdb = fd['dVda'], fd['dVdb']
        state = fd['state']
        traj_part = fd['traj_so_far']

        extent = [grid_bounds_2d[0][0], grid_bounds_2d[0][1],
                  grid_bounds_2d[1][0], grid_bounds_2d[1][1]]

        ax.imshow(V.T, origin='lower', extent=extent, aspect='equal',
                  cmap='RdYlBu', vmin=-vmax_abs, vmax=vmax_abs,
                  alpha=0.75, zorder=1)

        # V=0 contour
        a_vals = np.linspace(extent[0], extent[1], V.shape[0])
        b_vals = np.linspace(extent[2], extent[3], V.shape[1])
        Ag, Bg = np.meshgrid(a_vals, b_vals)
        ax.contour(Ag, Bg, V.T, levels=[0.0], colors='black',
                   linewidths=1.2, linestyles='-', zorder=6)

        # Quiver arrows: -∇V (steepest descent)
        s = quiver_stride
        u_q = -dVda[::s, ::s]
        v_q = -dVdb[::s, ::s]
        mag = np.sqrt(u_q**2 + v_q**2) + 1e-12
        ax.quiver(A[::s, ::s], B[::s, ::s], u_q / mag, v_q / mag,
                  color='k', alpha=0.30, scale=30, width=0.003,
                  headwidth=3.5, zorder=7)

        _draw_target_on_slice(ax, self.dynamics, plane)

        ax.plot(traj_part[:, idx_a], traj_part[:, idx_b],
                '-', color='cyan', linewidth=1.5, alpha=0.85, zorder=8)
        ax.plot(state[idx_a], state[idx_b], 'o', color='magenta',
                markersize=6, markeredgecolor='white', markeredgewidth=1.0,
                zorder=9)

        fixed_val = state[idx_fixed]
        ax.text(0.03, 0.96,
                f"{fixed_name}={fixed_val:+.2f}",
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='gray', alpha=0.85), zorder=10)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{plane.upper()} Slice", fontsize=10, pad=3)
        ax.grid(True, alpha=0.15, zorder=0)

    # ------------------------------------------------------------------ #
    #  Internal: draw the 3-D isometric panel
    # ------------------------------------------------------------------ #

    def _draw_3d_panel(self, ax, state, traj_so_far, X3, Y3, Z3, V3,
                       axis_limits):
        """Render the 3-D BRT + target + chaser on a 3-D axis."""
        self.brt_viz.render_matplotlib(
            X3, Y3, Z3, V3, state,
            ax=ax, trajectory=traj_so_far, title='',
            axis_limits=axis_limits)
        ax.view_init(elev=45, azim=45)
        ax.set_title('3-D Isometric', fontsize=10, pad=3)
        # Smaller tick labels for compact layout
        ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def generate(self, result: dict, output_path: str, *,
                 max_frames: int = 50, resolution: int = 80,
                 resolution_3d: int = 30, quiver_stride: int = 6,
                 fps: int = 5):
        """Generate the combined 4-panel GIF.

        Parameters
        ----------
        result        : simulation result dict.
        output_path   : path for the output .gif file.
        max_frames    : maximum number of animation frames.
        resolution    : grid resolution for 2-D slices.
        resolution_3d : grid resolution for 3-D BRT evaluation.
        quiver_stride : stride for quiver arrows in 2-D slices.
        fps           : frames per second.
        """
        traj = np.asarray(result['trajectory'])
        t_queries = get_t_queries(result)
        sim_times = np.asarray(result.get('sim_times',
                                          np.arange(len(traj)) * 0.1))
        key_idx = select_key_frames(len(traj), max_frames)

        planes = ['xy', 'xz', 'yz']

        # Pre-compute 2-D grid bounds per plane
        slice_bounds = {}
        for plane in planes:
            idx_a, idx_b, _ = _IDX_MAP[plane]
            slice_bounds[plane] = self._slice_extent(traj, idx_a, idx_b)

        # Pre-compute 3-D grid bounds (unified across trajectory)
        scene_lim = compute_scene_limits(traj, self.dynamics, padding=1.5)
        grid_bounds_3d = [
            (scene_lim['xlim'][0], scene_lim['xlim'][1]),
            (scene_lim['ylim'][0], scene_lim['ylim'][1]),
            (scene_lim['zlim'][0], scene_lim['zlim'][1]),
        ]

        # ---- Pre-evaluate all frames -------------------------------- #
        n_frames = len(key_idx)
        print(f"\n[CombinedGIF] Evaluating {n_frames} frames "
              f"(2-D: {resolution}², 3-D: {resolution_3d}³)...")

        frame_data = []  # list of dicts, one per frame
        for count, ki in enumerate(key_idx):
            state = traj[ki]
            t_q = float(t_queries[ki])
            fd = {
                'state': state,
                'sim_time': sim_times[ki],
                'traj_so_far': traj[:ki + 1],
            }
            # 2-D slices
            for plane in planes:
                A, B, V, dVda, dVdb = self.brt_viz.evaluate_brt_grid_2d(
                    state, t_q, plane=plane,
                    grid_bounds_2d=slice_bounds[plane],
                    resolution=resolution)
                fd[plane] = {'A': A, 'B': B, 'V': V,
                             'dVda': dVda, 'dVdb': dVdb}

            # 3-D grid
            X3, Y3, Z3, V3 = self.brt_viz.evaluate_brt_grid(
                state, t_q, grid_bounds_3d, resolution_3d)
            fd['3d'] = (X3, Y3, Z3, V3)

            frame_data.append(fd)
            if (count + 1) % 5 == 0 or count == n_frames - 1:
                print(f"  [{count + 1}/{n_frames}]")

        # ---- Build animation ----------------------------------------- #
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[1, 0])
        ax_3d = fig.add_subplot(gs[1, 1], projection='3d')

        vmax_abs = 1.0
        plane_axes = {'xy': ax_xy, 'xz': ax_xz, 'yz': ax_yz}

        # Shared colorbar (top-right of figure)
        norm = mcolors.Normalize(vmin=-vmax_abs, vmax=vmax_abs)
        sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_xy, ax_xz], shrink=0.70,
                            pad=0.02, location='right')
        cbar.set_label('V', fontsize=10)

        def _draw_frame(frame_idx):
            fd = frame_data[frame_idx]
            sim_t = fd['sim_time']
            state = fd['state']

            # 2-D panels
            for plane in planes:
                ax = plane_axes[plane]
                ax.clear()
                self._draw_slice_panel(
                    ax, {**fd[plane],
                         'state': state,
                         'traj_so_far': fd['traj_so_far']},
                    plane, slice_bounds[plane], quiver_stride, vmax_abs)

            # 3-D panel
            ax_3d.clear()
            X3, Y3, Z3, V3 = fd['3d']
            self._draw_3d_panel(ax_3d, state, fd['traj_so_far'],
                                X3, Y3, Z3, V3, scene_lim)

            # Shared time annotation
            fig.suptitle(
                f"t = {sim_t:.2f} s   |   "
                + _rotation_text(state).replace('\n', '   '),
                fontsize=11, y=0.98)

        # Initial draw
        _draw_frame(0)

        anim = FuncAnimation(fig, _draw_frame, frames=n_frames,
                             interval=1000 // fps, repeat=True)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"[CombinedGIF] Saved → {output_path}")
