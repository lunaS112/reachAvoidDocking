"""
2-D BRT Slice Animated GIFs for Docking13D

Produces animated GIFs showing the value-function heat-map on three
orthogonal position slices (XY, XZ, YZ) as the chaser moves along
its trajectory.  Each frame shows:

  * The value-function heat-map (RdYlBu, saturated at ±1).
  * Black V=0 contour (BRT boundary).
  * Quiver arrows showing in-plane spatial gradient ∇V.
  * Target body + docking post geometry (2D projection).
  * Gold goal-region overlay.
  * Chaser position marker.
  * Large text annotation for the held-out coordinate value.
  * Small text showing chaser quaternion and angular velocity.

Usage (from run_controller_13d.py, via --viz_gifs flag):

    from utils.controllers.slice_gif_13d import SliceGIF13D
    gif = SliceGIF13D(brt_viz, dynamics)
    gif.generate(result, output_dir, max_frames=50, resolution=80)
"""
from __future__ import annotations

import os

from utils.controllers.anim_utils import select_key_frames, get_t_queries
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

from utils.brt_visualization_13d import BRTVisualizer13D


# ───────────────────────────────────────────────────────────────────── #
#  Target-geometry drawing helpers (2-D projections)
# ───────────────────────────────────────────────────────────────────── #

def _draw_target_on_slice(ax, dynamics, plane: str):
    """Draw target body + docking post + goal region on a 2-D axis.

    Parameters
    ----------
    ax       : matplotlib Axes
    dynamics : Docking13D instance
    plane    : one of ``'xy'``, ``'xz'``, ``'yz'``
    """
    d = dynamics
    hw = d.w_t / 2.0      # half-width x
    hd = d.d_t / 2.0      # half-depth z
    post_hw_x = d.post_hw_x
    post_hw_z = d.post_hw_z
    post_len = d.post_length

    if plane == 'xy':
        # Body [−hw, hw] × [0, h_t]
        ax.add_patch(plt.Rectangle((-hw, 0), d.w_t, d.h_t,
                                   lw=1.2, edgecolor='gray',
                                   facecolor='silver', alpha=0.30, zorder=3))
        # Post [−post_hw_x, post_hw_x] × [−post_len, 0]
        ax.add_patch(plt.Rectangle((-post_hw_x, -post_len),
                                   2 * post_hw_x, post_len,
                                   lw=1.0, edgecolor='green',
                                   facecolor='limegreen', alpha=0.25, zorder=4))
        # Goal band
        if hasattr(d, 'goal_y_min'):
            ax.add_patch(plt.Rectangle((-d.eps_p, d.goal_y_min),
                                       2 * d.eps_p,
                                       d.goal_y_max - d.goal_y_min,
                                       lw=1.0, edgecolor='gold',
                                       facecolor='gold', alpha=0.18,
                                       linestyle='--', zorder=5))

    elif plane == 'xz':
        # Body [−hw, hw] × [−hd, hd]
        ax.add_patch(plt.Rectangle((-hw, -hd), d.w_t, d.d_t,
                                   lw=1.2, edgecolor='gray',
                                   facecolor='silver', alpha=0.30, zorder=3))
        # Post cross-section [−post_hw_x, post_hw_x] × [−post_hw_z, post_hw_z]
        ax.add_patch(plt.Rectangle((-post_hw_x, -post_hw_z),
                                   2 * post_hw_x, 2 * post_hw_z,
                                   lw=1.0, edgecolor='green',
                                   facecolor='limegreen', alpha=0.25, zorder=4))
        # Goal circle (eps_p in x-z)
        if hasattr(d, 'eps_p'):
            ax.add_patch(plt.Circle((0, 0), d.eps_p, lw=1.0,
                                    edgecolor='gold', facecolor='gold',
                                    alpha=0.18, linestyle='--', zorder=5))

    elif plane == 'yz':
        # Body: y ∈ [0, h_t], z ∈ [−hd, hd]
        ax.add_patch(plt.Rectangle((0, -hd), d.h_t, d.d_t,
                                   lw=1.2, edgecolor='gray',
                                   facecolor='silver', alpha=0.30, zorder=3))
        # Post: y ∈ [−post_len, 0], z ∈ [−post_hw_z, post_hw_z]
        ax.add_patch(plt.Rectangle((-post_len, -post_hw_z),
                                   post_len, 2 * post_hw_z,
                                   lw=1.0, edgecolor='green',
                                   facecolor='limegreen', alpha=0.25, zorder=4))
        # Goal band: y ∈ [goal_y_min, goal_y_max], z ∈ [−eps_p, eps_p]
        if hasattr(d, 'goal_y_min'):
            ax.add_patch(plt.Rectangle(
                (d.goal_y_min, -d.eps_p),
                d.goal_y_max - d.goal_y_min, 2 * d.eps_p,
                lw=1.0, edgecolor='gold', facecolor='gold',
                alpha=0.18, linestyle='--', zorder=5))


# ───────────────────────────────────────────────────────────────────── #
#  Rotation state text helper
# ───────────────────────────────────────────────────────────────────── #

def _rotation_text(state: np.ndarray) -> str:
    """Return a compact string summarising quaternion & angular vel."""
    q = state[6:10]
    w = state[10:13]
    # Angle-axis magnitude (total rotation angle)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    angle_rad = 2.0 * np.arccos(np.clip(np.abs(q_norm[0]), 0, 1))
    angle_deg = np.degrees(angle_rad)
    omega_mag = np.linalg.norm(w)
    return (f"q=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}]  "
            f"θ={angle_deg:.1f}°\n"
            f"ω=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]  "
            f"|ω|={omega_mag:.3f} rad/s")


# ───────────────────────────────────────────────────────────────────── #
#  Main generator class
# ───────────────────────────────────────────────────────────────────── #

_AXIS_LABELS = {
    'xy': ('x (m)', 'y (m)', 'z'),
    'xz': ('x (m)', 'z (m)', 'y'),
    'yz': ('y (m)', 'z (m)', 'x'),
}

_IDX_MAP = {  # (in-plane axis a, in-plane axis b, held-out axis)
    'xy': (0, 1, 2),
    'xz': (0, 2, 1),
    'yz': (1, 2, 0),
}


class SliceGIF13D:
    """Animate 2-D BRT slices along the docking trajectory.

    Parameters
    ----------
    brt_viz  : BRTVisualizer13D
        Must own the trained model and dynamics.
    dynamics : Docking13D
        Used for target-geometry drawing & reach/avoid evaluation.
    """

    def __init__(self, brt_viz: BRTVisualizer13D, dynamics):
        self.brt_viz = brt_viz
        self.dynamics = dynamics

    # Key-frame / time-query helpers imported from anim_utils

    # ------------------------------------------------------------------ #
    #  Single-slice GIF generation
    # ------------------------------------------------------------------ #

    def _generate_single_gif(self, result: dict, plane: str,
                             output_path: str, max_frames: int = 50,
                             resolution: int = 80, quiver_stride: int = 6,
                             fps: int = 5):
        """Create one animated GIF for a given slice plane.

        Parameters
        ----------
        result      : dict with 'trajectory', 'sim_times', etc.
        plane       : 'xy', 'xz', or 'yz'
        output_path : file path for the .gif
        max_frames  : maximum animation frames
        resolution  : grid resolution for V evaluation
        quiver_stride : stride for quiver arrows (every Nth point)
        fps         : frames per second
        """
        traj = np.asarray(result['trajectory'])
        t_queries = get_t_queries(result)
        sim_times = np.asarray(result.get('sim_times',
                                          np.arange(len(traj)) * 0.1))
        key_idx = select_key_frames(len(traj), max_frames)

        idx_a, idx_b, idx_fixed = _IDX_MAP[plane]
        xlabel, ylabel, fixed_name = _AXIS_LABELS[plane]

        # Compute consistent grid bounds across all frames
        pos_traj = traj[:, :3]  # (N, 3)
        margin = 2.0
        clamp = 12.0
        a_min = np.clip(pos_traj[:, idx_a].min() - margin, -clamp, clamp)
        a_max = np.clip(pos_traj[:, idx_a].max() + margin, -clamp, clamp)
        b_min = np.clip(pos_traj[:, idx_b].min() - margin, -clamp, clamp)
        b_max = np.clip(pos_traj[:, idx_b].max() + margin, -clamp, clamp)
        # Ensure we include some context around target (origin)
        a_min = min(a_min, -5.0)
        a_max = max(a_max, 5.0)
        b_min = min(b_min, -5.0)
        b_max = max(b_max, 5.0)
        grid_bounds_2d = [(a_min, a_max), (b_min, b_max)]

        # Pre-evaluate all frames
        print(f"  [{plane.upper()}] Evaluating {len(key_idx)} frames "
              f"({resolution}×{resolution})...")
        frame_data = []
        for count, ki in enumerate(key_idx):
            state = traj[ki]
            t_q = float(t_queries[ki])
            A, B, V, dVda, dVdb = self.brt_viz.evaluate_brt_grid_2d(
                state, t_q, plane=plane,
                grid_bounds_2d=grid_bounds_2d, resolution=resolution)
            frame_data.append({
                'A': A, 'B': B, 'V': V,
                'dVda': dVda, 'dVdb': dVdb,
                'state': state,
                'sim_time': sim_times[ki],
                'traj_so_far': traj[:ki + 1],
            })
            if (count + 1) % 10 == 0 or count == len(key_idx) - 1:
                print(f"    [{count + 1}/{len(key_idx)}]")

        # ---- Build animation ----------------------------------------- #
        fig, ax = plt.subplots(figsize=(9, 8))
        vmax_abs = 1.0  # symmetric colorbar

        # Create colorbar once (attached to a fixed ScalarMappable)
        norm = mcolors.Normalize(vmin=-vmax_abs, vmax=vmax_abs)
        sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('V (value function)', fontsize=10)

        def _draw_frame(frame_idx):
            ax.clear()
            fd = frame_data[frame_idx]
            A, B, V = fd['A'], fd['B'], fd['V']
            dVda, dVdb = fd['dVda'], fd['dVdb']
            state = fd['state']
            sim_t = fd['sim_time']
            traj_part = fd['traj_so_far']

            # Heat-map
            extent = [grid_bounds_2d[0][0], grid_bounds_2d[0][1],
                      grid_bounds_2d[1][0], grid_bounds_2d[1][1]]
            # A is (res, res) with indexing='ij' → need transpose for imshow
            ax.imshow(V.T, origin='lower', extent=extent,
                      aspect='equal', cmap='RdYlBu',
                      vmin=-vmax_abs, vmax=vmax_abs, alpha=0.75,
                      zorder=1)

            # V=0 contour
            a_vals = np.linspace(extent[0], extent[1], V.shape[0])
            b_vals = np.linspace(extent[2], extent[3], V.shape[1])
            Ag, Bg = np.meshgrid(a_vals, b_vals)
            ax.contour(Ag, Bg, V.T, levels=[0.0], colors='black',
                       linewidths=1.5, linestyles='-', zorder=6)

            # Quiver arrows: -∇V (steepest descent, toward BRT interior)
            s = quiver_stride
            a_q = A[::s, ::s]
            b_q = B[::s, ::s]
            u_q = -dVda[::s, ::s]
            v_q = -dVdb[::s, ::s]
            mag = np.sqrt(u_q**2 + v_q**2) + 1e-12
            u_q = u_q / mag
            v_q = v_q / mag
            ax.quiver(a_q, b_q, u_q, v_q, color='k', alpha=0.35,
                      scale=25, width=0.003, headwidth=3.5, zorder=7)

            # Target geometry
            _draw_target_on_slice(ax, self.dynamics, plane)

            # Trajectory so far
            ax.plot(traj_part[:, idx_a], traj_part[:, idx_b],
                    '-', color='cyan', linewidth=1.8, alpha=0.85, zorder=8)
            # Current position marker
            ax.plot(state[idx_a], state[idx_b], 'o', color='magenta',
                    markersize=8, markeredgecolor='white', markeredgewidth=1.2,
                    zorder=9)

            # Held-out coordinate label
            fixed_val = state[idx_fixed]
            ax.text(0.02, 0.97,
                    f"{fixed_name} = {fixed_val:+.3f} m",
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec='gray', alpha=0.85),
                    zorder=10)

            # Rotation state text
            ax.text(0.98, 0.97, _rotation_text(state),
                    transform=ax.transAxes, fontsize=8, fontfamily='monospace',
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                              ec='gray', alpha=0.80),
                    zorder=10)

            # Time label
            ax.text(0.02, 0.02, f"t = {sim_t:.2f} s",
                    transform=ax.transAxes, fontsize=11,
                    va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='gray', alpha=0.80),
                    zorder=10)

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"BRT Value Slice ({plane.upper()})  —  V(x, t)",
                         fontsize=13)
            ax.grid(True, alpha=0.15, zorder=0)

        # Initial draw
        _draw_frame(0)
        fig.tight_layout()

        anim = FuncAnimation(fig, _draw_frame, frames=len(frame_data),
                             interval=1000 // fps, repeat=True)
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  [{plane.upper()}] Saved → {output_path}")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def generate(self, result: dict, output_dir: str,
                 max_frames: int = 50, resolution: int = 80,
                 fps: int = 5, planes: list[str] | None = None):
        """Generate animated GIFs for the requested slice planes.

        Parameters
        ----------
        result     : simulation result dict
        output_dir : directory to write gif files
        max_frames : max animation frames per GIF
        resolution : grid resolution per axis
        fps        : frames per second
        planes     : list of 'xy', 'xz', 'yz' (default: all three)
        """
        if planes is None:
            planes = ['xy', 'xz', 'yz']

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[SliceGIF] Generating {len(planes)} GIF(s) "
              f"(res={resolution}, {fps} fps, ≤{max_frames} frames)")

        for plane in planes:
            gif_path = os.path.join(output_dir,
                                    f'brt_slice_{plane}.gif')
            self._generate_single_gif(
                result, plane, gif_path,
                max_frames=max_frames, resolution=resolution, fps=fps)

        print("[SliceGIF] Done.")
