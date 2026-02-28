"""
3D Controller Animation for Docking13D

Generates time-varying visualisations that combine the docking trajectory with
the evolving BRT blob.  Supports two output modes:

  • **Interactive HTML** (Plotly) — time-slider for BRT evolution, camera
    rotation/zoom in the browser.
  • **MP4 animation** (Matplotlib) — fixed-camera frame-by-frame rendering
    with a data panel showing value, phase, and controls.

Usage:
    from utils.controllers.controller_animation_13d import ControllerAnimation13D
    from utils.brt_visualization_13d import BRTVisualizer13D

    brt_viz = BRTVisualizer13D(controller, backend='plotly')
    anim = ControllerAnimation13D(brt_viz)
    anim.generate_interactive_html(result, 'output.html')
    anim.generate_mp4(result, 'output.mp4')
"""

from __future__ import annotations

import numpy as np
import warnings

from utils.brt_visualization_13d import compute_scene_limits, state_range_limits, _inject_play_pause_js


class ControllerAnimation13D:
    """Combines simulation result with BRT blob for animated output.

    Parameters
    ----------
    brt_viz : BRTVisualizer13D
        Fully initialised visualiser (owns controller, model, dynamics).
    grid_resolution : int
        Resolution per axis for BRT grid evaluation during animation.
        Lower values speed up frame generation at the cost of detail.
    grid_bounds : list | None
        [(xmin,xmax), (ymin,ymax), (zmin,zmax)].  *None* → auto from
        state_test_range.
    """

    def __init__(self, brt_viz, grid_resolution: int = 35,
                 grid_bounds: list | None = None):
        self.brt_viz = brt_viz
        self.resolution = grid_resolution
        self.grid_bounds = grid_bounds

    # ------------------------------------------------------------------ #
    #  Key-frame selection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_key_frames(n_total: int, max_frames: int = 60) -> np.ndarray:
        """Return at most *max_frames* evenly-spaced frame indices."""
        if n_total <= max_frames:
            return np.arange(n_total)
        return np.linspace(0, n_total - 1, max_frames, dtype=int)

    # ------------------------------------------------------------------ #
    #  Time-query logic (Phase-aware)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_t_queries(result: dict) -> np.ndarray:
        """Derive the time-query array used for each simulation step.

        For controllers that track phase and t_remaining (BRT, MPC-terminal):
            Phase 1 → tMax,  Phase 2 → t_remaining.
        For pure-MPC: constant tMax (only used for the blob, not cost).
        """
        if 't_remaining' in result and 'phases' in result:
            return np.asarray(result['t_remaining'], dtype=np.float64)
        # Fallback: use tMax from controller metadata
        tMax = result.get('tMax', 14.0)
        return np.full(len(result['trajectory']), tMax)

    # ------------------------------------------------------------------ #
    #  Interactive Plotly HTML
    # ------------------------------------------------------------------ #

    @staticmethod
    def _auto_center_bounds(pos, half_range=8.0, clamp=15.0):
        """Compute grid bounds centred on *pos* (3,), clamped to ±clamp."""
        lo = np.clip(pos[:3] - half_range, -clamp, clamp)
        hi = np.clip(pos[:3] + half_range, -clamp, clamp)
        return [(lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])]

    def generate_interactive_html(self, result: dict, output_path: str,
                                  max_frames: int = 50):
        """Write an interactive HTML file with a time slider.

        Uses visibility-toggling (not go.Frame animation) so that all 3-D
        trace types (Isosurface, Volume, Mesh3d, Surface) update correctly
        when the slider is moved.

        Parameters
        ----------
        result : dict from ``controller.simulate_docking()``.
        output_path : file path for output HTML.
        max_frames : maximum number of slider positions (keeps file size
                     manageable).
        """
        import plotly.graph_objects as go

        traj = result['trajectory']
        t_queries = self._get_t_queries(result)
        times = result['times']
        key_idx = self._select_key_frames(len(traj), max_frames)

        # --- Pre-compute unified grid bounds + scene axis limits ------ #
        if self.grid_bounds is not None:
            unified_bounds = self.grid_bounds
        else:
            positions = traj[key_idx, :3]
            half, clamp = 8.0, 15.0
            unified_lo = np.clip(positions.min(axis=0) - half, -clamp, clamp)
            unified_hi = np.clip(positions.max(axis=0) + half, -clamp, clamp)
            unified_bounds = [(unified_lo[0], unified_hi[0]),
                              (unified_lo[1], unified_hi[1]),
                              (unified_lo[2], unified_hi[2])]

        scene_limits = compute_scene_limits(
            traj, self.brt_viz.dynamics, padding=1.5)

        # --- Pre-compute all traces for every key frame --------------- #
        print(f"[Animation] Evaluating BRT grids for {len(key_idx)} frames "
              f"(resolution={self.resolution})...")
        all_traces = []          # flat list of every trace across all frames
        traces_per_frame = []    # (start_idx, count) for each frame group
        first_layout = None

        for count, idx in enumerate(key_idx):
            state = traj[idx]
            t_q = float(t_queries[idx])
            X, Y, Z, V = self.brt_viz.evaluate_brt_grid(
                state, t_q, unified_bounds, self.resolution)

            frame_fig = self.brt_viz.render_plotly(
                X, Y, Z, V, state,
                trajectory=traj[:idx + 1],
                title='',
                axis_range=scene_limits,
            )

            start_idx = len(all_traces)
            frame_trace_list = list(frame_fig.data)
            is_first = (count == 0)
            for tr in frame_trace_list:
                tr.visible = is_first
                # Suppress per-trace legend entries for non-first frames
                if not is_first:
                    tr.showlegend = False
            all_traces.extend(frame_trace_list)
            traces_per_frame.append((start_idx, len(frame_trace_list)))

            if first_layout is None:
                first_layout = frame_fig.layout

            if (count + 1) % 10 == 0 or count == len(key_idx) - 1:
                print(f"  Frame {count + 1}/{len(key_idx)} done.")

        # --- Build slider steps: each toggles one frame group --------- #
        total_traces = len(all_traces)
        slider_steps = []
        for fi, idx in enumerate(key_idx):
            vis = [False] * total_traces
            start, cnt = traces_per_frame[fi]
            for j in range(start, start + cnt):
                vis[j] = True
            slider_steps.append(dict(
                args=[{"visible": vis},
                      {"title": f"t = {times[idx]:.1f}s"}],
                label=f"t={times[idx]:.1f}s",
                method="update",
            ))

        # --- Assemble figure ------------------------------------------ #
        fig = go.Figure(data=all_traces)
        fig.update_layout(first_layout)
        # Force fixed axis range across all slider positions
        fig.update_layout(
            title=f"t = {times[key_idx[0]]:.1f}s",
            scene=dict(
                aspectmode='data',
                xaxis=dict(range=list(scene_limits['xlim'])),
                yaxis=dict(range=list(scene_limits['ylim'])),
                zaxis=dict(range=list(scene_limits['zlim'])),
            ),
            sliders=[dict(
                active=0,
                steps=slider_steps,
                currentvalue=dict(prefix='Time: '),
                pad=dict(t=50),
            )],
        )

        fig.write_html(output_path)

        # Inject Play / Pause JS that cycles through slider steps
        _inject_play_pause_js(output_path, interval_ms=500)

        print(f"[Animation] HTML written to {output_path}")

    # ------------------------------------------------------------------ #
    #  MP4 animation (Matplotlib)
    # ------------------------------------------------------------------ #

    def generate_mp4(self, result: dict, output_path: str,
                     fps: int = 10, max_frames: int = 120):
        """Write an MP4 animation with BRT blob + data panel.

        Parameters
        ----------
        result     : dict from ``controller.simulate_docking()``.
        output_path: file path for output MP4.
        fps        : frames per second.
        max_frames : maximum number of rendered frames.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        traj = result['trajectory']
        controls = result['controls']
        values = result['values']
        sim_times = result['times']
        t_queries = self._get_t_queries(result)
        key_idx = self._select_key_frames(len(traj), max_frames)

        # Pre-compute unified grid bounds for MP4
        if self.grid_bounds is not None:
            unified_bounds = self.grid_bounds
        else:
            positions = traj[key_idx, :3]
            half, clamp = 8.0, 15.0
            unified_lo = np.clip(positions.min(axis=0) - half, -clamp, clamp)
            unified_hi = np.clip(positions.max(axis=0) + half, -clamp, clamp)
            unified_bounds = [(unified_lo[0], unified_hi[0]),
                              (unified_lo[1], unified_hi[1]),
                              (unified_lo[2], unified_hi[2])]

        print(f"[Animation] Pre-computing {len(key_idx)} BRT grids for MP4...")
        grids = []
        for count, idx in enumerate(key_idx):
            X, Y, Z, V = self.brt_viz.evaluate_brt_grid(
                traj[idx], float(t_queries[idx]),
                unified_bounds, self.resolution)
            grids.append((X, Y, Z, V))
            if (count + 1) % 20 == 0 or count == len(key_idx) - 1:
                print(f"  Grid {count + 1}/{len(key_idx)} done.")

        # --- Pre-compute full-run data for fixed axis limits ---------- #
        all_dists = np.linalg.norm(traj[:, :3], axis=1)
        val_pad = max(0.05, (values.max() - values.min()) * 0.10)
        val_lo, val_hi = values.min() - val_pad, values.max() + val_pad
        dist_hi = all_dists.max() * 1.1 + 0.5

        # 3D axis limits: use dynamics state_range_ for full training domain
        scene_limits = state_range_limits(self.brt_viz.dynamics)
        x_lo, x_hi = scene_limits['xlim']
        y_lo, y_hi = scene_limits['ylim']
        z_lo, z_hi = scene_limits['zlim']

        # --- Set up figure layout ------------------------------------- #
        fig = plt.figure(figsize=(18, 9))
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_data = fig.add_subplot(122)
        ax_data2 = ax_data.twinx()  # create ONCE

        # Persistent data-panel elements
        ax_data.set_xlim(sim_times[0], sim_times[-1])
        ax_data.set_ylim(val_lo, val_hi)
        ax_data.set_xlabel('Time (s)')
        ax_data.set_ylabel('Value', color='b')
        ax_data.axhline(0, color='gray', ls='--', lw=0.5)

        ax_data2.set_ylim(0, dist_hi)
        ax_data2.set_ylabel('Distance (m)', color='r')

        line_val, = ax_data.plot([], [], 'b-', lw=1.2, label='V(x,t)')
        line_dist, = ax_data2.plot([], [], 'r-', lw=1.0, alpha=0.7,
                                   label='Distance')
        ax_data.legend(loc='upper left', fontsize=8)
        ax_data2.legend(loc='upper right', fontsize=8)
        title_text = ax_data.set_title('', fontsize=10)

        def update(frame_num):
            idx = key_idx[frame_num]
            state = traj[idx]
            X, Y, Z, V = grids[frame_num]

            ax_3d.clear()
            self.brt_viz.render_matplotlib(
                X, Y, Z, V, state,
                trajectory=traj[:idx + 1],
                title=f"t = {sim_times[idx]:.1f}s",
                ax=ax_3d,
                axis_limits=dict(xlim=(x_lo, x_hi),
                                 ylim=(y_lo, y_hi),
                                 zlim=(z_lo, z_hi)),
            )

            # Data panel — update line data, no clearing
            t_slice = sim_times[:idx + 1]
            line_val.set_data(t_slice, values[:idx + 1])
            line_dist.set_data(t_slice, all_dists[:idx + 1])

            # Phase indicator
            if 'phases' in result:
                phase = result['phases'][idx]
                title_text.set_text(f"Phase {phase}  |  "
                                    f"t_q = {t_queries[idx]:.2f}s")
            else:
                title_text.set_text(f"t = {sim_times[idx]:.1f}s")

        print("[Animation] Rendering MP4 frames...")
        anim = FuncAnimation(fig, update, frames=len(key_idx),
                             interval=1000 / fps)
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(title='Docking 13D'))
            anim.save(output_path, writer=writer)
            print(f"[Animation] MP4 written to {output_path}")
        except FileNotFoundError:
            warnings.warn("ffmpeg not found — cannot produce MP4. "
                          "Install ffmpeg or use generate_interactive_html().")
        finally:
            plt.close(fig)
