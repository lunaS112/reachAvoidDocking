"""
Trajectory-Only 3D Animation for Docking13D (no BRT)

Lightweight animation for controllers without a learned value function
(e.g. MPCController13D).  Renders:

  - Target spacecraft + dock hemisphere
  - Oriented chaser cube advancing along the trajectory
  - Trajectory line
  - Data panel (value / distance over time)

No BRT grid evaluation is performed, making these animations orders of
magnitude faster than the full BRT animation.

Usage:
    from utils.controllers.trajectory_only_animation_13d import TrajectoryAnimation13D

    anim = TrajectoryAnimation13D(dynamics)
    anim.generate_interactive_html(result, 'output.html')
    anim.generate_mp4(result, 'output.mp4')
"""

from __future__ import annotations

import numpy as np
import warnings

from utils.brt_visualization_13d import (
    add_target_plotly, add_reach_sphere_plotly, add_chaser_plotly,
    add_chaser_wireframe_plotly, add_chaser_axes_plotly,
    add_dock_rim_plotly, add_goal_ghost_plotly,
    add_trajectory_plotly,
    draw_target_mpl, draw_reach_sphere_mpl, draw_chaser_mpl,
    compute_scene_limits, state_range_limits,
    _inject_play_pause_js,
)

# Dynamic trace counting is used (like ControllerAnimation13D) because
# some Plotly trace types (e.g. go.Cone) serialize to HTML with extra
# internal sub-traces, making a fixed count unreliable.


class TrajectoryAnimation13D:
    """Trajectory-only animated output (no BRT grid evaluation).

    Parameters
    ----------
    dynamics : Docking13D
        Dynamics instance (provides target/chaser geometry).
    """

    def __init__(self, dynamics):
        self.dynamics = dynamics

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
    #  Interactive Plotly HTML
    # ------------------------------------------------------------------ #

    def generate_interactive_html(self, result: dict, output_path: str,
                                  max_frames: int = 50):
        """Write an interactive HTML file with a time slider.

        Uses visibility-toggling so that Mesh3d / Surface traces update
        correctly when the slider is moved.

        Parameters
        ----------
        result      : dict from ``controller.simulate_docking()``.
        output_path : file path for output HTML.
        max_frames  : maximum number of slider positions.
        """
        import plotly.graph_objects as go

        traj = result['trajectory']
        times = result['times']
        key_idx = self._select_key_frames(len(traj), max_frames)

        scene_limits = compute_scene_limits(
            traj, self.dynamics, padding=1.5)

        print(f"[Animation] Building {len(key_idx)} frames (trajectory only)...")
        all_traces = []
        traces_per_frame = []    # (start_idx, count) for each frame group

        for count, idx in enumerate(key_idx):
            state = traj[idx]
            fig_tmp = go.Figure()

            # Traces 0-2: Target (body + wireframe + dock)
            add_target_plotly(fig_tmp, self.dynamics)
            # Trace 3: Dock rim circle
            add_dock_rim_plotly(fig_tmp, self.dynamics)
            # Trace 4: Goal ghost wireframe
            add_goal_ghost_plotly(fig_tmp, self.dynamics)
            # Trace 5: Reach tolerance sphere
            add_reach_sphere_plotly(fig_tmp, self.dynamics)
            # Trace 6: Chaser cube (face-coloured)
            add_chaser_plotly(fig_tmp, self.dynamics, state)
            # Trace 7: Chaser wireframe edges
            add_chaser_wireframe_plotly(fig_tmp, self.dynamics, state)
            # Traces 8-9: Chaser orientation arrows
            add_chaser_axes_plotly(fig_tmp, self.dynamics, state)
            # Trace 10: Trajectory line
            add_trajectory_plotly(fig_tmp, traj[:idx + 1])

            start_idx = len(all_traces)
            frame_trace_list = list(fig_tmp.data)
            is_first = (count == 0)
            for tr in frame_trace_list:
                tr.visible = is_first
                if not is_first:
                    tr.showlegend = False
            all_traces.extend(frame_trace_list)
            traces_per_frame.append((start_idx, len(frame_trace_list)))

            if (count + 1) % 20 == 0 or count == len(key_idx) - 1:
                print(f"  Frame {count + 1}/{len(key_idx)} done.")

        # --- Build slider steps --------------------------------------- #
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
        # Use aspectmode='manual' with 1:1:1 ratio because the axis ranges
        # are already equalised by compute_scene_limits().
        # ('data' mode derives aspect from trace data extents which are
        #  sparse / lopsided when there is no dense BRT grid.)
        scene_kwargs = dict(
            xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=list(scene_limits['xlim'])),
            yaxis=dict(range=list(scene_limits['ylim'])),
            zaxis=dict(range=list(scene_limits['zlim'])),
        )
        fig = go.Figure(data=all_traces)
        fig.update_layout(
            title=f"t = {times[key_idx[0]]:.1f}s",
            scene=scene_kwargs,
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=0, r=0, t=40, b=0),
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
        """Write an MP4 animation with trajectory + data panel.

        Parameters
        ----------
        result      : dict from ``controller.simulate_docking()``.
        output_path : file path for output MP4.
        fps         : frames per second.
        max_frames  : maximum number of rendered frames.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        traj = result['trajectory']
        sim_times = result['times']
        values = result.get('values', None)
        key_idx = self._select_key_frames(len(traj), max_frames)

        all_dists = np.linalg.norm(traj[:, :3], axis=1)
        dist_hi = all_dists.max() * 1.1 + 0.5

        scene_limits = state_range_limits(self.dynamics)
        x_lo, x_hi = scene_limits['xlim']
        y_lo, y_hi = scene_limits['ylim']
        z_lo, z_hi = scene_limits['zlim']

        # --- Set up figure layout ------------------------------------- #
        fig = plt.figure(figsize=(18, 9))
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_data = fig.add_subplot(122)

        has_values = values is not None and len(values) > 0

        if has_values:
            ax_data2 = ax_data.twinx()
            val_pad = max(0.05, (values.max() - values.min()) * 0.10)
            val_lo, val_hi = values.min() - val_pad, values.max() + val_pad
            ax_data.set_ylim(val_lo, val_hi)
            ax_data.set_ylabel('Cost / Value', color='b')
            ax_data.axhline(0, color='gray', ls='--', lw=0.5)
            line_val, = ax_data.plot([], [], 'b-', lw=1.2, label='Value')
            ax_data2.set_ylim(0, dist_hi)
            ax_data2.set_ylabel('Distance (m)', color='r')
            line_dist, = ax_data2.plot([], [], 'r-', lw=1.0, alpha=0.7,
                                       label='Distance')
            ax_data.legend(loc='upper left', fontsize=8)
            ax_data2.legend(loc='upper right', fontsize=8)
        else:
            ax_data.set_ylim(0, dist_hi)
            ax_data.set_ylabel('Distance (m)', color='r')
            line_dist, = ax_data.plot([], [], 'r-', lw=1.2, label='Distance')
            ax_data.legend(loc='upper right', fontsize=8)

        ax_data.set_xlim(sim_times[0], sim_times[-1])
        ax_data.set_xlabel('Time (s)')
        title_text = ax_data.set_title('', fontsize=10)

        def update(frame_num):
            idx = key_idx[frame_num]
            state = traj[idx]

            ax_3d.clear()
            draw_target_mpl(ax_3d, self.dynamics)
            draw_reach_sphere_mpl(ax_3d, self.dynamics)
            draw_chaser_mpl(ax_3d, self.dynamics, state)

            if idx > 0:
                ax_3d.plot(traj[:idx + 1, 0], traj[:idx + 1, 1],
                           traj[:idx + 1, 2], 'r-', lw=2, label='Trajectory')

            ax_3d.set_xlabel('x (m)')
            ax_3d.set_ylabel('y (m)')
            ax_3d.set_zlabel('z (m)')
            ax_3d.set_title(f"t = {sim_times[idx]:.1f}s")
            ax_3d.set_xlim3d(x_lo, x_hi)
            ax_3d.set_ylim3d(y_lo, y_hi)
            ax_3d.set_zlim3d(z_lo, z_hi)
            ax_3d.set_box_aspect([1, 1, 1])
            ax_3d.legend(loc='upper left', fontsize=8)

            # Data panel
            t_slice = sim_times[:idx + 1]
            line_dist.set_data(t_slice, all_dists[:idx + 1])
            if has_values:
                line_val.set_data(t_slice, values[:idx + 1])
            title_text.set_text(f"t = {sim_times[idx]:.1f}s")

        print("[Animation] Rendering MP4 frames (trajectory only)...")
        anim = FuncAnimation(fig, update, frames=len(key_idx),
                             interval=1000 / fps)
        try:
            writer = FFMpegWriter(fps=fps,
                                  metadata=dict(title='Docking 13D — MPC'))
            anim.save(output_path, writer=writer)
            print(f"[Animation] MP4 written to {output_path}")
        except FileNotFoundError:
            warnings.warn("ffmpeg not found — cannot produce MP4. "
                          "Install ffmpeg or use generate_interactive_html().")
        finally:
            plt.close(fig)
