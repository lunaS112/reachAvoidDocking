"""
Unified animation and static plotting for any combination of controllers.

Supports visualising one trajectory (with oriented chaser) or overlaying all
three controller types (BRAT, MPC, MPC+Terminal) for side-by-side comparison.

Usage:
    from utils.controllers.controller_animation import animate_trajectories, plot_trajectories_static

    results = {"BRAT": brat_result, "MPC": mpc_result, "MPC+Terminal": term_result}
    animate_trajectories(results, dynamics, "comparison.mp4")
    plot_trajectories_static(results, dynamics, "comparison.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation
from tqdm import tqdm
import os

# Default colour palette per controller type
CONTROLLER_COLORS = {
    'brat': '#1f77b4',           # blue
    'mpc': '#ff7f0e',            # orange
    'mpc_terminal': '#2ca02c',   # green
    'grid_based': '#d62728',     # red
    'rl': '#9467bd',             # purple
}

CONTROLLER_LABELS = {
    'brat': 'BRAT',
    'mpc': 'MPC',
    'mpc_terminal': 'MPC+Terminal',
    'grid_based': 'Grid-Based HJ',
    'rl': 'RL (DDQN)',
}


def _get_color(result):
    """Return the colour for a result dict based on controller_type."""
    ctype = result.get('controller_type', 'brat')
    return CONTROLLER_COLORS.get(ctype, '#1f77b4')


def _get_label(result):
    ctype = result.get('controller_type', 'brat')
    return CONTROLLER_LABELS.get(ctype, ctype)


def _draw_scene(ax, dynamics):
    """Draw the static docking scene: target spacecraft, docking post, goal."""
    w_t = dynamics.w_t
    h_t = dynamics.h_t
    post_hw_x = dynamics.post_hw_x
    post_length = dynamics.post_length
    eps_p = dynamics.eps_p

    # Target spacecraft body
    target_body = mpatches.Rectangle(
        (-w_t / 2, 0), w_t, h_t,
        facecolor='gray', edgecolor='black', alpha=0.8, zorder=20,
        label='Target Spacecraft')
    ax.add_patch(target_body)

    # Docking post
    post = mpatches.Rectangle(
        (-post_hw_x, -post_length), 2 * post_hw_x, post_length,
        facecolor='gray', edgecolor='black', alpha=0.8, zorder=20)
    ax.add_patch(post)

    # Docking target marker at post tip
    dock_marker = mpatches.Circle(
        (0, -post_length), radius=0.15,
        facecolor='red', edgecolor='black', alpha=1.0, zorder=22)
    ax.add_patch(dock_marker)

    # Goal set (band below inflated post tip)
    goal_y_min = dynamics.goal_y_min
    goal_y_max = dynamics.goal_y_max
    goal_set = mpatches.Rectangle(
        (-eps_p, goal_y_min), 2 * eps_p, goal_y_max - goal_y_min,
        facecolor='green', edgecolor='darkgreen', alpha=0.4, zorder=15,
        label='Goal Set')
    ax.add_patch(goal_set)


# ---------------------------------------------------------------------------
# Animated visualisation
# ---------------------------------------------------------------------------

def animate_trajectories(results_dict, dynamics, save_path,
                         skip_frames=5, dt=0.1, fps=None):
    """
    Create an mp4 animation showing 1-3 controller trajectories.

    Args:
        results_dict: dict mapping display-name -> sim_result dict
                      e.g. {"BRAT": brat_result, "MPC": mpc_result, ...}
        dynamics:     Docking6D dynamics instance (for geometry parameters).
        save_path:    Where to save the mp4.
        skip_frames:  Frame decimation factor.
        dt:           Timestep between trajectory entries.
        fps:          Frames per second in video (auto if None).

    Returns:
        (fig, ani) matplotlib objects.
    """
    multi = len(results_dict) > 1
    w_c = dynamics.w_c
    h_c = dynamics.h_c

    # Determine the maximum trajectory length across all controllers
    max_len = max(len(r['trajectory']) for r in results_dict.values())

    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    title = 'Controller Comparison' if multi else list(results_dict.keys())[0]
    ax.set_title(title, fontsize=14)

    _draw_scene(ax, dynamics)

    # --- Per-controller artists ---
    traces = {}
    dots = {}
    chasers = {}
    markers = {}

    for name, result in results_dict.items():
        color = _get_color(result)
        label = _get_label(result)

        # Trajectory trace
        line, = ax.plot([], [], '--', linewidth=2, color=color, zorder=25,
                        label=label)
        traces[name] = line

        if multi:
            # Position dot
            dot, = ax.plot([], [], 'o', markersize=8, color=color, zorder=30)
            dots[name] = dot
        else:
            # Full oriented chaser
            chaser = mpatches.Rectangle(
                (-w_c / 2, -h_c / 2), w_c, h_c,
                facecolor=color, edgecolor='black', alpha=0.9, zorder=30)
            ax.add_patch(chaser)
            chasers[name] = chaser

            marker = mpatches.RegularPolygon(
                (0, 0), 3, radius=w_c / 3,
                orientation=0, facecolor='red', edgecolor='black',
                alpha=0.9, zorder=31)
            ax.add_patch(marker)
            markers[name] = marker

    # Text overlay
    time_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    info_text = ax.text(
        0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    safety_filter_text = ax.text(
        0.02, 0.82, '', transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # Only show when single controller with safety filter; hide initially
    safety_filter_text.set_visible(False)

    ax.legend(loc='upper right', fontsize=10)

    def animate(k):
        sim_time = k * dt
        time_text.set_text(f'Time: {sim_time:.2f}s')
        info_lines = []

        for name, result in results_dict.items():
            traj = result['trajectory']
            n = len(traj)
            idx = min(k, n - 1)

            px = traj[:idx + 1, 0]
            py = traj[:idx + 1, 1]
            traces[name].set_data(px, py)

            cx, cy = traj[idx, 0], traj[idx, 1]
            theta = traj[idx, 4]

            if multi:
                dots[name].set_data([cx], [cy])
            else:
                t_chaser = (mtransforms.Affine2D()
                            .rotate(theta)
                            .translate(cx, cy) + ax.transData)
                chasers[name].set_transform(t_chaser)

                mx = cx + (w_c / 2) * np.cos(theta)
                my = cy + (w_c / 2) * np.sin(theta)
                t_marker = (mtransforms.Affine2D()
                            .rotate(theta + np.pi / 2)
                            .translate(mx, my) + ax.transData)
                markers[name].set_transform(t_marker)

            dist = np.sqrt(cx**2 + cy**2)
            label = _get_label(result)
            info_lines.append(f'{label}: d={dist:.2f}m')

            # Safety filter highlight (single-controller mode only)
            if not multi:
                sf_log = result.get('safety_filter_log', [])
                sf_mode = result.get('safety_filter_mode', 0)
                sf_active = False
                if sf_mode > 0 and idx < len(sf_log):
                    entry = sf_log[idx]
                    if sf_mode == 1:
                        sf_active = entry.get('filter_active', False)
                    else:
                        sf_active = entry.get('alpha_effective', 0) > 1e-6
                if sf_mode > 0:
                    safety_filter_text.set_visible(True)
                    if sf_active:
                        safety_filter_text.set_text('Safety filter: ACTIVE')
                        safety_filter_text.set_bbox(dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9))
                        chasers[name].set_facecolor('#ff6b6b')
                        chasers[name].set_edgecolor('red')
                    else:
                        safety_filter_text.set_text('Safety filter: inactive')
                        safety_filter_text.set_bbox(dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))
                        chasers[name].set_facecolor(_get_color(result))
                        chasers[name].set_edgecolor('black')
                else:
                    safety_filter_text.set_visible(False)

        info_text.set_text('\n'.join(info_lines))

    frames = range(0, max_len, skip_frames)

    ani = animation.FuncAnimation(
        fig, animate, frames, interval=dt * 1000 * skip_frames, blit=False)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)

    if fps is None:
        fps = min(30, int(1 / (dt * skip_frames)))
    writer = animation.FFMpegWriter(fps=fps)

    frame_list = list(frames)
    print(f"Rendering animation ({len(frame_list)} frames) -> {save_path}")
    try:
        with tqdm(total=len(frame_list), desc="Rendering") as pbar:
            ani.save(save_path, writer=writer,
                     progress_callback=lambda i, n: pbar.update(1))
    except Exception as e:
        print(f"Progress bar error ({e}), retrying without...")
        ani.save(save_path, writer=writer)

    print(f"Animation saved to {save_path}")
    plt.close(fig)
    return fig, ani


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------

def plot_trajectories_static(results_dict, dynamics, save_path=None):
    """
    Create a two-panel static figure:
      Left:  2D trajectories overlaid on the docking scene.
      Right: Distance-to-goal over time for each controller.

    Args:
        results_dict: dict mapping display-name -> sim_result dict.
        dynamics:     Docking6D dynamics instance.
        save_path:    Where to save the PNG (optional).

    Returns:
        fig: Matplotlib figure.
    """
    fig, (ax_traj, ax_dist) = plt.subplots(1, 2, figsize=(18, 8))

    # ---- Left: 2D trajectory plot ----
    ax_traj.set_xlim([-15, 15])
    ax_traj.set_ylim([-15, 15])
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, linestyle='--', alpha=0.5)
    ax_traj.set_xlabel('x (m)', fontsize=12)
    ax_traj.set_ylabel('y (m)', fontsize=12)
    ax_traj.set_title('Docking Trajectories', fontsize=14)

    _draw_scene(ax_traj, dynamics)

    for name, result in results_dict.items():
        traj = result['trajectory']
        color = _get_color(result)
        label = _get_label(result)

        ax_traj.plot(traj[:, 0], traj[:, 1], '-', linewidth=2, color=color,
                     label=label, zorder=25)
        ax_traj.plot(traj[0, 0], traj[0, 1], 'o', markersize=8, color=color,
                     zorder=35)
        ax_traj.plot(traj[-1, 0], traj[-1, 1], '*', markersize=12, color=color,
                     zorder=35)

    ax_traj.legend(loc='upper right', fontsize=10)

    # ---- Right: Distance to goal over time ----
    ax_dist.set_xlabel('Time (s)', fontsize=12)
    ax_dist.set_ylabel('Distance to Goal (m)', fontsize=12)
    ax_dist.set_title('Distance to Goal', fontsize=14)
    ax_dist.grid(True, linestyle='--', alpha=0.5)

    for name, result in results_dict.items():
        traj = result['trajectory']
        times = result['times']
        dist = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        color = _get_color(result)
        label = _get_label(result)
        ax_dist.plot(times, dist, '-', linewidth=2, color=color, label=label)

    ax_dist.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_simulation_data_multi(results_dict, save_path=None):
    """
    Multi-panel time-series plot (position, velocity, orientation, controls)
    for 1-3 controllers overlaid.

    Args:
        results_dict: dict mapping display-name -> sim_result dict.
        save_path:    Optional PNG path.

    Returns:
        fig: Matplotlib figure.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for name, result in results_dict.items():
        traj = result['trajectory']
        controls = result['controls']
        times = result['times']
        color = _get_color(result)
        label = _get_label(result)

        # Position
        axes[0, 0].plot(times, traj[:, 0], '-', color=color, alpha=0.8,
                        label=f'{label} px')
        axes[0, 0].plot(times, traj[:, 1], '--', color=color, alpha=0.6,
                        label=f'{label} py')

        # Velocity
        axes[0, 1].plot(times, traj[:, 2], '-', color=color, alpha=0.8,
                        label=f'{label} vx')
        axes[0, 1].plot(times, traj[:, 3], '--', color=color, alpha=0.6,
                        label=f'{label} vy')

        # Orientation
        axes[1, 0].plot(times, traj[:, 4], '-', color=color, alpha=0.8,
                        label=f'{label} theta')
        axes[1, 0].plot(times, traj[:, 5], '--', color=color, alpha=0.6,
                        label=f'{label} omega')

        # Controls
        n_ctrl = min(len(times), len(controls))
        axes[1, 1].plot(times[:n_ctrl], controls[:n_ctrl, 0], '-',
                        color=color, alpha=0.8, label=f'{label} u_x')
        axes[1, 1].plot(times[:n_ctrl], controls[:n_ctrl, 1], '--',
                        color=color, alpha=0.6, label=f'{label} u_y')
        if controls.shape[1] > 2:
            axes[1, 1].plot(times[:n_ctrl], controls[:n_ctrl, 2], ':',
                            color=color, alpha=0.6, label=f'{label} u_theta')

        # Distance to goal
        dist = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        axes[2, 0].plot(times, dist, '-', color=color, linewidth=2,
                        label=label)

        # Value / cost
        values = result.get('values', np.zeros(len(times)))
        axes[2, 1].plot(times, values, '-', color=color, linewidth=2,
                        label=label)

    titles = ['Position (m)', 'Velocity (m/s)', 'Orientation (rad)',
              'Controls', 'Distance to Goal (m)', 'Value / Cost']
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel('Time (s)')
    
    # Add goal reference lines
    # Position goal: (0, 0)
    axes[0, 0].axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='Goal')
    
    # Velocity goal: (0, 0)
    axes[0, 1].axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='Goal')
    
    # Orientation goal: theta = pi/2, omega = 0
    axes[1, 0].axhline(np.pi/2, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='Goal theta')
    axes[1, 0].axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    # Controls: no specific goal
    
    # Distance to goal: 0
    axes[2, 0].axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='Goal')
    
    # Value: BRAT boundary at 0
    axes[2, 1].axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='BRAT boundary')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig
