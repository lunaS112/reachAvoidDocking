"""
Static visualisation plots for Docking13D simulation results.

Three public functions, each producing one PNG:

  • ``plot_trajectory_13d``   → ``trajectory.png``
      2×2 grid: 3D trajectory, XY projection, XZ projection, distance-to-dock.

  • ``plot_states_13d``       → ``simulation_states.png``
      5×2 grid: position, velocity, quaternion, angular velocity,
      distance, value function, phase, summary text, BRAT time.

  • ``plot_controls_13d``     → ``simulation_controls.png``
      2×2 grid: forces, torques, cumulative effort, force/torque magnitude.

Usage::

    from utils.controllers.static_plots_13d import (
        plot_trajectory_13d, plot_states_13d, plot_controls_13d,
    )
    plot_trajectory_13d(result, dynamics, 'outputs/trajectory.png')
    plot_states_13d(result, dynamics, 'outputs/simulation_states.png')
    plot_controls_13d(result, dynamics, 'outputs/simulation_controls.png')
"""

from __future__ import annotations

import os
import numpy as np


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _safe_mkdir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _q_goal_np(dynamics):
    """Return (4,) numpy array of the goal quaternion."""
    qg = dynamics.q_goal
    if hasattr(qg, 'cpu'):
        qg = qg.detach().cpu().numpy()
    return np.asarray(qg, dtype=np.float64)


def _draw_target_2d(ax, dynamics, plane='xy'):
    """Draw 2-D projection of the target body + docking post + goal band.

    *plane* is one of ``'xy'``, ``'xz'``.
    """
    import matplotlib.pyplot as plt

    d = dynamics
    hw = d.w_t / 2
    hd = d.d_t / 2
    post_hw_x = d.post_hw_x
    post_hw_z = d.post_hw_z
    post_len = d.post_length

    if plane == 'xy':
        # Target body [−hw, hw] × [0, h_t]
        rect = plt.Rectangle((-hw, 0), d.w_t, d.h_t,
                              linewidth=1.5, edgecolor='gray',
                              facecolor='silver', alpha=0.35, zorder=5)
        ax.add_patch(rect)
        # Docking post [−post_hw_x, post_hw_x] × [−post_len, 0]
        post = plt.Rectangle((-post_hw_x, -post_len),
                              2 * post_hw_x, post_len,
                              linewidth=1.2, edgecolor='green',
                              facecolor='limegreen', alpha=0.30, zorder=6)
        ax.add_patch(post)
        # Goal band
        if hasattr(d, 'goal_y_min'):
            goal_band = plt.Rectangle((-d.eps_p, d.goal_y_min),
                                       2 * d.eps_p,
                                       d.goal_y_max - d.goal_y_min,
                                       linewidth=1.2, edgecolor='gold',
                                       facecolor='gold', alpha=0.20,
                                       linestyle='--', zorder=7)
            ax.add_patch(goal_band)
    elif plane == 'xz':
        # Target body [−hw, hw] × [−hd, hd]
        rect = plt.Rectangle((-hw, -hd), d.w_t, d.d_t,
                              linewidth=1.5, edgecolor='gray',
                              facecolor='silver', alpha=0.35, zorder=5)
        ax.add_patch(rect)
        # Post cross-section [−post_hw_x, post_hw_x] × [−post_hw_z, post_hw_z]
        post = plt.Rectangle((-post_hw_x, -post_hw_z),
                              2 * post_hw_x, 2 * post_hw_z,
                              linewidth=1.2, edgecolor='green',
                              facecolor='limegreen', alpha=0.30, zorder=6)
        ax.add_patch(post)
        # Goal circle (eps_p radius in x-z)
        if hasattr(d, 'eps_p'):
            circ = plt.Circle((0, 0), d.eps_p, linewidth=1.2,
                               edgecolor='gold', facecolor='gold',
                               alpha=0.20, linestyle='--', zorder=7)
            ax.add_patch(circ)


def _draw_target_3d(ax, dynamics):
    """Draw 3-D target body + docking post + goal band box."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = dynamics
    hw, hd = d.w_t / 2, d.d_t / 2

    # --- Target body box centred at (0, h_t/2, 0) ---
    from utils.brt_visualization_13d import _oriented_box_mesh
    _, faces = _oriented_box_mesh(
        [0, d.h_t / 2, 0], (hw, d.h_t / 2, hd))
    mesh = Poly3DCollection(faces, alpha=0.35, linewidths=0.5)
    mesh.set_facecolor('silver')
    mesh.set_edgecolor('gray')
    ax.add_collection3d(mesh)

    # --- Docking post box centred at (0, -post_length/2, 0) ---
    post_hw_x = d.post_hw_x
    post_hw_z = d.post_hw_z
    post_len = d.post_length
    _, pfaces = _oriented_box_mesh(
        [0, -post_len / 2, 0], (post_hw_x, post_len / 2, post_hw_z))
    pmesh = Poly3DCollection(pfaces, alpha=0.30, linewidths=0.5)
    pmesh.set_facecolor('limegreen')
    pmesh.set_edgecolor('green')
    ax.add_collection3d(pmesh)

    # --- Goal band box ---
    if hasattr(d, 'goal_y_min'):
        goal_cy = (d.goal_y_min + d.goal_y_max) / 2
        goal_hy = (d.goal_y_max - d.goal_y_min) / 2
        _, gfaces = _oriented_box_mesh(
            [0, goal_cy, 0], (d.eps_p, goal_hy, d.eps_p))
        gmesh = Poly3DCollection(gfaces, alpha=0.15, linewidths=0.8)
        gmesh.set_facecolor('gold')
        gmesh.set_edgecolor('goldenrod')
        ax.add_collection3d(gmesh)


# --------------------------------------------------------------------------- #
#  1. Trajectory plot
# --------------------------------------------------------------------------- #

def plot_trajectory_13d(result, dynamics, save_path=None):
    """Create a 2×2 trajectory figure and optionally save it.

    Panels:
      (0,0) 3D trajectory with target geometry.
      (0,1) XY projection.
      (1,0) XZ projection.
      (1,1) Distance-to-dock over time.

    Returns ``matplotlib.figure.Figure``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    traj = result['trajectory']
    times = result['times']
    # Distance to goal center (matches reach_fn / goal band definition)
    goal_center = np.array([0.0, dynamics.goal_y_center, 0.0])
    dists = np.linalg.norm(traj[:, :3] - goal_center, axis=1)
    d = dynamics

    pad = 3.0
    x_lo = min(traj[:, 0].min(), -d.w_t / 2) - pad
    x_hi = max(traj[:, 0].max(),  d.w_t / 2) + pad
    y_lo = min(traj[:, 1].min(), 0.0) - pad
    y_hi = max(traj[:, 1].max(), d.h_t) + pad
    z_lo = min(traj[:, 2].min(), -d.d_t / 2) - pad
    z_hi = max(traj[:, 2].max(),  d.d_t / 2) + pad

    fig = plt.figure(figsize=(16, 14))

    # ---- (0,0) 3D trajectory ----------------------------------------- #
    ax3d = fig.add_subplot(221, projection='3d')
    _draw_target_3d(ax3d, dynamics)
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', lw=2,
              label='Trajectory')
    ax3d.plot([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], 'go',
              markersize=8, label='Start', zorder=35)
    ax3d.plot([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], 'r*',
              markersize=12, label='End', zorder=35)
    ax3d.set_xlim3d(x_lo, x_hi)
    ax3d.set_ylim3d(y_lo, y_hi)
    ax3d.set_zlim3d(z_lo, z_hi)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_xlabel('x (m)')
    ax3d.set_ylabel('y (m)')
    ax3d.set_zlabel('z (m)')
    ax3d.set_title('3D Trajectory', fontsize=13)
    ax3d.legend(fontsize=8, loc='upper left')

    # ---- (0,1) XY projection ---------------------------------------- #
    ax_xy = fig.add_subplot(222)
    _draw_target_2d(ax_xy, dynamics, plane='xy')
    ax_xy.plot(traj[:, 0], traj[:, 1], 'r-', lw=2, label='Trajectory',
               zorder=25)
    ax_xy.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start',
               zorder=35)
    ax_xy.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=12, label='End',
               zorder=35)
    ax_xy.set_xlim(x_lo, x_hi)
    ax_xy.set_ylim(y_lo, y_hi)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('x (m)')
    ax_xy.set_ylabel('y (m)')
    ax_xy.set_title('XY Projection', fontsize=13)
    ax_xy.grid(True, ls='--', alpha=0.5)
    ax_xy.legend(fontsize=8)

    # ---- (1,0) XZ projection ---------------------------------------- #
    ax_xz = fig.add_subplot(223)
    _draw_target_2d(ax_xz, dynamics, plane='xz')
    ax_xz.plot(traj[:, 0], traj[:, 2], 'r-', lw=2, label='Trajectory',
               zorder=25)
    ax_xz.plot(traj[0, 0], traj[0, 2], 'go', markersize=8, label='Start',
               zorder=35)
    ax_xz.plot(traj[-1, 0], traj[-1, 2], 'r*', markersize=12, label='End',
               zorder=35)
    ax_xz.set_xlim(x_lo, x_hi)
    ax_xz.set_ylim(z_lo, z_hi)
    ax_xz.set_aspect('equal')
    ax_xz.set_xlabel('x (m)')
    ax_xz.set_ylabel('z (m)')
    ax_xz.set_title('XZ Projection', fontsize=13)
    ax_xz.grid(True, ls='--', alpha=0.5)
    ax_xz.legend(fontsize=8)

    # ---- (1,1) Distance to dock -------------------------------------- #
    ax_dist = fig.add_subplot(224)
    ax_dist.plot(times, dists, 'b-', lw=2, label='Distance')
    ax_dist.axhline(d.eps_p, color='lime', ls='--', lw=1.2,
                     label=f'eps_p = {d.eps_p}m')
    if result.get('docked', False):
        dock_idx = np.argmax(dists <= d.eps_p)
        if dock_idx > 0:
            ax_dist.axvline(times[dock_idx], color='green', ls='--', lw=1,
                            alpha=0.6, label='Docked')
    ax_dist.set_xlabel('Time (s)')
    ax_dist.set_ylabel('Distance (m)')
    ax_dist.set_title('Distance to Target', fontsize=13)
    ax_dist.grid(True, ls='--', alpha=0.5)
    ax_dist.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        _safe_mkdir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Static] Trajectory figure saved to {save_path}")

    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
#  2. States plot
# --------------------------------------------------------------------------- #

def plot_states_13d(result, dynamics, save_path=None):
    """Create a 5×2 state time-series figure.

    Panels:
      (0,0) Position (x,y,z)        (0,1) Velocity (vx,vy,vz)
      (1,0) Quaternion (q0-q3)      (1,1) Angular velocity (ωx,ωy,ωz)
      (2,0) Distance to target      (2,1) Value function
      (3,0) Phase                   (3,1) Summary text
      (4,0) BRAT time (t*)          (4,1) (empty)

    Returns ``matplotlib.figure.Figure``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    traj = result['trajectory']
    times = result['times']
    values = result.get('values', np.zeros(len(times)))
    d = dynamics
    q_goal = _q_goal_np(dynamics)

    fig, axes = plt.subplots(5, 2, figsize=(16, 22))

    # ---- (0,0) Position ---------------------------------------------- #
    ax = axes[0, 0]
    pos_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (lbl, c) in enumerate(zip(['x', 'y', 'z'], pos_colors)):
        ax.plot(times, traj[:, i], color=c, lw=1.5, label=lbl)
    # Goal bands: x and z centered at 0 ± eps_p, y is [goal_y_min, goal_y_max]
    ax.fill_between(times, -d.eps_p, d.eps_p,
                    color='tab:blue', alpha=0.10, label=f'x,z goal ±{d.eps_p}m')
    ax.fill_between(times, -d.eps_p, d.eps_p,
                    color='tab:green', alpha=0.07)
    ax.fill_between(times, d.goal_y_min, d.goal_y_max,
                    color='tab:orange', alpha=0.12,
                    label=f'y goal [{d.goal_y_min:.2f}, {d.goal_y_max:.2f}]')
    # Green highlight where all position states are simultaneously in-tolerance
    pos_in_tol = (
        (np.abs(traj[:, 0]) <= d.eps_p) &
        (np.abs(traj[:, 2]) <= d.eps_p) &
        (traj[:, 1] >= d.goal_y_min) & (traj[:, 1] <= d.goal_y_max)
    )
    ax.set_title('Position (m)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ymin, ymax = ax.get_ylim()
    ax.fill_between(times, ymin, ymax, where=pos_in_tol,
                    color='lime', alpha=0.15, zorder=0, label='All in tol')
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=7)

    # ---- (0,1) Velocity ---------------------------------------------- #
    ax = axes[0, 1]
    vel_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (lbl, c) in enumerate(zip(['vx', 'vy', 'vz'], vel_colors)):
        ax.plot(times, traj[:, 3 + i], color=c, lw=1.5, label=lbl)
    # Goal bands: vx,vz at 0 ± eps_v_lateral; vy uses positive axial band
    ax.fill_between(times, -d.eps_v_lateral, d.eps_v_lateral,
                    color='tab:blue', alpha=0.10,
                    label=f'vx,vz goal ±{d.eps_v_lateral}')
    ax.fill_between(times, -d.eps_v_lateral, d.eps_v_lateral,
                    color='tab:green', alpha=0.07)
    ax.fill_between(times, d.eps_v_axial_lo, d.eps_v_axial_hi,
                    color='tab:orange', alpha=0.12,
                    label=f'vy goal [{d.eps_v_axial_lo}, {d.eps_v_axial_hi}]')
    # Green highlight where all velocity states are simultaneously in-tolerance
    vel_in_tol = (
        (np.abs(traj[:, 3]) <= d.eps_v_lateral) &
        (np.abs(traj[:, 5]) <= d.eps_v_lateral) &
        (traj[:, 4] >= d.eps_v_axial_lo) & (traj[:, 4] <= d.eps_v_axial_hi)
    )
    ax.set_title('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ymin, ymax = ax.get_ylim()
    ax.fill_between(times, ymin, ymax, where=vel_in_tol,
                    color='lime', alpha=0.15, zorder=0, label='All in tol')
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=7)

    # ---- (1,0) Quaternion -------------------------------------------- #
    ax = axes[1, 0]
    q_labels = ['q\u2080', 'q\u2081', 'q\u2082', 'q\u2083']
    q_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(4):
        ax.plot(times, traj[:, 6 + i], color=q_colors[i], lw=1.5,
                label=q_labels[i])
        ax.fill_between(times, q_goal[i] - d.eps_q, q_goal[i] + d.eps_q,
                        color=q_colors[i], alpha=0.08)
    # Green highlight where all quaternion components are simultaneously in-tolerance
    quat_in_tol = np.ones(len(times), dtype=bool)
    for i in range(4):
        quat_in_tol &= (np.abs(traj[:, 6 + i] - q_goal[i]) <= d.eps_q)
    ax.set_title(f'Quaternion (bands = q_goal ±{d.eps_q:.3f} rad)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ymin, ymax = ax.get_ylim()
    ax.fill_between(times, ymin, ymax, where=quat_in_tol,
                    color='lime', alpha=0.15, zorder=0, label='All in tol')
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=8, ncol=2)

    # ---- (1,1) Angular velocity -------------------------------------- #
    ax = axes[1, 1]
    omega_colors = ['tab:blue', 'tab:orange', 'tab:green']
    omega_labels = ['\u03c9x', '\u03c9y', '\u03c9z']
    for i, (lbl, c) in enumerate(zip(omega_labels, omega_colors)):
        ax.plot(times, traj[:, 10 + i], color=c, lw=1.5, label=lbl)
    # Goal bands: wx (roll) at 0 ± eps_omega_roll; wy,wz at 0 ± eps_omega_pitchyaw
    ax.fill_between(times, -d.eps_omega_roll, d.eps_omega_roll,
                    color='tab:blue', alpha=0.10,
                    label=f'\u03c9x goal ±{d.eps_omega_roll:.4f}')
    ax.fill_between(times, -d.eps_omega_pitchyaw, d.eps_omega_pitchyaw,
                    color='tab:orange', alpha=0.10,
                    label=f'\u03c9y,\u03c9z goal ±{d.eps_omega_pitchyaw:.4f}')
    ax.fill_between(times, -d.eps_omega_pitchyaw, d.eps_omega_pitchyaw,
                    color='tab:green', alpha=0.07)
    # Green highlight where all angular velocity states are simultaneously in-tolerance
    omg_in_tol = (
        (np.abs(traj[:, 10]) <= d.eps_omega_roll) &
        (np.abs(traj[:, 11]) <= d.eps_omega_pitchyaw) &
        (np.abs(traj[:, 12]) <= d.eps_omega_pitchyaw)
    )
    ax.set_title('Angular velocity (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ymin, ymax = ax.get_ylim()
    ax.fill_between(times, ymin, ymax, where=omg_in_tol,
                    color='lime', alpha=0.15, zorder=0, label='All in tol')
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=7)

    # ---- (2,0) Distance ---------------------------------------------- #
    ax = axes[2, 0]
    goal_center = np.array([0.0, d.goal_y_center, 0.0])
    dists = np.linalg.norm(traj[:, :3] - goal_center, axis=1)
    ax.plot(times, dists, 'b-', lw=2, label='Distance')
    ax.axhline(d.eps_p, color='lime', ls='--', lw=1.2,
               label=f'eps_p = {d.eps_p}m')
    ax.set_title('Distance to Target (m)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (2,1) Value function ---------------------------------------- #
    ax = axes[2, 1]
    ax.plot(times, values, 'b-', lw=1.5, label='V(x,t)')
    ax.axhline(0, color='k', ls='--', lw=1.2, alpha=0.5, label='BRT boundary')
    # Background shading: green where V<0 (inside BRT)
    v_arr = np.asarray(values)
    if len(v_arr) == len(times):
        ax.fill_between(times, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else v_arr.min() - 0.1,
                         v_arr, where=v_arr < 0,
                         color='green', alpha=0.08, label='Inside BRT')
    ax.set_title('Value Function')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (3,0) Phase ------------------------------------------------- #
    ax = axes[3, 0]
    if 'phases' in result:
        phases = np.asarray(result['phases'])
        ax.step(times[:len(phases)], phases, where='post', color='purple',
                lw=2.0, label='Phase')
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Phase 1\n(Reach)', 'Phase 2\n(Avoid)'])
    else:
        ax.text(0.5, 0.5, 'No phase info\n(pure MPC or BRT)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
    ax.set_title('Controller Phase')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (3,1) Summary text ------------------------------------------ #
    ax = axes[3, 1]
    ax.axis('off')
    dists_final = dists[-1] if len(dists) > 0 else float('nan')
    lines = [
        f"Docked:          {result.get('docked', 'N/A')}",
        f"Collision:       {result.get('collision', 'N/A')}",
        f"Success:         {result.get('success', 'N/A')}",
        f"Final distance:  {dists_final:.3f} m",
        f"Control effort:  {result.get('control_effort', 0.0):.2f}",
        f"Wall time:       {result.get('wall_time', 0.0):.2f} s",
        f"Sim duration:    {times[-1]:.1f} s" if len(times) > 0 else "",
        f"Steps:           {len(traj)}",
    ]
    summary = '\n'.join(lines)
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Summary')

    # ---- (4,0) BRAT time t* ----------------------------------------- #
    ax = axes[4, 0]
    t_rem = result.get('t_remaining')
    phases_arr = np.asarray(result['phases']) if 'phases' in result else None

    if t_rem is not None and phases_arr is not None:
        t_rem = np.asarray(t_rem, dtype=np.float64)
        n = min(len(times), len(t_rem), len(phases_arr))
        t_plot = times[:n]
        t_rem_plot = t_rem[:n]
        ph_plot = phases_arr[:n]

        # Background shading for Phase 1 (grey) vs Phase 2 (white)
        phase1_mask = ph_plot == 1
        if np.any(phase1_mask):
            ax.fill_between(t_plot, 0, t_rem_plot.max() * 1.1,
                            where=phase1_mask, color='gray', alpha=0.15,
                            label='Phase 1', step='post')

        # Plot t* line only during Phase 2 (mask Phase 1 as NaN)
        t_star_line = t_rem_plot.copy()
        t_star_line[phase1_mask] = np.nan
        ax.plot(t_plot, t_star_line, 'b-', lw=1.5, label='t* (strict)')

        # Overlay status markers from brt_time_adjustments (non-strict events)
        adjustments = result.get('brt_time_adjustments', [])
        argmin_times_list = []
        argmin_tstar_list = []
        hold_times_list = []
        hold_tstar_list = []
        for adj in adjustments:
            st = adj.get('status', '')
            if st == 'argmin':
                argmin_times_list.append(adj['sim_time'])
                argmin_tstar_list.append(adj['t_star'])
            elif st == 'hold':
                hold_times_list.append(adj['sim_time'])
                hold_tstar_list.append(adj['t_star'])
        if argmin_times_list:
            ax.scatter(argmin_times_list, argmin_tstar_list,
                       c='red', s=20, zorder=5, label='argmin')
        if hold_times_list:
            ax.scatter(hold_times_list, hold_tstar_list,
                       c='goldenrod', s=20, zorder=5, label='hold')

        ax.set_ylabel('t* (s)')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No t_remaining data',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
    ax.set_title('BRAT Time Horizon t*')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    # ---- (4,1) empty ------------------------------------------------- #
    axes[4, 1].axis('off')

    plt.tight_layout()

    if save_path:
        _safe_mkdir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Static] States figure saved to {save_path}")

    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
#  3. Controls plot
# --------------------------------------------------------------------------- #

def plot_controls_13d(result, dynamics, save_path=None):
    """Create a 2×2 control time-series figure.

    Panels:
      (0,0) Forces (Fx, Fy, Fz)     (0,1) Torques (τx, τy, τz)
      (1,0) Cumulative effort        (1,1) Force / torque magnitudes

    Returns ``matplotlib.figure.Figure``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    controls = result['controls']
    times = result['times']
    d = dynamics
    n_ctrl = min(len(times), len(controls))
    t_ctrl = times[:n_ctrl]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Saturate if attrs available
    F_max = getattr(d, 'F_bar', None)
    tau_max = getattr(d, 'tau_bar', None)

    # ---- (0,0) Forces ------------------------------------------------ #
    ax = axes[0, 0]
    for i, (lbl, c) in enumerate(zip(['Fx', 'Fy', 'Fz'],
                                      ['tab:blue', 'tab:orange', 'tab:green'])):
        ax.plot(t_ctrl, controls[:n_ctrl, i], color=c, lw=1.2, label=lbl)
    if F_max is not None:
        ax.axhline(F_max, color='red', ls='--', lw=0.8, alpha=0.5,
                   label=f'±F_max={F_max}')
        ax.axhline(-F_max, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='k', ls=':', lw=0.5, alpha=0.3)
    ax.set_title('Body-Frame Forces (N)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (0,1) Torques ----------------------------------------------- #
    ax = axes[0, 1]
    for i, (lbl, c) in enumerate(zip(['τx', 'τy', 'τz'],
                                      ['tab:blue', 'tab:orange', 'tab:green'])):
        ax.plot(t_ctrl, controls[:n_ctrl, 3 + i], color=c, lw=1.2, label=lbl)
    if tau_max is not None:
        ax.axhline(tau_max, color='red', ls='--', lw=0.8, alpha=0.5,
                   label=f'±τ_max={tau_max}')
        ax.axhline(-tau_max, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='k', ls=':', lw=0.5, alpha=0.3)
    ax.set_title('Body-Frame Torques (N·m)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (1,0) Cumulative effort ------------------------------------- #
    ax = axes[1, 0]
    effort_per_step = np.linalg.norm(controls[:n_ctrl], axis=1)
    cumulative = np.cumsum(effort_per_step)
    ax.plot(t_ctrl, cumulative, 'b-', lw=2, label='Cumulative ||u||')
    ax.set_title('Cumulative Control Effort')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Σ||u||')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (1,1) Force and torque magnitudes --------------------------- #
    ax = axes[1, 1]
    f_mag = np.linalg.norm(controls[:n_ctrl, :3], axis=1)
    t_mag = np.linalg.norm(controls[:n_ctrl, 3:], axis=1)
    ax.plot(t_ctrl, f_mag, 'b-', lw=1.5, label='||F|| (N)')
    ax.plot(t_ctrl, t_mag, 'r-', lw=1.5, label='||τ|| (N·m)')
    if F_max is not None:
        ax.axhline(F_max * np.sqrt(3), color='blue', ls=':', lw=0.8,
                   alpha=0.4, label=f'F_max·√3={F_max * np.sqrt(3):.1f}')
    ax.set_title('Force / Torque Magnitudes')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        _safe_mkdir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Static] Controls figure saved to {save_path}")

    plt.close(fig)
    return fig
