import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patches as mpatches
import torch
from tqdm import tqdm
from pathlib import Path as pathlib


def draw_target_body(ax, dynamics_, color='red', linewidth=2, alpha=0.7):
    """Draw the target body + docking post outline for the 13D geometry.
    
    Body: rectangle x ∈ [-w_t/2, w_t/2], y ∈ [0, h_t]
    Post: rectangle x ∈ [-post_hw_x, post_hw_x], y ∈ [-post_length, 0]
    """
    w_t = dynamics_.w_t
    h_t = dynamics_.h_t
    post_hw_x = dynamics_.post_hw_x
    post_length = dynamics_.post_length

    verts = []
    codes = []

    # Start at body bottom-left, go clockwise around body+post
    verts.append((-w_t/2, 0));                codes.append(Path.MOVETO)
    verts.append((-w_t/2, h_t));              codes.append(Path.LINETO)
    verts.append((w_t/2, h_t));               codes.append(Path.LINETO)
    verts.append((w_t/2, 0));                 codes.append(Path.LINETO)
    # Step down to post right side
    verts.append((post_hw_x, 0));             codes.append(Path.LINETO)
    verts.append((post_hw_x, -post_length));  codes.append(Path.LINETO)
    verts.append((-post_hw_x, -post_length)); codes.append(Path.LINETO)
    verts.append((-post_hw_x, 0));            codes.append(Path.LINETO)
    # Close back to start
    verts.append((-w_t/2, 0));                codes.append(Path.LINETO)

    path = Path(verts, codes)
    patch = patches.PathPatch(path, fill=False, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(patch)

    # Also draw the goal band
    if hasattr(dynamics_, 'goal_y_min'):
        goal_rect = patches.Rectangle(
            (-dynamics_.eps_p, dynamics_.goal_y_min), 2*dynamics_.eps_p,
            dynamics_.goal_y_max - dynamics_.goal_y_min,
            linewidth=1.5, edgecolor='green', facecolor='green', alpha=0.15,
            linestyle='--', label='Goal band'
        )
        ax.add_patch(goal_rect)


def plotBRTImages(costs, dynamics_, initial_condition_tensor, x_resolution, y_resolution,
                   x_min, x_max, y_min, y_max, save_def: pathlib):
    fig = plt.figure(figsize=(6, 6))
    fig2 = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1)
    reach_value = dynamics_.reach_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T

    BRT_img = costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.nanmax(BRT_img)
    min_value = np.nanmin(BRT_img)

    x_coords = np.linspace(x_min, x_max, x_resolution)
    y_coords = np.linspace(y_min, y_max, y_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    greys = np.full((*BRT_img.shape, 3), 70, dtype=np.uint8)
    imshow_kwargs = {
        'vmax': max_value,
        'vmin': min_value,
        'cmap': 'RdYlBu',
        'extent': (x_min, x_max, y_min, y_max),
        'origin': 'lower',
        'aspect': 'equal'
    }
    ax.imshow(greys)
    s1 = ax.imshow(BRT_img, **imshow_kwargs)

    level = 0.1
    ax.contour(X, Y, BRT_img, levels=[0.0], colors='black', linewidths=1.5, linestyles='-.')
    ax.contour(X, Y, BRT_img, levels=[level], colors='blue', linewidths=1.5, linestyles='--')

    ax.contour(X, Y, reach_value, levels=[0.0], colors='green', linewidths=1.5, linestyles='-')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(s1)
    ax.set_title('BRT Heatmap with Level Sets (x-y slice)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)
    handles = [
        mpatches.Patch(color='green', label='Reach Set'),
        mpatches.Patch(color='black', label='0 Level Set'),
        mpatches.Patch(color='blue', label=f'{level} Level Set')
    ]
    ax.legend(handles=handles)

    ax2 = fig2.add_subplot(1, 1, 1)
    binary_img = ax2.imshow(1 * (BRT_img <= 0), cmap='bwr',
                            origin='lower', extent=(x_min, x_max, y_min, y_max),
                            aspect='equal')
    ax2.contour(X, Y, BRT_img, levels=[0.0], colors='black', linewidths=1.5, linestyles='-')
    ax2.set_title('Binary Reachability (BRT ≤ 0)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__heatmap_.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"{save_def}__BRT.png", dpi=300, bbox_inches='tight')


def plotGoalAvoid(costs, dynamics_, initial_condition_tensor, x_resolution, y_resolution,
                   x_min, x_max, y_min, y_max, save_def: pathlib):
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1)
    reach_value = dynamics_.reach_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = dynamics_.avoid_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    boundry_value = dynamics_.boundary_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T

    BRT_img = costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.nanmax(BRT_img)
    min_value = np.nanmin(BRT_img)

    x_coords = np.linspace(x_min, x_max, x_resolution)
    y_coords = np.linspace(y_min, y_max, y_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    greys = np.full((*BRT_img.shape, 3), 70, dtype=np.uint8)
    imshow_kwargs = {
        'vmax': max_value,
        'vmin': min_value,
        'cmap': 'RdYlBu',
        'extent': (x_min, x_max, y_min, y_max),
        'origin': 'lower',
        'aspect': 'equal'
    }
    ax.imshow(greys)
    s1 = ax.imshow(BRT_img, **imshow_kwargs)

    ax.contour(X, Y, reach_value, levels=[0.0], colors='black', linewidths=1.5, linestyles='-')
    ax.contour(X, Y, avoid_value, levels=[0.0], colors='purple', linewidths=1.5, linestyles='-.')
    ax.contour(X, Y, boundry_value, levels=[0.0], colors='orange', linewidths=1.5, linestyles=':')

    if hasattr(dynamics_, 'avoid_fn') and hasattr(dynamics_, 'w_t'):
        draw_target_body(ax, dynamics_, color='red', linewidth=2, alpha=0.7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(s1)
    ax.set_title('BRT Heatmap with Level Sets (x-y slice)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)
    handles = [
        mpatches.Patch(color='black', label='Reach Set'),
        mpatches.Patch(color='purple', label='Avoid Set'),
        mpatches.Patch(color='orange', label='Boundary Set')
    ]
    ax.legend(handles=handles)

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__ReadAvoidSet.png", dpi=300, bbox_inches='tight')


def plotMPCTrajectories(mpc, initial_conditions, T, save_def: pathlib, max_trajs=10):
    n_trajs = min(len(initial_conditions), max_trajs)
    ic_subset = initial_conditions[:n_trajs]

    costs, state_trajs, _, _ = mpc.get_batch_data(ic_subset, T)

    state_trajs_np = state_trajs.detach().cpu().numpy()
    costs_np = costs.detach().cpu().numpy()

    actual_T = state_trajs_np.shape[1] - 1
    time_steps = np.arange(actual_T + 1) * mpc.dT

    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful = reach_values <= 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MPC Trajectory Analysis 13D (T={actual_T}, dt={mpc.dT})', fontsize=16)

    colors = plt.cm.tab20(np.linspace(0, 1, n_trajs))

    # 1) XY position
    ax1 = axes[0, 0]
    for i in range(n_trajs):
        x = state_trajs_np[i, :, 0]
        y = state_trajs_np[i, :, 1]
        ax1.plot(x, y, color=colors[i], alpha=0.8, linewidth=1.5, label=f'IC {i+1} (cost {costs_np[i]:.3f})')
        ax1.scatter(x[0], y[0], color=colors[i], s=25, marker='o')
        ax1.scatter(x[-1], y[-1], color=colors[i], s=25, marker='s')
    if hasattr(mpc.dynamics_, 'eps_p'):
        eps_p = mpc.dynamics_.eps_p
        ax1.add_patch(patches.Circle((0, 0), eps_p, fill=False, color='green', linewidth=1, linestyle='--'))
    if hasattr(mpc.dynamics_, 'w_t'):
        draw_target_body(ax1, mpc.dynamics_, color='red', linewidth=2, alpha=0.7)
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)'); ax1.set_title('XY Position'); ax1.grid(True, alpha=0.3); ax1.axis('equal')

    # 2) XY velocity
    ax2 = axes[0, 1]
    for i in range(n_trajs):
        vx = state_trajs_np[i, :, 3]
        vy = state_trajs_np[i, :, 4]
        ax2.plot(vx, vy, color=colors[i], alpha=0.8, linewidth=1.5)
        ax2.scatter(vx[0], vy[0], color=colors[i], s=25, marker='o')
        ax2.scatter(vx[-1], vy[-1], color=colors[i], s=25, marker='s')
    ax2.set_xlabel('vx (m/s)'); ax2.set_ylabel('vy (m/s)'); ax2.set_title('Velocity in XY'); ax2.grid(True, alpha=0.3)

    # 3) Altitude over time
    ax3 = axes[0, 2]
    for i in range(n_trajs):
        z = state_trajs_np[i, :, 2]
        ax3.plot(time_steps, z, color=colors[i], alpha=0.8, linewidth=1.5)
    ax3.set_xlabel('time (s)'); ax3.set_ylabel('z (m)'); ax3.set_title('Altitude vs Time'); ax3.grid(True, alpha=0.3)

    # 4) Attitude error angle vs time
    ax4 = axes[1, 0]
    q_goal = mpc.dynamics_.q_goal.detach().cpu().numpy()
    for i in range(n_trajs):
        q = state_trajs_np[i, :, 6:10]
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        # angle between q and q_goal: 2*acos(|q0_rel|)
        # q_rel = q_goal^{-1} ⊗ q
        q_conj = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
        q_rel = np.stack([
            q_conj[0]*q[:,0] - q_conj[1]*q[:,1] - q_conj[2]*q[:,2] - q_conj[3]*q[:,3],
            q_conj[0]*q[:,1] + q_conj[1]*q[:,0] + q_conj[2]*q[:,3] - q_conj[3]*q[:,2],
            q_conj[0]*q[:,2] - q_conj[1]*q[:,3] + q_conj[2]*q[:,0] + q_conj[3]*q[:,1],
            q_conj[0]*q[:,3] + q_conj[1]*q[:,2] - q_conj[2]*q[:,1] + q_conj[3]*q[:,0]
        ], axis=-1)
        q0_abs = np.clip(np.abs(q_rel[:,0]), 0.0, 1.0)
        ang = 2.0 * np.arccos(q0_abs)
        ax4.plot(time_steps, ang, color=colors[i], alpha=0.8, linewidth=1.5)
    ax4.set_xlabel('time (s)'); ax4.set_ylabel('attitude error (rad)'); ax4.set_title('Attitude Error vs Time'); ax4.grid(True, alpha=0.3)

    # 5) Angular velocity components
    ax5 = axes[1, 1]
    for i in range(n_trajs):
        wx = state_trajs_np[i, :, 10]; wy = state_trajs_np[i, :, 11]; wz = state_trajs_np[i, :, 12]
        ax5.plot(time_steps, wx, color=colors[i], alpha=0.8, linewidth=1.0)
        ax5.plot(time_steps, wy, color=colors[i], alpha=0.8, linewidth=1.0, linestyle='--')
        ax5.plot(time_steps, wz, color=colors[i], alpha=0.8, linewidth=1.0, linestyle=':')
    ax5.set_xlabel('time (s)'); ax5.set_ylabel('ω (rad/s)'); ax5.set_title('Angular Velocity Components'); ax5.grid(True, alpha=0.3)

    # 6) Docking success bars (reach values)
    ax6 = axes[1, 2]
    bars = ax6.bar(range(n_trajs), reach_values, color=colors, alpha=0.8)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('Trajectory'); ax6.set_ylabel('Reach Function Value'); ax6.set_title('Docking Success Analysis')
    ax6.set_xticks(range(n_trajs)); ax6.set_xticklabels([f'IC {i+1}' for i in range(n_trajs)])
    for i, (bar, rv) in enumerate(zip(bars, reach_values)):
        height = bar.get_height(); y_pos = height + 0.01 if height >= 0 else height - 0.01; va = 'bottom' if height >= 0 else 'top'
        ax6.text(bar.get_x() + bar.get_width()/2, y_pos, f'{rv:.3f}', ha='center', va=va, fontsize=10)

    plt.tight_layout()

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__fullTraj.png", dpi=300, bbox_inches='tight')

    return fig, state_trajs_np, costs_np, successful, reach_values


def compute_control_from_transition_13D(state_curr, state_next, dt, dynamics):
    """
    Approximate control [Fx,Fy,Fz,tx,ty,tz] given consecutive states and dt for Docking13D.
    Uses the Docking13D model equations to invert for controls.
    """
    # unpack
    x, y, z = state_curr[0], state_curr[1], state_curr[2]
    vx, vy, vz = state_curr[3], state_curr[4], state_curr[5]
    q = state_curr[6:10]
    omega = state_curr[10:13]

    vx_n, vy_n, vz_n = state_next[3], state_next[4], state_next[5]
    omega_n = state_next[10:13]

    # estimate accelerations
    ax_est = (vx_n - vx) / dt
    ay_est = (vy_n - vy) / dt
    az_est = (vz_n - vz) / dt

    # remove drift terms to get LVLH accel from body force
    n = dynamics.n
    mc = dynamics.mc

    ax_ctrl = ax_est - (2*n*vy + 3*(n**2)*x)
    ay_ctrl = ay_est - (-2*n*vx)
    az_ctrl = az_est - (-(n**2)*z)

    # rotate to body and scale by mass: F_body = m * R(q) * a_LVLH
    q = q / (torch.norm(q).item() + 1e-12) if isinstance(q, torch.Tensor) else q / (np.linalg.norm(q) + 1e-12)
    if isinstance(q, np.ndarray):
        q_t = torch.tensor(q, dtype=torch.float32, device=dynamics.state_range_.device)
    else:
        q_t = q
    R_L2B = dynamics.quat_to_R_LVLH_to_body(q_t.unsqueeze(0)).squeeze(0)
    aL = torch.tensor([ax_ctrl, ay_ctrl, az_ctrl], dtype=torch.float32, device=R_L2B.device)
    Fb = torch.matmul(R_L2B, aL) * mc

    # torque from rotational dynamics: tau = I * omega_dot + omega x (I omega)
    if isinstance(omega, np.ndarray):
        omega_t = torch.tensor(omega, dtype=torch.float32, device=R_L2B.device)
        omega_n_t = torch.tensor(omega_n, dtype=torch.float32, device=R_L2B.device)
    else:
        omega_t = omega
        omega_n_t = omega_n
    omega_dot = (omega_n_t - omega_t) / dt
    I = dynamics.I
    tau = torch.matmul(I, omega_dot.unsqueeze(-1)).squeeze(-1) + torch.cross(omega_t, torch.matmul(I, omega_t.unsqueeze(-1)).squeeze(-1), dim=-1)

    return torch.cat([Fb, tau], dim=-1)


def plotMPCControls(mpc, initial_conditions, T, save_def: pathlib, max_trajs=10):
    n_trajs = min(len(initial_conditions), max_trajs)
    ic_subset = initial_conditions[:n_trajs]

    costs, state_trajs, _, _ = mpc.get_batch_data(ic_subset, T)

    state_trajs_np = state_trajs.detach().cpu().numpy()

    actual_T = state_trajs_np.shape[1] - 1
    time_steps = np.arange(actual_T) * mpc.dT

    controls_np = np.zeros((n_trajs, actual_T, 6))  # [Fx,Fy,Fz,tx,ty,tz]

    for i in range(n_trajs):
        for t in range(actual_T):
            s_cur = torch.tensor(state_trajs_np[i, t, :]).to(mpc.device)
            s_nxt = torch.tensor(state_trajs_np[i, t+1, :]).to(mpc.device)
            u = compute_control_from_transition_13D(s_cur, s_nxt, mpc.dT, mpc.dynamics_)
            controls_np[i, t, :] = u.detach().cpu().numpy()

    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful = reach_values <= 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MPC Control Analysis 13D (T={actual_T}, dt={mpc.dT})', fontsize=16)

    colors = plt.cm.tab20(np.linspace(0, 1, n_trajs))

    # Fx
    ax1 = axes[0, 0]
    for i in range(n_trajs):
        ax1.plot(time_steps, controls_np[i, :, 0], color=colors[i], alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('time (s)'); ax1.set_ylabel('Fx (N)'); ax1.set_title('Fx'); ax1.grid(True, alpha=0.3)
    if hasattr(mpc.dynamics_, 'F_bar'):
        u = mpc.dynamics_.F_bar; ax1.axhline(y=u, color='black', linestyle='--', alpha=0.5); ax1.axhline(y=-u, color='black', linestyle='--', alpha=0.5)

    # Fy
    ax2 = axes[0, 1]
    for i in range(n_trajs):
        ax2.plot(time_steps, controls_np[i, :, 1], color=colors[i], alpha=0.8, linewidth=1.5)
    ax2.set_xlabel('time (s)'); ax2.set_ylabel('Fy (N)'); ax2.set_title('Fy'); ax2.grid(True, alpha=0.3)
    if hasattr(mpc.dynamics_, 'F_bar'):
        u = mpc.dynamics_.F_bar; ax2.axhline(y=u, color='black', linestyle='--', alpha=0.5); ax2.axhline(y=-u, color='black', linestyle='--', alpha=0.5)

    # Fz
    ax3 = axes[0, 2]
    for i in range(n_trajs):
        ax3.plot(time_steps, controls_np[i, :, 2], color=colors[i], alpha=0.8, linewidth=1.5)
    ax3.set_xlabel('time (s)'); ax3.set_ylabel('Fz (N)'); ax3.set_title('Fz'); ax3.grid(True, alpha=0.3)
    if hasattr(mpc.dynamics_, 'F_bar'):
        u = mpc.dynamics_.F_bar; ax3.axhline(y=u, color='black', linestyle='--', alpha=0.5); ax3.axhline(y=-u, color='black', linestyle='--', alpha=0.5)

    # tx
    ax4 = axes[1, 0]
    for i in range(n_trajs):
        ax4.plot(time_steps, controls_np[i, :, 3], color=colors[i], alpha=0.8, linewidth=1.5)
    ax4.set_xlabel('time (s)'); ax4.set_ylabel('tx (N·m)'); ax4.set_title('tx'); ax4.grid(True, alpha=0.3)
    if hasattr(mpc.dynamics_, 'tau_bar'):
        u = mpc.dynamics_.tau_bar; ax4.axhline(y=u, color='black', linestyle='--', alpha=0.5); ax4.axhline(y=-u, color='black', linestyle='--', alpha=0.5)

    # ty
    ax5 = axes[1, 1]
    for i in range(n_trajs):
        ax5.plot(time_steps, controls_np[i, :, 4], color=colors[i], alpha=0.8, linewidth=1.5)
    ax5.set_xlabel('time (s)'); ax5.set_ylabel('ty (N·m)'); ax5.set_title('ty'); ax5.grid(True, alpha=0.3)
    if hasattr(mpc.dynamics_, 'tau_bar'):
        u = mpc.dynamics_.tau_bar; ax5.axhline(y=u, color='black', linestyle='--', alpha=0.5); ax5.axhline(y=-u, color='black', linestyle='--', alpha=0.5)

    # tz
    ax6 = axes[1, 2]
    for i in range(n_trajs):
        ax6.plot(time_steps, controls_np[i, :, 5], color=colors[i], alpha=0.8, linewidth=1.5)
    ax6.set_xlabel('time (s)'); ax6.set_ylabel('tz (N·m)'); ax6.set_title('tz'); ax6.grid(True, alpha=0.3)
    if hasattr(mpc.dynamics_, 'tau_bar'):
        u = mpc.dynamics_.tau_bar; ax6.axhline(y=u, color='black', linestyle='--', alpha=0.5); ax6.axhline(y=-u, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__control.png", dpi=300, bbox_inches='tight')

    return fig, controls_np, successful


def compute_all_brt_slices(mpc, dynamics_, device, T, resolution):
    """Compute BRT for key 13D slices: position (x,y), velocity (vx,vy), rotation (delta_yaw, wz)."""
    plot_cfg = dynamics_.plot_config()
    state_range = dynamics_.state_test_range()

    def compute_brt_batch(initial_conditions, slice_name):
        costs = []
        for i in tqdm(range(4), desc=f"Computing {slice_name} BRT"):
            batch_size = len(initial_conditions) // 4
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size if i < 3 else len(initial_conditions)
            costs_batch, _, _, _ = mpc.get_batch_data(initial_conditions[start_idx:end_idx], T)
            costs.append(costs_batch)
        return torch.cat(costs, dim=0)

    # Position slice: x-y
    x_min, x_max = state_range[plot_cfg['x_axis_idx']]
    y_min, y_max = state_range[plot_cfg['y_axis_idx']]
    xs = torch.linspace(x_min, x_max, resolution[0])
    ys = torch.linspace(y_min, y_max, resolution[1])
    xys = torch.cartesian_prod(xs, ys).to(device)

    pos_ics = torch.zeros(resolution[0] * resolution[1], dynamics_.state_dim, device=device)
    pos_ics[:, :] = torch.tensor(plot_cfg['state_slices'], device=device)
    pos_ics[:, plot_cfg['x_axis_idx']] = xys[:, 0]
    pos_ics[:, plot_cfg['y_axis_idx']] = xys[:, 1]
    position_costs = compute_brt_batch(pos_ics, 'position')

    # Velocity slice: vx-vy
    vx_min, vx_max = state_range[3]
    vy_min, vy_max = state_range[4]
    vxs = torch.linspace(vx_min, vx_max, resolution[2])
    vys = torch.linspace(vy_min, vy_max, resolution[3])
    vxvys = torch.cartesian_prod(vxs, vys).to(device)

    vel_ics = torch.zeros(resolution[2] * resolution[3], dynamics_.state_dim, device=device)
    vel_ics[:, :] = torch.tensor(plot_cfg['state_slices'], device=device)
    vel_ics[:, 3] = vxvys[:, 0]
    vel_ics[:, 4] = vxvys[:, 1]
    velocity_costs = compute_brt_batch(vel_ics, 'velocity')

    # Rotation slice: delta yaw vs wz (use quaternion about z-axis)
    theta_min, theta_max = -math.pi, math.pi
    wz_min, wz_max = state_range[12]
    thetas = torch.linspace(theta_min, theta_max, resolution[4])
    wzs = torch.linspace(wz_min, wz_max, resolution[5])
    th_wzs = torch.cartesian_prod(thetas, wzs).to(device)

    rot_ics = torch.zeros(resolution[4] * resolution[5], dynamics_.state_dim, device=device)
    rot_ics[:, :] = torch.tensor(plot_cfg['state_slices'], device=device)
    # construct q = q_goal ⊗ q_delta_z(theta)
    q_goal = dynamics_.q_goal.to(device)
    def quat_mul(q, p):
        q0,q1,q2,q3 = q[...,0],q[...,1],q[...,2],q[...,3]
        p0,p1,p2,p3 = p[...,0],p[...,1],p[...,2],p[...,3]
        return torch.stack([
            q0*p0 - q1*p1 - q2*p2 - q3*p3,
            q0*p1 + q1*p0 + q2*p3 - q3*p2,
            q0*p2 - q1*p3 + q2*p0 + q3*p1,
            q0*p3 + q1*p2 - q2*p1 + q3*p0
        ], dim=-1)
    def quat_from_axis_angle(axis, angle):
        axis = axis / (torch.norm(axis) + 1e-12)
        half = 0.5 * angle
        if not isinstance(half, torch.Tensor):
            half = torch.tensor(half, device=axis.device)
        q0 = torch.cos(half)
        qv = axis * torch.sin(half)
        return torch.stack([q0, qv[0], qv[1], qv[2]], dim=-1)
    axis_z = torch.tensor([0.0, 0.0, 1.0], device=device)
    for i in range(rot_ics.shape[0]):
        ang = th_wzs[i, 0]
        q_delta = quat_from_axis_angle(axis_z, ang)
        q = quat_mul(q_goal, q_delta)
        q = q / (torch.norm(q) + 1e-12)
        rot_ics[i, 6:10] = q
        rot_ics[i, 12] = th_wzs[i, 1]  # wz
    rotation_costs = compute_brt_batch(rot_ics, 'rotation')

    return {
        'position': {
            'costs': position_costs,
            'coords': (x_min, x_max, y_min, y_max),
            'initial_conditions': pos_ics
        },
        'velocity': {
            'costs': velocity_costs,
            'coords': (vx_min, vx_max, vy_min, vy_max),
            'initial_conditions': vel_ics
        },
        'rotation': {
            'costs': rotation_costs,
            'coords': (theta_min, theta_max, wz_min, wz_max),
            'initial_conditions': rot_ics
        }
    }


def plotBRTPosition(mpc, interesting_ics_tensor, brt_data, x_resolution, y_resolution,
                    T, save_def: pathlib, level_sets=[0.0, 0.3, 0.6]):
    position_costs = brt_data['position']['costs']
    x_min, x_max, y_min, y_max = brt_data['position']['coords']
    init_cond = brt_data['position']['initial_conditions']

    costs, state_trajs, _, _ = mpc.get_batch_data(interesting_ics_tensor, T)
    state_trajs_np = state_trajs.detach().cpu().numpy()

    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful = reach_values <= 0

    BRT_img = position_costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.nanmax(BRT_img)
    min_value = np.nanmin(BRT_img)

    x_coords = np.linspace(x_min, x_max, x_resolution)
    y_coords = np.linspace(y_min, y_max, y_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    im = ax.imshow(BRT_img, vmax=max_value, vmin=min_value, cmap='RdYlBu',
                   extent=(x_min, x_max, y_min, y_max), origin='lower', aspect='equal', alpha=0.7)

    level_colors = ['purple', 'blue', 'orange', 'cyan', 'magenta', 'yellow']
    level_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    level_handles = []
    for i, level in enumerate(level_sets):
        color = level_colors[i % len(level_colors)]
        style = level_styles[i % len(level_styles)]
        ax.contour(X, Y, BRT_img, levels=[level], colors=[color], linewidths=1.5, linestyles=style)
        level_handles.append(mpatches.Patch(color=color, label=f'Level {level}'))

    colors = plt.cm.tab20(np.linspace(0, 1, len(interesting_ics_tensor)))
    traj_handles = []
    for i in range(len(interesting_ics_tensor)):
        x = state_trajs_np[i, :, 0]
        y = state_trajs_np[i, :, 1]
        success_label = 'Success' if successful[i] else 'Failed'
        line, = ax.plot(x, y, color=colors[i], linewidth=1.5, alpha=0.9, label=f'IC {i+1} ({success_label})')
        traj_handles.append(line)
        ax.scatter(x[0], y[0], color=colors[i], s=25, marker='o', edgecolors='black', linewidth=1, zorder=10)
        ax.scatter(x[-1], y[-1], color=colors[i], s=25, marker='s', edgecolors='black', linewidth=1, zorder=10)

    reach_value = mpc.dynamics_.reach_fn(init_cond).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = mpc.dynamics_.avoid_fn(init_cond).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    ax.contour(X, Y, reach_value, levels=[0.0], colors='green', linewidths=1.5, linestyles='-')
    ax.contour(X, Y, avoid_value, levels=[0.0], colors='red', linewidths=1.5, linestyles='-.')

    reach_avoid_handles = [mpatches.Patch(color='green', label='Reach Set'), mpatches.Patch(color='red', label='Avoid Set')]

    if hasattr(mpc.dynamics_, 'w_t'):
        draw_target_body(ax, mpc.dynamics_, color='red', linewidth=2, alpha=0.7)
        reach_avoid_handles.append(mpatches.Patch(facecolor='none', edgecolor='red', label='Target body + post'))

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_title('MPC Trajectories on Position BRT Heatmap'); ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('BRT Value')
    ax.legend(handles=(reach_avoid_handles + level_handles + traj_handles), bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__traj_position.png", dpi=300, bbox_inches='tight')

    return fig, state_trajs_np, successful


def plotBRTVelocity(mpc, interesting_ics_tensor, brt_data, x_resolution, y_resolution,
                    T, save_def: pathlib, level_sets=[0.0, 0.3, 0.6]):
    velocity_costs = brt_data['velocity']['costs']
    vx_min, vx_max, vy_min, vy_max = brt_data['velocity']['coords']
    vel_init = brt_data['velocity']['initial_conditions']

    costs, state_trajs, _, _ = mpc.get_batch_data(interesting_ics_tensor, T)
    state_trajs_np = state_trajs.detach().cpu().numpy()

    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful = reach_values <= 0

    BRT_img = velocity_costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.nanmax(BRT_img)
    min_value = np.nanmin(BRT_img)

    VX = np.linspace(vx_min, vx_max, x_resolution)
    VY = np.linspace(vy_min, vy_max, y_resolution)
    VXg, VYg = np.meshgrid(VX, VY)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    im = ax.imshow(BRT_img, vmax=max_value, vmin=min_value, cmap='RdYlBu',
                   extent=(vx_min, vx_max, vy_min, vy_max), origin='lower', aspect='equal', alpha=0.7)

    level_colors = ['purple', 'blue', 'orange', 'cyan', 'magenta', 'yellow']
    level_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    level_handles = []
    for i, level in enumerate(level_sets):
        color = level_colors[i % len(level_colors)]
        style = level_styles[i % len(level_styles)]
        ax.contour(VXg, VYg, BRT_img, levels=[level], colors=[color], linewidths=1.5, linestyles=style)
        level_handles.append(mpatches.Patch(color=color, label=f'Level {level}'))

    colors = plt.cm.tab20(np.linspace(0, 1, len(interesting_ics_tensor)))
    traj_handles = []
    for i in range(len(interesting_ics_tensor)):
        vx = state_trajs_np[i, :, 3]
        vy = state_trajs_np[i, :, 4]
        success_label = 'Success' if successful[i] else 'Failed'
        line, = ax.plot(vx, vy, color=colors[i], linewidth=1.5, alpha=0.9, label=f'IC {i+1} ({success_label})')
        traj_handles.append(line)
        ax.scatter(vx[0], vy[0], color=colors[i], s=25, marker='o', edgecolors='black', linewidth=1, zorder=10)
        ax.scatter(vx[-1], vy[-1], color=colors[i], s=25, marker='s', edgecolors='black', linewidth=1, zorder=10)

    reach_value = mpc.dynamics_.reach_fn(vel_init).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = mpc.dynamics_.avoid_fn(vel_init).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    ax.contour(VXg, VYg, reach_value, levels=[0.0], colors='green', linewidths=2, linestyles='-')
    ax.contour(VXg, VYg, avoid_value, levels=[0.0], colors='red', linewidths=2, linestyles='-.')

    reach_avoid_handles = [mpatches.Patch(color='green', label='Reach Set (velocity slice)'), mpatches.Patch(color='red', label='Avoid Set (velocity slice)')]

    ax.set_xlim(vx_min, vx_max); ax.set_ylim(vy_min, vy_max)
    ax.set_xlabel('vx (m/s)'); ax.set_ylabel('vy (m/s)'); ax.set_title('MPC Velocity Trajectories on Velocity BRT Slice'); ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('BRT Value')
    ax.legend(handles=(reach_avoid_handles + level_handles + traj_handles), bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__traj_velocity.png", dpi=300, bbox_inches='tight')

    return fig, state_trajs_np, successful


def plotBRTRotation(mpc, interesting_ics_tensor, brt_data, x_resolution, y_resolution,
                    T, save_def: pathlib, level_sets=[0.0, 0.3, 0.6]):
    rotation_costs = brt_data['rotation']['costs']
    theta_min, theta_max, wz_min, wz_max = brt_data['rotation']['coords']
    rot_init = brt_data['rotation']['initial_conditions']

    costs, state_trajs, _, _ = mpc.get_batch_data(interesting_ics_tensor, T)
    state_trajs_np = state_trajs.detach().cpu().numpy()

    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful = reach_values <= 0

    BRT_img = rotation_costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.nanmax(BRT_img)
    min_value = np.nanmin(BRT_img)

    TH = np.linspace(theta_min, theta_max, x_resolution)
    WZ = np.linspace(wz_min, wz_max, y_resolution)
    THg, WZg = np.meshgrid(TH, WZ)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    im = ax.imshow(BRT_img, vmax=max_value, vmin=min_value, cmap='RdYlBu',
                   extent=(theta_min, theta_max, wz_min, wz_max), origin='lower', aspect='equal', alpha=0.7)

    level_colors = ['purple', 'blue', 'orange', 'cyan', 'magenta', 'yellow']
    level_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    level_handles = []
    for i, level in enumerate(level_sets):
        color = level_colors[i % len(level_colors)]
        style = level_styles[i % len(level_styles)]
        ax.contour(THg, WZg, BRT_img, levels=[level], colors=[color], linewidths=1.5, linestyles=style)
        level_handles.append(mpatches.Patch(color=color, label=f'Level {level}'))

    colors = plt.cm.tab20(np.linspace(0, 1, len(interesting_ics_tensor)))
    traj_handles = []
    # plot delta yaw (approx) from quaternion vs wz trajectory
    q_goal = mpc.dynamics_.q_goal.detach().cpu().numpy()
    for i in range(len(interesting_ics_tensor)):
        q = state_trajs_np[i, :, 6:10]
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        q_conj = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
        q_rel = np.stack([
            q_conj[0]*q[:,0] - q_conj[1]*q[:,1] - q_conj[2]*q[:,2] - q_conj[3]*q[:,3],
            q_conj[0]*q[:,1] + q_conj[1]*q[:,0] + q_conj[2]*q[:,3] - q_conj[3]*q[:,2],
            q_conj[0]*q[:,2] - q_conj[1]*q[:,3] + q_conj[2]*q[:,0] + q_conj[3]*q[:,1],
            q_conj[0]*q[:,3] + q_conj[1]*q[:,2] - q_conj[2]*q[:,1] + q_conj[3]*q[:,0]
        ], axis=-1)
        q0_abs = np.clip(np.abs(q_rel[:,0]), 0.0, 1.0)
        ang = 2.0 * np.arccos(q0_abs)
        wz = state_trajs_np[i, :, 12]
        success_label = 'Success' if successful[i] else 'Failed'
        line, = ax.plot(ang, wz, color=colors[i], linewidth=1.5, alpha=0.9, label=f'IC {i+1} ({success_label})')
        traj_handles.append(line)
        ax.scatter(ang[0], wz[0], color=colors[i], s=25, marker='o', edgecolors='black', linewidth=1, zorder=10)
        ax.scatter(ang[-1], wz[-1], color=colors[i], s=25, marker='s', edgecolors='black', linewidth=1, zorder=10)

    # reach/avoid on this slice (computed from rot_init)
    reach_value = mpc.dynamics_.reach_fn(rot_init).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = mpc.dynamics_.avoid_fn(rot_init).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    ax.contour(THg, WZg, reach_value, levels=[0.0], colors='green', linewidths=2, linestyles='-')
    ax.contour(THg, WZg, avoid_value, levels=[0.0], colors='red', linewidths=2, linestyles='-.')

    reach_avoid_handles = [mpatches.Patch(color='green', label='Reach Set (Δyaw/wz slice)'), mpatches.Patch(color='red', label='Avoid Set (Δyaw/wz slice)')]

    ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(wz_min, wz_max)
    ax.set_xlabel('Δ yaw (rad)')
    ax.set_ylabel('wz (rad/s)')
    ax.set_title('MPC Rotational Trajectories on Rotation BRT Slice (13D)')
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('BRT Value')
    ax.legend(handles=(reach_avoid_handles + level_handles + traj_handles), bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__traj_rotation.png", dpi=300, bbox_inches='tight')

    return fig, state_trajs_np, successful
