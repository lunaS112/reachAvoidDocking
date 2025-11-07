import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patches as mpatches
import torch
from tqdm import tqdm
from pathlib import Path as pathlib

def draw_target_body(ax, w_t, h_t, dock_rad, color='red', linewidth=2, alpha=0.7):
    theta = np.linspace(0, np.pi, 100)
    arc_x = dock_rad * np.cos(theta)
    arc_y = dock_rad * np.sin(theta)

    verts = []
    codes = []

    # Start at bottom-left corner
    verts.append((-w_t/2, 0));             codes.append(Path.MOVETO)
    # Left side up
    verts.append((-w_t/2, h_t));           codes.append(Path.LINETO)
    # Top across
    verts.append((w_t/2, h_t));            codes.append(Path.LINETO)
    # Right side down
    verts.append((w_t/2, 0));              codes.append(Path.LINETO)
    # Bottom-right segment to mouth
    verts.append((dock_rad, 0));           codes.append(Path.LINETO)
    # Semicircular mouth (from +R to -R)
    for x, y in zip(arc_x, arc_y):
        verts.append((x, y));              codes.append(Path.LINETO)
    # Bottom-left segment back to start
    verts.append((-w_t/2, 0));             codes.append(Path.LINETO)

    path = Path(verts, codes)
    patch = patches.PathPatch(path, fill=False, color=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(patch)

def plotBRTImages(costs, dynamics_, initial_condition_tensor, x_resolution, y_resolution, x_min, x_max, y_min, y_max, save_def:pathlib):
    fig = plt.figure(figsize=(6, 6))
    fig2 = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1 )
    reach_value = dynamics_.reach_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = dynamics_.avoid_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T

    BRT_img = costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.amax(BRT_img[~np.isnan(BRT_img)])
    min_value = np.amin(BRT_img[~np.isnan(BRT_img)])

    # create coordinate grids for contour plotting
    x_coords = np.linspace(x_min, x_max, x_resolution)
    y_coords = np.linspace(y_min, y_max, y_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    # We'll also create a grey background into which the pixels will fade
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
    s1=ax.imshow(BRT_img, **imshow_kwargs)

    # Define secondary level set 
    level = 0.1

    # Add level set contours
    ax.contour(X, Y, BRT_img, levels=[0.0], colors='black', linewidths=1.5, linestyles='-.')
    ax.contour(X, Y, BRT_img, levels=[level], colors='blue', linewidths=1.5, linestyles='--')

    ax.contour(X, Y, reach_value, levels=[0.0], colors='green', linewidths=1.5, linestyles='-')
    ax.contour(X, Y, avoid_value, levels=[0.0], colors='red', linewidths=1.5, linestyles='-')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(s1)
    ax.set_title('BRT Heatmap with Level Sets')
    ax.set_xlabel('px (m)')
    ax.set_ylabel('py (m)')
    ax.grid(True, alpha=0.3)
    handles = [mpatches.Patch(color='green', linestyle='-', label='Reach Set'),
                mpatches.Patch(color='red', label='Avoid Set'),
                mpatches.Patch(color='black', label='0 Level Set'),
                mpatches.Patch(color='blue', label=f'{level} Level Set')]
    ax.legend(handles=handles)
    
    # Second plot - binary reachability
    ax2 = fig2.add_subplot(1, 1, 1)
    binary_img = ax2.imshow(1*(BRT_img <= 0), cmap='bwr',
                origin='lower', extent=(x_min, x_max, y_min, y_max),
                aspect='equal')
    
    # Add 0-level contour to binary plot as well
    ax2.contour(X, Y, BRT_img, levels=[0.0], colors='black', linewidths=1.5, linestyles='-')
    ax2.set_title('Binary Reachability (BRT ≤ 0)')
    ax2.set_xlabel('px (m)')
    ax2.set_ylabel('py (m)')
    
    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__heatmap_.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"{save_def}__BRT.png", dpi=300, bbox_inches='tight')

def plotGoalAvoid(costs, dynamics_, initial_condition_tensor, x_resolution, y_resolution, x_min, x_max, y_min, y_max, save_def:pathlib):
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1 )
    reach_value = dynamics_.reach_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = dynamics_.avoid_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    boundry_value = dynamics_.boundary_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    
    BRT_img = costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.amax(BRT_img[~np.isnan(BRT_img)])
    min_value = np.amin(BRT_img[~np.isnan(BRT_img)])

    # create coordinate grids for contour plotting
    x_coords = np.linspace(x_min, x_max, x_resolution)
    y_coords = np.linspace(y_min, y_max, y_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    # We'll also create a grey background into which the pixels will fade
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
    s1=ax.imshow(BRT_img, **imshow_kwargs)

    ax.contour(X, Y, reach_value, levels=[0.0], colors='black', linewidths=1.5, linestyles='-')
    ax.contour(X, Y, avoid_value, levels=[0.0], colors='purple', linewidths=1.5, linestyles='-.')
    ax.contour(X, Y, boundry_value, levels=[0.0], colors='orange', linewidths=1.5, linestyles=':')

    # Define body of target spacecraft
    if hasattr(dynamics_, 'avoid_fn'):
        # Draw target spacecraft body with docking port cutout
        w_t = dynamics_.w_t
        h_t = dynamics_.h_t
        dock_rad = dynamics_.dock_rad
        draw_target_body(ax, w_t, h_t, dock_rad, color='red', linewidth=2, alpha=0.7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(s1)
    ax.set_title('BRT Heatmap with Level Sets')
    ax.set_xlabel('px (m)')
    ax.set_ylabel('py (m)')
    ax.grid(True, alpha=0.3)
    handles = [mpatches.Patch(color='black', label='Reach Set'),
                mpatches.Patch(color='purple', label='Avoid Set'),
                mpatches.Patch(color='orange', label='Boundary Set')]
    ax.legend(handles=handles)

    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__ReadAvoidSet.png", dpi=300, bbox_inches='tight')

def plotMPCTrajectories(mpc, initial_conditions, T, save_def:pathlib, max_trajs=10):
    import matplotlib.patches as patches
    
    # Limit number of trajectories for visualization clarity
    n_trajs = min(len(initial_conditions), max_trajs)
    ic_subset = initial_conditions[:n_trajs]
    
    # Get optimal trajectories
    costs, state_trajs, _, _ = mpc.get_batch_data(ic_subset, T)
    
    # Convert to numpy for plotting
    state_trajs_np = state_trajs.detach().cpu().numpy()  # Shape: (n_trajs, actual_T+1, state_dim)
    costs_np = costs.detach().cpu().numpy()
    
    # Get actual trajectory length from the data
    actual_T = state_trajs_np.shape[1] - 1  # Subtract 1 because it includes initial state
    time_steps = np.arange(actual_T + 1) * mpc.dT

    # Define successful dockings based on reach_fn
    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful_dockings = reach_values <= 0  # reach_fn <= 0 means within target region
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MPC Trajectory Analysis (T={actual_T}, dt={mpc.dT})', fontsize=16)
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_trajs))

    # 1. Position trajectories (px, py)
    ax1 = axes[0, 0]
    for i in range(n_trajs):
        px = state_trajs_np[i, :, 0]  # px
        py = state_trajs_np[i, :, 1]  # py
        success_label = "Success" if successful_dockings[i] else "Failed"
        ax1.plot(px, py, color=colors[i], alpha=0.7, linewidth=1.5, 
                label=f'IC {i+1} (cost: {costs_np[i]:.3f})')
        ax1.scatter(px[0], py[0], color=colors[i], s=25, marker='o')  # Start
        ax1.scatter(px[-1], py[-1], color=colors[i], s=25, marker='s')  # End
    
    # Add target region visualization based on reach_fn parameters
    # For Docking6D, the reach function defines a region with eps_p, eps_v, eps_theta, eps_omega tolerances
    eps_p = mpc.dynamics_.eps_p
    target_circle = patches.Circle((0, 0), eps_p, fill=False, color='green', linewidth=1, linestyle='--')
    ax1.add_patch(target_circle)
    
    # Add failure region (avoid_fn) visualization if available
    if hasattr(mpc.dynamics_, 'avoid_fn'):
        # Draw target spacecraft body with docking port cutout
        w_t = mpc.dynamics_.w_t
        h_t = mpc.dynamics_.h_t
        dock_rad = mpc.dynamics_.dock_rad
        
        draw_target_body(ax1, w_t, h_t, dock_rad, color='red', linewidth=2, alpha=0.7)
            
        ax1.set_xlabel('px (m)')
        ax1.set_ylabel('py (m)')
        ax1.set_title('Position Trajectories')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.axis('equal')
    
    # 2. Velocity trajectories (vx, vy)
    ax2 = axes[0, 1]
    for i in range(n_trajs):
        vx = state_trajs_np[i, :, 2]  # vx
        vy = state_trajs_np[i, :, 3]  # vy
        ax2.plot(vx, vy, color=colors[i], alpha=0.7, linewidth=2)
        ax2.scatter(vx[0], vy[0], color=colors[i], s=100, marker='o')
        ax2.scatter(vx[-1], vy[-1], color=colors[i], s=100, marker='s')
    
    ax2.set_xlabel('vx (m/s)')
    ax2.set_ylabel('vy (m/s)')
    ax2.set_title('Velocity Trajectories')
    ax2.grid(True, alpha=0.3)
    
    # 3. Attitude trajectories (theta, omega) - Fixed indexing
    ax3 = axes[0, 2]
    for i in range(n_trajs):
        theta = state_trajs_np[i, :, 4]  # theta
        ax3.plot(time_steps, theta, color=colors[i], alpha=0.7, linewidth=2, label=f'IC {i+1}')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('θ (rad)')
    ax3.set_title('Attitude Trajectories')
    ax3.grid(True, alpha=0.3)
    
    # 4. Angular velocity
    ax4 = axes[1, 0]
    for i in range(n_trajs):
        omega = state_trajs_np[i, :, 5]  # omega
        ax4.plot(time_steps, omega, color=colors[i], alpha=0.7, linewidth=2)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('ω (rad/s)')
    ax4.set_title('Angular Velocity Trajectories')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cost evolution
    ax5 = axes[1, 1]
    
    # Create bar chart showing reach values
    reach_bars = ax5.bar(range(n_trajs), reach_values, color=colors, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1, label='Success threshold')
    ax5.set_xlabel('Trajectory')
    ax5.set_ylabel('Reach Function Value')
    ax5.set_title('Docking Success Analysis')
    ax5.set_xticks(range(n_trajs))
    ax5.set_xticklabels([f'IC {i+1}' for i in range(n_trajs)])
    ax5.legend()
    
    # Add reach values on bars
    for i, (bar, reach_val) in enumerate(zip(reach_bars, reach_values)):
        height = bar.get_height()
        y_pos = height + 0.01 if height >= 0 else height - 0.01
        va = 'bottom' if height >= 0 else 'top'
        ax5.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{reach_val:.3f}', ha='center', va=va, fontsize=10)
    
    # 6. State magnitude evolution
    ax6 = axes[1, 2]
    for i in range(n_trajs):
        state_mag = np.linalg.norm(state_trajs_np[i, :, :2], axis=1)  # Position magnitude
        ax6.plot(time_steps, state_mag, color=colors[i], alpha=0.7, linewidth=2)
    
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('||position|| (m)')
    ax6.set_title('Distance to Target')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=eps_p, color='blue', linestyle='--', alpha=0.7, label=f'Target radius ({eps_p:.3f}m)')
    ax6.legend()
    
    plt.tight_layout()
    
    # Save the plot
    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__fullTraj.png", dpi=300, bbox_inches='tight')
    
    print("\n=== MPC Trajectories: Final State Analysis ===")
    print(f"{'Traj':<4} {'px_err':<8} {'py_err':<8} {'vx_err':<8} {'vy_err':<8} {'θ_err':<8} {'ω_err':<8} {'pos_dist':<9} {'vel_dist':<9} {'θ_dist':<8} {'ω_dist':<8} {'Success':<8}")
    print(f"{'#':<4} {'(m)':<8} {'(m)':<8} {'(m/s)':<8} {'(m/s)':<8} {'(rad)':<8} {'(rad/s)':<8} {'(m)':<9} {'(m/s)':<9} {'(rad)':<8} {'(rad/s)':<8} {'(Y/N)':<8}")
    print("-" * 105)

    goal_state = mpc.dynamics_.goal_state.to(mpc.device)

    for i in range(n_trajs):
        final_state = state_trajs_np[i, -1, :]
        px_f, py_f, vx_f, vy_f, theta_f, omega_f = final_state
        
        # Compute errors from goal state
        px_error = px_f - goal_state[0].item()
        py_error = py_f - goal_state[1].item()
        vx_error = vx_f - goal_state[2].item()
        vy_error = vy_f - goal_state[3].item()
        theta_error = theta_f - goal_state[4].item()
        omega_error = omega_f - goal_state[5].item()
        
        # Compute distances (magnitudes)
        pos_dist = np.sqrt(px_error**2 + py_error**2)
        vel_dist = np.sqrt(vx_error**2 + vy_error**2)
        theta_dist = abs(theta_error)
        omega_dist = abs(omega_error)
        
        success_str = "Y" if successful_dockings[i] else "N"
        print(f"{i+1:<4} {px_error:<8.3f} {py_error:<8.3f} {vx_error:<8.3f} {vy_error:<8.3f} {theta_error:<8.3f} {omega_error:<8.3f} "
            f"{pos_dist:<9.3f} {vel_dist:<9.3f} {theta_dist:<8.3f} {omega_dist:<8.3f} {success_str:<8}")

    print("-" * 105)
    print(f"Goal state: px={goal_state[0].item():.3f}, py={goal_state[1].item():.3f}, vx={goal_state[2].item():.3f}, vy={goal_state[3].item():.3f}, θ={goal_state[4].item():.3f}, ω={goal_state[5].item():.3f}")
    print(f"Tolerances: pos={mpc.dynamics_.eps_p:.3f}m, vel={mpc.dynamics_.eps_v:.3f}m/s, theta={mpc.dynamics_.eps_theta:.3f}rad, omega={mpc.dynamics_.eps_omega:.3f}rad/s")
    
    return fig, state_trajs_np, costs_np, successful_dockings, reach_values

def plotMPCControls(mpc, initial_conditions, T, save_def:pathlib, max_trajs=10):
    """
    Plot control inputs for MPC trajectories
    """
    import matplotlib.patches as patches
    
    # Limit number of trajectories for visualization clarity
    n_trajs = min(len(initial_conditions), max_trajs)
    ic_subset = initial_conditions[:n_trajs]
    
    # Get optimal trajectories
    costs, state_trajs, _, _ = mpc.get_batch_data(ic_subset, T)
    
    # Convert to numpy for plotting
    state_trajs_np = state_trajs.detach().cpu().numpy()
    costs_np = costs.detach().cpu().numpy()
    
    # Get actual trajectory length from the data
    actual_T = state_trajs_np.shape[1] - 1
    time_steps = np.arange(actual_T) * mpc.dT
    
    # Extract control inputs by computing them from state dynamics
    controls_np = np.zeros((n_trajs, actual_T, 3))  # [ux, uy, tau] for each trajectory
    
    for i in range(n_trajs):
        for t in range(actual_T):
            # Current state
            state_curr = torch.tensor(state_trajs_np[i, t, :]).to(mpc.device)
            state_next = torch.tensor(state_trajs_np[i, t+1, :]).to(mpc.device)
            
            # Compute control that would produce this state transition
            control = compute_control_from_transition(state_curr, state_next, mpc.dT, mpc.dynamics_)
            controls_np[i, t, :] = control.detach().cpu().numpy()
    
    # Define successful dockings based on reach_fn
    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful_dockings = reach_values <= 0
    
    # Create figure with subplots for different control components
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MPC Control Analysis (T={actual_T}, dt={mpc.dT})', fontsize=16)

    colors = plt.cm.tab20(np.linspace(0, 1, n_trajs))

    # 1. X-direction thrust (ux)
    ax1 = axes[0, 0]
    for i in range(n_trajs):
        ux = controls_np[i, :, 0]
        success_label = "Success" if successful_dockings[i] else "Failed"
        ax1.plot(time_steps, ux, color=colors[i], alpha=0.7, linewidth=2, 
                label=f'IC {i+1} ({success_label})')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('ux (N)')
    ax1.set_title('X-Direction Thrust')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add control limits if available
    if hasattr(mpc.dynamics_, 'u_bar'):
        u_max = mpc.dynamics_.u_bar
        ax1.axhline(y=u_max, color='black', linestyle='--', alpha=0.5, label=f'Max thrust: ±{u_max:.2f}N')
        ax1.axhline(y=-u_max, color='black', linestyle='--', alpha=0.5)
    
    # 2. Y-direction thrust (uy)
    ax2 = axes[0, 1]
    for i in range(n_trajs):
        uy = controls_np[i, :, 1]
        ax2.plot(time_steps, uy, color=colors[i], alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('uy (N)')
    ax2.set_title('Y-Direction Thrust')
    ax2.grid(True, alpha=0.3)
    
    # Add control limits
    if hasattr(mpc.dynamics_, 'u_bar'):
        u_max = mpc.dynamics_.u_bar
        ax2.axhline(y=u_max, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=-u_max, color='black', linestyle='--', alpha=0.5)
    
    # 3. Torque (tau)
    ax3 = axes[0, 2]
    for i in range(n_trajs):
        tau = controls_np[i, :, 2]
        ax3.plot(time_steps, tau, color=colors[i], alpha=0.7, linewidth=2)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('τ (N⋅m)')
    ax3.set_title('Torque')
    ax3.grid(True, alpha=0.3)
    
    # Add control limits
    if hasattr(mpc.dynamics_, 'u_theta_bar'):
        tau_max = mpc.dynamics_.u_theta_bar
        ax3.axhline(y=tau_max, color='black', linestyle='--', alpha=0.5)
        ax3.axhline(y=-tau_max, color='black', linestyle='--', alpha=0.5)
    
    # 4. Control magnitude over time
    ax4 = axes[1, 0]
    for i in range(n_trajs):
        thrust_mag = np.sqrt(controls_np[i, :, 0]**2 + controls_np[i, :, 1]**2)
        ax4.plot(time_steps, thrust_mag, color=colors[i], alpha=0.7, linewidth=2)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('||thrust|| (N)')
    ax4.set_title('Thrust Magnitude')
    ax4.grid(True, alpha=0.3)
    
    # 5. Control effort (cumulative)
    ax5 = axes[1, 1]
    for i in range(n_trajs):
        # Compute cumulative control effort
        thrust_effort = np.cumsum(np.sqrt(controls_np[i, :, 0]**2 + controls_np[i, :, 1]**2) * mpc.dT)
        ax5.plot(time_steps, thrust_effort, color=colors[i], alpha=0.7, linewidth=2)
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Cumulative Thrust Effort (N⋅s)')
    ax5.set_title('Control Effort Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Control phase plot (ux vs uy)
    ax6 = axes[1, 2]
    for i in range(n_trajs):
        ux = controls_np[i, :, 0]
        uy = controls_np[i, :, 1]
        ax6.plot(ux, uy, color=colors[i], alpha=0.7, linewidth=2)
        ax6.scatter(ux[0], uy[0], color=colors[i], s=100, marker='o')  # Start
        ax6.scatter(ux[-1], uy[-1], color=colors[i], s=100, marker='s')  # End
    
    ax6.set_xlabel('ux (N)')
    ax6.set_ylabel('uy (N)')
    ax6.set_title('Thrust Vector Phase Plot')
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    # Add control constraint circle if available
    if hasattr(mpc.dynamics_, 'u_bar'):
        u_max = mpc.dynamics_.u_bar
        constraint_circle = patches.Circle((0, 0), u_max, fill=False, color='black', 
                                         linewidth=2, linestyle='--', alpha=0.5)
        ax6.add_patch(constraint_circle)
    
    plt.tight_layout()
    
    # Save the plot
    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__control.png", dpi=300, bbox_inches='tight')

    return fig, controls_np, successful_dockings

def compute_control_from_transition(state_curr, state_next, dt, dynamics):
    """
    Compute the control input that would produce the given state transition
    This is an approximation based on the dynamics model
    """
    # For Docking6D dynamics, we need to solve for control given state transition
    # state = [px, py, vx, vy, theta, omega]
    # dynamics: d/dt [px, py, vx, vy, theta, omega] = f(state, control)
    
    # Extract current states
    px, py, vx, vy, theta, omega = state_curr
    px_next, py_next, vx_next, vy_next, theta_next, omega_next = state_next
    
    # From Clohessy-Wiltshire dynamics:
    # dvx/dt = 3*n^2*px + 2*n*vy + ux/mc
    # dvy/dt = -2*n*vx + uy/mc
    # domega/dt = tau/jc
    
    n = dynamics.n  # orbital rate (this should call the method)
    mc = dynamics.mc  # chaser mass
    jc = dynamics.jc  # chaser moment of inertia
    
    # Compute accelerations from state differences
    ax = (vx_next - vx) / dt
    ay = (vy_next - vy) / dt
    alpha = (omega_next - omega) / dt
    
    # Solve for control inputs based on the Docking6D dsdt method:
    # dsdt[..., 2] = 3 * self.mean_motion()**2 * state[..., 0] + 2 * self.mean_motion() * state[..., 3] + control[..., 0] / self.mc
    # dsdt[..., 3] = -2 * self.mean_motion() * state[..., 0] + control[..., 1] / self.mc
    # dsdt[..., 5] = control[..., 2] / self.moment_of_inertia()
    
    # Rearranging for controls:
    # ux = mc * (ax - 3*n^2*px - 2*n*vy)
    # uy = mc * (ay + 2*n*vx)
    # tau = jc * alpha
    
    ux = mc * (ax - 3*n**2*px - 2*n*vy)
    uy = mc * (ay + 2*n*vx)
    tau = jc * alpha
    
    return torch.tensor([ux, uy, tau])

def compute_all_brt_slices(mpc, dynamics_, device, T, resolution):
    """Compute BRT for all state space slices once"""
    plot_config = dynamics_.plot_config()
    state_test_range = dynamics_.state_test_range()
    
    def compute_brt_batch(initial_conditions, slice_name):
        """Helper function to compute BRT in batches"""
        print(f"Computing {slice_name} BRT...")
        costs = []
        for i in tqdm(range(4)):
            batch_size = len(initial_conditions) // 4
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size if i < 3 else len(initial_conditions)
            costs_batch, _, _, _ = mpc.get_batch_data(initial_conditions[start_idx:end_idx], T)
            costs.append(costs_batch)
        return torch.cat(costs, dim=0)
    
    # Position slice (your existing computation)
    x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    xs = torch.linspace(x_min, x_max, resolution[0])
    ys = torch.linspace(y_min, y_max, resolution[1])
    xys = torch.cartesian_prod(xs, ys).to(device)

    position_ics = torch.zeros(resolution[0] * resolution[1], dynamics_.state_dim).to(device)
    position_ics[:, :] = torch.tensor(plot_config['state_slices']).to(device)
    position_ics[:, plot_config['x_axis_idx']] = xys[:, 0]
    position_ics[:, plot_config['y_axis_idx']] = xys[:, 1]
    position_costs = compute_brt_batch(position_ics, "position")
    
    # Velocity slice
    vx_min, vx_max = state_test_range[2]  # vx is always index 2
    vy_min, vy_max = state_test_range[3]  # vy is always index 3
    vxs = torch.linspace(vx_min, vx_max, resolution[2])
    vys = torch.linspace(vy_min, vy_max, resolution[3])
    vxvys = torch.cartesian_prod(vxs, vys).to(device)

    velocity_ics = torch.zeros(resolution[2] * resolution[3], dynamics_.state_dim).to(device)
    velocity_ics[:, :] = torch.tensor(plot_config['state_slices']).to(device)
    velocity_ics[:, 2] = vxvys[:, 0]  # vx
    velocity_ics[:, 3] = vxvys[:, 1]  # vy
    velocity_costs = compute_brt_batch(velocity_ics, "velocity")
    
    # Rotation slice
    theta_min, theta_max = state_test_range[4]  # theta is always index 4
    omega_min, omega_max = state_test_range[5]  # omega is always index 5
    thetas = torch.linspace(theta_min, theta_max, resolution[4])
    omegas = torch.linspace(omega_min, omega_max, resolution[5])
    theta_omegas = torch.cartesian_prod(thetas, omegas).to(device)

    rotation_ics = torch.zeros(resolution[4] * resolution[5], dynamics_.state_dim).to(device)
    rotation_ics[:, :] = torch.tensor(plot_config['state_slices']).to(device)
    rotation_ics[:, 4] = theta_omegas[:, 0]  # theta
    rotation_ics[:, 5] = theta_omegas[:, 1]  # omega
    rotation_costs = compute_brt_batch(rotation_ics, "rotation")
    
    return {
        'position': {
            'costs': position_costs,
            'coords': (x_min, x_max, y_min, y_max),
            'initial_conditions': position_ics
        },
        'velocity': {
            'costs': velocity_costs,
            'coords': (vx_min, vx_max, vy_min, vy_max),
            'initial_conditions': velocity_ics
        },
        'rotation': {
            'costs': rotation_costs,
            'coords': (theta_min, theta_max, omega_min, omega_max),
            'initial_conditions': rotation_ics
        }
    }

def plotBRTPosition(mpc, interesting_ics_tensor, brt_data, x_resolution, y_resolution, 
                    T, save_def:pathlib, level_sets=[0.0, 0.3, 0.6]):
    import matplotlib.patches as patches
    
    # Get pre-computed position BRT data
    position_costs = brt_data['position']['costs']
    x_min, x_max, y_min, y_max = brt_data['position']['coords']
    initial_condition_tensor = brt_data['position']['initial_conditions']
    
    # Get trajectories for interesting initial conditions
    costs, state_trajs, _, _ = mpc.get_batch_data(interesting_ics_tensor, T)
    state_trajs_np = state_trajs.detach().cpu().numpy()
    
    # Evaluate reach function for success determination
    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful_dockings = reach_values <= 0

    print("\n=== Position Plot: Final State Analysis ===")
    print(f"{'Traj':<4} {'px_err':<8} {'py_err':<8} {'vx_err':<8} {'vy_err':<8} {'θ_err':<8} {'ω_err':<8} {'pos_dist':<9} {'vel_dist':<9} {'θ_dist':<8} {'ω_dist':<8} {'Success':<8}")
    print(f"{'#':<4} {'(m)':<8} {'(m)':<8} {'(m/s)':<8} {'(m/s)':<8} {'(rad)':<8} {'(rad/s)':<8} {'(m)':<9} {'(m/s)':<9} {'(rad)':<8} {'(rad/s)':<8} {'(Y/N)':<8}")
    print("-" * 105)

    n_trajs = len(interesting_ics_tensor)
    goal_state = mpc.dynamics_.goal_state.to(mpc.device)

    for i in range(n_trajs):
        final_state = state_trajs_np[i, -1, :]
        px_f, py_f, vx_f, vy_f, theta_f, omega_f = final_state
        
        # Compute errors from goal state
        px_error = px_f - goal_state[0].item()
        py_error = py_f - goal_state[1].item()
        vx_error = vx_f - goal_state[2].item()
        vy_error = vy_f - goal_state[3].item()
        theta_error = theta_f - goal_state[4].item()
        omega_error = omega_f - goal_state[5].item()
        
        # Compute distances (magnitudes)
        pos_dist = np.sqrt(px_error**2 + py_error**2)
        vel_dist = np.sqrt(vx_error**2 + vy_error**2)
        theta_dist = abs(theta_error)
        omega_dist = abs(omega_error)
        
        success_str = "Y" if successful_dockings[i] else "N"
        print(f"{i+1:<4} {px_error:<8.3f} {py_error:<8.3f} {vx_error:<8.3f} {vy_error:<8.3f} {theta_error:<8.3f} {omega_error:<8.3f} "
            f"{pos_dist:<9.3f} {vel_dist:<9.3f} {theta_dist:<8.3f} {omega_dist:<8.3f} {success_str:<8}")

    print("-" * 105)
    print(f"Goal state: px={goal_state[0].item():.3f}, py={goal_state[1].item():.3f}, vx={goal_state[2].item():.3f}, vy={goal_state[3].item():.3f}, θ={goal_state[4].item():.3f}, ω={goal_state[5].item():.3f}")
    print(f"Tolerances: pos={mpc.dynamics_.eps_p:.3f}m, vel={mpc.dynamics_.eps_v:.3f}m/s, theta={mpc.dynamics_.eps_theta:.3f}rad, omega={mpc.dynamics_.eps_omega:.3f}rad/s")

    # Prepare BRT data using pre-computed position costs
    BRT_img = position_costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.amax(BRT_img[~np.isnan(BRT_img)])
    min_value = np.amin(BRT_img[~np.isnan(BRT_img)])
    
    # Create coordinate grids for contour plotting
    x_coords = np.linspace(x_min, x_max, x_resolution)
    y_coords = np.linspace(y_min, y_max, y_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create the combined plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 1. Plot BRT heatmap as background
    imshow_kwargs = {
        'vmax': max_value,
        'vmin': min_value,
        'cmap': 'RdYlBu',
        'extent': (x_min, x_max, y_min, y_max),
        'origin': 'lower',
        'aspect': 'equal',
        'alpha': 0.7
    }
    im = ax.imshow(BRT_img, **imshow_kwargs)
    
    # 2. Add level set contours
    n_trajs = len(interesting_ics_tensor)
    level_colors = ['purple', 'blue', 'orange', 'cyan', 'magenta', 'yellow']
    level_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    level_set_handles = []
    for i, level in enumerate(level_sets):
        color = level_colors[i % len(level_colors)]
        style = level_styles[i % len(level_styles)]
        
        ax.contour(X, Y, BRT_img, levels=[level], colors=[color], 
                   linewidths=1.5, linestyles=style)
        
        level_set_handles.append(
            mpatches.Patch(color=color, label=f'Level {level}')
        )
    
    # 3. Plot MPC trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, n_trajs))
    traj_handles = []
    
    for i in range(n_trajs):
        px = state_trajs_np[i, :, 0]  # px trajectory
        py = state_trajs_np[i, :, 1]  # py trajectory

        success_label = "Success" if successful_dockings[i] else "Failed"

        line, = ax.plot(px, py, color=colors[i], linewidth=1.5, alpha=0.9, 
               label=f'IC {i+1} ({success_label})')
        traj_handles.append(line)
        
        # Add start and end markers
        ax.scatter(px[0], py[0], color=colors[i], s=25, marker='o', 
              edgecolors='black', linewidth=1, zorder=10)
        ax.scatter(px[-1], py[-1], color=colors[i], s=25, marker='s', 
              edgecolors='black', linewidth=1, zorder=10)
        
        # Add trajectory direction arrows
        if len(px) > 1:
            for frac in [0.25, 0.5, 0.75]:
                idx = int(frac * (len(px) - 1))
                if idx < len(px) - 1:
                    dx = px[idx+1] - px[idx]
                    dy = py[idx+1] - py[idx]
                    ax.arrow(px[idx], py[idx], dx*0.3, dy*0.3, 
                           head_width=0.05, head_length=0.03, 
                           fc=colors[i], ec=colors[i], alpha=0.7)
    
    # 4. Add target region visualization
    reach_value = mpc.dynamics_.reach_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = mpc.dynamics_.avoid_fn(initial_condition_tensor).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    ax.contour(X, Y, reach_value, levels=[0.0], colors='green', linewidths=1.5, linestyles='-')
    ax.contour(X, Y, avoid_value, levels=[0.0], colors='red', linewidths=1.5, linestyles='-.')

    reach_avoid_handles = [
        mpatches.Patch(color='green', label='Reach Set'),
        mpatches.Patch(color='red', label='Avoid Set')
    ]

    # Define body of target spacecraft
    if hasattr(mpc.dynamics_, 'avoid_fn'):
        w_t = mpc.dynamics_.w_t
        h_t = mpc.dynamics_.h_t
        dock_rad = mpc.dynamics_.dock_rad
        
        draw_target_body(ax, w_t, h_t, dock_rad, color='red', linewidth=2, alpha=0.7)
        reach_avoid_handles.append(
            mpatches.Patch(facecolor='none', edgecolor='black', label='Body of target')
        )

    # 5. Formatting and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('px (m)', fontsize=14)
    ax.set_ylabel('py (m)', fontsize=14)
    ax.set_title('MPC Trajectories Overlaid on Position BRT Heatmap', fontsize=16)
    ax.grid(True, alpha=0.3)

    all_handles = reach_avoid_handles + level_set_handles + traj_handles

    # Add colorbar for BRT values
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('BRT Value', fontsize=12)

    ax.legend(handles=all_handles, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__traj_position.png", dpi=300, bbox_inches='tight')
    
    return fig, state_trajs_np, successful_dockings

def plotBRTVelocity(mpc, interesting_ics_tensor, brt_data, x_resolution, y_resolution, 
                    T, save_def:pathlib, level_sets=[0.0, 0.3, 0.6]):
    import matplotlib.patches as patches
    
    # Get pre-computed velocity BRT data
    velocity_costs = brt_data['velocity']['costs']
    vx_min, vx_max, vy_min, vy_max = brt_data['velocity']['coords']
    velocity_initial_conditions = brt_data['velocity']['initial_conditions']
    
    # Get trajectories for interesting initial conditions
    costs, state_trajs, _, _ = mpc.get_batch_data(interesting_ics_tensor, T)
    state_trajs_np = state_trajs.detach().cpu().numpy()
    
    # Evaluate reach function for success determination
    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful_dockings = reach_values <= 0

    # Print final state analysis table for velocity plot
    print("\n=== Velocity Plot: Final State Analysis ===")
    print(f"{'Traj':<4} {'px_err':<8} {'py_err':<8} {'vx_err':<8} {'vy_err':<8} {'θ_err':<8} {'ω_err':<8} {'pos_dist':<9} {'vel_dist':<9} {'θ_dist':<8} {'ω_dist':<8} {'Success':<8}")
    print(f"{'#':<4} {'(m)':<8} {'(m)':<8} {'(m/s)':<8} {'(m/s)':<8} {'(rad)':<8} {'(rad/s)':<8} {'(m)':<9} {'(m/s)':<9} {'(rad)':<8} {'(rad/s)':<8} {'(Y/N)':<8}")
    print("-" * 105)

    n_trajs = len(interesting_ics_tensor)
    goal_state = mpc.dynamics_.goal_state.to(mpc.device)

    for i in range(n_trajs):
        final_state = state_trajs_np[i, -1, :]
        px_f, py_f, vx_f, vy_f, theta_f, omega_f = final_state
        
        # Compute errors from goal state
        px_error = px_f - goal_state[0].item()
        py_error = py_f - goal_state[1].item()
        vx_error = vx_f - goal_state[2].item()
        vy_error = vy_f - goal_state[3].item()
        theta_error = theta_f - goal_state[4].item()
        omega_error = omega_f - goal_state[5].item()
        
        # Compute distances (magnitudes)
        pos_dist = np.sqrt(px_error**2 + py_error**2)
        vel_dist = np.sqrt(vx_error**2 + vy_error**2)
        theta_dist = abs(theta_error)
        omega_dist = abs(omega_error)
        
        success_str = "Y" if successful_dockings[i] else "N"
        print(f"{i+1:<4} {px_error:<8.3f} {py_error:<8.3f} {vx_error:<8.3f} {vy_error:<8.3f} {theta_error:<8.3f} {omega_error:<8.3f} "
            f"{pos_dist:<9.3f} {vel_dist:<9.3f} {theta_dist:<8.3f} {omega_dist:<8.3f} {success_str:<8}")

    print("-" * 105)
    print(f"Goal state: px={goal_state[0].item():.3f}, py={goal_state[1].item():.3f}, vx={goal_state[2].item():.3f}, vy={goal_state[3].item():.3f}, θ={goal_state[4].item():.3f}, ω={goal_state[5].item():.3f}")
    print(f"Tolerances: pos={mpc.dynamics_.eps_p:.3f}m, vel={mpc.dynamics_.eps_v:.3f}m/s, theta={mpc.dynamics_.eps_theta:.3f}rad, omega={mpc.dynamics_.eps_omega:.3f}rad/s")

    # Prepare BRT data for velocity slice
    BRT_img = velocity_costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.amax(BRT_img[~np.isnan(BRT_img)])
    min_value = np.amin(BRT_img[~np.isnan(BRT_img)])
    
    # Create coordinate grids for contour plotting
    vx_coords_np = np.linspace(vx_min, vx_max, x_resolution)
    vy_coords_np = np.linspace(vy_min, vy_max, y_resolution)
    VX, VY = np.meshgrid(vx_coords_np, vy_coords_np)
    
    # Create the combined plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 1. Plot BRT heatmap for velocity slice
    imshow_kwargs = {
        'vmax': max_value,
        'vmin': min_value,
        'cmap': 'RdYlBu',
        'extent': (vx_min, vx_max, vy_min, vy_max),
        'origin': 'lower',
        'aspect': 'equal',
        'alpha': 0.7
    }
    im = ax.imshow(BRT_img, **imshow_kwargs)
    
    # 2. Add level set contours
    n_trajs = len(interesting_ics_tensor)
    level_colors = ['purple', 'blue', 'orange', 'cyan', 'magenta', 'yellow']
    level_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    level_set_handles = []
    for i, level in enumerate(level_sets):
        color = level_colors[i % len(level_colors)]
        style = level_styles[i % len(level_styles)]
        
        ax.contour(VX, VY, BRT_img, levels=[level], colors=[color], 
                   linewidths=1.5, linestyles=style)
        
        level_set_handles.append(
            mpatches.Patch(color=color, label=f'Level {level}')
        )
    
    # 3. Plot MPC trajectories in velocity space
    colors = plt.cm.tab20(np.linspace(0, 1, n_trajs))
    traj_handles = []
    
    for i in range(n_trajs):
        vx = state_trajs_np[i, :, 2]  # vx trajectory
        vy = state_trajs_np[i, :, 3]  # vy trajectory

        success_label = "Success" if successful_dockings[i] else "Failed"

        line, = ax.plot(vx, vy, color=colors[i], linewidth=1.5, alpha=0.9, 
               label=f'IC {i+1} ({success_label})')
        traj_handles.append(line)
        
        # Add start and end markers
        ax.scatter(vx[0], vy[0], color=colors[i], s=25, marker='o', 
              edgecolors='black', linewidth=1, zorder=10)
        ax.scatter(vx[-1], vy[-1], color=colors[i], s=25, marker='s', 
              edgecolors='black', linewidth=1, zorder=10)
        
        # Add trajectory direction arrows
        if len(vx) > 1:
            for frac in [0.25, 0.5, 0.75]:
                idx = int(frac * (len(vx) - 1))
                if idx < len(vx) - 1:
                    dvx = vx[idx+1] - vx[idx]
                    dvy = vy[idx+1] - vy[idx]
                    arrow_scale = 0.1
                    ax.arrow(vx[idx], vy[idx], dvx*arrow_scale, dvy*arrow_scale, 
                           head_width=0.02, head_length=0.01, 
                           fc=colors[i], ec=colors[i], alpha=0.7)
    
    # 4. Add target region visualization using actual reach_fn
    reach_value = mpc.dynamics_.reach_fn(velocity_initial_conditions).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = mpc.dynamics_.avoid_fn(velocity_initial_conditions).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    ax.contour(VX, VY, reach_value, levels=[0.0], colors='green', linewidths=2, linestyles='-')
    ax.contour(VX, VY, avoid_value, levels=[0.0], colors='red', linewidths=2, linestyles='-.')

    reach_avoid_handles = [
        mpatches.Patch(color='green', label='Reach Set (velocity slice)'),
        mpatches.Patch(color='red', label='Avoid Set (velocity slice)')
    ]

    # 5. Formatting and labels
    ax.set_xlim(vx_min, vx_max)
    ax.set_ylim(vy_min, vy_max)
    ax.set_xlabel('vx (m/s)', fontsize=14)
    ax.set_ylabel('vy (m/s)', fontsize=14)
    ax.set_title('MPC Velocity Trajectories on Velocity BRT Slice', fontsize=16)
    ax.grid(True, alpha=0.3)

    all_handles = reach_avoid_handles + level_set_handles + traj_handles

    # Add colorbar for BRT values
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('BRT Value', fontsize=12)

    ax.legend(handles=all_handles, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__traj_velocity.png", dpi=300, bbox_inches='tight')

    return fig, state_trajs_np, successful_dockings

def plotBRTRotation(mpc, interesting_ics_tensor, brt_data, x_resolution, y_resolution, 
                    T, save_def:pathlib, level_sets=[0.0, 0.3, 0.6]):
    import matplotlib.patches as patches
    
    # Get pre-computed rotation BRT data
    rotation_costs = brt_data['rotation']['costs']
    theta_min, theta_max, omega_min, omega_max = brt_data['rotation']['coords']
    rotation_initial_conditions = brt_data['rotation']['initial_conditions']
    
    # Get trajectories for interesting initial conditions
    costs, state_trajs, _, _ = mpc.get_batch_data(interesting_ics_tensor, T)
    state_trajs_np = state_trajs.detach().cpu().numpy()
    
    # Evaluate reach function for success determination
    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful_dockings = reach_values <= 0

    # Print final state analysis table for rotation plot
    print("\n=== Rotation Plot: Final State Analysis ===")
    print(f"{'Traj':<4} {'px_err':<8} {'py_err':<8} {'vx_err':<8} {'vy_err':<8} {'θ_err':<8} {'ω_err':<8} {'pos_dist':<9} {'vel_dist':<9} {'θ_dist':<8} {'ω_dist':<8} {'Success':<8}")
    print(f"{'#':<4} {'(m)':<8} {'(m)':<8} {'(m/s)':<8} {'(m/s)':<8} {'(rad)':<8} {'(rad/s)':<8} {'(m)':<9} {'(m/s)':<9} {'(rad)':<8} {'(rad/s)':<8} {'(Y/N)':<8}")
    print("-" * 105)

    n_trajs = len(interesting_ics_tensor)
    goal_state = mpc.dynamics_.goal_state.to(mpc.device)

    for i in range(n_trajs):
        final_state = state_trajs_np[i, -1, :]
        px_f, py_f, vx_f, vy_f, theta_f, omega_f = final_state
        
        # Compute errors from goal state
        px_error = px_f - goal_state[0].item()
        py_error = py_f - goal_state[1].item()
        vx_error = vx_f - goal_state[2].item()
        vy_error = vy_f - goal_state[3].item()
        theta_error = theta_f - goal_state[4].item()
        omega_error = omega_f - goal_state[5].item()
        
        # Compute distances (magnitudes)
        pos_dist = np.sqrt(px_error**2 + py_error**2)
        vel_dist = np.sqrt(vx_error**2 + vy_error**2)
        theta_dist = abs(theta_error)
        omega_dist = abs(omega_error)
        
        success_str = "Y" if successful_dockings[i] else "N"
        print(f"{i+1:<4} {px_error:<8.3f} {py_error:<8.3f} {vx_error:<8.3f} {vy_error:<8.3f} {theta_error:<8.3f} {omega_error:<8.3f} "
            f"{pos_dist:<9.3f} {vel_dist:<9.3f} {theta_dist:<8.3f} {omega_dist:<8.3f} {success_str:<8}")

    print("-" * 105)
    print(f"Goal state: px={goal_state[0].item():.3f}, py={goal_state[1].item():.3f}, vx={goal_state[2].item():.3f}, vy={goal_state[3].item():.3f}, θ={goal_state[4].item():.3f}, ω={goal_state[5].item():.3f}")
    print(f"Tolerances: pos={mpc.dynamics_.eps_p:.3f}m, vel={mpc.dynamics_.eps_v:.3f}m/s, theta={mpc.dynamics_.eps_theta:.3f}rad, omega={mpc.dynamics_.eps_omega:.3f}rad/s")

    # Prepare BRT data for rotation slice
    BRT_img = rotation_costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.amax(BRT_img[~np.isnan(BRT_img)])
    min_value = np.amin(BRT_img[~np.isnan(BRT_img)])
    
    # Create coordinate grids for contour plotting
    theta_coords_np = np.linspace(theta_min, theta_max, x_resolution)
    omega_coords_np = np.linspace(omega_min, omega_max, y_resolution)
    THETA, OMEGA = np.meshgrid(theta_coords_np, omega_coords_np)
    
    # Create the combined plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 1. Plot BRT heatmap for rotation slice
    imshow_kwargs = {
        'vmax': max_value,
        'vmin': min_value,
        'cmap': 'RdYlBu',
        'extent': (theta_min, theta_max, omega_min, omega_max),
        'origin': 'lower',
        'aspect': 'equal',
        'alpha': 0.7
    }
    im = ax.imshow(BRT_img, **imshow_kwargs)
    
    # 2. Add level set contours
    n_trajs = len(interesting_ics_tensor)
    level_colors = ['purple', 'blue', 'orange', 'cyan', 'magenta', 'yellow']
    level_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    level_set_handles = []
    for i, level in enumerate(level_sets):
        color = level_colors[i % len(level_colors)]
        style = level_styles[i % len(level_styles)]
        
        ax.contour(THETA, OMEGA, BRT_img, levels=[level], colors=[color], 
                   linewidths=1.5, linestyles=style)
        
        level_set_handles.append(
            mpatches.Patch(color=color, label=f'Level {level}')
        )
    
    # 3. Plot MPC trajectories in rotation space
    colors = plt.cm.tab20(np.linspace(0, 1, n_trajs))
    traj_handles = []
    
    for i in range(n_trajs):
        theta = state_trajs_np[i, :, 4]  # theta trajectory
        omega = state_trajs_np[i, :, 5]  # omega trajectory

        success_label = "Success" if successful_dockings[i] else "Failed"

        line, = ax.plot(theta, omega, color=colors[i], linewidth=1.5, alpha=0.9, 
               label=f'IC {i+1} ({success_label})')
        traj_handles.append(line)
        
        # Add start and end markers
        ax.scatter(theta[0], omega[0], color=colors[i], s=25, marker='o', 
              edgecolors='black', linewidth=1, zorder=10)
        ax.scatter(theta[-1], omega[-1], color=colors[i], s=25, marker='s', 
              edgecolors='black', linewidth=1, zorder=10)
        
        # Add trajectory direction arrows
        if len(theta) > 1:
            for frac in [0.25, 0.5, 0.75]:
                idx = int(frac * (len(theta) - 1))
                if idx < len(theta) - 1:
                    dtheta = theta[idx+1] - theta[idx]
                    domega = omega[idx+1] - omega[idx]
                    arrow_scale = 0.3
                    ax.arrow(theta[idx], omega[idx], dtheta*arrow_scale, domega*arrow_scale, 
                           head_width=0.1, head_length=0.05, 
                           fc=colors[i], ec=colors[i], alpha=0.7)
    
    # 4. Add target region visualization using actual reach_fn
    reach_value = mpc.dynamics_.reach_fn(rotation_initial_conditions).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    avoid_value = mpc.dynamics_.avoid_fn(rotation_initial_conditions).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    ax.contour(THETA, OMEGA, reach_value, levels=[0.0], colors='green', linewidths=2, linestyles='-')
    ax.contour(THETA, OMEGA, avoid_value, levels=[0.0], colors='red', linewidths=2, linestyles='-.')

    reach_avoid_handles = [
        mpatches.Patch(color='green', label='Reach Set (rotation slice)'),
        mpatches.Patch(color='red', label='Avoid Set (rotation slice)')
    ]

    # 5. Formatting and labels
    ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(omega_min, omega_max)
    ax.set_xlabel('θ (rad)', fontsize=14)
    ax.set_ylabel('ω (rad/s)', fontsize=14)
    ax.set_title('MPC Rotational Trajectories on Rotation BRT Slice', fontsize=16)
    ax.grid(True, alpha=0.3)

    all_handles = reach_avoid_handles + level_set_handles + traj_handles

    # Add colorbar for BRT values
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('BRT Value', fontsize=12)

    ax.legend(handles=all_handles, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    save_def.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_def}__traj_rotation.png", dpi=300, bbox_inches='tight')

    return fig, state_trajs_np, successful_dockings
