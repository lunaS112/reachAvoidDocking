import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patches as mpatches
import torch
from tqdm import tqdm
from dynamics import dynamics
from utils import MPC, modules
import math

# mpl.use('Agg')
torch.manual_seed(1)
np.random.seed(1)

ROLLOUT_NUM = 100

def plotBRTImages(costs, x_resolution, y_resolution,x_min, x_max, y_min, y_max):
    fig = plt.figure(figsize=(6, 6))
    fig2 = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1 )
    
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
        'aspect': 'auto'
    }
    ax.imshow(greys)
    s1=ax.imshow(BRT_img, **imshow_kwargs)

    # Define secondary level set 
    level = 0.1

    # Add level set contours
    contour_0 = ax.contour(X, Y, BRT_img, levels=[0.0], colors='black', linewidths=1.5, linestyles='-')
    contour_1 = ax.contour(X, Y, BRT_img, levels=[level], colors='red', linewidths=1.5, linestyles='--')

    # Add labels for the contours
    ax.clabel(contour_0, inline=True, fontsize=10, fmt='0-level')
    ax.clabel(contour_1, inline=True, fontsize=10, fmt=f'{level}-level')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(s1)
    ax.set_title('BRT Heatmap with Level Sets')
    ax.set_xlabel('px (m)')
    ax.set_ylabel('py (m)')
    
    # Second plot - binary reachability
    ax2 = fig2.add_subplot(1, 1, 1)
    binary_img = ax2.imshow(1*(BRT_img <= 0), cmap='bwr',
                origin='lower', extent=(x_min, x_max, y_min, y_max),
                aspect='auto')
    
    # Add 0-level contour to binary plot as well
    ax2.contour(X, Y, BRT_img, levels=[0.0], colors='black', linewidths=1.5, linestyles='-')
    
    ax2.set_title('Binary Reachability (BRT ≤ 0)')
    ax2.set_xlabel('px (m)')
    ax2.set_ylabel('py (m)')
    
    fig.savefig("./data/heatmapClassicMPC.png", dpi=300, bbox_inches='tight')
    fig2.savefig("./data/BRTClassicMPC.png", dpi=300, bbox_inches='tight')

def plotMPCTrajectories(mpc, initial_conditions, T, max_trajs=10, save_animation=False):
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
    
    print(f"State trajectories shape: {state_trajs_np.shape}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MPC Trajectory Analysis (T={actual_T}, dt={mpc.dT})', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_trajs))
    
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
        
        # Draw the main rectangular body
        body_rect = patches.Rectangle((-w_t/2, 0), w_t, h_t, 
                                    fill=False, color='red', linewidth=2, alpha=0.7, label='Avoid region')
        ax1.add_patch(body_rect)
        
        # Draw only the upper semicircle of the docking port (cutout)
        theta = np.linspace(0, np.pi, 100)  # Upper semicircle from 0 to π
        dock_x = dock_rad * np.cos(theta)
        dock_y = dock_rad * np.sin(theta)
        
        # Plot the semicircular cutout
        ax1.plot(dock_x, dock_y, color='red', linewidth=2, alpha=0.7)
        
        # Connect the ends of the semicircle to show the opening
        ax1.plot([-dock_rad, dock_rad], [0, 0], color='red', linewidth=2, alpha=0.7)
        
        # Optional: Add vertical lines connecting the rectangle to the docking port
        # This helps visualize that it's a cutout, not separate shapes
        if dock_rad < w_t/2:  # Only if docking port is smaller than spacecraft width
            ax1.plot([-dock_rad, -dock_rad], [0, 0], color='red', linewidth=2, alpha=0.7)
            ax1.plot([dock_rad, dock_rad], [0, 0], color='red', linewidth=2, alpha=0.7)
            
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
    import os
    os.makedirs('./data', exist_ok=True)
    fig.savefig("./data/mpc_trajectoriesClassicMPC.png", dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print("\n=== MPC Trajectory Summary ===")
    print(f"Number of trajectories: {n_trajs}")
    print(f"Time duration: {actual_T*mpc.dT:.1f} seconds")
    print(f"Cost range: [{costs_np.min():.3f}, {costs_np.max():.3f}]")
    print(f"Reach value range: [{reach_values.min():.3f}, {reach_values.max():.3f}]")
    print(f"Successful dockings (reach_fn ≤ 0): {np.sum(successful_dockings)}/{n_trajs}")
    
    # Analyze final states in terms of each component of reach_fn
    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    px = final_states_tensor[:, 0].cpu().numpy()
    py = final_states_tensor[:, 1].cpu().numpy()
    vx = final_states_tensor[:, 2].cpu().numpy()
    vy = final_states_tensor[:, 3].cpu().numpy()
    theta = final_states_tensor[:, 4].cpu().numpy()
    omega = final_states_tensor[:, 5].cpu().numpy()
    
    position_dist = np.sqrt(px**2 + py**2)
    velocity_dist = np.sqrt(vx**2 + vy**2)
    theta_dist = np.abs(theta - np.pi/2)  # Assuming target attitude is π/2
    omega_dist = np.abs(omega)
    
    print(f"Final position distances: [{position_dist.min():.3f}, {position_dist.max():.3f}] m (tolerance: {mpc.dynamics_.eps_p:.3f})")
    print(f"Final velocity magnitudes: [{velocity_dist.min():.3f}, {velocity_dist.max():.3f}] m/s (tolerance: {mpc.dynamics_.eps_v:.3f})")
    print(f"Final theta errors: [{theta_dist.min():.3f}, {theta_dist.max():.3f}] rad (tolerance: {mpc.dynamics_.eps_theta:.3f})")
    print(f"Final omega magnitudes: [{omega_dist.min():.3f}, {omega_dist.max():.3f}] rad/s (tolerance: {mpc.dynamics_.eps_omega:.3f})")
    
    return fig, state_trajs_np, costs_np, successful_dockings, reach_values

def plotTrajectoryOverlay(mpc, interesting_ics_tensor, T, costs_grid, x_resolution, y_resolution, 
                         x_min, x_max, y_min, y_max, level_sets=[0.0, 0.3, 0.6]):
    import matplotlib.patches as patches
    
    # Get trajectories for all interesting initial conditions
    costs, state_trajs, _, _ = mpc.get_batch_data(interesting_ics_tensor, T)
    state_trajs_np = state_trajs.detach().cpu().numpy()
    costs_np = costs.detach().cpu().numpy()
    
    # Evaluate reach function for success determination
    final_states_tensor = torch.tensor(state_trajs_np[:, -1, :]).to(mpc.device)
    reach_values = mpc.dynamics_.reach_fn(final_states_tensor).detach().cpu().numpy()
    successful_dockings = reach_values <= 0
    
    # Prepare BRT data
    BRT_img = costs_grid.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
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
        'alpha': 0.7  # Make heatmap semi-transparent so trajectories show clearly
    }
    im = ax.imshow(BRT_img, **imshow_kwargs)
    
    # 2. Add level set contours
    level_colors = ['black', 'red', 'blue']  # Colors for the three level sets
    level_styles = ['-', '--', '-.']  # Line styles for the three level sets
    level_widths = [1.5, 1.5, 1.5]  # Line widths for the three level sets
    
    for i, level in enumerate(level_sets):
        color = level_colors[i % len(level_colors)]
        style = level_styles[i % len(level_styles)]
        width = level_widths[i % len(level_widths)]
        
        contour = ax.contour(X, Y, BRT_img, levels=[level], colors=[color], 
                           linewidths=width, linestyles=style)
    
    # 3. Plot MPC trajectories
    n_trajs = len(interesting_ics_tensor)
    colors = plt.cm.Set1(np.linspace(0, 1, n_trajs))  # Use Set1 colormap for distinct colors
    
    for i in range(n_trajs):
        px = state_trajs_np[i, :, 0]  # px trajectory
        py = state_trajs_np[i, :, 1]  # py trajectory

        success_label = "Success" if successful_dockings[i] else "Failed"

        # Plot trajectory line
        ax.plot(px, py, color=colors[i], linewidth=1.5, alpha=0.9, 
               label=f'IC {i+1} ({success_label})')
        
        # Add start and end markers
        ax.scatter(px[0], py[0], color=colors[i], s=25, marker='o', 
              edgecolors='black', linewidth=1, zorder=10)  # Start
        ax.scatter(px[-1], py[-1], color=colors[i], s=25, marker='s', 
              edgecolors='black', linewidth=1, zorder=10)  # End
        
        # Add trajectory direction arrows
        if len(px) > 1:
            # Add arrows at quarter points along trajectory
            for frac in [0.25, 0.5, 0.75]:
                idx = int(frac * (len(px) - 1))
                if idx < len(px) - 1:
                    dx = px[idx+1] - px[idx]
                    dy = py[idx+1] - py[idx]
                    ax.arrow(px[idx], py[idx], dx*0.3, dy*0.3, 
                           head_width=0.05, head_length=0.03, 
                           fc=colors[i], ec=colors[i], alpha=0.7)
    
    # Add target region visualization based on reach_fn parameters
    # For Docking6D, the reach function defines a region with eps_p, eps_v, eps_theta, eps_omega tolerances
    eps_p = mpc.dynamics_.eps_p
    target_circle = patches.Circle((0, 0), eps_p, fill=False, color='green', linewidth=1, linestyle='--')
    ax.add_patch(target_circle)
    
    # Add failure region (avoid_fn) visualization if available
    if hasattr(mpc.dynamics_, 'avoid_fn'):
        # Draw target spacecraft body with docking port cutout
        w_t = mpc.dynamics_.w_t
        h_t = mpc.dynamics_.h_t
        dock_rad = mpc.dynamics_.dock_rad
        
        # Draw the main rectangular body
        body_rect = patches.Rectangle((-w_t/2, 0), w_t, h_t, 
                                    fill=False, color='red', linewidth=2, alpha=0.7, label='Avoid region')
        ax.add_patch(body_rect)
        
        # Draw only the upper semicircle of the docking port (cutout)
        theta = np.linspace(0, np.pi, 100)  # Upper semicircle from 0 to π
        dock_x = dock_rad * np.cos(theta)
        dock_y = dock_rad * np.sin(theta)
        
        # Plot the semicircular cutout
        ax.plot(dock_x, dock_y, color='red', linewidth=2, alpha=0.7)
        
        # Connect the ends of the semicircle to show the opening
        ax.plot([-dock_rad, dock_rad], [0, 0], color='red', linewidth=2, alpha=0.7)
        
        # Optional: Add vertical lines connecting the rectangle to the docking port
        # This helps visualize that it's a cutout, not separate shapes
        if dock_rad < w_t/2:  # Only if docking port is smaller than spacecraft width
            ax.plot([-dock_rad, -dock_rad], [0, 0], color='red', linewidth=2, alpha=0.7)
            ax.plot([dock_rad, dock_rad], [0, 0], color='red', linewidth=2, alpha=0.7)

    # 5. Formatting and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('px (m)', fontsize=14)
    ax.set_ylabel('py (m)', fontsize=14)
    ax.set_title('MPC Trajectories Overlaid on BRT Heatmap', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for BRT values
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('BRT Value', fontsize=12)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)

    fig.savefig("./data/trajectory_overlayClassicMPC.png", dpi=300, bbox_inches='tight')
    
    return fig, state_trajs_np, successful_dockings

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    dynamics_ = dynamics.Docking6D('reach_avoid')
    T = 6
    dt = 0.1
    x_res=101
    y_res=101
    plot_config = dynamics_.plot_config()
    state_test_range = dynamics_.state_test_range()
    x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    #x_min, x_max = -1, 1
    #y_min, y_max = -1, 1
    y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    z_min, z_max = state_test_range[plot_config['z_axis_idx']]

    xs = torch.linspace(x_min, x_max, x_res)
    ys = torch.linspace(y_min, y_max, y_res)
    xys = torch.cartesian_prod(xs, ys).to(device)
    initial_condition_tensor=torch.zeros(x_res*y_res, dynamics_.state_dim).to(device)
    initial_condition_tensor[:, :] = torch.tensor(plot_config['state_slices']).to(device)
    initial_condition_tensor[:, plot_config['x_axis_idx']] = xys[:, 0]
    initial_condition_tensor[:, plot_config['y_axis_idx']] = xys[:, 1]
    #initial_condition_tensor[:, plot_config['z_axis_idx']] = z_max*0.5

    # Try to use Receeding Syle MPC
    # - There may be a BUG (Try direct first and we can try receeding)
    mpc = MPC.MPC(horizon=None, receding_horizon=1, dT=dt, num_samples=100,
              dynamics_=dynamics_, device=device, mode="MPC", sample_mode="gaussian",
              style='receding',num_iterative_refinement=10, 
              cost_type="classic_mpc", mpc_percentage=0.8)

    costs=[]
    for i in tqdm(range(4)):
        batch_size = len(initial_condition_tensor) // 4
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < 3 else len(initial_condition_tensor)
        costs0, state_trajs, _, _ = mpc.get_batch_data(
            initial_condition_tensor[start_idx:end_idx,...], T)
        costs.append(costs0)
    
    costs=torch.cat(costs,dim=0)
    
    # Select initial conditions for visualization
    interesting_ics = []
    interesting_ics.append(torch.tensor([x_min*0.8, y_min*0.8, 0, 0, 0, 0]).to(device)) 
    interesting_ics.append(torch.tensor([x_max*0.8, y_max*0.8, 0, 0, 0, 0]).to(device)) 
    interesting_ics.append(torch.tensor([0, -1.5, 0, 0, 0, 0]).to(device))  
    interesting_ics.append(torch.tensor([0, 0.75, 0, 0, np.pi/4, 0]).to(device)) 
    interesting_ics.append(torch.tensor([4, -1.0, 0, 0, 0, 0]).to(device)) 
    interesting_ics.append(torch.tensor([1.0, -2.0, 0, 0, 0, 0]).to(device)) 
    interesting_ics_tensor = torch.stack(interesting_ics)

    print("Plotting Images")
    plotBRTImages(costs,x_resolution=x_res,y_resolution=y_res,x_min=x_min,x_max=x_max,y_min=y_min, y_max=y_max)
    #plotBRTImages(dynamics_.boundary_fn(initial_condition_tensor),x_resolution=x_res,y_resolution=y_res,x_min=x_min,x_max=x_max,y_min=y_min, y_max=y_max)
    plotMPCTrajectories(mpc, interesting_ics_tensor, T, max_trajs=5)
    plotTrajectoryOverlay(mpc, interesting_ics_tensor, T, costs, x_res, y_res, x_min, x_max, y_min, y_max, 
        level_sets=[0.0, 0.1, 0.3])

    print("Images saved successfully!") 

__all__ = ['run_quadrotor_mppi']
