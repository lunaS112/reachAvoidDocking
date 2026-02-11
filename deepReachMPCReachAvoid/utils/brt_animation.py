"""
Animation module for BRT-based docking visualization.

Creates animated visualizations showing:
- Time-varying learned value function (static in Phase 1, shrinking in Phase 2)
- Goal set (docking tolerance region)
- Failure set (target spacecraft body)
- Chaser trajectory with orientation

Inspired by gridBased6DImplementation/utils/animation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from tqdm import tqdm
import os


def create_deepreach_animation(controller, sim_result, save_path, 
                                skip_frames=10, resolution=40,
                                show_value_function=True):
    """
    Create animated docking visualization with time-varying DeepReach value function.
    
    The value function display changes based on control phase:
    - Phase 1 (Convergence): Static heatmap of V(x, tMax)
    - Phase 2 (Precision): Shrinking heatmap of V(x, t_remaining)
    
    Args:
        controller: BRTController instance with loaded dynamics
        sim_result: Dictionary from controller.simulate_docking() containing:
            - trajectory: (N, 6) state trajectory
            - controls: (N, 3) control inputs
            - values: (N,) value function along trajectory
            - phases: (N,) phase indicator (1 or 2)
            - t_remaining: (N,) time remaining
            - times: (N,) simulation times
        save_path: Path to save the animation (mp4)
        skip_frames: Number of frames to skip for faster animation
        resolution: Grid resolution for value function heatmap
        show_value_function: Whether to show value function heatmap
        
    Returns:
        fig, ani: Matplotlib figure and animation objects
    """
    # Extract data
    trajectory = sim_result['trajectory']
    phases = sim_result['phases']
    t_remaining_history = sim_result['t_remaining']
    times = sim_result['times']
    values = sim_result['values']
    
    px = trajectory[:, 0]
    py = trajectory[:, 1]
    theta = trajectory[:, 4]
    vx = trajectory[:, 2]
    vy = trajectory[:, 3]
    omega = trajectory[:, 5]
    
    # Get dynamics parameters
    dynamics = controller.dynamics
    tMax = controller.tMax
    
    # Geometry parameters
    w_t = dynamics.w_t  # Target width
    h_t = dynamics.h_t  # Target height
    dock_rad = dynamics.dock_rad  # Docking port radius
    w_c = dynamics.w_c  # Chaser width
    h_c = dynamics.h_c  # Chaser height
    eps_p = dynamics.eps_p  # Position tolerance
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    
    # Configure plot
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('DeepReach BRT-Based Docking Control', fontsize=14)
    
    # Precompute value function grid coordinates
    x_range = np.linspace(-15, 15, resolution)
    y_range = np.linspace(-15, 15, resolution)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    # Compute colormap range from sample of value function
    if show_value_function:
        print("Computing value function range...")
        sample_values = []
        sample_times = [tMax, tMax/2, tMax/4, 0.1]
        for t in sample_times:
            for i in range(0, resolution, 5):
                for j in range(0, resolution, 5):
                    state = np.array([X[i, j], Y[i, j], 0, 0, np.pi/2, 0])
                    try:
                        v = controller.get_value(state, t)
                        if np.isfinite(v):
                            sample_values.append(v)
                    except:
                        pass
        
        if sample_values:
            vmin = np.percentile(sample_values, 5)
            vmax = np.percentile(sample_values, 95)
            # Ensure reasonable range
            vmin = max(-2.0, vmin)
            vmax = min(2.0, vmax)
            if vmin >= vmax:
                vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -1.0, 1.0
        print(f"Value function range: [{vmin:.2f}, {vmax:.2f}]")
    else:
        vmin, vmax = -1.0, 1.0
    
    # Colormap
    cmap = plt.cm.RdBu_r  # Red=negative (inside BRT), Blue=positive (outside)
    
    # Add colorbar (will be updated dynamically)
    # Store as dictionary to allow modification in animate function
    colorbar_info = {'sm': None, 'cbar': None, 'vmin': vmin, 'vmax': vmax}
    if show_value_function:
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Value Function V(x, t)', fontsize=11)
        colorbar_info['sm'] = sm
        colorbar_info['cbar'] = cbar
    
    # ================== STATIC ELEMENTS ==================
    
    # Target spacecraft body (gray rectangle)
    target_body = mpatches.Rectangle(
        (-w_t/2, 0), w_t, h_t,
        facecolor='gray', edgecolor='black', alpha=0.8, zorder=20,
        label='Target Spacecraft'
    )
    ax.add_patch(target_body)
    
    # Docking port (white semicircle cutout)
    theta_angles = np.linspace(0, np.pi, 50)
    docking_x = dock_rad * np.cos(theta_angles)
    docking_y = dock_rad * np.sin(theta_angles)
    docking_bay = mpatches.Polygon(
        np.column_stack([docking_x, docking_y]),
        closed=True, facecolor='white', edgecolor='black', alpha=0.9, zorder=21
    )
    ax.add_patch(docking_bay)
    
    # Docking target marker (red dot at pi/2 orientation)
    dock_marker = mpatches.Circle(
        (0, dock_rad), radius=0.15,
        facecolor='red', edgecolor='black', alpha=1.0, zorder=22
    )
    ax.add_patch(dock_marker)
    
    # Goal set (green rectangle showing docking tolerance)
    goal_set = mpatches.Rectangle(
        (-eps_p, -eps_p), 2*eps_p, 2*eps_p,
        facecolor='green', edgecolor='darkgreen', alpha=0.4, zorder=15,
        label='Goal Set'
    )
    ax.add_patch(goal_set)
    
    # ================== DYNAMIC ELEMENTS ==================
    
    # Chaser spacecraft (blue rectangle)
    chaser = mpatches.Rectangle(
        (-w_c/2, -h_c/2), w_c, h_c,
        facecolor='blue', edgecolor='black', alpha=0.9, zorder=30
    )
    ax.add_patch(chaser)
    
    # Orientation marker (red triangle)
    marker = mpatches.RegularPolygon(
        (0, 0), 3, radius=w_c/3,
        orientation=0, facecolor='red', edgecolor='black', alpha=0.9, zorder=31
    )
    ax.add_patch(marker)
    
    # Trajectory trace
    trace, = ax.plot([], [], '--', linewidth=2, color='orange', zorder=25, label='Trajectory')
    
    # BRT boundary contour (will be updated)
    contour_artists = {'filled': None, 'lines': None}
    
    # Text elements
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    phase_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    value_text = ax.text(0.02, 0.82, '', transform=ax.transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Cache for value function grids (to avoid recomputation)
    value_cache = {}
    
    def compute_value_grid(t_query, current_vx, current_vy, current_theta, current_omega):
        """Compute value function grid at given time and velocity."""
        cache_key = (round(t_query, 2), round(current_vx, 1), round(current_vy, 1))
        
        if cache_key in value_cache:
            return value_cache[cache_key]
        
        V = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([X[i, j], Y[i, j], current_vx, current_vy, current_theta, current_omega])
                try:
                    V[i, j] = controller.get_value(state, t_query)
                except:
                    V[i, j] = np.nan
        
        # Limit cache size
        if len(value_cache) > 100:
            # Remove oldest entries
            keys_to_remove = list(value_cache.keys())[:50]
            for key in keys_to_remove:
                del value_cache[key]
        
        value_cache[cache_key] = V
        return V
    
    def animate(k):
        """Animation update function."""
        # Get current state
        current_theta = theta[k]
        current_vx = vx[k]
        current_vy = vy[k]
        current_omega = omega[k]
        current_phase = phases[k]
        current_t_remaining = t_remaining_history[k]
        current_time = times[k]
        current_value = values[k]
        
        # Update chaser position and orientation
        t_chaser = mtransforms.Affine2D().rotate(current_theta).translate(px[k], py[k]) + ax.transData
        chaser.set_transform(t_chaser)
        
        # Update orientation marker
        marker_x = px[k] + (w_c/2) * np.cos(current_theta)
        marker_y = py[k] + (w_c/2) * np.sin(current_theta)
        t_marker = mtransforms.Affine2D().rotate(current_theta + np.pi/2).translate(marker_x, marker_y) + ax.transData
        marker.set_transform(t_marker)
        
        # Update trajectory trace
        trace.set_data(px[:k+1], py[:k+1])
        
        # Update text
        time_text.set_text(f'Time: {current_time:.2f}s')
        
        if current_phase == 1:
            phase_text.set_text(f'Phase 1: Convergence\nUsing V(x, tMax={tMax:.1f}s)')
            phase_text.set_bbox(dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            phase_text.set_text(f'Phase 2: Precision\nt_remaining = {current_t_remaining:.2f}s')
            phase_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        value_text.set_text(f'V(x, t) = {current_value:.3f}')
        if current_value <= 0:
            value_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            value_text.set_bbox(dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Update value function heatmap
        if show_value_function:
            # Clear previous contours
            for tp in ['filled', 'lines']:
                if contour_artists[tp] is not None:
                    for coll in contour_artists[tp].collections:
                        try:
                            coll.remove()
                        except:
                            pass
            
            # Determine query time based on phase
            if current_phase == 1:
                t_query = tMax
            else:
                t_query = max(current_t_remaining, 0.1)
            
            # Compute value function grid
            V = compute_value_grid(t_query, current_vx, current_vy, current_theta, current_omega)
            
            # Compute dynamic vmin/vmax from current slice
            V_finite = V[np.isfinite(V)]
            if len(V_finite) > 0:
                current_vmin = np.percentile(V_finite, 5)
                current_vmax = np.percentile(V_finite, 95)
                # Ensure reasonable bounds and separation
                current_vmin = max(-3.0, current_vmin)
                current_vmax = min(3.0, current_vmax)
                if current_vmax - current_vmin < 0.1:
                    mid = (current_vmin + current_vmax) / 2
                    current_vmin = mid - 0.5
                    current_vmax = mid + 0.5
            else:
                current_vmin, current_vmax = -1.0, 1.0
            
            # Create contour plots with dynamic range
            try:
                contour_artists['filled'] = ax.contourf(
                    X, Y, V, levels=20, cmap=cmap, alpha=0.6,
                    vmin=current_vmin, vmax=current_vmax, zorder=5
                )
                contour_artists['lines'] = ax.contour(
                    X, Y, V, levels=[0], colors=['black'], linewidths=2.5, zorder=10
                )
                
                # Update colorbar to reflect current slice range
                if colorbar_info['sm'] is not None and colorbar_info['cbar'] is not None:
                    new_norm = Normalize(vmin=current_vmin, vmax=current_vmax)
                    colorbar_info['sm'].set_norm(new_norm)
                    colorbar_info['cbar'].update_normal(colorbar_info['sm'])
                    # Update colorbar label to show current time
                    colorbar_info['cbar'].set_label(f'V(x, t={t_query:.1f}s)', fontsize=11)
            except:
                pass
        
        return chaser, marker, trace, time_text, phase_text, value_text
    
    # Create animation
    print(f"Creating animation with {len(times)} frames (skip={skip_frames})...")
    frames = range(0, len(times), skip_frames)
    
    ani = animation.FuncAnimation(
        fig, animate, frames,
        interval=controller.dt * 1000 * skip_frames,
        blit=False  # Required for contour updates
    )
    
    # Save animation
    print(f"Saving animation to {save_path}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    writer = animation.FFMpegWriter(fps=min(30, int(1/(controller.dt * skip_frames))))
    
    try:
        with tqdm(total=len(list(frames)), desc="Rendering") as pbar:
            ani.save(save_path, writer=writer, 
                    progress_callback=lambda i, n: pbar.update(1))
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Error with progress bar, trying without: {e}")
        ani.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    
    plt.close(fig)
    return fig, ani


def plot_trajectory_static(controller, sim_result, save_path=None, show_brt=True):
    """
    Create static plot of docking trajectory with value function overlay.
    
    Args:
        controller: BRTController instance
        sim_result: Dictionary from controller.simulate_docking()
        save_path: Path to save figure (optional)
        show_brt: Whether to show BRT overlay
        
    Returns:
        fig: Matplotlib figure
    """
    trajectory = sim_result['trajectory']
    phases = sim_result['phases']
    
    px = trajectory[:, 0]
    py = trajectory[:, 1]
    
    # Get dynamics parameters
    dynamics = controller.dynamics
    w_t = dynamics.w_t
    h_t = dynamics.h_t
    dock_rad = dynamics.dock_rad
    eps_p = dynamics.eps_p
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Show BRT at tMax
    if show_brt:
        print("Computing BRT for static plot...")
        X, Y, V = controller.get_value_grid(controller.tMax, resolution=60)
        
        # Heatmap
        vmin = max(-2.0, np.percentile(V[np.isfinite(V)], 5))
        vmax = min(2.0, np.percentile(V[np.isfinite(V)], 95))
        
        contour = ax.contourf(X, Y, V, levels=20, cmap='RdBu_r', alpha=0.7,
                              vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label(f'Value Function V(x, tMax={controller.tMax}s)')
        
        # Zero level set
        ax.contour(X, Y, V, levels=[0], colors=['black'], linewidths=2)
    
    # Plot trajectory with phase coloring
    phase1_mask = phases == 1
    phase2_mask = phases == 2
    
    if np.any(phase1_mask):
        ax.plot(px[phase1_mask], py[phase1_mask], 'b-', linewidth=2, label='Phase 1: Convergence')
    if np.any(phase2_mask):
        ax.plot(px[phase2_mask], py[phase2_mask], 'g-', linewidth=2, label='Phase 2: Precision')
    
    # Start and end markers
    ax.plot(px[0], py[0], 'go', markersize=12, label='Start', zorder=35)
    ax.plot(px[-1], py[-1], 'r*', markersize=15, label='End', zorder=35)
    
    # Phase transition marker
    if sim_result['phase_transition_time'] is not None:
        transition_idx = np.argmin(np.abs(sim_result['times'] - sim_result['phase_transition_time']))
        ax.plot(px[transition_idx], py[transition_idx], 'mo', markersize=10, 
                label='BRT Entry', zorder=35)
    
    # Target spacecraft
    target = mpatches.Rectangle((-w_t/2, 0), w_t, h_t, 
                                 facecolor='gray', edgecolor='black', alpha=0.7, zorder=20)
    ax.add_patch(target)
    
    # Docking port
    theta_angles = np.linspace(0, np.pi, 50)
    docking_x = dock_rad * np.cos(theta_angles)
    docking_y = dock_rad * np.sin(theta_angles)
    docking_bay = mpatches.Polygon(np.column_stack([docking_x, docking_y]),
                                    closed=True, facecolor='white', edgecolor='black', 
                                    alpha=0.9, zorder=21)
    ax.add_patch(docking_bay)
    
    # Goal set
    goal = mpatches.Rectangle((-eps_p, -eps_p), 2*eps_p, 2*eps_p,
                               facecolor='green', edgecolor='darkgreen', alpha=0.5, zorder=15)
    ax.add_patch(goal)
    
    # Configure plot
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('DeepReach BRT-Based Docking Trajectory', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add results text
    success_str = "SUCCESS" if sim_result['success'] else "INCOMPLETE"
    results_text = f"Result: {success_str}\n"
    results_text += f"Final pos: ({px[-1]:.2f}, {py[-1]:.2f})\n"
    results_text += f"Sim time: {sim_result['times'][-1]:.1f}s"
    ax.text(0.02, 0.02, results_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_simulation_data(sim_result, save_path=None):
    """
    Create multi-panel plot showing simulation data over time.
    
    Args:
        sim_result: Dictionary from controller.simulate_docking()
        save_path: Path to save figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    trajectory = sim_result['trajectory']
    controls = sim_result['controls']
    values = sim_result['values']
    phases = sim_result['phases']
    t_remaining = sim_result['t_remaining']
    times = sim_result['times']
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Position
    ax = axes[0, 0]
    ax.plot(times, trajectory[:, 0], 'b-', label='px')
    ax.plot(times, trajectory[:, 1], 'g-', label='py')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity
    ax = axes[0, 1]
    ax.plot(times, trajectory[:, 2], 'b-', label='vx')
    ax.plot(times, trajectory[:, 3], 'g-', label='vy')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Orientation
    ax = axes[1, 0]
    ax.plot(times, trajectory[:, 4], 'r-', label='θ')
    ax.plot(times, trajectory[:, 5], 'b-', label='ω')
    ax.axhline(np.pi/2, color='g', linestyle='--', alpha=0.5, label='θ_target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad), Angular vel (rad/s)')
    ax.set_title('Orientation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control inputs
    ax = axes[1, 1]
    ax.plot(times, controls[:, 0], 'r-', label='u_x')
    ax.plot(times, controls[:, 1], 'g-', label='u_y')
    ax.plot(times, controls[:, 2], 'b-', label='u_θ')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.set_title('Control Inputs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Value function and phase
    ax = axes[2, 0]
    ax.plot(times, values, 'k-', linewidth=2, label='V(x, t)')
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='BRT boundary')
    ax.fill_between(times, values, 0, where=(phases == 2), 
                    alpha=0.3, color='green', label='Phase 2')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('Value Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # t_remaining
    ax = axes[2, 1]
    ax.plot(times, t_remaining, 'b-', linewidth=2)
    ax.fill_between(times, t_remaining, alpha=0.3, where=(phases == 2), color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('t_remaining (s)')
    ax.set_title('Time Remaining in BRT Phase')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig
