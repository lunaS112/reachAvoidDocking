import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation
from matplotlib.path import Path


def animate(t, px, py, d=None, extra=None, skip_frames=10):
    """Animate the docking system from given position data."""
    
    # Geometry of target spacecraft
    target_width = 6.0
    target_height = 3.0
    dock_rad = 1.0
    
    # Geometry of chaser spacecraft
    chaser_width = 1.0
    chaser_height = 1.0
    
    # Constants for accessing extra data
    AX_IDX = 0  # Index for axis in extra data
    Y_DATA = 1  # Index for y-data in extra data
    STYLES = 2  # Index for style dictionary in extra data
    
    # Initialize tracking of configured axes
    axes_configured = [False] * 6
    
    # Initialize dictionary to store extra lines
    extra_lines = {}
    if extra is not None:
        for name, item in extra.items():
            extra_lines[name] = (None, item[Y_DATA])
    
    # Create figure with 6 subplots (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=100)
    axes_flat = axes.flatten()
    
    # Main trajectory plot (top left)
    ax = axes_flat[0]  # Main animation axis
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Spacecraft Docking Trajectory')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Position plot (top right)
    ax_pos = axes_flat[1]
    ax_pos.set_xlabel('t (s)')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.set_title('Position vs Time')
    ax_pos.grid(True)
    
    # Velocity plot (middle left)
    ax_vel = axes_flat[2]
    ax_vel.set_xlabel('t (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('Velocity vs Time')
    ax_vel.grid(True)
    
    # Orientation plot (middle right)
    ax_theta = axes_flat[3]
    ax_theta.set_xlabel('t (s)')
    ax_theta.set_ylabel('θ (rad), ω (rad/s)')
    ax_theta.set_title('Orientation vs Time')
    ax_theta.grid(True)
    
    # Control inputs plot (bottom left)
    ax_control = axes_flat[4]
    ax_control.set_xlabel('t (s)')
    ax_control.set_ylabel('Control Input')
    ax_control.set_title('Control Inputs vs Time')
    ax_control.grid(True)
    
    # Distance to target plot (bottom right)
    ax_dist = axes_flat[5]
    ax_dist.set_xlabel('t (s)')
    ax_dist.set_ylabel('Distance (m)')
    ax_dist.set_title('Distance to Target')
    ax_dist.grid(True)
    dist_line = ax_dist.plot([], [], 'k-')[0]
    
    # Configure the simulation plot limits
    ax.set_xlim([-12.5, 12.5])
    ax.set_ylim([-12.5, 12.5])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Spacecraft Docking Simulation")
    axes_configured[0] = True

    # Configure additional plots
    if d is not None:
        ax_d = axes_flat[1]  # Using flat index instead of tuple
        ax_d.set_xlim([0, t[-1]])
        ax_d.set_ylim([min(d) - 0.1, max(d) + 0.1])
        ax_d.set_xlabel("t (s)")
        ax_d.set_ylabel("θ (rad)")
        ax_d.grid(True, linestyle='--', alpha=0.7)
        dst = ax_d.plot([], [], linewidth=1, color="k", label="θ (rad)")[0]
        axes_configured[1] = True
        
    if extra is not None:
        for name, item in extra.items():
            ax_idx = item[AX_IDX]
            ax_extra = axes_flat[ax_idx]  # Use axes_flat instead of axes
            if not axes_configured[ax_idx]:
                ax_extra.set_xlim([0, t[-1]])
                # Handle multi-dimensional data by flattening or taking appropriate bounds
                if item[Y_DATA].ndim > 1:
                    y_min = np.min(item[Y_DATA])
                    y_max = np.max(item[Y_DATA])
                else:
                    y_min = min(item[Y_DATA])
                    y_max = max(item[Y_DATA])
                ax_extra.set_ylim([y_min, y_max])
                ax_extra.set_xlabel("t (s)")
                ax_extra.set_ylabel(name)
                ax_extra.grid(True, linestyle='--', alpha=0.7)
                axes_configured[ax_idx] = True
                # For multi-dimensional data, plot each column separately
                if item[Y_DATA].ndim > 1:
                    for col in range(item[Y_DATA].shape[1]):
                        line = ax_extra.plot([], [], label=f"{name}_{col}", **item[STYLES])[0]
                        extra_lines[f"{name}_{col}"] = (line, item[Y_DATA][:, col])
                else:
                    line = ax_extra.plot([], [], label=name, **item[STYLES])[0]
                    extra_lines[name] = (line, item[Y_DATA])
            else:
                ylim = ax_extra.get_ylim()
                # Handle multi-dimensional data
                if item[Y_DATA].ndim > 1:
                    y_min = np.min(item[Y_DATA])
                    y_max = np.max(item[Y_DATA])
                else:
                    y_min = min(item[Y_DATA])
                    y_max = max(item[Y_DATA])
                ax_extra.set_ylim([
                    min(ylim[0], y_min),
                    max(ylim[1], y_max)
                ])
                ylabel = ax_extra.get_ylabel()
                ax_extra.set_ylabel(f"{ylabel}\n{name}" if name not in ylabel else ylabel)
                # For multi-dimensional data, plot each column separately
                if item[Y_DATA].ndim > 1:
                    for col in range(item[Y_DATA].shape[1]):
                        line = ax_extra.plot([], [], label=f"{name}_{col}", **item[STYLES])[0]
                        extra_lines[f"{name}_{col}"] = (line, item[Y_DATA][:, col])
                else:
                    line = ax_extra.plot([], [], label=name, **item[STYLES])[0]
                    extra_lines[name] = (line, item[Y_DATA])
                
    # Update legend for configured axes
    for idx, ax_subplot in enumerate(axes_flat):
        if axes_configured[idx]:
            ymin, ymax = ax_subplot.get_ylim()
            margin = 0.05 * (ymax - ymin)
            ax_subplot.set_ylim([ymin - margin, ymax + margin])
            _, labels = ax_subplot.get_legend_handles_labels()
            if labels:
                ax_subplot.legend()
    fig.tight_layout()

    # Create Target spacecraft (static, doesn't move)
    # Base rectangle
    target_body = mpatches.Rectangle(
        (-target_width/2, 0), 
        target_width, 
        target_height,
        facecolor='gray',
        edgecolor='black',
        alpha=0.7
    )
    
    # Create docking bay (semicircle indentation)
    theta_angles = np.linspace(0, np.pi, 50)  
    docking_x = dock_rad * np.cos(theta_angles)
    docking_y = dock_rad * np.sin(theta_angles)
    docking_vertices = np.column_stack([docking_x, docking_y])
    docking_bay = mpatches.Polygon(
        docking_vertices, 
        closed=True,
        facecolor='white',
        edgecolor='black',
        alpha=0.7
    )
    
    # Add a red dot marker at the pi/2 orientation point on docking bay
    dock_marker = mpatches.Circle(
        (0, dock_rad),  # Position at the bottom of the docking bay
        radius=0.15,     # Small marker
        facecolor='red',
        edgecolor='black',
        alpha=1.0,
        zorder=10        # Ensure it's drawn on top
    )
    ax.add_patch(dock_marker)
    
    
    # Create Chaser spacecraft (moves during animation)
    chaser = mpatches.Rectangle(
        (-chaser_width/2, -chaser_height/2),
        chaser_width,
        chaser_height,
        facecolor='blue',
        edgecolor='black',
        alpha=0.8
    )
    
    # Add directional marker (small triangle to indicate orientation)
    marker = mpatches.RegularPolygon(
        (0, 0), 
        3, 
        radius=chaser_width/3,
        orientation=0,
        facecolor='red',
        alpha=0.8
    )
    
    # Trace of chaser's path
    trace = ax.plot([], [], "--", linewidth=1.5, color="tab:orange")[0]
    
    # Time-stamp
    timestamp = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # Add all patches to the axes
    ax.add_patch(target_body)
    ax.add_patch(docking_bay)
    ax.add_patch(chaser)
    ax.add_patch(marker)
    
    # Draw target and safety boundaries
    safety_boundary = mpatches.Circle(
        (0, 0), 
        0.5,  # eps_p from the scoring function
        fill=False, 
        color='green', 
        linestyle='--',
        alpha=0.7
    )
    ax.add_patch(safety_boundary)
    
    def _animate(k, t, px, py, d):
        # Update chaser position and orientation
        theta = d[k] if d is not None else 0
        
        # Transform for chaser spacecraft - ensure correct positioning
        t_chaser = mtransforms.Affine2D().rotate(theta).translate(px[k], py[k]) + ax.transData
        chaser.set_transform(t_chaser)
        chaser.set_zorder(5)  # Ensure chaser is above other elements
        
        # Transform for orientation marker - ensure correct positioning
        marker_x = px[k] + (chaser_width/2) * np.cos(theta)
        marker_y = py[k] + (chaser_width/2) * np.sin(theta)
        t_marker = mtransforms.Affine2D().rotate(theta + np.pi/2).translate(marker_x, marker_y) + ax.transData
        marker.set_transform(t_marker)
        marker.set_zorder(6)  # Ensure marker is above chaser
        
        # Update trace
        trace.set_data(px[:k+1], py[:k+1])
        
        # Update timestamp
        timestamp.set_text(f"t = {t[k]:.2f} s")
        
        # Map extra data to specific plots based on name patterns
        for name, (line, data) in extra_lines.items():
            if line is not None:
                # Update the extra line with data
                line.set_data(t[:k+1], data[:k+1] if k+1 <= len(data) else data)
    
        # Collect all artists that need to be updated
        artists = [chaser, marker, trace, timestamp]
        artists.extend([line for line, _ in extra_lines.values() if line is not None])
        
        return tuple(artists)
    
    # Set time limits for all time-series plots
    for subplot in axes_flat[1:]:
        subplot.set_xlim([0, t[-1]])
    
    fig.tight_layout()
    
    # Create animation
    dt_val = t[1] - t[0] if len(t) > 1 else 0.1
    ani = animation.FuncAnimation(
        fig, _animate, range(0, t.size, skip_frames), fargs=(t, px, py, d),
        interval=dt_val * 1000 * skip_frames, blit=True
    )
    
    return fig, ani


def create_simple_animation(t, px, py, d, controller, save_path, skip_frames=10):
    """Create a simplified animation with time-varying heatmap overlay showing the docking maneuver.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    import matplotlib.animation as animation
    import numpy as np
    from tqdm import tqdm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Geometry
    target_width = 6.0
    target_height = 3.0
    dock_rad = 1.0
    chaser_width = 1.0
    chaser_height = 1.0
    
    # Configure the plot
    ax.set_xlim([-12.5, 12.5])
    ax.set_ylim([-12.5, 12.5])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Spacecraft Docking Simulation with Time-Varying Value Function")
    
    # Extract grid coordinates for BRT visualization
    X, Y = None, None
    vmin, vmax = -0.5, 0.5  # Default values
    grid_4D = None
    
    # Calculate velocities from position data
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    
    # Calculate velocities using central differences
    if len(t) > 2:
        dt_vals = np.diff(t)
        # Forward difference for first point
        vx[0] = (px[1] - px[0]) / dt_vals[0]
        vy[0] = (py[1] - py[0]) / dt_vals[0]
        
        # Central difference for middle points
        for i in range(1, len(t)-1):
            vx[i] = (px[i+1] - px[i-1]) / (t[i+1] - t[i-1])
            vy[i] = (py[i+1] - py[i-1]) / (t[i+1] - t[i-1])
        
        # Backward difference for last point
        vx[-1] = (px[-1] - px[-2]) / dt_vals[-1]
        vy[-1] = (py[-1] - py[-2]) / dt_vals[-1]
    
    if hasattr(controller, 'grid_4D') and hasattr(controller, 'values_4D'):
        grid_4D = controller.grid_4D
        
        # Get the coordinate vectors for plotting
        x_coords = np.array(grid_4D.coordinate_vectors[0])
        y_coords = np.array(grid_4D.coordinate_vectors[1])
        vx_coords = np.array(grid_4D.coordinate_vectors[2])
        vy_coords = np.array(grid_4D.coordinate_vectors[3])
        
        # Create a meshgrid for position visualization
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Calculate robust vmin, vmax range from a sample of slices
        all_values = []
        sampling_times = np.linspace(0, len(controller.times)-1, min(10, len(controller.times)), dtype=int)
        for i in sampling_times:
            if i < len(controller.values_4D):
                # Sample at various velocities
                for vx_idx in [0, len(vx_coords)//4, len(vx_coords)//2, 3*len(vx_coords)//4, -1]:
                    for vy_idx in [0, len(vy_coords)//4, len(vy_coords)//2, 3*len(vy_coords)//4, -1]:
                        try:
                            value_slice = np.array(controller.values_4D[i][:, :, vx_idx, vy_idx])
                            if np.any(np.isfinite(value_slice)):
                                all_values.extend(value_slice[np.isfinite(value_slice)].flatten())
                        except Exception:
                            pass
    
        # Calculate robust vmin, vmax range
        if all_values:
            p5 = np.percentile(all_values, 5) 
            p95 = np.percentile(all_values, 95)
            
            # Ensure vmin < vmax with a reasonable separation
            if abs(p95 - p5) < 0.1:
                mean_val = (p5 + p95) / 2
                vmin = mean_val - 0.1
                vmax = mean_val + 0.1
            else:
                vmin = max(-0.5, p5)
                vmax = min(0.5, p95)
            
            # Final safety check
            if vmin >= vmax:
                vmin = -0.5
                vmax = 0.5
    else:
        print("Warning: Controller does not have grid_4D or values_4D attributes. No BRT visualization will be shown.")
    
    # Create spacecraft elements
    target_body = mpatches.Rectangle(
        (-target_width/2, 0), 
        target_width, 
        target_height,
        facecolor='gray',
        edgecolor='black',
        alpha=0.7,
        zorder=20
    )
    
    # Create docking bay
    theta_angles = np.linspace(0, np.pi, 50)  
    docking_x = dock_rad * np.cos(theta_angles)
    docking_y = dock_rad * np.sin(theta_angles)
    docking_vertices = np.column_stack([docking_x, docking_y])
    docking_bay = mpatches.Polygon(
        docking_vertices, 
        closed=True,
        facecolor='white',
        edgecolor='black',
        alpha=0.7,
        zorder=21
    )
    
    # Add docking target marker
    dock_marker = mpatches.Circle(
        (0, dock_rad),
        radius=0.15,
        facecolor='red',
        edgecolor='black',
        alpha=1.0,
        zorder=22
    )
    
    # Target set visualization
    eps_p = controller.eps_p if hasattr(controller, 'eps_p') else 0.5
    target_box = mpatches.Rectangle(
        (-eps_p, -eps_p), 
        2*eps_p, 
        2*eps_p, 
        color='g', 
        alpha=0.3, 
        label='Target Set',
        zorder=15
    )
    
    # Add static patches to the axes
    ax.add_patch(target_body)
    ax.add_patch(docking_bay)
    ax.add_patch(dock_marker)
    ax.add_patch(target_box)
    
    # Create Chaser spacecraft
    chaser = mpatches.Rectangle(
        (-chaser_width/2, -chaser_height/2),
        chaser_width,
        chaser_height,
        facecolor='blue',
        edgecolor='black',
        alpha=0.8,
        zorder=25
    )
    ax.add_patch(chaser)
    
    # Add directional marker
    marker = mpatches.RegularPolygon(
        (0, 0), 
        3, 
        radius=chaser_width/3,
        orientation=0,
        facecolor='red',
        alpha=0.8,
        zorder=26
    )
    ax.add_patch(marker)
    
    # Trace of chaser's path
    trace = ax.plot([], [], "--", linewidth=1.5, color="tab:orange")[0]
    
    # Text elements
    timestamp = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.7))
    value_info = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7))
    vel_info = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.7))

    # Colormap
    cmap = plt.cm.RdBu_r  # Red for negative (inside BRT), Blue for positive (outside)
    
    ax.legend(loc='upper left')
    
    # Dictionary to keep track of contour artists
    contour_artists = {'filled': None, 'lines': None}
    
    def _simple_animate(k, ax, t, px, py, d, X=None, Y=None):
        # Current time and velocity
        current_time = t[k]
        current_vx = vx[k]
        current_vy = vy[k]
        
        # Update BRT visualization if we have grid and values
        if X is not None and grid_4D is not None and hasattr(controller, 'values_4D'):
            # Map simulation time to BRT time
            final_time = controller.final_time
            brt_time = max(final_time, min(0, final_time + current_time))
            time_idx = np.argmin(np.abs(controller.times - brt_time))
            
            if time_idx < len(controller.values_4D):
                # Clear previous contour collections
                for tp in ['filled', 'lines']:
                    if contour_artists[tp] is not None:
                        try:
                            for coll in contour_artists[tp].collections:
                                coll.remove()
                        except (AttributeError, ValueError) as e:
                            # If the above fails, clear all collections and redraw
                            for coll in ax.collections[:]:
                                try:
                                    coll.remove()
                                except Exception:
                                    pass
                
                try:
                    # Find closest velocity indices
                    vx_coords = np.array(grid_4D.coordinate_vectors[2])
                    vy_coords = np.array(grid_4D.coordinate_vectors[3])
                    vx_idx = np.argmin(np.abs(vx_coords - current_vx))
                    vy_idx = np.argmin(np.abs(vy_coords - current_vy))
                    
                    # Get slice of value function at current time and velocity
                    value_slice = np.array(controller.values_4D[time_idx][:, :, vx_idx, vy_idx])
                    
                    # Create new contour plots
                    contour_artists['filled'] = ax.contourf(
                        X, Y, value_slice, levels=20, 
                        cmap=cmap, alpha=0.6, 
                        vmin=vmin, vmax=vmax,
                        zorder=5  # Below other elements
                    )
                    contour_artists['lines'] = ax.contour(
                        X, Y, value_slice, levels=[0], 
                        colors=['black'], linewidths=2,
                        zorder=10
                    )
                    
                    # Update value function info
                    current_state = [px[k], py[k], current_vx, current_vy]
                    try:
                        value_at_pos = controller.value_at_state_4D(current_state, time_idx)
                        value_info.set_text(f"Value: {value_at_pos:.3f}")
                        vel_info.set_text(f"Velocity: vx={current_vx:.2f}, vy={current_vy:.2f}")
                    except Exception as e:
                        value_info.set_text("Value: N/A")
                        vel_info.set_text(f"Velocity: vx={current_vx:.2f}, vy={current_vy:.2f}")
                        
                except Exception as e:
                    print(f"Frame {k}: Error creating contour plot: {e}")
        
        # Update chaser position and orientation
        theta = d[k] if d is not None else 0
        
        # Transform for chaser spacecraft
        t_chaser = mtransforms.Affine2D().rotate(theta).translate(px[k], py[k]) + ax.transData
        chaser.set_transform(t_chaser)
        
        # Transform for orientation marker
        marker_x = px[k] + (chaser_width/2) * np.cos(theta)
        marker_y = py[k] + (chaser_width/2) * np.sin(theta)
        t_marker = mtransforms.Affine2D().rotate(theta + np.pi/2).translate(marker_x, marker_y) + ax.transData
        marker.set_transform(t_marker)
        
        # Update trace
        trace.set_data(px[:k+1], py[:k+1])
        
        # Update timestamp
        timestamp.set_text(f"t = {t[k]:.2f} s")
        
        # Return updated artists (not returning contours as they cause issues with blit=True)
        return chaser, marker, trace, timestamp, value_info, vel_info
    
    # Create animation with progress bar
    dt_val = t[1] - t[0] if len(t) > 1 else 0.1
    frames = range(0, t.size, skip_frames)
    
    # Setup animation - use blit=False for safety with contour plots
    print("Creating animation...")
    ani = animation.FuncAnimation(
        fig, _simple_animate, frames, 
        fargs=(ax, t, px, py, d, X, Y),
        interval=dt_val * 1000 * skip_frames, 
        blit=False  # More compatible but slightly slower
    )
    
    # Save the animation with progress bar
    print(f"Saving animation to {save_path}...")
    writer = animation.FFMpegWriter(fps=min(30, int(1/(dt_val*skip_frames))))
    
    try:
        with tqdm(total=len(frames)) as pbar:
            ani.save(
                save_path, 
                writer=writer,
                progress_callback=lambda i, n: pbar.update(1)
            )
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying alternative method without progress bar...")
        ani.save(save_path, writer=writer)
        print("Animation saved with alternate method.")
    
    plt.close(fig)  # Close the figure after saving
    return fig, ani