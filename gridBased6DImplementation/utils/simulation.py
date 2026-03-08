import numpy as np
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.animation import animate, create_simple_animation

def simulate(s0, u_fn, dt, nt, f):
    """Simulate the docking system.

    Args:
        s0 (np.ndarray): The initial state with shape (n)
        u_fn (callable): The control function u_fn(s, t) -> u with shape (m)
        dt (float): The time step
        nt (int): The number of time steps
        f (callable): The system dynamics f(s, u) -> dynamics with shape (n)

    Returns:
        s_history (np.ndarray): The state history with shape (nt, n)
        u_history (np.ndarray): The control history with shape (nt, m)
        t_history (np.ndarray): The time history with shape (nt)
    """
    s_history = np.full((nt, len(s0)), np.nan)
    t_history = np.full((nt,), np.nan)
    s_history[0] = s0
    
    # Get first control input to determine control dimension
    u0 = u_fn(s0, 0)
    u_history = np.full((nt, len(u0)), np.nan)
    u_history[0] = u0
    t_history[0] = 0
    
    # Add disturbance tracking
    d_history = []  # Will store disturbances if they're captured
    
    for i in tqdm(range(1, nt), desc='Simulating...'):
        # Capture disturbance if f is a function from run_d.py's random_disturbance
        # This won't affect other simulations
        if hasattr(f, '__closure__') and f.__closure__ is not None:
            for cell in f.__closure__:
                if 'disturbance_history' in cell.cell_contents.__globals__:
                    if len(cell.cell_contents.__globals__['disturbance_history']) >= i:
                        d_history.append(cell.cell_contents.__globals__['disturbance_history'][i-1])
                        
        s_history[i] = s_history[i-1] + dt*f(s_history[i-1], u_history[i-1])
        u_history[i] = u_fn(s_history[i], i*dt)
        t_history[i] = i*dt
        
    return s_history, u_history, t_history, d_history if d_history else None

def plot_trajectory(s_history, controller, save_path=None, show_brt=True):
    """Plot the trajectory of the system and highlight target and failure regions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    import os
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add BRT overlay if requested
    if show_brt:
        if hasattr(controller, 'values_4D_T'):
            # ComboControl BRT plotting
            grid_4D = controller.grid_4D
            vx_idx = len(grid_4D.coordinate_vectors[2]) // 2
            vy_idx = len(grid_4D.coordinate_vectors[3]) // 2
            
            # Extract the 2D slice at zero velocity
            brt_slice = np.array(controller.values_4D_T[:, :, vx_idx, vy_idx])
            
            # Get the coordinate vectors for plotting
            x_coords = np.array(grid_4D.coordinate_vectors[0])
            y_coords = np.array(grid_4D.coordinate_vectors[1])
            
        elif hasattr(controller, 'values_6D_T'):
            # SingleControl BRT plotting
            grid_6D = controller.grid_6D
            # Get middle indices for vx, vy, theta, omega (zero velocity, nominal orientation)
            vx_idx = len(grid_6D.coordinate_vectors[2]) // 2
            vy_idx = len(grid_6D.coordinate_vectors[3]) // 2
            theta_idx = len(grid_6D.coordinate_vectors[4]) // 2
            omega_idx = len(grid_6D.coordinate_vectors[5]) // 2
            
            # Extract 2D slice at zero velocities and nominal orientation
            brt_slice = np.array(controller.values_6D_T[:, :, vx_idx, vy_idx, theta_idx, omega_idx])
            
            # Get the coordinate vectors for plotting
            x_coords = np.array(grid_6D.coordinate_vectors[0])
            y_coords = np.array(grid_6D.coordinate_vectors[1])
        else:
            print("Warning: No BRT data found in controller - skipping BRT overlay")
            return fig
            
        # Create a meshgrid
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Add the heatmap visualization of the value function
        # Define appropriate range for the value function
        vmin = min(-0.5, np.min(brt_slice))
        vmax = max(0.5, np.max(brt_slice))
        
        # Create a colormap that emphasizes the zero level set
        cmap = plt.cm.RdBu_r  # Red for negative (inside BRT), Blue for positive (outside)
        
        # Add heatmap with transparency to see trajectory
        heatmap = ax.contourf(X, Y, brt_slice, levels=20, cmap=cmap, 
                              alpha=0.7, vmin=vmin, vmax=vmax)
                              
        # Add a colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Value Function')
        
        # Plot the zero level set contour more prominently
        zero_contour = ax.contour(X, Y, brt_slice, levels=[0], colors=['black'], linewidths=2)
        ax.plot([], [], color='black', linewidth=2, label='BRT Boundary (Zero Level Set)')
    
    # Plot the trajectory after heatmap so it's visible on top
    ax.plot(s_history[:, 0], s_history[:, 1], 'b-', label='Trajectory')
    ax.plot(s_history[0, 0], s_history[0, 1], 'go', markersize=8, label='Start')
    ax.plot(s_history[-1, 0], s_history[-1, 1], 'ro', markersize=8, label='End')
    
    # Draw failure set (target spacecraft body)
    w_t = controller.w_t
    h_t = controller.h_t
    dock_rad = controller.dock_rad
    
    # Draw the main body rectangle
    body = Rectangle((-w_t/2, 0), w_t, h_t, color='r', alpha=0.3, label='Failure Set')
    ax.add_patch(body)
    
    # Draw the docking indentation (semicircular cutout at the bottom)
    dock = Circle((0, 0), dock_rad, color='white', alpha=1.0)
    ax.add_patch(dock)
    
    # Draw target set 
    eps_p = controller.eps_p
    target_box = Rectangle((-eps_p, -eps_p), 2*eps_p, 2*eps_p, 
                          color='g', alpha=0.5, label='Target Set')  # Increased alpha for better visibility
    ax.add_patch(target_box)
    
    # Set plot properties
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Spacecraft Docking Trajectory with Value Function')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim([-12.5, 12.5])
    ax.set_ylim([-12.5, 12.5])
    
    # Add a legend
    ax.legend(loc='best')
    
    # Save figure if path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig

def dock(s0, controller, dt, nt, f, save_animation, animation_save_path, show_brt=True, disturbance_list=None):
    """Run full docking simulation.

    Args:
        s0 (np.ndarray): The initial state with shape (n)
        controller (Controller): The controller with functions
            controller.reset()
            controller.u_fn(s, t) -> u with shape (m)
            controller.data_to_visualize() -> data_to_visualize (dict)
        dt (float): The time step
        nt (int): The number of time steps
        f (callable): The system dynamics f(s, u) -> dynamics with shape (n)
        save_animation (bool): Whether to animate the simulation
        animation_save_path (str): The path to save the animation

    Returns:
        success (bool): Whether the chaser spacecraft entered the target docking zone
        distance (float): The distance from the target docking zone at the end of the simulation
    """
    # Check if initial state is inside BRT
    initial_in_brt = False
        
    # Check for ComboControl BRT (has both 4D translational and 2D rotational BRTs)
    if (hasattr(controller, 'values_4D_T') and hasattr(controller, 'value_at_state_4D') and 
        hasattr(controller, 'values_2D_T') and hasattr(controller, 'value_at_state_2D')):
        
        # Extract translational part of the initial state
        s0_translational = s0[:4]  # [px, py, vx, vy]
        brt_value_4D = controller.value_at_state_4D(s0_translational, -1)
        
        # Extract rotational part of the initial state
        s0_rotational = s0[4:6]  # [theta, omega]
        brt_value_2D = controller.value_at_state_2D(s0_rotational, -1)
        
        # Initial state is in BRT only if BOTH parts are in their respective BRTs
        translational_in_brt = brt_value_4D <= 0
        rotational_in_brt = brt_value_2D <= 0
        initial_in_brt = translational_in_brt and rotational_in_brt
        
        # Print detailed BRT information
        print(f"BRT Check: Translational BRT value = {brt_value_4D:.4f} ({'inside' if translational_in_brt else 'outside'})")
        print(f"BRT Check: Rotational BRT value = {brt_value_2D:.4f} ({'inside' if rotational_in_brt else 'outside'})")
        print(f"BRT Check: Initial state is {'' if initial_in_brt else 'NOT '} in combined BRT")

        
    # Check for SingleControl BRT
    elif hasattr(controller, 'values_6D_T') and hasattr(controller, 'value_at_state_6D'):
        brt_value = controller.value_at_state_6D(s0)
        initial_in_brt = brt_value <= 0
    
    # simulate with disturbance tracking
    controller.reset()
    s_history, u_history, t_history, d_history = simulate(s0, controller.u_fn, dt, nt, f)
    print('Done.')
    
    # If we have an external disturbance list, use it
    if disturbance_list is not None and len(disturbance_list) > 0:
        controller.disturbance_history = np.array(disturbance_list)
        print(f"Using {len(disturbance_list)} disturbance samples from external source")
    # Otherwise use any disturbances captured during simulation
    elif d_history is not None:
        controller.disturbance_history = np.array(d_history)
        print(f"Captured {len(d_history)} disturbance samples during simulation")

    # Now create animation with disturbance data
    if save_animation:
        skip_frames = int(0.1/dt) # number of frames to skip in the animation
        
        # Create full animation with all plots
        fig, ani = animate(
             t_history,
             s_history[:, 0],
             s_history[:, 1],
             d=s_history[:, 4] if s_history.shape[-1] > 4 else None,
             extra=controller.data_to_visualize(),
            skip_frames=skip_frames)
        with tqdm(total=t_history.size, desc="Animating full view...") as pbar:
                ani.save(animation_save_path, writer='ffmpeg', progress_callback=lambda i, n: pbar.update(skip_frames))
        print(f'Done. Animation saved to {animation_save_path}')
        
        # Create simplified animation with just spacecraft and heatmap
        simple_save_path = animation_save_path.replace('.mp4', '_simple.mp4')
        fig, ani = create_simple_animation(
            t_history,
            s_history[:, 0],
            s_history[:, 1],
            d=s_history[:, 4] if s_history.shape[-1] > 4 else None,
            controller=controller,
            save_path=simple_save_path,
            skip_frames=skip_frames
        )
        print(f'Simple animation saved to {simple_save_path}')
        
    # Generate a proper PNG file path for the static trajectory plot
    # This ensures we always use a valid image format regardless of the animation path
    if animation_save_path.endswith('.mp4'):
        trajectory_save_path = animation_save_path.replace('.mp4', '_trajectory.png')
    else:
        # Fallback in case the animation path doesn't end with .mp4
        trajectory_save_path = os.path.splitext(animation_save_path)[0] + '_trajectory.png'
        
    plot_trajectory(s_history, controller, save_path=trajectory_save_path, show_brt=show_brt)

    # Score
    eps = 0.01  # Tolerance for scoring, can be adjusted if needed
    # Define docking parameters (match those from RA_docking_6D.py)
    eps_p = controller.eps_p + eps*2   # Position tolerance for docking (m)
    eps_v = controller.eps_v + eps  # Velocity tolerance for docking (m/s)
    eps_theta = controller.eps_theta + eps # Angular position tolerance for docking (rad)
    eps_omega = controller.eps_omega  + eps # Angular velocity tolerance for docking (rad/s)
    
    # Get final state
    final_state = s_history[-1]
    
    # Check if final state is within all tolerance bounds
    px, py = final_state[0], final_state[1]
    vx, vy = final_state[2], final_state[3]
    theta, omega = final_state[4], final_state[5]
    
    target_theta = np.pi/2 
    
    # Calculate distance from each constraint
    position_distance = max(abs(px) - eps_p, abs(py) - eps_p)
    velocity_distance = max(abs(vx) - eps_v, abs(vy) - eps_v)
    orientation_distance = abs(theta - target_theta) - eps_theta
    angular_vel_distance = abs(omega) - eps_omega
    
    # Print out final state
    print(f'Final state: px={px:.2f}, py={py:.2f}, vx={vx:.2f}, vy={vy:.2f}, theta={theta - target_theta:.2f}, omega={omega:.2f}')
    
    # Maximum distance from target set (negative if inside)
    max_distance = max(position_distance, velocity_distance, orientation_distance, angular_vel_distance)
    
    # Check if spacecraft has successfully docked
    qualified = max_distance <= 0
    
    # Calculate fuel usage metrics
    u_history = np.array(controller.u_history)
    
    # Calculate translational fuel consumption (proportional to total impulse)
    # Using approximation: Δv = F*Δt/m = u*Δt/m
    translational_impulse = np.sum(np.sqrt(u_history[:,0]**2 + u_history[:,1]**2)) * dt
    translational_fuel = translational_impulse / (300 * 9.81)  # Assuming Isp = 300s
    
    # Calculate rotational fuel consumption
    rotational_impulse = np.sum(np.abs(u_history[:,2])) * dt
    rotational_fuel = rotational_impulse / (300 * 9.81)  # Assuming same Isp
    
    # Calculate throttle usage statistics
    u_bar_translational = controller.u_bar_4D if hasattr(controller, 'u_bar_4D') else 20.0
    u_bar_rotational = controller.u_bar_2D if hasattr(controller, 'u_bar_2D') else 1.5
    
    # Normalize control inputs by their maximum values to get throttle percentages
    throttle_x = np.abs(u_history[:,0]) / u_bar_translational * 100
    throttle_y = np.abs(u_history[:,1]) / u_bar_translational * 100
    throttle_theta = np.abs(u_history[:,2]) / u_bar_rotational * 100
    
    # Calculate throttle statistics
    avg_throttle_x = np.mean(throttle_x)
    avg_throttle_y = np.mean(throttle_y)
    avg_throttle_theta = np.mean(throttle_theta)
    max_throttle_x = np.max(throttle_x)
    max_throttle_y = np.max(throttle_y)
    max_throttle_theta = np.max(throttle_theta)
    
    # Calculate time spent at different throttle levels (binned)
    throttle_bins = [0, 20, 40, 60, 80, 100]
    time_at_x_throttle = np.histogram(throttle_x, bins=throttle_bins)[0] * dt
    time_at_y_throttle = np.histogram(throttle_y, bins=throttle_bins)[0] * dt
    time_at_theta_throttle = np.histogram(throttle_theta, bins=throttle_bins)[0] * dt
    
    # Adjust for simulation time to match exactly final_time
    sim_duration = -1*(controller.final_time)  # Use the intended simulation time
    adjustment_factor = sim_duration / (dt * nt) 

    # Apply adjustment to throttle times
    time_at_x_throttle *= adjustment_factor
    time_at_y_throttle *= adjustment_factor
    time_at_theta_throttle *= adjustment_factor
    
    
    # Display fuel and throttle metrics
    print()
    print('#####################')
    print('### FUEL METRICS ####')
    print('#####################')
    print()
    print(f'Translational fuel: {translational_fuel:.4f} kg')
    print(f'Rotational fuel:    {rotational_fuel:.4f} kg')
    print(f'Total fuel:         {translational_fuel + rotational_fuel:.4f} kg')
    print()
    print('######################')
    print('### THROTTLE USAGE ###')
    print('######################')
    print()
    print(f'Average X-axis throttle:       {avg_throttle_x:.1f}%')
    print(f'Average Y-axis throttle:       {avg_throttle_y:.1f}%')
    print(f'Average rotational throttle:   {avg_throttle_theta:.1f}%')
    print(f'Maximum X-axis throttle:       {max_throttle_x:.1f}%')
    print(f'Maximum Y-axis throttle:       {max_throttle_y:.1f}%')
    print(f'Maximum rotational throttle:   {max_throttle_theta:.1f}%')
    print()
    print('Time spent at throttle levels (seconds):')
    print('Throttle Range    |   X-axis  |   Y-axis  | Rotation')
    print('-------------------------------------------------')
    for i in range(len(throttle_bins)-1):
        print(f'{throttle_bins[i]:>3}-{throttle_bins[i+1]:<3}%        | {time_at_x_throttle[i]:>8.2f} | {time_at_y_throttle[i]:>8.2f} | {time_at_theta_throttle[i]:>8.2f}')
    
    # Check if trajectory ever entered failure set (body + docking post)
    # Uses corner-based check: compute 4 chaser corners from orientation,
    # check each against uninflated body and post rectangles.
    entered_failure = False
    hw = controller.w_c / 2.0
    hh = controller.h_c / 2.0
    for i in range(len(s_history)):
        state = s_history[i]
        px, py, theta = state[0], state[1], state[4]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        corners = [
            (px + hw * cos_t - hh * sin_t, py + hw * sin_t + hh * cos_t),
            (px - hw * cos_t - hh * sin_t, py - hw * sin_t + hh * cos_t),
            (px - hw * cos_t + hh * sin_t, py - hw * sin_t - hh * cos_t),
            (px + hw * cos_t + hh * sin_t, py + hw * sin_t - hh * cos_t),
        ]
        for cx, cy in corners:
            in_body = abs(cx) <= controller.w_t / 2.0 and 0.0 <= cy <= controller.h_t
            in_post = abs(cx) <= controller.post_hw_x and -controller.post_length <= cy <= 0.0
            if in_body or in_post:
                entered_failure = True
                break
        if entered_failure:
            break

    # Update the return values to include the new metrics
    return qualified, entered_failure, initial_in_brt, translational_fuel + rotational_fuel