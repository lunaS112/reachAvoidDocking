import numpy as np
np.random.seed(0)  # For reproducible results

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dynamics import f_disturbed
from utils.simulation import dock

from ComboControl import ComboController as Controller

# Store disturbances for visualization
disturbance_history = []

# Random disturbance generator function with history tracking
def random_disturbance(s, t, d_bar=0.01, d_theta_bar=0.01):
    """Generate random disturbance within specified bounds."""
    d_x = np.random.uniform(-d_bar, d_bar)
    d_y = np.random.uniform(-d_bar, d_bar)
    d_theta = np.random.uniform(-d_theta_bar, d_theta_bar)
    disturbance = np.array([d_x, d_y, d_theta])
    
    # Store for visualization
    disturbance_history.append(disturbance)
    
    return disturbance

# Initial state: [x, y, vx, vy, theta, omega]
s0 = [5.0, 4.0, -0.2, -0.3, -np.pi*5/4, -0.2]

# Create controller
controller = Controller()
final_time = abs(controller.final_time)  # Make positive for simulation
dt = 0.1
nt = int(final_time/dt + 1)

# Set disturbance parameters
d_bar = 0.01  # Position/velocity disturbance bound
d_theta_bar = 0.01  # Angular disturbance bound

# Animation settings
save_animation = True
animation_save_path = 'outputs/RA_animation_disturbed.mp4'

# Reset controller and run simulation
controller.reset()

# Clear history before simulation
disturbance_history.clear()

# Create disturbed dynamics function with history tracking
f = lambda s, u: f_disturbed(s, u, random_disturbance(s, 0, d_bar, d_theta_bar))

# Pass the disturbance history directly to the dock function
qualified, entered_failure, initial_in_brt, fuel_usage = dock(s0, controller, dt, nt, f,
                                  save_animation, animation_save_path,
                                  disturbance_list=disturbance_history)

# Display results
print()
print('###############')
print('### SUMMARY ###')
print('###############')
print()
print(f'Initial state within BRT:    {initial_in_brt}')
print(f'Entered failure:             {entered_failure}')
print(f'Finished in target:          {qualified}')
print(f'Using random disturbances with bounds:')
print(f' - Position/velocity: ±{d_bar} m/s²')
print(f' - Angular:          ±{d_theta_bar} rad/s²')
print(f"Total fuel usage: {fuel_usage:.4f} kg")