import numpy as np
np.random.seed(0)  # For reproducible results

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dynamics import f_disturbed
from utils.simulation import dock

from ComboControl import ComboController as Controller

# Define 5 different initial states to test
initial_states = [
    [-6.0, 2.0, 0.0, 0.0, -np.pi, 0.0],     # Original initial state
    [-8.0, 3.0, 0.1, -0.1, -np.pi/2, 0.1],  # Further away with some velocity
    [10.0, -10.0, -0.1, 0.2, -np.pi*3/4, -0.1], # Different quadrant
    [-10.0, -10.0, 0.2, 0.0, -np.pi/4, 0.05],  # Far approach
    [5.0, 4.0, -0.2, -0.3, -np.pi*1/4, -0.2]  # With significant initial velocities
]

# Create controller once
controller = Controller()
final_time = abs(controller.final_time)  # Make positive for simulation
dt = 0.1
nt = int(final_time/dt + 1)

# Set disturbance parameters
d_bar = 0.01  # Position/velocity disturbance bound
d_theta_bar = 0.01  # Angular disturbance bound

# Results tracking
results = []

# Run simulation for each initial state
for i, s0 in enumerate(initial_states):
    print(f"\n--- Running simulation {i+1}/5 ---")
    print(f"Initial state: {s0}")
    
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
    
    # Animation settings - create unique filenames for each run
    save_animation = False
    animation_save_path = f'outputs/RA_animation_run_{i+1}.mp4'
    
    # Reset controller and clear disturbance history
    controller.reset()
    disturbance_history.clear()
    
    # Create disturbed dynamics function with history tracking
    f = lambda s, u: f_disturbed(s, u, random_disturbance(s, 0, d_bar, d_theta_bar))
    
    # Run simulation
    qualified, entered_failure, initial_in_brt, fuel_usage = dock(s0, controller, dt, nt, f,
                                  save_animation, animation_save_path,
                                  disturbance_list=disturbance_history)
    
    # Store results
    results.append({
        'initial_state': s0,
        'initial_in_brt': initial_in_brt,
        'entered_failure': entered_failure,
        'reached_target': qualified,
        'fuel_usage': fuel_usage,
        'docked_safely': qualified and not entered_failure  # New metric
    })

# Display summary of all runs
print("\n\n#########################")
print("### OVERALL SUMMARY ###")
print("#########################\n")

# Add fuel_usage column to the table header
print("| Run | Initial Position | In BRT | Failure | Success | Docked Safely | Fuel (kg) |")
print("|-----|-----------------|--------|---------|---------|--------------|-----------|")
for i, result in enumerate(results):
    s0 = result['initial_state']
    position = f"({s0[0]:.1f}, {s0[1]:.1f})"
    fuel = f"{result['fuel_usage']:.4f}"
    print(f"| {i+1}   | {position:14} | {str(result['initial_in_brt']):6} | {str(result['entered_failure']):7} | " +
          f"{str(result['reached_target']):7} | {str(result['docked_safely']):12} | {fuel:9} |")

# Calculate additional fuel statistics
total_fuel = sum(r['fuel_usage'] for r in results)
avg_fuel = total_fuel / len(results)
success_fuel = [r['fuel_usage'] for r in results if r['reached_target']]
avg_success_fuel = sum(success_fuel) / len(success_fuel) if success_fuel else 0

# Include fuel statistics in the output
print(f"\nTotal fuel used: {total_fuel:.4f} kg")
print(f"Average fuel per run: {avg_fuel:.4f} kg")
if success_fuel:
    print(f"Average fuel for successful runs: {avg_success_fuel:.4f} kg")

# Calculate success statistics
success_count = sum(1 for r in results if r['reached_target'])
failure_count = sum(1 for r in results if r['entered_failure'])
brt_correct = sum(1 for r in results if r['initial_in_brt'] == r['reached_target'])

# Display these statistics in console too
print(f"\nSuccess rate: {success_count}/{len(results)} ({100 * success_count / len(results):.0f}%)")
print(f"Failure rate: {failure_count}/{len(results)} ({100 * failure_count / len(results):.0f}%)")
print(f"BRT prediction accuracy: {brt_correct}/{len(results)} ({100 * brt_correct / len(results):.0f}%)")

# Calculate safe docking success rate
safe_dock_count = sum(1 for r in results if r['docked_safely'])
print(f"\nSafe docking rate: {safe_dock_count}/{len(results)} ({100 * safe_dock_count / len(results):.0f}%)")

# Save results to a file
try:
    with open('outputs/simulation_results.txt', 'w') as f:
        f.write("Spacecraft Docking Simulation Results\n")
        f.write("===================================\n\n")
        
        for i, result in enumerate(results):
            s0 = result['initial_state']
            f.write(f"Run {i+1}:\n")
            f.write(f"  Initial state: {s0}\n")
            f.write(f"  Initial state in BRT: {result['initial_in_brt']}\n")
            f.write(f"  Entered failure set: {result['entered_failure']}\n")
            f.write(f"  Reached target set: {result['reached_target']}\n")
            f.write(f"  Docked safely: {result['docked_safely']}\n")  # Add this line
            f.write(f"  Fuel used: {result['fuel_usage']:.4f} kg\n\n")
        
        f.write("Summary:\n")
        f.write(f"  Success rate: {success_count}/{len(results)} ({100 * success_count / len(results):.0f}%)\n")
        f.write(f"  Safe docking rate: {safe_dock_count}/{len(results)} ({100 * safe_dock_count / len(results):.0f}%)\n")  # Add this line
        f.write(f"  Total fuel used: {total_fuel:.4f} kg\n")
        f.write(f"  Average fuel per run: {avg_fuel:.4f} kg\n")
        if success_fuel:
            f.write(f"  Average fuel for successful runs: {avg_success_fuel:.4f} kg\n")
        f.write(f"  Using disturbance bounds: ±{d_bar} m/s², ±{d_theta_bar} rad/s²\n")
    
    print(f"\nDetailed results saved to outputs/simulation_results.txt")
except:
    print("\nCould not save results to file")