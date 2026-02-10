"""
Test script for BRT-based optimal controller.

This script demonstrates the two-phase control strategy using the learned 
DeepReach value function for spacecraft docking.

Usage:
    python run_brt_controller.py --tMax 14.0 --sim_time 30.0
"""

import argparse
import numpy as np
import os
import sys

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.brt_controller import BRTController
from utils.brt_animation import (
    create_deepreach_animation, 
    plot_trajectory_static,
    plot_simulation_data
)


def main():
    parser = argparse.ArgumentParser(description='Run BRT-based docking controller')
    
    # Paths
    parser.add_argument('--checkpoint_path', type=str, 
                        default='runs/Docking6D_RA_14sec/training/checkpoints/model_epoch_145000.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/brt_control',
                        help='Directory to save outputs')
    
    # Controller parameters
    parser.add_argument('--tMax', type=float, default=14.0,
                        help='BRT time horizon (use lower than trained to avoid artifacts)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Control update frequency (seconds)')
    
    # Simulation parameters
    parser.add_argument('--sim_time', type=float, default=30.0,
                        help='Maximum simulation time (seconds)')
    parser.add_argument('--initial_px', type=float, default=2.0,
                        help='Initial x position (m)')
    parser.add_argument('--initial_py', type=float, default=10.0,
                        help='Initial y position (m)')
    parser.add_argument('--initial_vx', type=float, default=0.0,
                        help='Initial x velocity (m/s)')
    parser.add_argument('--initial_vy', type=float, default=0.0,
                        help='Initial y velocity (m/s)')
    parser.add_argument('--initial_theta', type=float, default=-1.57,
                        help='Initial orientation (rad), default pi/2')
    parser.add_argument('--initial_omega', type=float, default=0.0,
                        help='Initial angular velocity (rad/s)')
    
    # Animation parameters
    parser.add_argument('--skip_frames', type=int, default=5,
                        help='Frames to skip in animation')
    parser.add_argument('--resolution', type=int, default=40,
                        help='Value function grid resolution')
    parser.add_argument('--no_animation', action='store_true',
                        help='Skip animation generation (faster)')
    parser.add_argument('--no_value_function', action='store_true',
                        help='Skip value function heatmap in animation (faster)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        print("Please ensure you have a trained model in the runs/ directory.")
        return
    
    print("=" * 60)
    print("BRT-Based Optimal Controller for Docking6D")
    print("=" * 60)
    
    # Initialize controller
    print(f"\nLoading controller with tMax={args.tMax}s, dt={args.dt}s...")
    controller = BRTController(
        checkpoint_path=args.checkpoint_path,
        tMax=args.tMax,
        dt=args.dt
    )
    
    # Define initial state
    initial_state = np.array([
        args.initial_px,
        args.initial_py,
        args.initial_vx,
        args.initial_vy,
        args.initial_theta,
        args.initial_omega
    ])
    
    print(f"\nInitial state: px={initial_state[0]:.2f}, py={initial_state[1]:.2f}, "
          f"vx={initial_state[2]:.2f}, vy={initial_state[3]:.2f}, "
          f"θ={initial_state[4]:.2f}, ω={initial_state[5]:.2f}")
    
    # Check initial BRT status
    initial_value = controller.get_value(initial_state, args.tMax)
    print(f"Initial V(x, tMax={args.tMax}s) = {initial_value:.4f}")
    print(f"Initial state is {'INSIDE' if initial_value <= 0 else 'OUTSIDE'} the BRT")
    
    # Run simulation
    print(f"\nRunning simulation for {args.sim_time}s...")
    result = controller.simulate_docking(initial_state, args.sim_time)
    
    # Print results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Final state: {result['final_state']}")
    print(f"Simulation duration: {result['times'][-1]:.2f}s")
    
    if result['phase_transition_time'] is not None:
        print(f"Entered BRT (Phase 2) at t={result['phase_transition_time']:.2f}s")
    else:
        print("Never entered BRT (stayed in Phase 1)")
    
    # Count phase durations
    phase1_time = np.sum(result['phases'] == 1) * args.dt
    phase2_time = np.sum(result['phases'] == 2) * args.dt
    print(f"Time in Phase 1 (Convergence): {phase1_time:.1f}s")
    print(f"Time in Phase 2 (Precision): {phase2_time:.1f}s")
    
    # Final value
    final_value = controller.get_value(result['final_state'], 0.1)
    print(f"Final V(x, t=0.1s) = {final_value:.4f}")
    
    # Generate outputs
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)
    
    # Static trajectory plot
    trajectory_path = os.path.join(args.output_dir, 'trajectory.png')
    print(f"\nGenerating trajectory plot: {trajectory_path}")
    plot_trajectory_static(controller, result, save_path=trajectory_path, show_brt=True)
    
    # Simulation data plot
    data_path = os.path.join(args.output_dir, 'simulation_data.png')
    print(f"Generating data plots: {data_path}")
    plot_simulation_data(result, save_path=data_path)
    
    # Animation
    if not args.no_animation:
        animation_path = os.path.join(args.output_dir, 'docking_animation.mp4')
        print(f"\nGenerating animation: {animation_path}")
        print("(This may take a few minutes...)")
        create_deepreach_animation(
            controller, result, animation_path,
            skip_frames=args.skip_frames,
            resolution=args.resolution,
            show_value_function=not args.no_value_function
        )
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Outputs saved to: {args.output_dir}")
    

if __name__ == '__main__':
    main()
