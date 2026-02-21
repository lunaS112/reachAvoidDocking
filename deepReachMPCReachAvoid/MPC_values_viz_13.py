import argparse
import numpy as np
import torch
import datetime
from pathlib import Path
from dynamics import dynamics
from utils import MPC, MPC_viz_helper_13D as viz
from utils import trajectory_animation_13D as traj_anim

torch.manual_seed(1)
np.random.seed(1)

ROLLOUT_NUM = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description='13D MPC Values Visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step for MPC')
    parser.add_argument('--T', type=int, default=15,
                        help='Time horizon')
    parser.add_argument('--style', type=str, default='receding',
                        choices=['direct', 'receding'],
                        help='MPC style')
    parser.add_argument('--cost-type', type=str, default='reachability',
                        choices=['reachability', 'classic_mpc', 'mixed'],
                        help='Cost type for MPC')
    parser.add_argument('--mpc-percentage', type=float, default=0.8,
                        help='MPC percentage (only used if cost_type is mixed)')
    
    # Resolution parameters
    parser.add_argument('--resolution', type=int, default=51,
                        help='Resolution for all BRT slices')
    
    # Plot flags (all default to False, use flag to enable)
    parser.add_argument('--brt-heatmap', action='store_true',
                        help='Plot BRT heatmap images')
    parser.add_argument('--goal-avoid', action='store_true',
                        help='Plot goal/avoid set visualization')
    parser.add_argument('--position-brt', action='store_true',
                        help='Plot position (x-y) BRT slice')
    parser.add_argument('--velocity-brt', action='store_true',
                        help='Plot velocity (vx-vy) BRT slice')
    parser.add_argument('--rotation-brt', action='store_true',
                        help='Plot rotation (yaw-wz) BRT slice')
    parser.add_argument('--trajectories', action='store_true',
                        help='Plot MPC trajectories')
    parser.add_argument('--controls', action='store_true',
                        help='Plot MPC control inputs')
    parser.add_argument('--animation', action='store_true',
                        help='Generate 3D trajectory animation GIF')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plots and animations')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    dynamics_ = dynamics.Docking13D('reach_avoid')

    # Parameters from args
    T = args.T
    dt = args.dt
    style = args.style
    cost_type = args.cost_type.replace('-', '_')  # Handle hyphen in choice
    mpc_percentage = args.mpc_percentage
    
    # Resolution for BRT computation (13D state space)
    res = args.resolution
    x_res = y_res = vx_res = vy_res = theta_res = wz_res = res
    resolutions = [x_res, y_res, vx_res, vy_res, theta_res, wz_res]
    
    # Determine which plots to generate
    if args.all:
        plot_brt_heatmap = True
        plot_goal_avoid = True
        plot_position_brt = True
        plot_velocity_brt = True
        plot_rotation_brt = True
        plot_trajectories = True
        plot_controls = True
        plot_animation = True
    else:
        plot_brt_heatmap = args.brt_heatmap
        plot_goal_avoid = args.goal_avoid
        plot_position_brt = args.position_brt
        plot_velocity_brt = args.velocity_brt
        plot_rotation_brt = args.rotation_brt
        plot_trajectories = args.trajectories
        plot_controls = args.controls
        plot_animation = args.animation
    
    # Check if any plot is requested
    any_plot = (plot_brt_heatmap or plot_goal_avoid or plot_position_brt or 
                plot_velocity_brt or plot_rotation_brt or plot_trajectories or 
                plot_controls or plot_animation)
    
    if not any_plot:
        print("No plots requested. Use --help to see available options.")
        print("Example: python3 MPC_values_viz_13.py --dt 0.1 --T 15 --position-brt")
        exit(0)
    
    # Save definition root
    save_root = Path(__file__).parent
    daytime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_def = f"MPC_13D_"
    save_def = save_root / f"data/{daytime}__{style}_{cost_type}_{dt}_{T}_13D" / save_def
    
    print(f"Configuration: dt={dt}, T={T}, style={style}, cost_type={cost_type}")
    print(f"Output directory: {save_def.parent}")
    
    # Initialize MPC controller
    mpc = MPC.MPC(horizon=None, receding_horizon=1, dT=dt, num_samples=100,
                  dynamics_=dynamics_, device=device, mode="MPC", sample_mode="gaussian",
                  style=style, num_iterative_refinement=10, 
                  cost_type=cost_type, mpc_percentage=mpc_percentage)

    # Compute all BRT slices once using centralized function
    print("Computing all BRT slices for 13D dynamics...")
    brt_data = viz.compute_all_brt_slices(mpc, dynamics_, device, T, resolutions)
    
    # Get position data for backward compatibility with existing functions
    costs = brt_data['position']['costs']
    x_min, x_max, y_min, y_max = brt_data['position']['coords']
    initial_condition_tensor = brt_data['position']['initial_conditions']
    
    # Select initial conditions for visualization (13D state)
    # [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    # Use goal quaternion so trajectories only need position change to reach goal
    q_goal = dynamics_.q_goal.cpu()
    q0, q1, q2, q3 = float(q_goal[0]), float(q_goal[1]), float(q_goal[2]), float(q_goal[3])
    interesting_ics = []
    interesting_ics.append(torch.tensor([0.5, -0.5, 0.5, 0, 0, 0, q0, q1, q2, q3, 0, 0, 0]).to(device))  
    interesting_ics.append(torch.tensor([-0.5, 0.5, -0.5, 0, 0, 0, q0, q1, q2, q3, 0, 0, 0]).to(device))  
    interesting_ics_tensor = torch.stack(interesting_ics)

    # Generate requested plots
    if plot_brt_heatmap:
        print("Plotting BRT heatmap...")
        viz.plotBRTImages(costs, dynamics_, initial_condition_tensor, 
                          x_resolution=x_res, y_resolution=y_res, 
                          x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, 
                          save_def=save_def)

    if plot_goal_avoid:
        print("Plotting Goal/Avoid sets...")
        viz.plotGoalAvoid(costs, dynamics_, initial_condition_tensor, 
                          x_resolution=x_res, y_resolution=y_res,
                          x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                          save_def=save_def)

    if plot_position_brt:
        print("Generating Position BRT slice...")
        viz.plotBRTPosition(mpc, interesting_ics_tensor, brt_data, x_res, y_res, T,
            save_def=save_def, level_sets=[0.0, 0.1, 0.15])

    if plot_velocity_brt:
        print("Generating Velocity BRT slice...")
        viz.plotBRTVelocity(mpc, interesting_ics_tensor, brt_data, vx_res, vy_res, T,
            save_def=save_def, level_sets=[0.0, 0.1, 0.15])

    if plot_rotation_brt:
        print("Generating Rotation BRT slice...")
        viz.plotBRTRotation(mpc, interesting_ics_tensor, brt_data, theta_res, wz_res, T,
            save_def=save_def, level_sets=[0.0, 0.1, 0.15])

    if plot_trajectories:
        print("Generating MPC Trajectories plot...")
        viz.plotMPCTrajectories(mpc, interesting_ics_tensor, T, save_def=save_def, max_trajs=5)

    if plot_controls:
        print("Generating MPC Controls plot...")
        viz.plotMPCControls(mpc, interesting_ics_tensor, T, save_def=save_def, max_trajs=5)

    if plot_animation:
        print("Generating 3D trajectory animation...")
        traj_anim.animate_trajectory_13D(mpc, interesting_ics_tensor, T, dt, save_def=save_def)

    print("Done!")
    print(f"Outputs saved to: {save_def.parent}")
