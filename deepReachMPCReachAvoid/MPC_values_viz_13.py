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

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    dynamics_ = dynamics.Docking13D('reach_avoid')

    # Required parameters
    T = 30
    dt = 0.1
    style = "receding"  # Can be "direct" or "receding"
    cost_type = "reachability"  # Can be "reachability", "classic_mpc", or "mixed"
    mpc_percentage = 0.8  # Used only if cost_type is "mixed"
    
    # Resolution for BRT computation (13D state space)
    # The 13D helper computes 3 slices: position (x,y), velocity (vx,vy), rotation (delta_yaw, wz)
    # resolutions = [x_res, y_res, vx_res, vy_res, theta_res, wz_res]
    x_res = 51
    y_res = 51
    vx_res = 51
    vy_res = 51
    theta_res = 51  # delta yaw resolution
    wz_res = 51     # wz angular velocity resolution
    resolutions = [x_res, y_res, vx_res, vy_res, theta_res, wz_res]
    
    # Save definition root
    save_root = Path(__file__).parent
    daytime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_def = f"MPC_13D_"
    save_def = save_root / f"data/{daytime}__{style}_{cost_type}_{dt}_{T}_13D" / save_def
    
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
    interesting_ics = []
    # Identity quaternion: [1, 0, 0, 0]
    interesting_ics.append(torch.tensor([0.5, -0.5, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).to(device))  
    interesting_ics.append(torch.tensor([-0.5, 0.5, -0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).to(device))  
    interesting_ics_tensor = torch.stack(interesting_ics)

    print("Plotting BRT and Reach Avoid sets...")
    viz.plotBRTImages(costs, dynamics_, initial_condition_tensor, 
                      x_resolution=x_res, y_resolution=y_res, 
                      x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, 
                      save_def=save_def)
    viz.plotGoalAvoid(costs, dynamics_, initial_condition_tensor, 
                      x_resolution=x_res, y_resolution=y_res,
                      x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                      save_def=save_def)

    print("Generating BRT slice plots...")
    print("Generating Position BRT")
    viz.plotBRTPosition(mpc, interesting_ics_tensor, brt_data, x_res, y_res, T,
        save_def=save_def, level_sets=[0.0, 0.1, 0.15])
    print("Generating Velocity BRT")
    viz.plotBRTVelocity(mpc, interesting_ics_tensor, brt_data, vx_res, vy_res, T,
        save_def=save_def, level_sets=[0.0, 0.1, 0.15])
    print("Generating Rotation BRT")
    viz.plotBRTRotation(mpc, interesting_ics_tensor, brt_data, theta_res, wz_res, T,
        save_def=save_def, level_sets=[0.0, 0.1, 0.15])

    print("Generating MPC Trajectories plot...")
    viz.plotMPCTrajectories(mpc, interesting_ics_tensor, T, save_def=save_def, max_trajs=5)

    print("Generating MPC Controls plot...")
    viz.plotMPCControls(mpc, interesting_ics_tensor, T, save_def=save_def, max_trajs=5)

    print("Generating 3D trajectory animation...")
    traj_anim.animate_trajectory_13D(mpc, interesting_ics_tensor, T, dt, save_def=save_def)

    print("Images saved successfully!")
    print("13D dynamics visualization complete!")
