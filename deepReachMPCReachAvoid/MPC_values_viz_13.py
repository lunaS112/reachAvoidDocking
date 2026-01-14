import numpy as np
import torch
import datetime
from pathlib import Path
from dynamics import dynamics
from utils import MPC, MPC_viz_helper as viz
from utils import trajectory_animation as traj_anim

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
    T = 6
    dt = 0.5
    style = "receding"  # Can be "direct" or "receding"
    cost_type = "reachability"  # Can be "reachability", "classic_mpc", or "mixed"
    mpc_percentage = 0.8  # Used only if cost_type is "mixed"
    
    # Resolution for BRT computation (13D state space)
    # [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    x_res = 51
    y_res = 51
    z_res = 51
    vx_res = 51
    vy_res = 51
    vz_res = 51
    qw_res = 51
    qx_res = 51
    qy_res = 51
    qz_res = 51
    wx_res = 51
    wy_res = 51
    wz_res = 51
    resolutions = [x_res, y_res, z_res, vx_res, vy_res, vz_res, 
                   qw_res, qx_res, qy_res, qz_res, wx_res, wy_res, wz_res]
    
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
                      x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, 
                      save_def=save_def)
    viz.plotGoalAvoid(costs, dynamics_, initial_condition_tensor, 
                      x_resolution=x_res, y_resolution=y_res,
                      x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0,
                      save_def=save_def)

    print("Images saved successfully!")
    print("13D dynamics test complete!")
