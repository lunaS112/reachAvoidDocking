import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patches as mpatches
import torch
from tqdm import tqdm
from dynamics import dynamics
from utils import MPC, modules
from utils import MPC_viz_helper as viz
import math

# mpl.use('Agg')
torch.manual_seed(1)
np.random.seed(1)

ROLLOUT_NUM = 100

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    dynamics_ = dynamics.Docking6D('reach_avoid')

    # Required parameters
    T = 15
    dt = 0.5
    save_def = "MPC_R|Cost_Reach"

    # Resolution for BRT computation
    x_res = 201
    y_res = 201
    
    mpc = MPC.MPC(horizon=None, receding_horizon=1, dT=dt, num_samples=100,
              dynamics_=dynamics_, device=device, mode="MPC", sample_mode="gaussian",
              style='receding', num_iterative_refinement=10, 
              cost_type="reachability", mpc_percentage=0.8)

    # Compute all BRT slices once using centralized function
    print("Computing all BRT slices...")
    brt_data = viz.compute_all_brt_slices(mpc, dynamics_, device, T, x_res, y_res)
    
    # Get position data for backward compatibility with existing functions
    costs = brt_data['position']['costs']
    x_min, x_max, y_min, y_max = brt_data['position']['coords']
    initial_condition_tensor = brt_data['position']['initial_conditions']
    
    # Select initial conditions for visualization
    interesting_ics = []
    interesting_ics.append(torch.tensor([-0.5, 0.5, 0, 0, np.pi/4, 0]).to(device))  
    interesting_ics.append(torch.tensor([-0.5, -0.5, 0, 0, np.pi/2, 0]).to(device)) 
    interesting_ics_tensor = torch.stack(interesting_ics)

    print("Plotting BRT and Reach Avoid sets...")
    viz.plotBRTImages(costs, dynamics_, initial_condition_tensor, x_resolution=x_res, y_resolution=y_res, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, save_def=save_def)
    viz.plotGoalAvoid(costs, dynamics_, initial_condition_tensor, x_resolution=x_res, y_resolution=y_res, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    viz.plotMPCTrajectories(mpc, interesting_ics_tensor, T, max_trajs=5, save_def=save_def)

    print("Generating BRT slice plots...")
    print("Generating Position BRT")
    viz.plotBRTPosition(mpc, interesting_ics_tensor, brt_data, x_res, y_res, T,
        level_sets=[0.0, 0.1, 0.15], save_def=save_def)
    print("Generating Velocity BRT")
    viz.plotBRTVelocity(mpc, interesting_ics_tensor, brt_data, x_res, y_res, T,
        level_sets=[0.0, 0.1, 0.15], save_def=save_def)
    print("Generating Rotation BRT")
    viz.plotBRTRotation(mpc, interesting_ics_tensor, brt_data, x_res, y_res, T,
        level_sets=[0.0, 0.1, 0.15], save_def=save_def)
    print("Generating MPC Controls Histograms")
    viz.plotMPCControls(mpc, interesting_ics_tensor, T, max_trajs=6, save_def=save_def)
    print("Images saved successfully!")