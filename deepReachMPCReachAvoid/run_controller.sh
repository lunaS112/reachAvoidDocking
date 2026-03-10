#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_15sec-fine/training/checkpoints/model_final.pth"
CKPT="runs/Docking6D_RA_15sec-fine_newGeom/training/checkpoints/model_horizon_12.50.pth"
CKPT_INNER="runs/Docking6D_3sec/training/checkpoints/model_final.pth"d
CKPT_AVOID="runs/Docking6D_RA_avoid/training/checkpoints/model_final.pth"

########################### Single controller runs ###########################



python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --initial_px -10.442982320340697 --initial_py -2.0512017498686443 --initial_theta 1.205599463157637 \
  --initial_vx 0.6868342952257529 --initial_vy 0.04974792745952561 --initial_omega -0.18448436899393705 \
 --output_dir ./outputs/brt_safety_filter_1_timeout_1

python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --initial_px 3.1188486779158744 --initial_py 8.553503388304648 --initial_theta -2.701630548995958 --initial_omega -0.01365488906296819 \
  --initial_vx -0.5148129080308737 --initial_vy -0.7221356967338857 \
  --output_dir ./outputs/brt_filter_1_test

  "initial_state": [
          3.1188486779158744,
          8.553503388304648,
          -0.5148129080308737,
          -0.7221356967338857,
          -2.701630548995958,
          -0.01365488906296819

python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --safety_filter_mode 0\
  --tMax 12.5  --max_sim_time 60.0 \
  --output_dir ./outputs/brt_newGeom

# Cascaded MPC+Terminal single run
python run_controller.py single --controller cascaded_mpc_terminal \
  --checkpoint_path $CKPT --inner_checkpoint_path $CKPT_INNER \
  --tMax 15.0 --inner_tMax 3.0 --effective_horizon 3.0 --max_sim_time 60.0 \
  --num_samples 500 --num_refinement 10 \
  --output_dir ./outputs/single_cascaded_mpc_terminal


#################### Control effort penalty (fuel minimization) ##############

# MPC+Terminal with light effort penalty (safety-first, slight fuel savings)
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
  --num_samples 100 --num_refinement 10 \
  --effort_weight 0.005 \
  --output_dir ./outputs/single_mpc_terminal_effort_light

# MPC+Terminal with moderate effort penalty
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
  --num_samples 100 --num_refinement 10 \
  --effort_weight 0.05 \
  --output_dir ./outputs/single_mpc_terminal_effort_moderate

# Cascaded MPC+Terminal with effort penalty
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT \
  --tMax 15.0 --inner_tMax 3.0 --effective_horizon 3.0 --max_sim_time 60.0 \
  --num_samples 100 --num_refinement 10 \
  --effort_weight 0.005 \
  --output_dir ./outputs/single_cascaded_mpc_terminal_effort

########################### Comparison runs ##################################

# Quick comparison
python run_controller.py compare --controllers brt \
  --checkpoint_path $CKPT --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 1000 --tMax 15.0 --max_sim_time 60.0 --effort_weight 0.005\
  --sampling_method uniform --output_dir ./outputs/BRT_1000safety_filter_1 

  python run_controller.py compare --controllers brt \
  --checkpoint_path $CKPT \
  --n_rollouts 5 --tMax 12.5 --max_sim_time 60.0 --effort_weight 0.005\
  --sampling_method brt --output_dir ./outputs/BRT_newGeom_BRAT
   
# Volume comparison 
python volume_comparison.py \
    --checkpoint_path $CKPT \
    --time_horizons 5 10 15 --n_monte_carlo 500000
