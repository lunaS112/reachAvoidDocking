#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST
CKPT="runs/Docking6D_RA_15_114000/training/checkpoints/model_epoch_114000.pth"
CKPT="runs/Docking6D_RA_15_145000/training/checkpoints/model_epoch_145000.pth"
CKPT="runs/Docking6D_RA_15sec-fine/training/checkpoints/model_final.pth"
CKPT="runs/Docking6D_RA_15sec-fine/training/checkpoints/model_epoch_138000.pth"
CKPT_INNER="runs/Docking6D_3sec/training/checkpoints/model_final.pth"

########################### Single controller runs ###########################

# BRT single run (default initial condition)
python run_controller.py single --controller brt \
   --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
   --initial_px -8.1572345 --initial_theta -0.50771584 --initial_py -4.0154211\
   --initial_vx -0.20646505 --initial_vy 0.07763347 --initial_omega 0.27782925\
  --output_dir ./outputs/single_brt

python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
  --initial_px 0 --initial_py 0.0 --initial_theta -3 \
  --initial_vx 0.0 --initial_vy 0.0 --initial_omega 0.375 \
  --output_dir ./outputs/odd3_brt

# MPC single run
python run_controller.py single --controller mpc \
   --checkpoint_path $CKPT --mpc_dt 0.1 --max_sim_time 60.0 \
   --num_samples 100 --num_refinement 10 \
   --output_dir ./outputs/single_mpc

# MPC+Terminal single run
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
  --num_samples 100 --num_refinement 10 --mpc_dt 0.5\
  --output_dir ./outputs/single_mpc_terminal

python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
  --initial_px -5.981874814109322 --initial_py 10.293041673097736 --initial_theta 1.026933217676297 --initial_omega 0.18254358031368267 \
  --initial_vx -0.14381762025741018 --initial_vy 0.9296800942967711 \
  --output_dir ./outputs/single_brt_failed

########################### Cascaded controller runs #########################

# Cascaded BRT single run
python run_controller.py single --controller cascaded_brt \
  --checkpoint_path $CKPT --inner_checkpoint_path $CKPT_INNER \
  --tMax 15.0 --inner_tMax 3.0 --max_sim_time 60.0 \
  --output_dir ./outputs/single_cascaded_brt

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
python run_controller.py single --controller cascaded_mpc_terminal \
  --checkpoint_path $CKPT --inner_checkpoint_path $CKPT_INNER \
  --tMax 15.0 --inner_tMax 3.0 --effective_horizon 3.0 --max_sim_time 60.0 \
  --num_samples 500 --num_refinement 10 \
  --effort_weight 0.01 \
  --output_dir ./outputs/single_cascaded_mpc_terminal_effort

########################### Comparison runs ##################################

# Quick comparison
python run_controller.py compare --controllers brt \
  --checkpoint_path $CKPT  \
  --n_rollouts 10000 --tMax 15.0 --max_sim_time 60.0 \
  --sampling_method brt --output_dir ./outputs/comparison_BRT_IC 
   
# Volume comparison 
python volume_comparison.py \
    --checkpoint_path $CKPT \
    --time_horizons 5 10 15 --n_monte_carlo 500000
