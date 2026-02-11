#!/bin/bash
# RUN IN TERMINAL FIRST
CKPT="runs/Docking6D_RA_15_114000/training/checkpoints/model_epoch_114000.pth"
CKPT="runs/Docking6D_RA_15_145000/training/checkpoints/model_epoch_145000.pth"
CKPT_INNER="runs/Docking6D_3sec/training/checkpoints/model_final.pth"

########################### Single controller runs ###########################

# BRT single run (default initial condition)
python run_controller.py single --controller brt \
   --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
   --initial_px -8.1572345 --initial_theta -0.50771584 --initial_py -4.0154211\
   --initial_vx -0.20646505 --initial_vy 0.07763347 --initial_omega 0.27782925\
  --output_dir ./outputs/single_brt

# MPC single run
python run_controller.py single --controller mpc \
   --checkpoint_path $CKPT --planning_horizon 20.0 --mpc_dt 0.5 --max_sim_time 30.0 \
   --initial_px 5.0 --initial_theta 0.0 --initial_py 4.0\
   --num_samples 100 --num_refinement 10 \
   --output_dir ./outputs/single_mpc

# MPC+Terminal single run
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 15.0 --effective_horizon 3.0 --max_sim_time 45.0 \
  --num_samples 500 --num_refinement 10 \
  --output_dir ./outputs/single_mpc_terminal

# BRT single run with custom initial condition and no animation
python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 14.0 --max_sim_time 30.0 \
  --initial_px 5.0 --initial_py -8.0 --initial_theta 0.0 \
  --no_animation \
  --output_dir ./outputs/single_brt_custom

########################### Cascaded controller runs #########################
# --no_value_function --no_animation
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

########################### Comparison runs ##################################

# Quick comparison
python run_controller.py compare --controllers brt mpc mpc_terminal cascaded_brt cascaded_mpc_terminal\
  --checkpoint_path $CKPT --n_rollouts 3 --tMax 15.0 --max_sim_time 60.0 \
  --output_dir ./outputs/comparison_quick --animate 

# Full comparison
python run_controller.py compare --controllers brt mpc mpc_terminal cascaded_brt cascaded_mpc_terminal \
  --checkpoint_path $CKPT --inner_checkpoint_path $CKPT_INNER --n_rollouts 50 \
  --tMax 15.0 --inner_tMax 3.0 --max_sim_time 60.0 \
  --output_dir ./outputs/comparison_full

# BRT vs MPC only
python run_controller.py compare --controllers brt mpc \
  --checkpoint_path $CKPT --n_rollouts 20 --tMax 14.0 --max_sim_time 30.0 \
  --output_dir ./outputs/comparison_brt_vs_mpc

# BRT vs MPC+Terminal only
python run_controller.py compare --controllers brt mpc_terminal \
  --checkpoint_path $CKPT --n_rollouts 20 --tMax 14.0 --max_sim_time 30.0 \
  --output_dir ./outputs/comparison_brt_vs_terminal
