#!/bin/bash
# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_15sec-fine/training/checkpoints/model_final.pth"
CKPT_INNER="runs/Docking6D_3sec/training/checkpoints/model_final.pth"d
CKPT_AVOID="runs/Docking6D_RA_avoid/training/checkpoints/model_final.pth"

########################### Single controller runs ###########################

# BRT single run (default initial condition)
python run_controller.py single --controller brt \
   --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 \
   --initial_px -8.1572345 --initial_theta -0.50771584 --initial_py -4.0154211\
   --initial_vx -0.20646505 --initial_vy 0.07763347 --initial_omega 0.27782925\
  --output_dir ./outputs/single_brt

python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --initial_px 0.41228244118177493 --initial_py -0.5943343367056588 --initial_theta 0.27652373189896373 \
  --initial_vx -0.6946567118136735 --initial_vy 0.243612463480831 --initial_omega 0.2312060204561165 \
 --output_dir ./outputs/brt_safety_filter_fail

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
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT \
  --tMax 15.0 --inner_tMax 3.0 --effective_horizon 3.0 --max_sim_time 60.0 \
  --num_samples 100 --num_refinement 10 \
  --effort_weight 0.005 \
  --output_dir ./outputs/single_cascaded_mpc_terminal_effort

########################### Comparison runs ##################################

# Quick comparison
python run_controller.py compare --controllers brt \
  --checkpoint_path $CKPT --safety_filter_mode 2 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 10000 --tMax 15.0 --max_sim_time 60.0 --effort_weight 0.005\
  --sampling_method uniform --output_dir ./outputs/BRT_10000safety_filter_2 
   
# Volume comparison 
python volume_comparison.py \
    --checkpoint_path $CKPT \
    --time_horizons 5 10 15 --n_monte_carlo 500000