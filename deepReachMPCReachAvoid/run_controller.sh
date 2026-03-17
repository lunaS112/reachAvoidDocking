#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_10sec_FixedScaling/training/checkpoints/model_final.pth"
CKPT_AVOID="runs/Docking6D_RA_avoid/training/checkpoints/model_final.pth"

########################### Single controller runs ###########################

python run_controller.py single --controller brat \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 90.0 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --gradient_fallback --grad_threshold 0.01 --avoid_proximity_margin 1.0 --skip_frames 10\
  --initial_state -2.09787499 -3.60878275 -0.19819089 0.35746812 1.11985678 0.02817976 \
 --output_dir ./outputs/Collision_BRAT_SF_1

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

########################### Comparison runs ##################################

# UNIFORM IC
python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 10000 --tMax 10.0 --max_sim_time 90.0 --gradient_fallback --grad_threshold 0.01\
  --sampling_method uniform --output_dir ./outputs/BRAT_10000_uniform_IC_SF-1_FixedScaling
# BRAT IC
python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 0 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 10000 --tMax 10 --max_sim_time 60.0 --gradient_fallback --grad_threshold 0.01\
  --sampling_method brat --output_dir ./outputs/BRAT_10000_brat_IC_SF-0_FixedScaling

