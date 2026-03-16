#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_10sec_HighSamp/training/checkpoints/model_final.pth"
CKPT_AVOID="runs/Docking6D_RA_avoid/training/checkpoints/model_final.pth"

########################### Single controller runs ###########################

python run_controller.py single --controller brat \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --initial_px -10.442982320340697 --initial_py -2.0512017498686443 --initial_theta 1.205599463157637 \
  --initial_vx 0.6868342952257529 --initial_vy 0.04974792745952561 --initial_omega -0.18448436899393705 \
 --output_dir ./outputs/brt_safety_filter_1_timeout_1

python run_controller.py single --controller brat \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 90.0 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --gradient_fallback --grad_threshold 0.01 --avoid_proximity_margin 1.0 --skip_frames 10\
  --initial_px -7.6842415069805465 --initial_py 9.83105334616458  --initial_vx -0.7089186102031108  \
  --initial_vy 0.2557012652676034 --initial_theta -0.5195892507411801 --initial_omega 0.05868982844575166 \
 --output_dir ./outputs/brat_timeout_1_gradient_fallback

"initial_state": [
          -7.6842415069805465,
          9.83105334616458,
          -0.7089186102031108,
          0.2557012652676034,
          -0.5195892507411801,
          0.05868982844575166
        ],
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

# Quick comparison
python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 1000 --tMax 15.0 --max_sim_time 60.0 --effort_weight 0.005\
  --sampling_method uniform --output_dir ./outputs/BRT_1000safety_filter_1 

python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 0 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 10000 --tMax 10 --max_sim_time 60.0 --gradient_fallback --grad_threshold 0.01\
  --sampling_method uniform --output_dir ./outputs/BRAT_10000_uniform_IC_sampling_SF-1_GradFallback

