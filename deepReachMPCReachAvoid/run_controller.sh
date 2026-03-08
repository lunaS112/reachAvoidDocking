#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_15sec-fine/training/checkpoints/model_final.pth"
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
  --initial_px -3.407509855901665 --initial_py -2.9689989713225415 --initial_theta 0.49913273937622415 --initial_omega -0.017424178189499262 \
  --initial_vx -1.026343555401426 --initial_vy -0.046729930795811336 \
  --output_dir ./outputs/brt_filter_1_test

  "initial_state": [
          -3.407509855901665,
          -2.9689989713225415,
          0.49913273937622415,
          -0.017424178189499262,
          -1.026343555401426,
          -0.046729930795811336
        ],

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
  --checkpoint_path $CKPT --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 1000 --tMax 15.0 --max_sim_time 60.0 --effort_weight 0.005\
  --sampling_method uniform --output_dir ./outputs/BRT_1000safety_filter_1 
   
# Volume comparison 
python volume_comparison.py \
    --checkpoint_path $CKPT \
    --time_horizons 5 10 15 --n_monte_carlo 500000
