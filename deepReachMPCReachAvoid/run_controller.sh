#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_10sec_FixedScaling/training/checkpoints/model_final.pth"
CKPT_AVOID="runs/Docking6D_RA_avoid/training/checkpoints/model_final.pth"

########################### Single controller runs ###########################

python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 60.0 --safety_filter_margin 0.05 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --initial_px -10.442982320340697 --initial_py -2.0512017498686443 --initial_theta 1.205599463157637 \
  --initial_vx 0.6868342952257529 --initial_vy 0.04974792745952561 --initial_omega -0.18448436899393705 \
 --output_dir ./outputs/brt_safety_filter_1_timeout_margin_0.05_test

python run_controller.py single --controller brt \
  --checkpoint_path $CKPT --tMax 15.0 --max_sim_time 70.0 --safety_filter_margin 0.02 --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --initial_px 4.849024119721175 --initial_py 8.700267469331695 --initial_theta 3.0716048110691005 --initial_omega 0.2481656543798394 \
  --initial_vx -0.7225675839837122 --initial_vy 0.37521647241745115 \
  --output_dir ./outputs/brt_filter_1_test

  "initial_state": [
          4.849024119721175,
          8.700267469331695,
          -0.7225675839837122,
          0.37521647241745115,
          3.0716048110691005,
          0.2481656543798394
        ]

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

########################### 6D Geometry BRAT run #############################

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
  --n_rollouts 10000 --tMax 15.0 --max_sim_time 60.0 --effort_weight 0.005\
  --sampling_method uniform --output_dir ./outputs/BRT_10000_safety_filter_1

# Volume comparison
python volume_comparison.py \
    --checkpoint_path $CKPT \
    --time_horizons 5 10 15 --n_monte_carlo 500000

# 6D Geometry: UNIFORM IC
python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 10000 --tMax 10.0 --max_sim_time 90.0 --gradient_fallback --grad_threshold 0.01\
  --sampling_method uniform --output_dir ./outputs/BRAT_10000_uniform_IC_SF-1_FixedScaling
# 6D Geometry: BRAT IC
python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 0 --safety_checkpoint_path $CKPT_AVOID\
  --n_rollouts 10000 --tMax 10 --max_sim_time 60.0 --gradient_fallback --grad_threshold 0.01\
  --sampling_method brat --output_dir ./outputs/BRAT_10000_brat_IC_SF-0_FixedScaling
