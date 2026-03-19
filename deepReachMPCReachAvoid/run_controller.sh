#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_10sec_HighSamp/training/checkpoints/model_final.pth"
CKPT_AVOID="runs/Docking6D_RA_avoid/training/checkpoints/model_final.pth"

#################### Gradient MPC Baseline (analytical cost only) ############

python run_controller.py single --controller brat \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID \
  --initial_state  4.538664617848674 7.794189959332044 -0.6292057093074211 -0.40244653427131194 -1.837042145950629 0.4173335630816306 \
  --skip_frames 10 \
  --output_dir ./outputs/BRAT_collision_test

# Shorter horizon, faster per-step (less accurate but quicker)
python run_controller.py single --controller mpc \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID \
  --planning_horizon 2.0 --mpc_dt 0.5 \
  --gradient_iters 25 --num_restarts 4 --gradient_lr 1.0 --goal_weight 0.01 \
  --skip_frames 10 \
  --output_dir ./outputs/MPC_gradient_short_horizon_Test

#################### Gradient MPC + Terminal Cost ############################

# Single run: gradient MPC + learned terminal cost
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --effective_horizon 1.0 \
  --gradient_iters 50 --num_restarts 8 --gradient_lr 1.0 \
  --skip_frames 10 \
  --output_dir ./outputs/MPC_terminal_gradient

# MPC+Terminal with light effort penalty (fuel savings)
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --effective_horizon 1.0 \
  --gradient_iters 50 --num_restarts 8 --gradient_lr 1.0 \
  --effort_weight 0.005 \
  --skip_frames 10 \
  --output_dir ./outputs/MPC_terminal_gradient_effort_light

# MPC+Terminal with moderate effort penalty
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --effective_horizon 1.0 \
  --gradient_iters 50 --num_restarts 8 --gradient_lr 1.0 \
  --effort_weight 0.05 \
  --skip_frames 10 \
  --output_dir ./outputs/MPC_terminal_gradient_effort_moderate

# MPC+Terminal with safety filter
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --effective_horizon 1.0 --effort_weight 0.05 --planning_horizon 2.0 --mpc_dt 0.5\
  --gradient_iters 10 --num_restarts 3 --gradient_lr 1.0 \
  --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID \
  --skip_frames 10 \
  --output_dir ./outputs/MPC_terminal_gradient_SF_fast

########################### Comparison runs ##################################

# Quick 3-way comparison: BRAT vs MPC baseline vs MPC+Terminal (50 ICs)
python run_controller.py compare --controllers brat grid_based\
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID \
  --mpc_gradient_iters 30 --mpc_num_restarts 4 --gradient_lr 1.0 --goal_weight 0.01 \
  --mpc_terminal_gradient_iters 10 --mpc_terminal_num_restarts 1 \
  --planning_horizon 2.0 --mpc_dt 0.5 --effective_horizon 1.0 \
  --n_rollouts 10 --seed 42 --sampling_method uniform \
  --output_dir ./outputs/2_way_comparison_10_uniform_IC_SF-1_HighSamp

# Large-scale BRAT-only baseline (uniform IC)
python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID \
  --n_rollouts 10000 --tMax 10.0 --max_sim_time 90.0 --gradient_fallback --grad_threshold 0.01 \
  --sampling_method uniform --output_dir ./outputs/BRAT_10000_uniform_IC_SF-1_FixedScaling

# Large-scale BRAT-only baseline (BRAT IC)
python run_controller.py compare --controllers brat \
  --checkpoint_path $CKPT --safety_filter_mode 0 --safety_checkpoint_path $CKPT_AVOID \
  --n_rollouts 10000 --tMax 10 --max_sim_time 60.0 --gradient_fallback --grad_threshold 0.01 \
  --sampling_method brat --output_dir ./outputs/BRAT_10000_brat_IC_SF-0_FixedScaling

