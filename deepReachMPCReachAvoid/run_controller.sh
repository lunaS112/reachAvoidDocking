#!/bin/bash
# Activate venv (provides 'python' command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# RUN IN TERMINAL FIRST

CKPT="runs/Docking6D_RA_10sec_HighSamp/training/checkpoints/model_final.pth"
CKPT_AVOID="runs/Docking6D_RA_avoid/training/checkpoints/model_final.pth"

#################### Gradient MPC Baseline (analytical cost only) ############

# Shorter horizon, faster per-step (less accurate but quicker)
python run_controller.py single --controller mpc \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID \
  --planning_horizon 2.0 --mpc_dt 0.5 \
  --gradient_iters 50 --num_restarts 4 --gradient_lr 1.0 --goal_weight 0.01 \
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

# Quick 3-way comparison: BRAT vs MPC baseline vs MPC+Terminal (10 ICs)
python run_controller.py compare --controllers brat mpc mpc_terminal \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path $CKPT_AVOID \
  --gradient_iters 50 --num_restarts 8 --gradient_lr 1.0 --goal_weight 0.01 \
  --planning_horizon 10.0 --mpc_dt 0.5 --effective_horizon 1.0 \
  --n_rollouts 10 --seed 1 --sampling_method uniform \
  --output_dir ./outputs/comparison_3way_quick

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

