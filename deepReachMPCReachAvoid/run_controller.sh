#!/bin/bash
# ============================================================================
#  6D planar docking — controller rollout commands.
#
#  Edit the CKPT_* variables to point at your own trained checkpoints.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

CKPT="runs/Docking6D_RA/training/checkpoints/model_final.pth"
CKPT_VANILLA="runs/Docking6D_Vanilla/training/checkpoints/model_final.pth"
CKPT_AVOID="runs/Docking6D_Avoid/training/checkpoints/model_final.pth"
CKPT_RL="../RLBaseline/experiments/docking6d/model/Q-400221.pth"

# ---------------------------------------------------------------------------
#  Single rollouts
# ---------------------------------------------------------------------------

# Full two-phase BRAT controller with safety filter
python run_controller.py single --controller brat \
  --checkpoint_path "$CKPT" --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path "$CKPT_AVOID" \
  --output_dir ./outputs/single_brat

# Vanilla DeepReach baseline (no MPC supervision)
python run_controller.py single --controller vanilla_brat \
  --checkpoint_path "$CKPT" --vanilla_checkpoint_path "$CKPT_VANILLA" \
  --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path "$CKPT_AVOID" \
  --output_dir ./outputs/single_vanilla_brat

# Gradient MPC with learned terminal cost (our strongest MPC baseline)
python run_controller.py single --controller mpc_terminal \
  --checkpoint_path "$CKPT" --tMax 10.0 --max_sim_time 60.0 \
  --effective_horizon 1.0 --gradient_iters 50 --num_restarts 8 --gradient_lr 1.0 \
  --safety_filter_mode 1 --safety_checkpoint_path "$CKPT_AVOID" \
  --output_dir ./outputs/single_mpc_terminal

# ---------------------------------------------------------------------------
#  Multi-controller comparison over sampled ICs (paper Table 1 / Fig 5)
# ---------------------------------------------------------------------------
python run_controller.py compare \
  --controllers grid_based brat mpc_terminal mpc vanilla_brat rl \
  --checkpoint_path "$CKPT" --vanilla_checkpoint_path "$CKPT_VANILLA" \
  --rl_checkpoint_path "$CKPT_RL" \
  --tMax 10.0 --max_sim_time 60.0 \
  --safety_filter_mode 1 --safety_checkpoint_path "$CKPT_AVOID" \
  --mpc_gradient_iters 50 --mpc_num_restarts 8 --gradient_lr 1.0 --goal_weight 0.01 \
  --mpc_terminal_gradient_iters 20 --mpc_terminal_num_restarts 1 \
  --planning_horizon 2.0 --mpc_dt 0.5 --effective_horizon 1.0 \
  --n_rollouts 500 --seed 19 --sampling_method uniform \
  --safety_margin_non_brat 0.01 \
  --output_dir ./outputs/6_way_comparison
