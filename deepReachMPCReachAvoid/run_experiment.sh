#!/bin/bash
# ============================================================================
#  Training commands for the BRAT (Backward Reach-Avoid Tube) experiments.
#
#  Replace `YOUR_WANDB_ACCOUNT` with your wandb entity (or drop the
#  `--use_wandb` flag and the `--wandb_*` options to disable logging).
#
#  Checkpoints and logs are written to ./runs/<experiment_name>/.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# ---------------------------------------------------------------------------
#  6D planar docking — BRAT with MPC supervision (main result, ~24 h on RTX 4090)
# ---------------------------------------------------------------------------
python3 run_experiment.py --mode train \
  --experiment_name Docking6D_RA --dynamics_class Docking6D --tMax 10 \
  --pretrain --num_target_samples 5000 --pretrain_iters 1000 \
  --num_epochs 150000 --pause_epoch 2000 --counter_end 100000 --num_nl 512 \
  --set_mode reach_avoid --lr 2e-5 \
  --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 100 \
  --num_MPC_data_samples 10000 --numpoints 50000 --mpc_ground_truth_frequency 15 \
  --MPC_style receding --MPC_receding_horizon 1 --MPC_dt 0.1 \
  --deepReach_model exact --time_till_refinement 0.5 --cost_type reachability \
  --use_wandb --wandb_project BRAT --wandb_name Docking6D_RA \
  --wandb_group Docking6D --wandb_entity YOUR_WANDB_ACCOUNT

# ---------------------------------------------------------------------------
#  6D planar docking — Vanilla DeepReach baseline (no MPC supervision)
# ---------------------------------------------------------------------------
python3 run_experiment.py --mode train \
  --experiment_name Docking6D_Vanilla --dynamics_class Docking6D --tMax 10 \
  --pretrain --num_target_samples 5000 --pretrain_iters 1000 \
  --num_epochs 120000 --pause_epoch 1 --counter_end 100000 --num_nl 512 \
  --set_mode reach_avoid --lr 2e-5 \
  --not_use_MPC --mpc_ground_truth_frequency 0 \
  --deepReach_model vanilla --time_till_refinement 0.5 --cost_type reachability \
  --use_wandb --wandb_project BRAT --wandb_name Docking6D_Vanilla \
  --wandb_group Docking6D --wandb_entity YOUR_WANDB_ACCOUNT

# ---------------------------------------------------------------------------
#  13D full docking (pos + vel + quaternion + angular vel) — BRAT
# ---------------------------------------------------------------------------
python3 run_experiment.py --mode train \
  --experiment_name Docking13D_RA --dynamics_class Docking13D --tMax 10 \
  --pretrain --num_target_samples 32500 --pretrain_iters 2000 \
  --num_epochs 200000 --pause_epoch 3000 --counter_end 120000 --num_nl 512 \
  --set_mode reach_avoid --lr 1e-5 \
  --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 150 \
  --num_MPC_data_samples 7500 --numpoints 65000 --mpc_ground_truth_frequency 15 \
  --MPC_style receding --MPC_receding_horizon 1 --MPC_dt 0.1 \
  --deepReach_model exact --time_till_refinement 0.5 --cost_type reachability \
  --use_wandb --wandb_project BRAT --wandb_name Docking13D_RA \
  --wandb_group Docking13D --wandb_entity YOUR_WANDB_ACCOUNT

# ---------------------------------------------------------------------------
#  13D — Avoid-only BRT for the safety filter (used by --safety_checkpoint_path)
# ---------------------------------------------------------------------------
python3 run_experiment.py --mode train \
  --experiment_name Docking13D_Avoid --dynamics_class Docking13D --tMax 10 \
  --pretrain --num_target_samples 32500 --pretrain_iters 2000 \
  --num_epochs 200000 --pause_epoch 3000 --counter_end 120000 --num_nl 512 \
  --set_mode avoid --lr 1e-5 --avoid_target_sampling boundary \
  --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 150 \
  --num_MPC_data_samples 7500 --numpoints 75000 --mpc_ground_truth_frequency 0 \
  --MPC_style receding --MPC_receding_horizon 1 --MPC_dt 0.1 \
  --deepReach_model exact --time_till_refinement 0.5 --cost_type reachability \
  --use_wandb --wandb_project BRAT --wandb_name Docking13D_Avoid \
  --wandb_group Docking13D --wandb_entity YOUR_WANDB_ACCOUNT
