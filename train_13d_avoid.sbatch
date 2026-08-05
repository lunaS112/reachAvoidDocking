#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=Docking13D_Avoid
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

python3 run_experiment.py --mode train \
  --experiment_name Docking13D_Avoid --dynamics_class Docking13D --tMax 10 \
  --pretrain --num_target_samples 32500 --pretrain_iters 2000 \
  --num_epochs 200000 --pause_epoch 3000 --counter_end 120000 --num_nl 512 \
  --set_mode avoid --lr 1e-5 --avoid_target_sampling boundary \
  --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 150 \
  --num_MPC_data_samples 7500 --numpoints 75000 --mpc_ground_truth_frequency 0 \
  --MPC_style receding --MPC_receding_horizon 1 --MPC_dt 0.1 \
  --deepReach_model exact --time_till_refinement 0.5 --cost_type reachability

echo "TRAIN_DONE"
