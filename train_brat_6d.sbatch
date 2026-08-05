#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=Docking6D_RA
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

python3 run_experiment.py --mode train \
  --experiment_name Docking6D_RA --dynamics_class Docking6D --tMax 10 \
  --pretrain --num_target_samples 5000 --pretrain_iters 1000 \
  --num_epochs 150000 --pause_epoch 2000 --counter_end 100000 --num_nl 512 \
  --set_mode reach_avoid --lr 2e-5 \
  --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 100 \
  --num_MPC_data_samples 10000 --numpoints 50000 --mpc_ground_truth_frequency 15 \
  --MPC_style receding --MPC_receding_horizon 1 --MPC_dt 0.1 \
  --deepReach_model exact --time_till_refinement 0.5 --cost_type reachability

echo "TRAIN_DONE"
