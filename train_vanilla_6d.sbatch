#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=Docking6D_Vanilla
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

python3 run_experiment.py --mode train \
  --experiment_name Docking6D_Vanilla --dynamics_class Docking6D --tMax 10 \
  --pretrain --num_target_samples 5000 --pretrain_iters 1000 \
  --num_epochs 120000 --pause_epoch 1 --counter_end 100000 --num_nl 512 \
  --set_mode reach_avoid --lr 2e-5 \
  --not_use_MPC --mpc_ground_truth_frequency 0 \
  --deepReach_model vanilla --time_till_refinement 0.5 --cost_type reachability

echo "TRAIN_DONE"
