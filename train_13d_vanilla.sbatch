#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=Docking13D_Vanilla
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

# Not an official run_experiment.sh recipe -- constructed by the same
# transformation the repo's own Docking6D_Vanilla used relative to
# Docking6D_RA (strip MPC, --deepReach_model vanilla, keep everything else
# from the matching 13D BRAT recipe, incl. its lr=1e-5).
python3 run_experiment.py --mode train \
  --experiment_name Docking13D_Vanilla --dynamics_class Docking13D --tMax 10 \
  --pretrain --num_target_samples 32500 --pretrain_iters 2000 \
  --num_epochs 160000 --pause_epoch 1 --counter_end 120000 --num_nl 512 \
  --set_mode reach_avoid --lr 1e-5 \
  --not_use_MPC --mpc_ground_truth_frequency 0 \
  --deepReach_model vanilla --time_till_refinement 0.5 --cost_type reachability

echo "TRAIN_DONE"
