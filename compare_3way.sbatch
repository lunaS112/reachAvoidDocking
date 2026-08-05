#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=Compare_3way_6D
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

python3 run_controller.py compare \
  --controllers grid_based brat vanilla_brat \
  --checkpoint_path runs/Docking6D_RA/training/checkpoints/model_final.pth \
  --vanilla_checkpoint_path runs/Docking6D_Vanilla/training/checkpoints/model_final.pth \
  --tMax 10.0 --max_sim_time 60.0 \
  --n_rollouts 500 --seed 19 --sampling_method uniform \
  --grid_cache_dir ./outputs/grid_cache \
  --output_dir ./outputs/3way_comparison

echo "COMPARE_DONE"
