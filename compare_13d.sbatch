#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=Compare_13D_Safety_Vanilla
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

python3 run_controller_13d.py compare \
  --controllers brat_safety_13d vanilla_brat_13d \
  --checkpoint_path runs/Docking13D_RA/training/checkpoints/model_final.pth \
  --vanilla_checkpoint_path runs/Docking13D_Vanilla/training/checkpoints/model_final.pth \
  --safety_checkpoint_path runs/Docking13D_Avoid/training/checkpoints/model_final.pth \
  --safety_filter_mode 1 \
  --tMax 10.0 --max_sim_time 60.0 --dt 0.1 \
  --num_rollouts 500 --seed 19 --sampling_method uniform \
  --output_dir ./outputs/13d_comparison_safety

echo COMPARE_13D_DONE
