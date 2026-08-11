#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=Hybrid_Compare_17s
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

echo "Step 3: Compare Hybrid (17s trained) vs Baseline (17s) @ tMax=17.0, dt=0.5"
python3 run_controller.py compare \
  --controllers grid_based brat vanilla_brat \
  --checkpoint_path runs/Docking6D_RA/training/checkpoints/model_final.pth \
  --vanilla_checkpoint_path runs/Docking6D_Vanilla/training/checkpoints/model_final.pth \
  --tMax 17.0 --max_sim_time 60.0 --dt 0.5 \
  --n_rollouts 500 --seed 19 --sampling_method uniform \
  --grid_cache_dir ./outputs/grid_cache \
  --output_dir ./outputs/hybrid_vs_baseline_17s

echo "HYBRID_17S_COMPARE_DONE"
