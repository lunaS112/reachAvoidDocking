#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=test_hybrid_learning_ctrl
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

python3 run_controller.py single --controller hybrid_learning \
  --hybrid_learning_checkpoint_path /scratch/dandans/cmpt720_hybrid_hj/artifacts/results/deepreach_hybrid_sampled/docking4d_hybrid/best_model.pth \
  --tMax 10.0 --max_sim_time 60.0 \
  --initial_px 2.0 --initial_py 10.0 --initial_vx 0.0 --initial_vy 0.0 --initial_theta -1.57 --initial_omega 0.0 \
  --grid_cache_dir ./outputs/grid_cache \
  --output_dir ./outputs/test_hybrid_learning_single \
  --no_animation

echo "TEST_DONE"
