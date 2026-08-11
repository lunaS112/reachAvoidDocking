#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=Grid_Baseline_17s_Direct
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT/gridBased6DImplementation"

echo "=========================================="
echo "Generating Grid-Based HJ Baseline (tMax=17.0s)"
echo "=========================================="
echo "Time: $(date)"
echo ""

python3 generate_grid_baseline_17s.py

echo ""
echo "=========================================="
echo "Grid baseline generation complete"
echo "=========================================="
