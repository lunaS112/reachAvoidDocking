#!/bin/bash
#
# Generic SBATCH Template for MPC Evaluation Framework
#
# Usage: Copy this template and adjust parameters for your cluster
# Then run: sbatch your_custom_job.sbatch
#

# ============================================================================
# CLUSTER-SPECIFIC PARAMETERS (EDIT THESE FOR YOUR CLUSTER)
# ============================================================================

# Account/project (required by most clusters)
#SBATCH --account=YOUR_ACCOUNT_ID

# Partition/queue name (depends on cluster)
#SBATCH --partition=gpu

# GPU type and count (adjust based on your cluster)
#SBATCH --gres=gpu:h100:1              # h100 / v100 / a100 / rtx_a5000 etc.

# CPU and memory (adjust based on available resources)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Time limit (hours:minutes:seconds)
#SBATCH --time=04:00:00                # Increase if tasks timeout

# ============================================================================
# JOB CONFIGURATION (USUALLY DON'T CHANGE)
# ============================================================================

#SBATCH --job-name=MPC_Evaluation
#SBATCH --output=job_logs/%x-%j.out
#SBATCH --error=job_logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

set -euo pipefail  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT_ROOT/job_logs"

echo "=========================================="
echo "MPC Evaluation - Job Setup"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# ============================================================================
# LOAD ENVIRONMENT MODULES (CLUSTER-DEPENDENT)
# ============================================================================

# Common module loading patterns:

# Pattern 1: Compute Canada / Alliance
# module load python/3.11 cuda/12.2 cudnn

# Pattern 2: XSEDE/ACCESS
# module load python cuda/11.8

# Pattern 3: Local cluster (might not have modules)
# export PATH="/opt/cuda/bin:$PATH"
# export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"

# Uncomment the pattern for your cluster:
# module load python/3.11
# module load cuda/12.2

echo "Environment modules loaded"

# ============================================================================
# PYTHON ENVIRONMENT
# ============================================================================

# Option A: Use virtual environment
if [ -d ".venv" ]; then
    echo "Activating Python venv..."
    source .venv/bin/activate
else
    echo "Creating Python venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "Python: $(python3 --version)"
echo "PyTorch available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ============================================================================
# ACTUAL WORK (CUSTOMIZE BASED ON YOUR TASK)
# ============================================================================

cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

echo ""
echo "=========================================="
echo "Running MPC Evaluation"
echo "=========================================="
echo ""

# Example 1: Compare hybrid vs baseline at 10s
# python3 run_controller.py compare \
#   --controllers grid_based brat vanilla_brat \
#   --checkpoint_path runs/Docking6D_RA/training/checkpoints/model_final.pth \
#   --vanilla_checkpoint_path runs/Docking6D_Vanilla/training/checkpoints/model_final.pth \
#   --tMax 10.0 --max_sim_time 60.0 --dt 0.5 \
#   --n_rollouts 500 --seed 19 --sampling_method uniform \
#   --grid_cache_dir ./outputs/grid_cache \
#   --output_dir ./outputs/hybrid_vs_baseline_10s

# Example 2: Grid-based ground truth at 17s
# python3 run_controller.py compare \
#   --controllers grid_based \
#   --tMax 17.0 --max_sim_time 60.0 --dt 0.5 \
#   --n_rollouts 500 --seed 19 --sampling_method uniform \
#   --grid_cache_dir ./outputs/grid_cache \
#   --output_dir ./outputs/grid_baseline_17s

# REPLACE ABOVE WITH YOUR ACTUAL COMMAND:

python3 run_controller.py compare \
  --controllers grid_based brat vanilla_brat \
  --checkpoint_path runs/Docking6D_RA/training/checkpoints/model_final.pth \
  --vanilla_checkpoint_path runs/Docking6D_Vanilla/training/checkpoints/model_final.pth \
  --tMax 10.0 --max_sim_time 60.0 --dt 0.5 \
  --n_rollouts 500 --seed 19 --sampling_method uniform \
  --grid_cache_dir ./outputs/grid_cache \
  --output_dir ./outputs/hybrid_vs_baseline_10s

# ============================================================================
# CLEANUP & SUMMARY
# ============================================================================

echo ""
echo "=========================================="
echo "Job Complete"
echo "=========================================="
echo "Output saved to: ./outputs/"
echo "Job logs saved to: $PROJECT_ROOT/job_logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.out"
echo "=========================================="

# Deactivate venv on exit
if [ ! -z "${VIRTUAL_ENV:-}" ]; then
    deactivate
fi

exit 0
