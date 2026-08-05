#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=profile_vanilla6d
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/profile_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/profile_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/deepReachMPCReachAvoid"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 1 \
    > "$PROJECT_ROOT/profile_logs/gpu_util_${SLURM_JOB_ID}.csv" &
NVSMI_PID=$!

python3 run_experiment.py --mode train \
  --experiment_name ProfileTest_Vanilla --dynamics_class Docking6D --tMax 10 \
  --pretrain --num_target_samples 5000 --pretrain_iters 50 \
  --num_epochs 400 --pause_epoch 1 --counter_end 100000 --num_nl 512 \
  --set_mode reach_avoid --lr 2e-5 \
  --not_use_MPC --mpc_ground_truth_frequency 0 \
  --deepReach_model vanilla --time_till_refinement 0.5 --cost_type reachability \
  --epochs_til_ckpt 100000 --steps_til_summary 20 \
  2>&1 | tee "$PROJECT_ROOT/profile_logs/train_${SLURM_JOB_ID}.log"

kill $NVSMI_PID 2>/dev/null || true
echo "PROFILE_DONE"
