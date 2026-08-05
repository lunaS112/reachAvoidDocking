#!/bin/bash
#SBATCH --account=def-mochen_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=GroundTruth_4D2D
#SBATCH --output=/scratch/dandans/Deepreach_mpc/reachAvoidDocking/job_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=/scratch/dandans/Deepreach_mpc/reachAvoidDocking
mkdir -p "$PROJECT_ROOT/job_logs"

module load python/3.11
module load cuda/12.2
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/gridBased6DImplementation/4D_2D"

python3 -c "import jax; print('JAX devices:', jax.devices())"

echo "=== Running 4D translational ground truth (grid 51x51x31x31, 251 timesteps) ==="
python3 Docking4D.py

echo "=== Running 2D rotational ground truth (grid 361x141, 251 timesteps) ==="
python3 Docking2D.py

echo "GROUND_TRUTH_DONE"
