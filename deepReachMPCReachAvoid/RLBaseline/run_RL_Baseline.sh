#!/bin/bash
# Training launch script for RL Baseline (DRABE/DDQN) on docking dynamics.
#
# Improvements over default training:
#   1. Gamma annealing (-g 0.9 -a): starts gamma=0.9, anneals to 0.9999
#      → 1000x stronger reach signal early in training
#   2. Importance sampling (--importance_sampling): 20% of episodes start
#      near the goal region
#   3. 13D: PD attitude controller (--pd_attitude) reduces actions 729 → 27,
#      tighter IC range (--tight_ic), slower target tracking (--tau 0.005),
#      higher importance sampling (--target_ratio 0.3)

set -euo pipefail
cd "$(dirname "$0")"

# === 6D Docking ===
echo "=== Training 6D Docking ==="
python train_docking.py --dynamics 6d -wi 5000 -w -g 0.9 -a \
    --importance_sampling --target_ratio 0.2 \
    -n 6d_improved -sf

# === 13D Docking ===
echo "=== Training 13D Docking ==="
python train_docking.py --dynamics 13d -wi 5000 -w -g 0.9 -a \
    --pd_attitude --tight_ic \
    --importance_sampling --target_ratio 0.3 \
    --tau 0.005 \
    -arc 512 512 512 -mc 100000 -ms 500 -mu 400000 \
    -n 13d_improved -sf
