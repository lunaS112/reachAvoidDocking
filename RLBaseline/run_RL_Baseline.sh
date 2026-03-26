#!/bin/bash
# Training launch script for RL Baseline (DRABE/DDQN) on docking dynamics.
#
# Improvements over default training:
#   1. Gamma annealing (-g 0.9 -a): starts gamma=0.9, anneals to 0.9999
#      → 1000x stronger reach signal early in training
#   2. Importance sampling (--importance_sampling): 20% of episodes start
#      near the goal region
#   3. 13D: larger network (-arc 512 512 512), bigger buffer (-mc 200000),
#      longer episodes (-ms 500), more updates (-mu 1200000)

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
    --importance_sampling --target_ratio 0.15 \
    -arc 512 512 512 -mc 200000 -ms 300 -mu 4000000 \
    -n 13d_improved -sf
