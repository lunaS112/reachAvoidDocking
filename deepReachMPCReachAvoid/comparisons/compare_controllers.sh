#!/bin/bash
# Compare grid-based (ground truth) vs DeepReach (learned) controllers
#
# Run from deepReachMPCReachAvoid/ directory:
#   bash comparisons/compare_controllers.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../.venv/bin/activate"

CKPT="runs/Docking6D_RA_10sec/training/checkpoints/model_final.pth"
CKPT="runs/Docking6D_RA_10sec_HighSamp/training/checkpoints/model_final.pth"

# ---- Script 1: Gradient quality outside learned BRAT ----
# IC inside 25s grid BRAT, outside 10s DeepReach BRAT
python comparisons/gradient_quality_comparison.py \
    --checkpoint_path $CKPT \
    --tMax 10 \
    --grid_time_horizon 25 \
    --max_sim_time 60.0 \
    --n_ics 5 \
    --n_candidates 500000 \
    --device cuda \
    --output_dir ./outputs/gradient_comparison

# ---- Script 2: Value function approximation quality ----
# IC inside both 10s BRATs
python comparisons/value_function_comparison.py \
    --checkpoint_path $CKPT \
    --tMax 10 \
    --comparison_horizon 10 \
    --max_sim_time 30.0 \
    --n_ics 5 \
    --n_candidates 500000 \
    --device cuda \
    --output_dir ./outputs/value_comparison

# ---- Script 3: Volume comparison ----
python comparisons/volume_comparison.py \
    --checkpoint_path $CKPT --tMax 10 \
    --time_horizons 3 6 8 10 --n_monte_carlo 500000
