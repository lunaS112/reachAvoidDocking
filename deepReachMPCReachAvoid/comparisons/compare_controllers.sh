#!/bin/bash
# ============================================================================
#  Paper comparison experiments: learned BRAT (DeepReach) vs grid-based
#  ground truth, vanilla DeepReach, and the RL baseline.
#
#  Run from deepReachMPCReachAvoid/ :
#      bash comparisons/compare_controllers.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../.venv/bin/activate"

CKPT="runs/Docking6D_RA/training/checkpoints/model_final.pth"
CKPT_VANILLA="runs/Docking6D_Vanilla/training/checkpoints/model_final.pth"
CKPT_RL="./RLBaseline/experiments/docking6d/model/Q-400221.pth"
RL_META="./RLBaseline/experiments/docking6d/train.pkl"

# Gradient quality: ICs inside the 25 s grid BRAT but outside the 10 s learned BRAT
python comparisons/gradient_quality_comparison.py \
    --checkpoint_path "$CKPT" --tMax 10 --grid_time_horizon 25 \
    --max_sim_time 60.0 --n_ics 15 --n_candidates 500000 \
    --device cuda --seed 19 \
    --output_dir ./outputs/gradient_comparison

# Value-function approximation error on ICs inside both BRATs
python comparisons/value_function_comparison.py \
    --checkpoint_path "$CKPT" --tMax 10 --comparison_horizon 10 \
    --max_sim_time 30.0 --n_ics 5 --n_candidates 500000 \
    --device cuda \
    --output_dir ./outputs/value_comparison

# BRAT volume (Monte-Carlo) — learned vs grid ground truth
python comparisons/volume_comparison.py \
    --checkpoint_path "$CKPT" --tMax 10 --time_horizons 10 \
    --n_monte_carlo 500000 \
    --output_dir ./outputs/volume_comparison

# Classification metrics (FPR / TPR vs grid ground truth) for learned, vanilla, and RL
python comparisons/brat_metrics.py \
    --learned_checkpoint "$CKPT" \
    --vanilla_checkpoint "$CKPT_VANILLA" \
    --rl_checkpoint "$CKPT_RL" --rl_metadata "$RL_META" \
    --tMax 10.0 --n_samples 500000 \
    --output_dir ./outputs/brat_metrics
