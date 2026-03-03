#!/bin/bash
# ============================================================================
#  13D Docking Controller — Example Commands
# ============================================================================
#  Uncomment the command you want to run; execute with:
#     bash run_controller_13d.sh
# ============================================================================

# Activate venv (provides 'python' on PATH)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# ---------------------------------------------------------------------------
#  Checkpoints
# ---------------------------------------------------------------------------
# Short-horizon (tMax = 5 s, 4 hidden layers, 5× importance sampling)
CKPT_5s="runs/Docking13D_RA_5s_5xSampling/training/checkpoints/model_final.pth"

# Longer-horizon (tMax = 10 s, 3 hidden layers — larger BRAT)
CKPT_10s="runs/Docking13D_RA-Might-Be-Usableish/training/checkpoints/model_epoch_104000.pth"

# >>> Choose which checkpoint to use for the commands below <<<
CKPT="$CKPT_10s"
TMAX=10.0

# ===========================================================================
#  1.  Single-controller runs
# ===========================================================================

# --- BRT 13D : custom initial condition ---
# python run_controller_13d.py single \
#   --controller brt_13d \
#   --checkpoint_path "$CKPT" --tMax $TMAX --max_sim_time 60.0 \
#   --initial_state "[-8.0, 10.0, -3.5, 0.05, -0.02, 0.01, 0.7071, 0.0, 0.0, 0.7071, 0.1, -0.05, 0.02]" \
#   --viz_html --viz_resolution 30 --viz_max_frames 40 \
#   --output_dir ./outputs/13d_single_brt_custom

# --- BRT 13D : random IC sampled from BRAT ---
# python run_controller_13d.py single \
#   --controller brt_13d \
#   --checkpoint_path "$CKPT" --tMax $TMAX --max_sim_time 60.0 \
#   --sampling_method brt --seed 42 \
#   --viz_html --viz_resolution 30 --viz_max_frames 40 \
#   --output_dir ./outputs/13d_single_brt_brat

# --- BRT 13D : default IC (10 m away, at rest, goal attitude) ---
# python run_controller_13d.py single \
#   --controller brt_13d \
#   --checkpoint_path "$CKPT" --tMax $TMAX --max_sim_time 60.0 \
#   --viz_html --viz_resolution 30 --viz_max_frames 40 \
#   --output_dir ./outputs/13d_single_brt_default

# --- MPC 13D (no value function) ---
# python run_controller_13d.py single \
#   --controller mpc_13d \
#   --checkpoint_path "$CKPT" --dt 0.1 --max_sim_time 60.0 \
#   --planning_horizon 3.0 --mpc_dt 0.5 \
#   --num_samples 100 --num_refinement 10 \
#   --viz_html \
#   --output_dir ./outputs/13d_single_mpc

# --- MPC + Terminal 13D ---
# python run_controller_13d.py single \
#   --controller mpc_terminal_13d \
#   --checkpoint_path "$CKPT" --tMax $TMAX --max_sim_time 60.0 \
#   --effective_horizon 3.0 --effort_weight 0.0 \
#   --num_samples 100 --num_refinement 10 \
#   --viz_html --viz_resolution 30 --viz_max_frames 40 \
#   --output_dir ./outputs/13d_single_mpc_terminal

# ===========================================================================
#  2.  Stagnation-escape tuning (MPC+Terminal only)
# ===========================================================================

# python run_controller_13d.py single \
#   --controller mpc_terminal_13d \
#   --checkpoint_path "$CKPT" --tMax $TMAX --max_sim_time 60.0 \
#   --effective_horizon 2.0 \
#   --num_samples 500 --num_refinement 10 \
#   --exploration_factor 5.0 --exploration_patience 1 --escape_thresh 0.3 \
#   --output_dir ./outputs/13d_single_mpc_terminal_aggressive

# ===========================================================================
#  3.  Multi-controller comparison runs
# ===========================================================================

# Uniform IC sampling (geometric constraints only)
# python run_controller_13d.py compare \
#   --controllers brt_13d mpc_13d mpc_terminal_13d \
#   --checkpoint_path "$CKPT" --tMax $TMAX --max_sim_time 60.0 \
#   --num_rollouts 3 --seed 42 \
#   --num_samples 500 --num_refinement 10 \
#   --output_dir ./outputs/13d_comparison_uniform

# BRAT IC sampling (ICs inside the learned backward reachable-avoid tube)
# python run_controller_13d.py compare \
#   --controllers brt_13d mpc_terminal_13d \
#   --checkpoint_path "$CKPT" --tMax $TMAX --max_sim_time 60.0 \
#   --num_rollouts 5 --seed 42 --sampling_method brt \
#   --num_samples 500 --num_refinement 10 \
#   --output_dir ./outputs/13d_comparison_brat
