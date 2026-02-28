#!/bin/bash
# Checkpoint path for leanred value function (for BRT and MPC+Terminal controllers)
CKPT="runs/Docking13D_RA_GOOD/training/checkpoints/model_final.pth"

#
cd /home/santiagothorup/Documents/reachAvoidDocking/deepReachMPCReachAvoid/outputs/13d_single_brt_custom && python -m http.server 8765

########################### Single controller runs ###########################

# --- BRT 13D: custom initial condition ---
python run_controller_13d.py single --controller brt_13d \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --initial_state "[-8.0, 10.0, -3.5, 0.05, -0.02, 0.01, 0.7071, 0.0, 0.0, 0.7071, 0.1, -0.05, 0.02]" \
  --viz_html --viz_mp4 --viz_resolution 30 --viz_max_frames 40 \
  --output_dir ./outputs/13d_single_brt_custom

# --- BRT 13D: ---
python run_controller_13d.py single --controller brt_13d \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --viz_html --viz_mp4 --viz_resolution 30 --viz_max_frames 40 \
  --output_dir ./outputs/13d_single_brt_viz

# --- MPC 13D ---
python run_controller_13d.py single --controller mpc_13d \
  --checkpoint_path $CKPT --dt 0.1 --max_sim_time 60.0 \
  --planning_horizon 3.0 --mpc_dt 0.5 --viz_html --viz_mp4 \
  --num_samples 100 --num_refinement 10 \
  --output_dir ./outputs/13d_single_mpc

# --- MPC+Terminal 13D---
python run_controller_13d.py single --controller mpc_terminal_13d \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --effective_horizon 3.0 --effort_weight 0.0\
  --num_samples 100 --num_refinement 10 \
  --viz_html --viz_mp4 --viz_resolution 30 --viz_max_frames 40 \
  --output_dir ./outputs/13d_single_mpc_terminal_viz

#################### Stagnation-escape tuning ################################

# MPC+Terminal 13D: aggressive exploration (higher factor, lower patience)
python run_controller_13d.py single --controller mpc_terminal_13d \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --effective_horizon 2.0 \
  --num_samples 500 --num_refinement 10 \
  --exploration_factor 5.0 --exploration_patience 1 --escape_thresh 0.3 \
  --output_dir ./outputs/13d_single_mpc_terminal_aggressive_explore


########################### Comparison runs ##################################

# Comparison
python run_controller_13d.py compare \
  --controllers brt_13d mpc_13d mpc_terminal_13d \
  --checkpoint_path $CKPT --tMax 10.0 --max_sim_time 60.0 \
  --num_rollouts 3 --seed 42 \
  --num_samples 500 --num_refinement 10 \
  --output_dir ./outputs/13d_comparison_quick


