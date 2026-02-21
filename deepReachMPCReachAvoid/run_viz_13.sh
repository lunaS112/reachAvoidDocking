#!/bin/bash
# 13D MPC Values Visualization Commands
# Copy and paste the command you want to run

############################################# Position BRT only (default) ##########################################
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --position-brt

############################################# Custom dt and T ##########################################
python3 MPC_values_viz_13.py --dt 0.2 --T 30 --position-brt

############################################# Multiple BRT slices ##########################################
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --position-brt --velocity-brt --rotation-brt

############################################# With trajectories and controls ##########################################
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --position-brt --trajectories --controls

############################################# With 3D animation GIF ##########################################
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --position-brt --animation

############################################# All plots and animations ##########################################
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --all

############################################# Higher resolution ##########################################
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --resolution 101 --position-brt

############################################# Different MPC styles ##########################################
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --style direct --position-brt
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --cost-type classic_mpc --position-brt
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --cost-type mixed --mpc-percentage 0.5 --position-brt

