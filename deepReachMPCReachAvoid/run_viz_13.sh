#!/bin/bash
# ============================================================================
#  13D MPC value-function visualisation (BRAT slices & animations).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# Default: 2-D position BRAT slice
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --position-brt

# All plots + animations (position / velocity / rotation slices + 3-D gif)
python3 MPC_values_viz_13.py --dt 0.1 --T 15 --all
