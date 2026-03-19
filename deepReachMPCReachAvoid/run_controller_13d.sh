#!/usr/bin/env bash
# ============================================================================
#  13-D Docking Controller Runner
# ============================================================================
#
#  Quick-start:
#    ./run_controller_13d.sh                    # BRT single run, default IC
#    ./run_controller_13d.sh --viz              # + HTML animation & slice GIFs
#    ./run_controller_13d.sh --easy             # easy IC (close, aligned)
#    ./run_controller_13d.sh --compare          # multi-controller comparison
#    ./run_controller_13d.sh --help             # this message
#
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.venv/bin/activate"

# ---------------------------------------------------------------------------
#  Configuration — edit these to match your setup
# ---------------------------------------------------------------------------
CKPT="runs/Docking13D_RA_new_goal_set2/training/checkpoints/model_horizon_10.00.pth"
TMAX=10.0
MAX_SIM=60.0
OUTDIR="./outputs/13d_run"

# Easy initial condition: close, aligned, gentle upward drift
EASY_IC="[0,-3.0,0,0,0.05,0,0.7071,0,0,0.7071,0,0,0]"

# ---------------------------------------------------------------------------
#  Parse flags
# ---------------------------------------------------------------------------
MODE="single"
VIZ_FLAGS=""
CONTROLLER="brt_13d"
IC_FLAG=""
EXTRA_ARGS=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --help, -h          Show this help message
  --viz               Enable all visualisations (HTML + slice GIFs)
  --viz-html          Enable only HTML animation
  --viz-gifs          Enable only 2-D slice GIFs
  --easy              Use an easy IC (close, aligned, gentle drift)
  --ic "JSON_ARRAY"   Custom 13-element initial state, e.g. "[10,0,0,...]"
  --compare           Run multi-controller comparison instead of single run
  --controller NAME   Controller: brt_13d | mpc_13d | mpc_terminal_13d
  --ckpt PATH         Checkpoint path (default: \$CKPT)
  --tmax FLOAT        Time horizon (default: $TMAX)
  --outdir DIR        Output directory (default: $OUTDIR)

Examples:
  $(basename "$0")                              # default BRT run
  $(basename "$0") --easy --viz                 # easy IC with full viz
  $(basename "$0") --ic "[5,-5,2,0,0,0,0.7071,0,0,0.7071,0,0,0]" --viz
  $(basename "$0") --compare --outdir ./outputs/13d_cmp
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)      usage ;;
    --viz)          VIZ_FLAGS="--viz_html --viz_gifs"; shift ;;
    --viz-html)     VIZ_FLAGS="${VIZ_FLAGS} --viz_html"; shift ;;
    --viz-gifs)     VIZ_FLAGS="${VIZ_FLAGS} --viz_gifs"; shift ;;
    --easy)         IC_FLAG="--initial_state '${EASY_IC}'"; shift ;;
    --ic)           IC_FLAG="--initial_state '$2'"; shift 2 ;;
    --compare)      MODE="compare"; shift ;;
    --controller)   CONTROLLER="$2"; shift 2 ;;
    --ckpt)         CKPT="$2"; shift 2 ;;
    --tmax)         TMAX="$2"; shift 2 ;;
    --outdir)       OUTDIR="$2"; shift 2 ;;
    *)              EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
  esac
done

# Validate checkpoint exists
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: Checkpoint not found: $CKPT" >&2
  echo "  Set the CKPT variable in this script or use --ckpt PATH" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
#  Run
# ---------------------------------------------------------------------------
if [[ "$MODE" == "compare" ]]; then
  echo "=== Multi-controller comparison ==="
  eval python3 run_controller_13d.py compare \
    --controllers brt_13d mpc_terminal_13d \
    --checkpoint_path "$CKPT" --tMax "$TMAX" --max_sim_time "$MAX_SIM" \
    --num_rollouts 5 --seed 42 \
    --num_samples 500 --num_refinement 10 \
    --output_dir "$OUTDIR" \
    $EXTRA_ARGS
else
  echo "=== Single run: $CONTROLLER ==="
  eval python3 run_controller_13d.py single \
    --controller "$CONTROLLER" \
    --checkpoint_path "$CKPT" --tMax "$TMAX" --max_sim_time "$MAX_SIM" \
    $IC_FLAG \
    $VIZ_FLAGS \
    --viz_resolution 30 --viz_max_frames 40 \
    --output_dir "$OUTDIR" \
    $EXTRA_ARGS
fi
