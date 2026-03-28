#!/usr/bin/env python3
"""
BRAT Classification Metrics: FPR and TPR vs. Grid-Based Ground Truth

Compares learned BRAT approximations to the numerical grid-based BRAT
(Hamilton-Jacobi, 4D+2D decomposition) using binary classification metrics.

Terminology
-----------
"Safe"   / Positive  := inside the BRAT  (value ≤ 0), i.e., reachable while
                        avoiding obstacles.
"Unsafe" / Negative  := outside the BRAT (value > 0), i.e., not reachable.

Confusion matrix (ground truth = grid-based HJ value):

              pred = safe   pred = unsafe
  gt = safe  |     TP     |      FN     |   (grid says reachable)
  gt = unsafe|     FP     |      TN     |   (grid says not reachable)

TPR = TP / (TP + FN)
    Fraction of grid-safe states correctly identified as safe.
    Higher TPR → less conservative (method captures more of the true BRAT).

FPR = FP / (FP + TN)
    Fraction of grid-unsafe states incorrectly claimed safe.
    Higher FPR → more optimistic (method includes states the grid says are
    NOT reachable — a potential safety concern).

Methods evaluated:
    1. Learned BRAT   — DeepReach with MPC refinement ("our method")
    2. Vanilla BRAT   — DeepReach without MPC refinement  (optional)
    3. RL BRAT        — DDQNSingle, V(s) = min_a Q(s,a)  (optional, time-independent)

Ground truth: ComboController (4D+2D decomposed Hamilton-Jacobi).

Usage
-----
    python comparisons/brat_metrics.py \\
        --learned_checkpoint ./runs/Docking6D_RA_10sec_HighSamp/training/checkpoints/model_final.pth \\
        --vanilla_checkpoint ./runs/Docking6D_10sec_Vanilla_BaseLine/training/checkpoints/model_final.pth \\
        --rl_checkpoint      ./RLBaseline/experiments/docking-6d-DDQN/6d_improved-TF/model/Q-400221.pth \\
        --rl_metadata        ./RLBaseline/experiments/docking-6d-DDQN/6d_improved-TF/train.pkl \\
        --tMax 10.0 \\
        --n_samples 500000 \\
        --output_dir ./outputs/brat_metrics
"""

import argparse
import json
import os
import pickle
import sys
import time as _time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — run from deepReachMPCReachAvoid/ or comparisons/
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEEPREACH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT  = os.path.dirname(DEEPREACH_DIR)
RL_DIR        = os.path.join(DEEPREACH_DIR, 'RLBaseline')

for _p in (DEEPREACH_DIR,
           os.path.join(PROJECT_ROOT, 'gridBased6DImplementation'),
           RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.controllers import BRATController

_combo_module = None


def _get_combo_module():
    global _combo_module
    if _combo_module is None:
        import importlib
        _combo_module = importlib.import_module('ComboControl')
    return _combo_module


# ---------------------------------------------------------------------------
# Sampling bounds
# ---------------------------------------------------------------------------
# Focused around the BRAT region for the 6D docking problem.
# These match the MC_BOUNDS in volume_comparison.py for consistency.
MC_BOUNDS = np.array([
    [ -5.0,   5.0],   # px  (BRAT ⊂ ±5 m at t ≤ 15 s)
    [ -5.0,   2.0],   # py  (goal ≈ −1.2 m; obstacle above y ≈ 0)
    [ -1.5,   1.5],   # vx
    [ -1.5,   1.5],   # vy
    [-np.pi, np.pi],  # theta
    [ -1.0,   1.0],   # omega
])

# Tighter bounds for sanity / near-goal validation
FOCUSED_BOUNDS = np.array([
    [-3.0,  3.0],
    [-3.0,  0.5],
    [-0.5,  0.5],
    [-0.5,  0.5],
    [ 0.5,  2.6],
    [-0.3,  0.3],
])

# RL observation normalization  — must match Docking6DEnv._obs_center/half_range,
# which are derived from Docking6D.state_range_.
_RL_STATE_RANGE = np.array([
    [-15.0,  15.0],   # px
    [-15.0,  15.0],   # py
    [ -1.5,   1.5],   # vx
    [ -2.0,   2.0],   # vy   (dynamics range; env normalizes using this)
    [-np.pi, np.pi],  # theta
    [ -1.0,   1.0],   # omega
])
_RL_OBS_CENTER     = (_RL_STATE_RANGE[:, 0] + _RL_STATE_RANGE[:, 1]) / 2.0
_RL_OBS_HALF_RANGE = (_RL_STATE_RANGE[:, 1] - _RL_STATE_RANGE[:, 0]) / 2.0


# ---------------------------------------------------------------------------
# Grid helpers (mirrors volume_comparison.py)
# ---------------------------------------------------------------------------

def physical_time_to_index(t_physical: float, times_array: np.ndarray) -> int:
    """Map a positive physical time horizon to the nearest grid time index.

    The HJ solver stores times in reverse: times[0] = 0 (terminal),
    times[-1] = final_time (e.g. −30 s).  A horizon of T seconds
    corresponds to times[idx] ≈ −T.
    """
    target = -abs(t_physical)
    return int(np.argmin(np.abs(times_array - target)))


def _grid_batch_4D(combo, states_4d: np.ndarray, time_idx: int) -> np.ndarray:
    from scipy.interpolate import interpn
    coord_vecs = [np.array(v) for v in combo.grid_4D.coordinate_vectors]
    grid_vals  = np.array(combo.values_4D[time_idx])
    return interpn(coord_vecs, grid_vals, np.atleast_2d(states_4d),
                   method='linear', bounds_error=False, fill_value=np.inf)


def _grid_batch_2D(combo, states_2d: np.ndarray, time_idx: int) -> np.ndarray:
    from scipy.interpolate import interpn
    coord_vecs = [np.array(v) for v in combo.grid_2D.coordinate_vectors]
    grid_vals  = np.array(combo.values_2D[time_idx])
    return interpn(coord_vecs, grid_vals, np.atleast_2d(states_2d),
                   method='linear', bounds_error=False, fill_value=np.inf)


def grid_value_6D(combo, states_6d: np.ndarray, time_idx: int,
                  chunk: int = 50_000) -> np.ndarray:
    """V_6D(s, t) = max(V_4D(s_trans), V_2D(s_rot)), evaluated in chunks."""
    out = []
    for i in range(0, len(states_6d), chunk):
        s = states_6d[i:i + chunk]
        v4 = _grid_batch_4D(combo, s[:, :4], time_idx)
        v2 = _grid_batch_2D(combo, s[:, 4:], time_idx)
        out.append(np.maximum(v4, v2))
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# RL Q-network loading and evaluation
# ---------------------------------------------------------------------------

def load_rl_q_network(rl_checkpoint: str, rl_metadata: str, device: str):
    """Load a saved DDQNSingle Q-network.

    The checkpoint is Q_network.state_dict() saved by RARL.utils.save_model.
    Architecture and normalization are reconstructed from train.pkl.

    Args:
        rl_checkpoint: path to Q-<step>.pth
        rl_metadata:   path to train.pkl  (has 'dimList' and 'args' keys)
        device:        torch device string

    Returns:
        model:          nn.Module, in eval mode on ``device``
        obs_center:     (6,) float64 array — subtracted before normalising
        obs_half_range: (6,) float64 array — divided after subtracting center
    """
    with open(rl_metadata, 'rb') as fh:
        meta = pickle.load(fh)

    dim_list = meta['dimList']          # e.g. [6, 256, 256, 27]
    act_type = meta['args'].get('actType', 'Tanh')

    from RARL.model import Model
    model = Model(dim_list, act_type)
    ckpt = torch.load(rl_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    print(f"  RL model: dimList={dim_list}, actType={act_type}")
    print(f"  RL checkpoint: {rl_checkpoint}")
    return model, _RL_OBS_CENTER.copy(), _RL_OBS_HALF_RANGE.copy()


def rl_value_batch(model, states_6d: np.ndarray,
                   obs_center: np.ndarray, obs_half: np.ndarray,
                   device: str, chunk: int = 50_000) -> np.ndarray:
    """V(s) = min_a Q(normalize(s), a) for a batch of raw 6D states.

    The RL value is time-independent; the *sign convention* matches the HJ
    formulation (V ≤ 0 ↔ state can reach the goal while avoiding obstacles).

    Args:
        states_6d: (N, 6) un-normalised raw states.

    Returns:
        (N,) array of min Q-values.  in_BRAT ↔ value ≤ 0.
    """
    norm = (states_6d - obs_center) / obs_half   # (N, 6), in [−1, 1]
    out  = []
    for i in range(0, len(norm), chunk):
        batch = torch.tensor(norm[i:i + chunk], dtype=torch.float32,
                             device=device)
        with torch.no_grad():
            q = model(batch)                   # (chunk, n_actions)
            v = q.min(dim=1).values            # (chunk,)
        out.append(v.cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# DeepReach batched value evaluation (adds chunking on top of
# BRATController.get_values_batch_states, which does a single forward pass)
# ---------------------------------------------------------------------------

def brat_value_batch(ctrl: BRATController, states_6d: np.ndarray,
                     t_physical: float, chunk: int = 50_000) -> np.ndarray:
    """Chunked wrapper around BRATController.get_values_batch_states.

    Without chunking, N = 500 000 can OOM on GPU at the coord_to_input step.
    """
    out = []
    for i in range(0, len(states_6d), chunk):
        v = ctrl.get_values_batch_states(states_6d[i:i + chunk], t_physical)
        out.append(v)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_in_brat: np.ndarray,
                    gt_in_brat:   np.ndarray) -> dict:
    """Compute FPR, TPR and related binary classification metrics.

    Positive class = "inside BRAT" (safe / reachable).

    Args:
        pred_in_brat: boolean (N,) — method predicts state is inside BRAT.
        gt_in_brat:   boolean (N,) — grid ground truth (inside BRAT = True).

    Returns dict with:
        TP, FP, TN, FN (int counts),
        TPR, FPR, TNR, FNR, precision, accuracy, balanced_accuracy (float),
        n_gt_safe, n_gt_unsafe (int),
        pred_safe_frac (float).
    """
    pred = pred_in_brat.astype(bool)
    gt   = gt_in_brat.astype(bool)

    TP = int(np.sum( pred &  gt))
    FP = int(np.sum( pred & ~gt))
    TN = int(np.sum(~pred & ~gt))
    FN = int(np.sum(~pred &  gt))

    n_gt_safe   = TP + FN   # total positives in ground truth
    n_gt_unsafe = FP + TN   # total negatives in ground truth
    n_total     = TP + FP + TN + FN

    # Guard against empty classes
    TPR = TP / n_gt_safe   if n_gt_safe   > 0 else float('nan')
    FPR = FP / n_gt_unsafe if n_gt_unsafe > 0 else float('nan')
    TNR = TN / n_gt_unsafe if n_gt_unsafe > 0 else float('nan')
    FNR = FN / n_gt_safe   if n_gt_safe   > 0 else float('nan')

    prec = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    acc  = (TP + TN) / n_total

    # Balanced accuracy = (TPR + TNR) / 2
    if not (np.isnan(TPR) or np.isnan(TNR)):
        bal_acc = (TPR + TNR) / 2.0
    else:
        bal_acc = float('nan')

    return {
        'TP':                TP,
        'FP':                FP,
        'TN':                TN,
        'FN':                FN,
        'TPR':               float(TPR),
        'FPR':               float(FPR),
        'TNR':               float(TNR),
        'FNR':               float(FNR),
        'precision':         float(prec),
        'accuracy':          float(acc),
        'balanced_accuracy': float(bal_acc),
        'n_gt_safe':         n_gt_safe,
        'n_gt_unsafe':       n_gt_unsafe,
        'pred_safe_frac':    float(pred.sum() / n_total),
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check(combo, grid_times, t_physical,
                  learned_ctrl, vanilla_ctrl, rl_model,
                  rl_obs_center, rl_obs_half, device):
    """Assert both value functions are negative at the known goal state."""
    goal = np.array([[0.0, -1.2, 0.0, 0.0, np.pi / 2, 0.0]])
    t_idx = physical_time_to_index(t_physical, grid_times)

    gv  = grid_value_6D(combo, goal, t_idx)[0]
    drv = brat_value_batch(learned_ctrl, goal, t_physical)[0]

    print(f"  Sanity check at goal state (t={t_physical:.1f}s):")
    print(f"    Grid     V = {gv:.4f}  {'OK' if gv <= 0 else 'WARNING: should be <= 0'}")
    print(f"    Learned  V = {drv:.4f}  {'OK' if drv <= 0 else 'WARNING: should be <= 0'}")

    if vanilla_ctrl is not None:
        vanv = brat_value_batch(vanilla_ctrl, goal, t_physical)[0]
        print(f"    Vanilla  V = {vanv:.4f}  {'OK' if vanv <= 0 else 'WARNING: should be <= 0'}")

    if rl_model is not None:
        rlv = rl_value_batch(rl_model, goal, rl_obs_center, rl_obs_half, device)[0]
        print(f"    RL       V = {rlv:.4f}  {'OK' if rlv <= 0 else 'WARNING: should be <= 0'}")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_at_horizon(combo, grid_times, t_physical,
                        learned_ctrl, vanilla_ctrl,
                        rl_model, rl_obs_center, rl_obs_half,
                        n_samples: int, seed: int,
                        bounds: np.ndarray, device: str) -> dict:
    """Evaluate FPR/TPR metrics for all methods at a single time horizon.

    Args:
        rl_model: pass None to skip the RL method.
        vanilla_ctrl: pass None to skip the vanilla BRAT.

    Returns:
        dict mapping method name → metrics dict from compute_metrics().
    """
    rng = np.random.RandomState(seed)
    states = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, 6))

    # ---- Ground truth: grid-based BRAT ----
    t_idx = physical_time_to_index(t_physical, grid_times)
    print(f"  Grid time index: {t_idx}  (stored time ≈ {grid_times[t_idx]:.3f}s "
          f"vs requested {t_physical:.1f}s)")

    gt_raw = grid_value_6D(combo, states, t_idx)
    n_nonfinite_gt = int(np.sum(~np.isfinite(gt_raw)))
    if n_nonfinite_gt > 0:
        print(f"  WARNING: {n_nonfinite_gt:,} non-finite grid values "
              f"— treated as outside BRAT")
    gt_raw    = np.where(np.isfinite(gt_raw), gt_raw, np.inf)
    gt_in_brat = gt_raw <= 0

    n_safe   = int(gt_in_brat.sum())
    n_unsafe = int((~gt_in_brat).sum())
    print(f"  Ground truth: {n_safe:,} safe ({100*n_safe/n_samples:.2f}%), "
          f"{n_unsafe:,} unsafe ({100*n_unsafe/n_samples:.2f}%)")

    results = {}

    # ---- Learned BRAT ----
    print("  Evaluating Learned BRAT ...")
    t0 = _time.time()
    dr_raw = brat_value_batch(learned_ctrl, states, t_physical)
    dr_raw = np.where(np.isfinite(dr_raw), dr_raw, np.inf)
    results['learned'] = compute_metrics(dr_raw <= 0, gt_in_brat)
    print(f"    done ({_time.time()-t0:.1f}s) — "
          f"pred_safe={results['learned']['pred_safe_frac']*100:.2f}%")

    # ---- Vanilla BRAT ----
    if vanilla_ctrl is not None:
        print("  Evaluating Vanilla BRAT ...")
        t0 = _time.time()
        van_raw = brat_value_batch(vanilla_ctrl, states, t_physical)
        van_raw = np.where(np.isfinite(van_raw), van_raw, np.inf)
        results['vanilla'] = compute_metrics(van_raw <= 0, gt_in_brat)
        print(f"    done ({_time.time()-t0:.1f}s) — "
              f"pred_safe={results['vanilla']['pred_safe_frac']*100:.2f}%")

    # ---- RL BRAT (time-independent) ----
    if rl_model is not None:
        print("  Evaluating RL BRAT (V = min_a Q, time-independent) ...")
        t0 = _time.time()
        rl_raw = rl_value_batch(rl_model, states, rl_obs_center,
                                rl_obs_half, device)
        rl_raw = np.where(np.isfinite(rl_raw), rl_raw, np.inf)
        results['rl'] = compute_metrics(rl_raw <= 0, gt_in_brat)
        print(f"    done ({_time.time()-t0:.1f}s) — "
              f"pred_safe={results['rl']['pred_safe_frac']*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_METHOD_LABELS = {
    'learned': 'Learned BRAT',
    'vanilla': 'Vanilla BRAT',
    'rl':      'RL BRAT     ',
}


def print_metrics_table(all_results: dict) -> None:
    """Print a formatted results table to stdout.

    Args:
        all_results: {t_physical -> {method_name -> metrics_dict}}
    """
    col_w = 14
    header = (
        f"{'Method':<{col_w}} {'Time(s)':>7}  "
        f"{'TPR':>7}  {'FPR':>7}  {'TNR':>7}  {'FNR':>7}  "
        f"{'Prec':>7}  {'Acc':>7}  {'BalAcc':>7}  "
        f"{'GT_safe':>9}  {'GT_unsafe':>11}"
    )
    sep = '─' * len(header)

    print()
    print(sep)
    print(' BRAT CLASSIFICATION METRICS  '
          '(positive class = inside BRAT = safe/reachable)')
    print(' Ground truth: grid-based Hamilton-Jacobi (4D+2D decomposition)')
    print(sep)
    print()
    print(' Metric definitions:')
    print('   TPR = TP/(TP+FN)  high → less conservative (captures more of true BRAT)')
    print('   FPR = FP/(FP+TN)  high → more optimistic   (claims unsafe states are safe)')
    print()
    print(sep)
    print(header)
    print(sep)

    def _f(x, digits=4):
        if isinstance(x, float) and not np.isnan(x):
            return f'{x:.{digits}f}'
        return '   NaN '

    for t_phys in sorted(all_results.keys()):
        method_results = all_results[t_phys]
        for mname, m in method_results.items():
            label = _METHOD_LABELS.get(mname, mname)
            row = (
                f"{label:<{col_w}} {t_phys:>7.1f}  "
                f"{_f(m['TPR']):>7}  {_f(m['FPR']):>7}  "
                f"{_f(m['TNR']):>7}  {_f(m['FNR']):>7}  "
                f"{_f(m['precision']):>7}  {_f(m['accuracy']):>7}  "
                f"{_f(m['balanced_accuracy']):>7}  "
                f"{m['n_gt_safe']:>9,}  {m['n_gt_unsafe']:>11,}"
            )
            print(row)
        if len(method_results) > 1:
            print()  # blank line between time horizons

    print(sep)
    print()
    print(' TP = pred safe  & gt safe  |  FP = pred safe  & gt unsafe')
    print(' TN = pred unsafe & gt unsafe |  FN = pred unsafe & gt safe')
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='BRAT classification metrics: FPR/TPR vs. grid ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # -- Method checkpoints --
    parser.add_argument(
        '--learned_checkpoint', required=True,
        help='Path to learned BRAT model checkpoint (model_final.pth)')
    parser.add_argument(
        '--vanilla_checkpoint', default=None,
        help='Path to vanilla BRAT model checkpoint (optional)')
    parser.add_argument(
        '--rl_checkpoint', default=None,
        help='Path to RL Q-network checkpoint (Q-*.pth, optional)')
    parser.add_argument(
        '--rl_metadata', default=None,
        help='Path to RL train.pkl (required when --rl_checkpoint is given)')

    # -- Query parameters --
    parser.add_argument(
        '--tMax', type=float, default=10.0,
        help='Training tMax used for BRATController initialisation (default: 10.0)')
    parser.add_argument(
        '--time_horizons', type=float, nargs='+', default=None,
        help='Physical time horizons (s) to evaluate (default: [tMax])')

    # -- Sampling --
    parser.add_argument(
        '--n_samples', type=int, default=500_000,
        help='Number of Monte Carlo samples (default: 500,000)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility')
    parser.add_argument(
        '--use_focused_bounds', action='store_true',
        help='Use tight near-goal sampling bounds (FOCUSED_BOUNDS) instead of MC_BOUNDS')

    # -- Grid parameters --
    parser.add_argument(
        '--grid_cache_dir', default=None,
        help='Directory for HJ grid cache (default: <output_dir>/grid_cache)')
    parser.add_argument(
        '--grid_final_time', type=float, default=-30.0,
        help='Grid backward time horizon in seconds (negative, default: -30.0)')

    # -- Output --
    parser.add_argument(
        '--output_dir', default='./outputs/brat_metrics',
        help='Directory for output files')
    parser.add_argument(
        '--device', default='cuda',
        help='Torch device (default: cuda, falls back to cpu if unavailable)')

    args = parser.parse_args()

    # Resolve device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        args.device = 'cpu'

    # Validate RL args
    if args.rl_checkpoint is not None and args.rl_metadata is None:
        parser.error('--rl_metadata is required when --rl_checkpoint is provided')

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.grid_cache_dir or os.path.join(args.output_dir, 'grid_cache')

    time_horizons = args.time_horizons if args.time_horizons else [args.tMax]
    bounds        = FOCUSED_BOUNDS if args.use_focused_bounds else MC_BOUNDS
    bounds_name   = 'focused' if args.use_focused_bounds else 'mc'
    hypervolume   = float(np.prod(bounds[:, 1] - bounds[:, 0]))

    print('=' * 65)
    print(f'  BRAT Metrics Evaluation')
    print(f'  Time horizons : {time_horizons}')
    print(f'  N samples     : {args.n_samples:,}')
    print(f'  Bounds        : {bounds_name}  (hypervolume = {hypervolume:.1f})')
    print(f'  Device        : {args.device}')
    print('=' * 65)

    # ------------------------------------------------------------------ #
    # 1. Load grid-based BRAT (ground truth)
    # ------------------------------------------------------------------ #
    print('\n── Grid-based BRAT (ground truth) ──')
    t0 = _time.time()
    ComboController = _get_combo_module().ComboController
    combo = ComboController(
        # Parameters aligned with Docking6D defaults in volume_comparison.py
        mc=200.0,
        orbit_alt=400,
        post_hw_x=0.6,
        post_length=0.2,
        w_t=6, h_t=3,
        w_c=1.0, h_c=1.0,
        eps_p=0.1,
        eps_v=0.1,
        eps_theta=0.04,
        eps_omega=0.05,
        u_bar_4D=20.0,
        u_bar_2D=1.5,
        d_bar_4D=0.0,   # no disturbance — matches DeepReach training
        d_bar_2D=0.0,
        final_time=args.grid_final_time,
        cache_dir=cache_dir,
    )
    grid_times = np.array(combo.times)
    print(f"  Ready in {_time.time()-t0:.1f}s  |  "
          f"{len(grid_times)} time steps  "
          f"[{grid_times[0]:.2f}, {grid_times[-1]:.2f}]")

    # ------------------------------------------------------------------ #
    # 2. Load learned BRAT ("our method")
    # ------------------------------------------------------------------ #
    print('\n── Learned BRAT (DeepReach) ──')
    learned_ctrl = BRATController(
        checkpoint_path=args.learned_checkpoint,
        tMax=args.tMax,
        device=args.device,
    )

    # ------------------------------------------------------------------ #
    # 3. Load vanilla BRAT (optional baseline)
    # ------------------------------------------------------------------ #
    vanilla_ctrl = None
    if args.vanilla_checkpoint:
        print('\n── Vanilla BRAT (DeepReach, no MPC) ──')
        vanilla_ctrl = BRATController(
            checkpoint_path=args.vanilla_checkpoint,
            tMax=args.tMax,
            device=args.device,
        )

    # ------------------------------------------------------------------ #
    # 4. Load RL BRAT (optional baseline)
    # ------------------------------------------------------------------ #
    rl_model = rl_obs_center = rl_obs_half = None
    if args.rl_checkpoint:
        print('\n── RL BRAT (DDQNSingle) ──')
        rl_model, rl_obs_center, rl_obs_half = load_rl_q_network(
            args.rl_checkpoint, args.rl_metadata, args.device)

    # ------------------------------------------------------------------ #
    # 5. Sanity check at the known goal state
    # ------------------------------------------------------------------ #
    print('\n── Sanity check ──')
    _sanity_check(combo, grid_times, time_horizons[0],
                  learned_ctrl, vanilla_ctrl,
                  rl_model, rl_obs_center, rl_obs_half,
                  args.device)

    # ------------------------------------------------------------------ #
    # 6. Evaluate metrics at each requested time horizon
    # ------------------------------------------------------------------ #
    all_results = {}   # t_phys -> {method_name -> metrics}

    for t_phys in time_horizons:
        print(f'\n── t = {t_phys:.1f}s  (N={args.n_samples:,}, bounds={bounds_name}) ──')
        t0 = _time.time()
        results = evaluate_at_horizon(
            combo, grid_times, t_phys,
            learned_ctrl, vanilla_ctrl,
            rl_model, rl_obs_center, rl_obs_half,
            args.n_samples, args.seed, bounds, args.device,
        )
        all_results[t_phys] = results
        print(f"  Horizon done in {_time.time()-t0:.1f}s")

    # ------------------------------------------------------------------ #
    # 7. Print results table
    # ------------------------------------------------------------------ #
    print_metrics_table(all_results)

    # ------------------------------------------------------------------ #
    # 8. Save results to JSON
    # ------------------------------------------------------------------ #
    json_out = {
        '_metadata': {
            'learned_checkpoint': args.learned_checkpoint,
            'vanilla_checkpoint': args.vanilla_checkpoint,
            'rl_checkpoint':      args.rl_checkpoint,
            'tMax':               args.tMax,
            'time_horizons':      time_horizons,
            'n_samples':          args.n_samples,
            'seed':               args.seed,
            'bounds_used':        bounds_name,
            'bounds':             bounds.tolist(),
            'hypervolume':        hypervolume,
            'grid_final_time':    args.grid_final_time,
            'notes': [
                'TPR = TP/(TP+FN): fraction of grid-safe states found safe. '
                'Higher = less conservative.',
                'FPR = FP/(FP+TN): fraction of grid-unsafe states wrongly '
                'claimed safe. Higher = more optimistic / less safe.',
                'RL value is time-independent: V(s) = min_a Q(normalize(s),a). '
                'Same result repeated at every time horizon.',
                'Non-finite values treated as outside BRAT (positive).',
                'Grid params: d_bar=0 (no disturbance), matched to DeepReach training.',
            ],
            'metric_definitions': {
                'TP': 'pred_in_BRAT=True  AND gt_in_BRAT=True',
                'FP': 'pred_in_BRAT=True  AND gt_in_BRAT=False  (overly optimistic)',
                'TN': 'pred_in_BRAT=False AND gt_in_BRAT=False',
                'FN': 'pred_in_BRAT=False AND gt_in_BRAT=True   (overly conservative)',
                'TPR': 'TP / (TP+FN)',
                'FPR': 'FP / (FP+TN)',
                'TNR': 'TN / (FP+TN)  = 1 - FPR',
                'FNR': 'FN / (TP+FN)  = 1 - TPR',
                'balanced_accuracy': '(TPR + TNR) / 2',
            },
        },
        'results': {
            str(t): {m: v for m, v in res.items()}
            for t, res in all_results.items()
        },
    }

    json_path = os.path.join(args.output_dir, 'brat_metrics.json')
    with open(json_path, 'w') as fh:
        json.dump(json_out, fh, indent=2)
    print(f"Results saved → {json_path}")

    print('=' * 65)
    print('  Done.')
    print('=' * 65)


if __name__ == '__main__':
    main()
