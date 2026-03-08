#!/usr/bin/env python3
"""
Volume comparison between grid-based (ground truth) and DeepReach (learned)
value functions for the 6D docking reach-avoid problem.

The grid-based value function is solved via a 4D+2D decomposition
(hj_reachability library) and combined as V_6D = max(V_4D, V_2D).
The DeepReach value function is a SIREN neural network trained via
BRAT HJI-VI PDE loss.

This script:
  1. Loads / solves the grid-based value functions (with caching).
  2. Loads the DeepReach model via BRTController.
  3. Evaluates both value functions on 2D slices through state space.
  4. Computes a Monte-Carlo estimate of 6D BRT volumes.
  5. Produces side-by-side contour plots and a volume summary table.

Known limitations (documented per plan):
  - Target-set norm differs: grid uses L-inf; DeepReach uses L2-weighted.
  - Avoid-set geometry differs: grid uses semicircle centered at
    (chaser_buffer, 0) with offset; DeepReach uses origin-centered
    semicircle with cutout term and asymmetric scaling.
  - These are inherent modelling differences and are expected to produce
    some disagreement between the two value functions.

Usage:
    python volume_comparison.py \
        --checkpoint_path runs/Docking6D_RA_14sec/training/checkpoints/model_epoch_145000.pth \
        --tMax 15 \
        --time_horizons 5 10 15 \
        --n_monte_carlo 500000 \
        --slice_resolution 200 \
        --output_dir ./outputs/volume_comparison \
        --device cuda
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---- path setup ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'gridBased6DImplementation'))

from utils.controllers import BRTController

# Grid-based imports (deferred to avoid import errors when JAX not available)
_combo_module = None

def _get_combo_module():
    global _combo_module
    if _combo_module is None:
        import importlib
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'gridBased6DImplementation'))
        _combo_module = importlib.import_module('ComboControl')
    return _combo_module


# ======================================================================
#  Helper: physical time -> grid time index
# ======================================================================
def physical_time_to_index(t_physical, times_array):
    """Convert a positive physical time horizon (seconds) to the closest
    index into the grid solver's ``times`` array.

    The grid solver stores times backwards: times[0] = 0 (terminal) and
    times[-1] = final_time (e.g. -30).  A physical horizon of T seconds
    corresponds to ``times[idx] ≈ -T``.
    """
    target = -abs(t_physical)
    idx = int(np.argmin(np.abs(np.array(times_array) - target)))
    return idx


# ======================================================================
#  Grid value query helpers (batched)
# ======================================================================
def grid_value_batch_4D(combo, states_4d, time_idx):
    """Query V_4D for (N, 4) states at a single time index.

    Uses scipy.interpolate.interpn under the hood (same as
    ComboController.value_at_state_4D but batched).
    """
    from scipy.interpolate import interpn
    coord_vecs = [np.array(v) for v in combo.grid_4D.coordinate_vectors]
    grid_vals = np.array(combo.values_4D[time_idx])
    pts = np.atleast_2d(states_4d)
    vals = interpn(coord_vecs, grid_vals, pts,
                   method='linear', bounds_error=False, fill_value=np.inf)
    return vals  # (N,)


def grid_value_batch_2D(combo, states_2d, time_idx):
    """Query V_2D for (N, 2) states at a single time index."""
    from scipy.interpolate import interpn
    coord_vecs = [np.array(v) for v in combo.grid_2D.coordinate_vectors]
    grid_vals = np.array(combo.values_2D[time_idx])
    pts = np.atleast_2d(states_2d)
    vals = interpn(coord_vecs, grid_vals, pts,
                   method='linear', bounds_error=False, fill_value=np.inf)
    return vals  # (N,)


def grid_value_6D(combo, states_6d, time_idx):
    """Combined 6D grid value: V_6D = max(V_4D, V_2D)."""
    v4 = grid_value_batch_4D(combo, states_6d[:, :4], time_idx)
    v2 = grid_value_batch_2D(combo, states_6d[:, 4:], time_idx)
    return np.maximum(v4, v2)


# ======================================================================
#  DeepReach value query helper (batched)
# ======================================================================
def deepreach_value_6D(brt_ctrl, states_6d, t_physical):
    """Query the DeepReach value function for (N, 6) states at time t."""
    return brt_ctrl.get_values_batch_states(states_6d, t_physical)


# ======================================================================
#  Slice definitions
# ======================================================================
DEFAULT_SLICES = [
    {
        'name': 'px_py',
        'vary': [0, 1],
        'fixed': {2: 0, 3: 0, 4: np.pi/2, 5: 0.0},
        'labels': ('px (m)', 'py (m)'),
        'title': 'Position slice (vx=vy=0, θ=π/2, ω=0)',
    },
    {
        'name': 'vx_vy',
        'vary': [2, 3],
        'fixed': {0: 0.0, 1: -1.2, 4: np.pi/2, 5: 0.0},
        'labels': ('vx (m/s)', 'vy (m/s)'),
        'title': 'Velocity slice (px=0, py=-1.2, θ=π/2, ω=0)',
    },
    {
        'name': 'theta_omega',
        'vary': [4, 5],
        'fixed': {0: 0.0, 1: -1.2, 2: 0.0, 3: 0.0},
        'labels': ('θ (rad)', 'ω (rad/s)'),
        'title': 'Attitude slice (px=0, py=-1.2, vx=vy=0)',
    },
]

# State bounds for each dim (used for slice grids and Monte Carlo)
STATE_BOUNDS = np.array([
    [-15.0,  15.0],  # px
    [-15.0,  15.0],  # py
    [ -1.5,   1.5],  # vx
    [ -1.5,   1.5],  # vy
    [-np.pi, np.pi], # theta
    [ -1.0,   1.0],  # omega
])


# ======================================================================
#  Slice evaluation
# ======================================================================
def evaluate_slice(combo, brt_ctrl, slice_def, t_physical, resolution,
                   grid_times):
    """Evaluate both value functions on a 2D grid for a given slice.

    Returns:
        dict with keys: grid_vals, dr_vals, xx, yy, extent
    """
    vary = slice_def['vary']
    fixed = slice_def['fixed']
    lo0 = STATE_BOUNDS[vary[0], 0]
    hi0 = STATE_BOUNDS[vary[0], 1]
    lo1 = STATE_BOUNDS[vary[1], 0]
    hi1 = STATE_BOUNDS[vary[1], 1]

    ax0 = np.linspace(lo0, hi0, resolution)
    ax1 = np.linspace(lo1, hi1, resolution)
    xx, yy = np.meshgrid(ax0, ax1, indexing='xy')
    flat0 = xx.ravel()
    flat1 = yy.ravel()
    N = len(flat0)

    # Build full 6D states
    states = np.zeros((N, 6))
    for dim, val in fixed.items():
        states[:, dim] = val
    states[:, vary[0]] = flat0
    states[:, vary[1]] = flat1

    # Grid value
    t_idx = physical_time_to_index(t_physical, grid_times)
    gv = grid_value_6D(combo, states, t_idx).reshape(xx.shape)

    # DeepReach value
    dv = deepreach_value_6D(brt_ctrl, states, t_physical).reshape(xx.shape)

    extent = [lo0, hi0, lo1, hi1]
    return {
        'grid_vals': gv,
        'dr_vals': dv,
        'xx': xx,
        'yy': yy,
        'extent': extent,
    }


# ======================================================================
#  Monte Carlo volume estimation
# ======================================================================
def monte_carlo_volume(combo, brt_ctrl, t_physical, grid_times, n_samples,
                       seed=42):
    """Estimate 6D BRT volumes via uniform random sampling.

    Returns dict with keys:
        grid_frac, dr_frac, overlap_frac, grid_only_frac, dr_only_frac,
        jaccard, n_samples
    """
    rng = np.random.RandomState(seed)
    states = rng.uniform(STATE_BOUNDS[:, 0], STATE_BOUNDS[:, 1],
                         size=(n_samples, 6))

    t_idx = physical_time_to_index(t_physical, grid_times)

    # Evaluate in chunks to avoid OOM
    chunk = 50_000
    gv_all = []
    dv_all = []
    for i in range(0, n_samples, chunk):
        s = states[i:i+chunk]
        gv_all.append(grid_value_6D(combo, s, t_idx))
        dv_all.append(deepreach_value_6D(brt_ctrl, s, t_physical))

    gv = np.concatenate(gv_all)
    dv = np.concatenate(dv_all)

    in_grid = gv <= 0
    in_dr   = dv <= 0
    overlap = in_grid & in_dr
    grid_only = in_grid & ~in_dr
    dr_only   = in_dr & ~in_grid
    union = in_grid | in_dr

    n = float(n_samples)
    jaccard = float(overlap.sum()) / float(union.sum()) if union.sum() > 0 else 0.0

    return {
        'grid_frac':      float(in_grid.sum() / n),
        'dr_frac':        float(in_dr.sum() / n),
        'overlap_frac':   float(overlap.sum() / n),
        'grid_only_frac': float(grid_only.sum() / n),
        'dr_only_frac':   float(dr_only.sum() / n),
        'jaccard':        jaccard,
        'n_samples':      n_samples,
    }


# ======================================================================
#  Plotting
# ======================================================================
def plot_slice_comparison(slice_result, slice_def, t_physical, save_path):
    """Three-panel plot: grid V, DeepReach V, and difference / overlap."""
    gv = slice_result['grid_vals']
    dv = slice_result['dr_vals']
    xx = slice_result['xx']
    yy = slice_result['yy']
    xlabel, ylabel = slice_def['labels']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{slice_def['title']}  |  t = {t_physical:.0f}s", fontsize=14)

    # Common value range for consistent colour bar
    vmin = min(gv.min(), dv.min())
    vmax = max(gv.max(), dv.max())
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None
    cmap = 'RdBu_r'

    # Panel 1: Grid
    c0 = axes[0].contourf(xx, yy, gv, levels=60, cmap=cmap, norm=norm)
    axes[0].contour(xx, yy, gv, levels=[0], colors='k', linewidths=1.5)
    axes[0].set_title('Grid-based V(x, t)')
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    plt.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: DeepReach
    c1 = axes[1].contourf(xx, yy, dv, levels=60, cmap=cmap, norm=norm)
    axes[1].contour(xx, yy, dv, levels=[0], colors='k', linewidths=1.5)
    axes[1].set_title('DeepReach V(x, t)')
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel(ylabel)
    plt.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlap / disagreement
    in_grid = (gv <= 0).astype(float)
    in_dr   = (dv <= 0).astype(float)
    # 0 = neither, 1 = grid only, 2 = DR only, 3 = both
    zone = in_grid + 2 * in_dr
    from matplotlib.colors import ListedColormap
    cmap_zone = ListedColormap(['#f0f0f0', '#fc8d62', '#66c2a5', '#8da0cb'])
    axes[2].imshow(zone, origin='lower', aspect='auto',
                   extent=slice_result['extent'], cmap=cmap_zone,
                   vmin=0, vmax=3, interpolation='nearest')
    axes[2].set_title('BRT overlap')
    axes[2].set_xlabel(xlabel); axes[2].set_ylabel(ylabel)
    # Legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='#f0f0f0', label='Neither'),
        mpatches.Patch(color='#fc8d62', label='Grid only'),
        mpatches.Patch(color='#66c2a5', label='DeepReach only'),
        mpatches.Patch(color='#8da0cb', label='Both'),
    ]
    axes[2].legend(handles=patches, fontsize=8, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved slice plot: {save_path}")


def plot_volume_vs_time(volume_results, save_path):
    """Line chart of BRT volume fraction vs. time horizon for both methods."""
    times = sorted(volume_results.keys())
    grid_fracs = [volume_results[t]['grid_frac'] for t in times]
    dr_fracs   = [volume_results[t]['dr_frac']   for t in times]
    overlap    = [volume_results[t]['overlap_frac'] for t in times]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, grid_fracs, 'o-', color='#fc8d62', label='Grid BRT', lw=2)
    ax.plot(times, dr_fracs,   's-', color='#66c2a5', label='DeepReach BRT', lw=2)
    ax.plot(times, overlap,    '^--', color='#8da0cb', label='Overlap', lw=1.5)
    ax.set_xlabel('Time Horizon (s)')
    ax.set_ylabel('BRT Volume Fraction')
    ax.set_title('BRT Volume vs. Time Horizon (Monte Carlo)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved volume-vs-time plot: {save_path}")


def print_volume_table(volume_results):
    """Print a formatted volume comparison table to stdout."""
    header = (f"{'Time(s)':>8} {'Grid BRT%':>10} {'DR BRT%':>10} "
              f"{'Overlap%':>10} {'Grid-only%':>11} {'DR-only%':>10} "
              f"{'Jaccard':>8}")
    sep = '-' * len(header)
    print('\n' + sep)
    print('VOLUME COMPARISON  (Monte Carlo)')
    print(sep)
    print(header)
    print(sep)
    for t in sorted(volume_results.keys()):
        v = volume_results[t]
        print(f"{t:>8.0f} {v['grid_frac']*100:>9.2f}% {v['dr_frac']*100:>9.2f}% "
              f"{v['overlap_frac']*100:>9.2f}% {v['grid_only_frac']*100:>10.2f}% "
              f"{v['dr_only_frac']*100:>9.2f}% {v['jaccard']:>8.4f}")
    print(sep + '\n')


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Volume comparison: grid-based vs DeepReach value functions',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to DeepReach model checkpoint (.pth)')
    parser.add_argument('--tMax', type=float, default=15.0,
                        help='DeepReach tMax for value queries')
    parser.add_argument('--time_horizons', type=float, nargs='+',
                        default=[5.0, 10.0, 15.0],
                        help='Time horizons (seconds) to evaluate')
    parser.add_argument('--n_monte_carlo', type=int, default=500_000,
                        help='Number of random samples for volume estimation')
    parser.add_argument('--slice_resolution', type=int, default=200,
                        help='Grid resolution per axis for 2D slice plots')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/volume_comparison',
                        help='Directory for output files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device for DeepReach queries')
    parser.add_argument('--grid_cache_dir', type=str, default=None,
                        help='Directory for caching grid HJ solutions. '
                             'Defaults to <output_dir>/grid_cache')
    parser.add_argument('--grid_final_time', type=float, default=-30.0,
                        help='Grid solver backward time horizon (negative)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for Monte Carlo sampling')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.grid_cache_dir or os.path.join(args.output_dir, 'grid_cache')

    # ---- 1. Load DeepReach model ----
    print('=' * 60)
    print('Loading DeepReach model ...')
    print('=' * 60)
    t0 = time.time()
    brt_ctrl = BRTController(
        checkpoint_path=args.checkpoint_path,
        tMax=args.tMax,
        device=args.device,
    )
    print(f"  DeepReach model loaded in {time.time() - t0:.1f}s")

    # ---- 2. Load / solve grid-based value functions ----
    print('\n' + '=' * 60)
    print('Loading grid-based value functions (aligned to DeepReach params) ...')
    print('=' * 60)
    t0 = time.time()
    ComboController = _get_combo_module().ComboController
    combo = ComboController(
        # Aligned to DeepReach Docking6D defaults
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
        d_bar_4D=0.0,   # no disturbance (match DeepReach)
        d_bar_2D=0.0,
        final_time=args.grid_final_time,
        cache_dir=cache_dir,
    )
    print(f"  Grid solver ready in {time.time() - t0:.1f}s")

    grid_times = np.array(combo.times)

    # ---- 3. Slice evaluations ----
    print('\n' + '=' * 60)
    print('Evaluating 2D slices ...')
    print('=' * 60)
    slice_results = {}  # (slice_name, t) -> result dict
    for sl in DEFAULT_SLICES:
        for t_phys in args.time_horizons:
            print(f"  Slice '{sl['name']}' at t={t_phys:.0f}s ...")
            res = evaluate_slice(combo, brt_ctrl, sl, t_phys,
                                 args.slice_resolution, grid_times)
            slice_results[(sl['name'], t_phys)] = res

            fname = f"slice_{sl['name']}_t{t_phys:.0f}s.png"
            plot_slice_comparison(res, sl, t_phys,
                                 os.path.join(args.output_dir, fname))

    # ---- 4. Monte Carlo volume estimation ----
    print('\n' + '=' * 60)
    print('Monte Carlo volume estimation ...')
    print('=' * 60)
    volume_results = {}
    for t_phys in args.time_horizons:
        print(f"  t={t_phys:.0f}s  (N={args.n_monte_carlo:,}) ...")
        t0 = time.time()
        vol = monte_carlo_volume(combo, brt_ctrl, t_phys, grid_times,
                                 args.n_monte_carlo, seed=args.seed)
        print(f"    done in {time.time() - t0:.1f}s  |  "
              f"grid={vol['grid_frac']*100:.2f}%  "
              f"DR={vol['dr_frac']*100:.2f}%  "
              f"Jaccard={vol['jaccard']:.4f}")
        volume_results[t_phys] = vol

    print_volume_table(volume_results)

    # ---- 5. Volume-vs-time plot ----
    vol_plot_path = os.path.join(args.output_dir, 'volume_vs_time.png')
    plot_volume_vs_time(volume_results, vol_plot_path)

    # ---- 6. Save all results to JSON ----
    json_data = {
        '_metadata': {
            'checkpoint_path': args.checkpoint_path,
            'tMax': args.tMax,
            'time_horizons': args.time_horizons,
            'n_monte_carlo': args.n_monte_carlo,
            'slice_resolution': args.slice_resolution,
            'grid_final_time': args.grid_final_time,
            'seed': args.seed,
            'notes': [
                'Grid uses L-inf target sets; DeepReach uses weighted max.',
                'Body+post obstacle geometry consistent between grid and DeepReach.',
                'Grid params aligned to DeepReach: post_hw_x=0.6, post_length=0.2, eps_p=0.1, d_bar=0.',
            ],
        },
        'volume': {str(k): v for k, v in volume_results.items()},
    }
    json_path = os.path.join(args.output_dir, 'volume_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    print('\n' + '=' * 60)
    print('Volume comparison complete.')
    print(f'All outputs in: {args.output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
