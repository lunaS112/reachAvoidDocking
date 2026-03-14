#!/usr/bin/env python3
"""
Gradient Quality Comparison: Grid-Based (25s) vs DeepReach (10s)

Picks ICs that lie INSIDE the 25s grid-based BRAT but OUTSIDE the 10s
learned BRAT.  At these states the DeepReach controller operates in
Phase 1 (static gradient at tMax).  We compare:
  - Trajectories and control inputs from both controllers
  - Direct gradient cosine similarity and sign agreement on the
    control-relevant components (vx, vy, omega)

Usage:
    python gradient_quality_comparison.py \
        --checkpoint_path ../runs/Docking6D_RA_10sec_NEW/training/checkpoints/model_final.pth \
        --tMax 10 --grid_time_horizon 25 --n_ics 5 --device cuda
"""

import argparse
import json
import os
import sys
import time as _time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- path setup (comparisons/ -> deepReachMPCReachAvoid/ -> project root) ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPREACH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(DEEPREACH_DIR)
sys.path.insert(0, DEEPREACH_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'gridBased6DImplementation'))

from utils.controllers import BRATController
from comparisons.volume_comparison import (
    physical_time_to_index, grid_value_6D, deepreach_value_6D, MC_BOUNDS,
)

# Lazy import for grid-based controller
_combo_module = None
def _get_combo_module():
    global _combo_module
    if _combo_module is None:
        import importlib
        _combo_module = importlib.import_module('ComboControl')
    return _combo_module


# ======================================================================
#  Shared dynamics (numpy CW equations)
# ======================================================================
def _load_dynamics_fn():
    """Return the numpy CW dynamics f(s, u) from the grid-based utils."""
    import importlib.util
    dynamics_path = os.path.join(
        PROJECT_ROOT, 'gridBased6DImplementation', 'utils', 'dynamics.py')
    spec = importlib.util.spec_from_file_location('grid_dynamics', dynamics_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.f


# ======================================================================
#  IC selection
# ======================================================================
def select_ics(combo, brat_ctrl, grid_times, grid_horizon, n_ics,
               n_candidates=500_000, seed=42):
    """Sample ICs inside 25s grid BRAT but outside 10s DeepReach BRAT.

    Returns (n_ics, 6) array of selected ICs, or fewer if not enough found.
    """
    rng = np.random.RandomState(seed)
    states = rng.uniform(MC_BOUNDS[:, 0], MC_BOUNDS[:, 1],
                         size=(n_candidates, 6))

    # Grid value at the specified horizon (e.g. 25s)
    t_idx = physical_time_to_index(grid_horizon, grid_times)
    gv = grid_value_6D(combo, states, t_idx)

    # DeepReach value at tMax
    dv = deepreach_value_6D(brat_ctrl, states, brat_ctrl.tMax)

    # Inside grid BRAT (<=0) AND outside DeepReach BRAT (>0)
    mask = (gv <= 0) & (dv > 0)
    candidates = states[mask]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No ICs found inside grid BRAT (t={grid_horizon}s) but outside "
            f"DeepReach BRAT (tMax={brat_ctrl.tMax}s). Try increasing n_candidates.")

    # Prefer ICs with moderate positive DeepReach value (clearly outside but not extreme)
    dv_cand = dv[mask]
    order = np.argsort(dv_cand)  # smallest positive first (barely outside)
    selected = candidates[order[:n_ics]]
    print(f"  IC selection: {mask.sum()} candidates found, selected {len(selected)}")
    return selected


# ======================================================================
#  Unified simulation
# ======================================================================
def simulate_both(ic, combo, brat_ctrl, dynamics_fn, dt, max_sim_time,
                  grid_times):
    """Simulate both controllers from the same IC using shared dynamics.

    Returns (result_grid, result_dr) dicts with keys:
        trajectory, controls, times, values, gradients, success
    """
    num_steps = int(max_sim_time / dt)

    def _run_one(controller_fn, value_fn, gradient_fn):
        states = np.zeros((num_steps + 1, 6))
        controls = np.zeros((num_steps, 3))
        values = np.zeros(num_steps)
        grads = np.zeros((num_steps, 6))
        times = np.zeros(num_steps)

        states[0] = ic.copy()

        for k in range(num_steps):
            t = k * dt
            s = states[k]

            # Get control from this controller
            u = controller_fn(s, t)
            controls[k] = u
            times[k] = t

            # Record value and gradient
            values[k] = value_fn(s, t)
            grads[k] = gradient_fn(s, t)

            # Euler integration with shared dynamics
            s_next = s + dt * dynamics_fn(s, u)
            # Wrap theta to [-pi, pi]
            s_next[4] = (s_next[4] + np.pi) % (2 * np.pi) - np.pi
            states[k + 1] = s_next

        return {
            'trajectory': states,  # (num_steps+1, 6)
            'controls': controls,  # (num_steps, 3)
            'times': times,        # (num_steps,)
            'values': values,      # (num_steps,)
            'gradients': grads,    # (num_steps, 6)
        }

    # --- Grid controller ---
    combo.reset()

    def grid_value(s, t):
        brt_time = max(combo.final_time, min(0, combo.final_time + t))
        tidx = int(np.argmin(np.abs(np.array(combo.times) - brt_time)))
        s4d, s2d = s[:4], s[4:]
        v4 = combo.value_at_state_4D(s4d, tidx)
        v2 = combo.value_at_state_2D(s2d, tidx)
        return max(v4, v2)

    def grid_gradient(s, t):
        brt_time = max(combo.final_time, min(0, combo.final_time + t))
        tidx = int(np.argmin(np.abs(np.array(combo.times) - brt_time)))
        g4 = combo.grad_at_state_4D(s[:4], tidx)  # (4,)
        g2 = combo.grad_at_state_2D(s[4:], tidx)  # (2,)
        return np.concatenate([g4, g2])  # (6,)

    result_grid = _run_one(combo.u_fn, grid_value, grid_gradient)
    result_grid['controller_type'] = 'grid'

    # --- DeepReach controller ---
    brat_ctrl.reset()

    def dr_value(s, t):
        return brat_ctrl.get_value(s, brat_ctrl.tMax)

    def dr_gradient(s, t):
        return brat_ctrl.get_gradient(s, brat_ctrl.tMax)

    result_dr = _run_one(brat_ctrl.u_fn, dr_value, dr_gradient)
    result_dr['controller_type'] = 'deepreach'

    return result_grid, result_dr


# ======================================================================
#  Gradient metrics
# ======================================================================
def compute_gradient_metrics(grads_grid, grads_dr):
    """Compute cosine similarity and sign agreement between gradient arrays.

    Args:
        grads_grid, grads_dr: (N, 6) gradient arrays

    Returns dict with:
        cosine_sim: (N,) cosine similarity of full 6D gradient
        sign_agreement: (N, 3) bool for control-relevant components (vx, vy, omega)
        sign_agreement_rate: (3,) fraction of steps where signs agree
        mean_cosine_sim: float
    """
    # Cosine similarity of full 6D gradient
    dot = np.sum(grads_grid * grads_dr, axis=1)
    norm_g = np.linalg.norm(grads_grid, axis=1) + 1e-12
    norm_d = np.linalg.norm(grads_dr, axis=1) + 1e-12
    cosine_sim = dot / (norm_g * norm_d)

    # Sign agreement on control-relevant indices: vx(2), vy(3), omega(5)
    ctrl_idx = [2, 3, 5]
    sign_grid = np.sign(grads_grid[:, ctrl_idx])
    sign_dr = np.sign(grads_dr[:, ctrl_idx])
    sign_agreement = (sign_grid == sign_dr)
    sign_agreement_rate = sign_agreement.mean(axis=0)

    return {
        'cosine_sim': cosine_sim,
        'sign_agreement': sign_agreement,
        'sign_agreement_rate': sign_agreement_rate,
        'mean_cosine_sim': float(np.mean(cosine_sim)),
    }


# ======================================================================
#  Trajectory deviation
# ======================================================================
def trajectory_deviation(traj_grid, traj_dr):
    """L2 distance between trajectories at each step (position only)."""
    pos_grid = traj_grid[:, :2]
    pos_dr = traj_dr[:, :2]
    min_len = min(len(pos_grid), len(pos_dr))
    return np.linalg.norm(pos_grid[:min_len] - pos_dr[:min_len], axis=1)


# ======================================================================
#  Plotting
# ======================================================================
def _draw_docking_scene(ax):
    """Draw target body, docking post, and goal region on ax."""
    from matplotlib.patches import Rectangle
    # Target body (6m x 3m, bottom at y=0)
    body = Rectangle((-3, 0), 6, 3, color='gray', alpha=0.3, label='Target body')
    ax.add_patch(body)
    # Docking post (1.2m x 0.2m, extends in -y)
    post = Rectangle((-0.6, -0.2), 1.2, 0.2, color='gray', alpha=0.5, label='Docking post')
    ax.add_patch(post)
    # Goal region (aligned with DeepReach: y in [-1.4, -0.9], x in [-0.1, 0.1])
    goal = Rectangle((-0.1, -1.4), 0.2, 0.5, color='green', alpha=0.3, label='Goal')
    ax.add_patch(goal)


def _add_brt_contours(ax, combo, brat_ctrl, grid_times, grid_horizon, dr_horizon,
                       px_range=(-6, 6), py_range=(-6, 6), resolution=120):
    """Draw zero level set contours of grid and DeepReach BRTs on a px-py plot.

    Evaluates at vx=0, vy=0, theta=pi/2, omega=0 (nominal docking state).
    """
    px = np.linspace(*px_range, resolution)
    py = np.linspace(*py_range, resolution)
    PX, PY = np.meshgrid(px, py)
    flat = np.column_stack([
        PX.ravel(), PY.ravel(),
        np.zeros(PX.size),          # vx = 0
        np.zeros(PX.size),          # vy = 0
        np.full(PX.size, np.pi/2),  # theta = pi/2
        np.zeros(PX.size),          # omega = 0
    ])

    # Grid BRT
    grid_idx = physical_time_to_index(grid_horizon, grid_times)
    v_grid = grid_value_6D(combo, flat, grid_idx).reshape(PX.shape)
    ax.contour(PX, PY, v_grid, levels=[0], colors='purple', linewidths=1,
               linestyles='dashed', label=f'Grid BRAT (t={grid_horizon}s)')

    # DeepReach BRT
    v_dr = deepreach_value_6D(brat_ctrl, flat, dr_horizon).reshape(PX.shape)
    ax.contour(PX, PY, v_dr, levels=[0], colors='green', linewidths=1,
               linestyles='dashed', label=f'DeepReach BRAT (t={dr_horizon}s)')


def plot_trajectory_overlay(result_grid, result_dr, ic_idx, save_path,
                            combo=None, brat_ctrl=None, grid_times=None,
                            grid_horizon=None, dr_horizon=None):
    """2D position trajectory overlay with docking scene."""
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_docking_scene(ax)

    # Draw BRT zero level set contours if controllers are provided
    if combo is not None and brat_ctrl is not None:
        _add_brt_contours(ax, combo, brat_ctrl, grid_times,
                          grid_horizon, dr_horizon)

    tg = result_grid['trajectory']
    td = result_dr['trajectory']
    ax.plot(tg[:, 0], tg[:, 1], 'b-', lw=1.5, label='Grid (25s)')
    ax.plot(td[:, 0], td[:, 1], 'r--', lw=1.5, label='DeepReach (10s)')
    ax.plot(tg[0, 0], tg[0, 1], 'ko', ms=8, label='Start')
    ax.plot(tg[-1, 0], tg[-1, 1], 'bs', ms=7)
    ax.plot(td[-1, 0], td[-1, 1], 'r^', ms=7)

    ax.set_xlabel('px (m)')
    ax.set_ylabel('py (m)')
    ax.set_title(f'Trajectory Comparison — IC {ic_idx}')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_states(result_grid, result_dr, ic_idx, save_path):
    """6 states vs time (3x2 subplots)."""
    labels = ['px (m)', 'py (m)', 'vx (m/s)', 'vy (m/s)', 'θ (rad)', 'ω (rad/s)']
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    tg = result_grid['trajectory']
    td = result_dr['trajectory']
    times_g = np.arange(tg.shape[0]) * (result_grid['times'][1] - result_grid['times'][0])
    times_d = np.arange(td.shape[0]) * (result_dr['times'][1] - result_dr['times'][0])

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(times_g, tg[:, i], 'b-', lw=1.2, label='Grid')
        ax.plot(times_d, td[:, i], 'r--', lw=1.2, label='DeepReach')
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    axes[-2].set_xlabel('Time (s)')
    fig.suptitle(f'State Comparison — IC {ic_idx}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_controls(result_grid, result_dr, ic_idx, save_path):
    """3 controls vs time."""
    labels = ['u_x (N)', 'u_y (N)', 'u_θ (N·m)']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.step(result_grid['times'], result_grid['controls'][:, i],
                'b-', lw=1.2, where='post', label='Grid')
        ax.step(result_dr['times'], result_dr['controls'][:, i],
                'r--', lw=1.2, where='post', label='DeepReach')
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Control Comparison — IC {ic_idx}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_gradient_metrics(grad_metrics, times, ic_idx, save_path):
    """Cosine similarity and sign agreement vs time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Cosine similarity
    ax = axes[0]
    ax.plot(times, grad_metrics['cosine_sim'], 'k-', lw=1.0)
    ax.axhline(1.0, color='green', ls=':', alpha=0.5)
    ax.axhline(0.0, color='gray', ls=':', alpha=0.3)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Gradient Comparison — IC {ic_idx}  |  '
                 f'Mean cos-sim = {grad_metrics["mean_cosine_sim"]:.4f}')
    ax.set_ylim(-0.2, 1.1)
    ax.grid(alpha=0.3)

    # Per-component sign agreement
    ax = axes[1]
    ctrl_labels = ['vx', 'vy', 'ω']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for j, (lbl, c) in enumerate(zip(ctrl_labels, colors)):
        agree = grad_metrics['sign_agreement'][:, j].astype(float)
        # Smooth with a rolling window for readability
        window = min(20, len(agree) // 5) if len(agree) > 20 else 1
        if window > 1:
            kernel = np.ones(window) / window
            smooth = np.convolve(agree, kernel, mode='same')
        else:
            smooth = agree
        ax.plot(times, smooth, color=c, lw=1.2,
                label=f'{lbl} ({grad_metrics["sign_agreement_rate"][j]*100:.1f}%)')
    ax.set_ylabel('Sign Agreement (smoothed)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_aggregate(all_metrics, save_path):
    """Bar chart of aggregated metrics across multiple ICs."""
    n = len(all_metrics)
    cos_sims = [m['mean_cosine_sim'] for m in all_metrics]
    sign_rates = np.array([m['sign_agreement_rate'] for m in all_metrics])
    traj_devs = [m['mean_traj_deviation'] for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Cosine similarity
    ax = axes[0]
    ax.bar(range(n), cos_sims, color='steelblue')
    ax.axhline(np.mean(cos_sims), color='red', ls='--',
               label=f'Mean={np.mean(cos_sims):.4f}')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Gradient Cosine Similarity')
    ax.legend()

    # Sign agreement per component
    ax = axes[1]
    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, sign_rates[:, 0], w, label='vx', color='tab:blue')
    ax.bar(x, sign_rates[:, 1], w, label='vy', color='tab:orange')
    ax.bar(x + w, sign_rates[:, 2], w, label='ω', color='tab:green')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Sign Agreement Rate')
    ax.set_title('Control Sign Agreement')
    ax.legend()

    # Trajectory deviation
    ax = axes[2]
    ax.bar(range(n), traj_devs, color='coral')
    ax.axhline(np.mean(traj_devs), color='red', ls='--',
               label=f'Mean={np.mean(traj_devs):.3f}m')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Mean Position Deviation (m)')
    ax.set_title('Trajectory Deviation')
    ax.legend()

    fig.suptitle(f'Aggregate Metrics — {n} ICs', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Gradient quality comparison: grid-based vs DeepReach',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--tMax', type=float, default=10.0,
                        help='DeepReach tMax for value/gradient queries')
    parser.add_argument('--grid_final_time', type=float, default=-30.0,
                        help='Grid solver backward time horizon (negative)')
    parser.add_argument('--grid_time_horizon', type=float, default=25.0,
                        help='Grid BRAT horizon for IC selection (seconds)')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--max_sim_time', type=float, default=60.0)
    parser.add_argument('--n_ics', type=int, default=1)
    parser.add_argument('--n_candidates', type=int, default=500_000)
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/gradient_comparison')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--grid_cache_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.grid_cache_dir or os.path.join(args.output_dir, 'grid_cache')

    # ---- 1. Load DeepReach model ----
    print('=' * 60)
    print('Loading DeepReach model ...')
    t0 = _time.time()
    brat_ctrl = BRATController(
        checkpoint_path=args.checkpoint_path,
        tMax=args.tMax,
        device=args.device,
    )
    print(f"  Loaded in {_time.time() - t0:.1f}s")

    # ---- 2. Load grid-based controller ----
    print('\n' + '=' * 60)
    print('Loading grid-based controller (filter_mode=None for pure optimal) ...')
    t0 = _time.time()
    ComboController = _get_combo_module().ComboController
    combo = ComboController(
        mc=200.0, orbit_alt=400,
        post_hw_x=0.6, post_length=0.2,
        w_t=6, h_t=3, w_c=1.0, h_c=1.0,
        eps_p=0.1, eps_v=0.1, eps_theta=0.04, eps_omega=0.05,
        u_bar_4D=20.0, u_bar_2D=1.5,
        d_bar_4D=0.0, d_bar_2D=0.0,
        final_time=args.grid_final_time,
        cache_dir=cache_dir,
        filter_mode=None,
    )
    grid_times = np.array(combo.times)
    print(f"  Ready in {_time.time() - t0:.1f}s")

    # ---- 3. Select ICs ----
    print('\n' + '=' * 60)
    print(f'Selecting {args.n_ics} ICs (inside grid {args.grid_time_horizon}s '
          f'BRAT, outside DeepReach {args.tMax}s BRAT) ...')
    ics = select_ics(combo, brat_ctrl, grid_times, args.grid_time_horizon,
                     args.n_ics, args.n_candidates, args.seed)

    # ---- 4. Load shared dynamics ----
    dynamics_fn = _load_dynamics_fn()

    # ---- 5. Run comparisons ----
    print('\n' + '=' * 60)
    print(f'Running {len(ics)} comparison(s) ...')
    all_metrics = []
    all_results = []

    for i, ic in enumerate(ics):
        print(f'\n--- IC {i}: {ic} ---')
        combo.reset()
        brat_ctrl.reset()

        result_grid, result_dr = simulate_both(
            ic, combo, brat_ctrl, dynamics_fn, args.dt, args.max_sim_time,
            grid_times)

        # Gradient metrics
        grad_met = compute_gradient_metrics(
            result_grid['gradients'], result_dr['gradients'])

        # Trajectory deviation
        tdev = trajectory_deviation(result_grid['trajectory'],
                                    result_dr['trajectory'])
        grad_met['mean_traj_deviation'] = float(np.mean(tdev))
        grad_met['max_traj_deviation'] = float(np.max(tdev))

        all_metrics.append(grad_met)
        all_results.append((result_grid, result_dr))

        # Print per-IC summary
        print(f"  Cosine similarity:  {grad_met['mean_cosine_sim']:.4f}")
        print(f"  Sign agreement:     vx={grad_met['sign_agreement_rate'][0]*100:.1f}%  "
              f"vy={grad_met['sign_agreement_rate'][1]*100:.1f}%  "
              f"ω={grad_met['sign_agreement_rate'][2]*100:.1f}%")
        print(f"  Trajectory dev:     mean={grad_met['mean_traj_deviation']:.3f}m  "
              f"max={grad_met['max_traj_deviation']:.3f}m")

        # Per-IC plots
        ic_dir = os.path.join(args.output_dir, f'ic_{i}')
        os.makedirs(ic_dir, exist_ok=True)
        plot_trajectory_overlay(result_grid, result_dr, i,
                                os.path.join(ic_dir, 'trajectory.png'),
                                combo=combo, brat_ctrl=brat_ctrl,
                                grid_times=grid_times,
                                grid_horizon=args.grid_time_horizon,
                                dr_horizon=args.tMax)
        plot_states(result_grid, result_dr, i,
                    os.path.join(ic_dir, 'states.png'))
        plot_controls(result_grid, result_dr, i,
                      os.path.join(ic_dir, 'controls.png'))
        plot_gradient_metrics(grad_met, result_grid['times'], i,
                              os.path.join(ic_dir, 'gradient_metrics.png'))

    # ---- 6. Aggregate results ----
    if len(ics) > 1:
        print('\n' + '=' * 60)
        print('Aggregate metrics:')
        cos_sims = [m['mean_cosine_sim'] for m in all_metrics]
        sign_rates = np.array([m['sign_agreement_rate'] for m in all_metrics])
        tdevs = [m['mean_traj_deviation'] for m in all_metrics]
        print(f"  Cosine similarity:  {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}")
        print(f"  Sign agreement vx:  {np.mean(sign_rates[:,0])*100:.1f}% ± {np.std(sign_rates[:,0])*100:.1f}%")
        print(f"  Sign agreement vy:  {np.mean(sign_rates[:,1])*100:.1f}% ± {np.std(sign_rates[:,1])*100:.1f}%")
        print(f"  Sign agreement ω:   {np.mean(sign_rates[:,2])*100:.1f}% ± {np.std(sign_rates[:,2])*100:.1f}%")
        print(f"  Trajectory dev:     {np.mean(tdevs):.3f} ± {np.std(tdevs):.3f}m")

        plot_aggregate(all_metrics,
                       os.path.join(args.output_dir, 'aggregate_metrics.png'))

    # ---- 7. Save JSON results ----
    json_data = {
        '_metadata': {
            'checkpoint_path': args.checkpoint_path,
            'tMax': args.tMax,
            'grid_final_time': args.grid_final_time,
            'grid_time_horizon': args.grid_time_horizon,
            'dt': args.dt,
            'max_sim_time': args.max_sim_time,
            'n_ics': len(ics),
            'seed': args.seed,
        },
        'ics': ics.tolist(),
        'per_ic': [
            {
                'mean_cosine_sim': m['mean_cosine_sim'],
                'sign_agreement_rate_vx': float(m['sign_agreement_rate'][0]),
                'sign_agreement_rate_vy': float(m['sign_agreement_rate'][1]),
                'sign_agreement_rate_omega': float(m['sign_agreement_rate'][2]),
                'mean_traj_deviation': m['mean_traj_deviation'],
                'max_traj_deviation': m['max_traj_deviation'],
            }
            for m in all_metrics
        ],
    }
    if len(ics) > 1:
        json_data['aggregate'] = {
            'mean_cosine_sim': float(np.mean(cos_sims)),
            'std_cosine_sim': float(np.std(cos_sims)),
            'mean_sign_agreement_vx': float(np.mean(sign_rates[:, 0])),
            'mean_sign_agreement_vy': float(np.mean(sign_rates[:, 1])),
            'mean_sign_agreement_omega': float(np.mean(sign_rates[:, 2])),
            'mean_traj_deviation': float(np.mean(tdevs)),
            'std_traj_deviation': float(np.std(tdevs)),
        }

    json_path = os.path.join(args.output_dir, 'gradient_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f'All outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()
