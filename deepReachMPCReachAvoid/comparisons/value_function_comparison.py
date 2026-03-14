#!/usr/bin/env python3
"""
Value Function Approximation Quality: Grid-Based vs DeepReach

Picks ICs that lie INSIDE both the grid-based and DeepReach BRATs at
the same time horizon.  Both controllers should successfully dock from
these ICs.  We compare:
  - Trajectories and control inputs
  - Cross-evaluated value functions along each trajectory
  - Value function error |V_DR - V_grid|

Usage:
    python value_function_comparison.py \
        --checkpoint_path ../runs/Docking6D_RA_10sec_NEW/training/checkpoints/model_final.pth \
        --tMax 10 --comparison_horizon 10 --n_ics 5 --device cuda
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

# ---- path setup ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPREACH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(DEEPREACH_DIR)
sys.path.insert(0, DEEPREACH_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'gridBased6DImplementation'))

from utils.controllers import BRATController
from comparisons.volume_comparison import (
    physical_time_to_index, grid_value_6D, deepreach_value_6D, MC_BOUNDS,
)

_combo_module = None
def _get_combo_module():
    global _combo_module
    if _combo_module is None:
        import importlib
        _combo_module = importlib.import_module('ComboControl')
    return _combo_module


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
def select_ics(combo, brat_ctrl, grid_times, comparison_horizon, n_ics,
               n_candidates=500_000, seed=42):
    """Sample ICs inside both grid-based and DeepReach BRATs at comparison_horizon.

    Returns (n_ics, 6) array.
    """
    rng = np.random.RandomState(seed)
    states = rng.uniform(MC_BOUNDS[:, 0], MC_BOUNDS[:, 1],
                         size=(n_candidates, 6))

    t_idx = physical_time_to_index(comparison_horizon, grid_times)
    gv = grid_value_6D(combo, states, t_idx)
    dv = deepreach_value_6D(brat_ctrl, states, brat_ctrl.tMax)

    # Inside BOTH BRATs
    mask = (gv <= 0) & (dv <= 0)
    candidates = states[mask]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No ICs found inside both BRATs at t={comparison_horizon}s. "
            f"Try increasing n_candidates.")

    # Prefer ICs near the BRAT boundary (combined value closest to zero)
    combined_val = np.maximum(gv[mask], dv[mask])
    order = np.argsort(-combined_val)  # least negative first (closest to zero level set)
    selected = candidates[order[:n_ics]]
    print(f"  IC selection: {mask.sum()} candidates found, selected {len(selected)}")
    print(f"  Selected value range: [{combined_val[order[:n_ics]].min():.4f}, {combined_val[order[:n_ics]].max():.4f}]")
    return selected


# ======================================================================
#  Unified simulation with value cross-evaluation
# ======================================================================
def simulate_both(ic, combo, brat_ctrl, dynamics_fn, dt, max_sim_time,
                  grid_times, comparison_horizon):
    """Simulate both controllers and cross-evaluate value functions.

    Returns (result_grid, result_dr) dicts with keys:
        trajectory, controls, times, v_grid, v_dr (both value functions
        evaluated along this controller's trajectory)
    """
    num_steps = int(max_sim_time / dt)

    def _grid_value_at(s, sim_time, horizon):
        """Query grid value function at a given remaining time."""
        remaining = max(0.01, horizon - sim_time)
        tidx = physical_time_to_index(remaining, grid_times)
        s4d, s2d = s[:4], s[4:]
        v4 = combo.value_at_state_4D(s4d, tidx)
        v2 = combo.value_at_state_2D(s2d, tidx)
        return max(v4, v2)

    def _dr_value_at(s, sim_time, horizon):
        """Query DeepReach value function at remaining time."""
        remaining = max(0.01, horizon - sim_time)
        return brat_ctrl.get_value(s, remaining)

    def _run_one(controller_fn, horizon):
        states = np.zeros((num_steps + 1, 6))
        controls = np.zeros((num_steps, 3))
        v_grid = np.zeros(num_steps)
        v_dr = np.zeros(num_steps)
        times = np.zeros(num_steps)

        states[0] = ic.copy()

        for k in range(num_steps):
            t = k * dt
            s = states[k]

            u = controller_fn(s, t)
            controls[k] = u
            times[k] = t

            # Cross-evaluate both value functions at this state
            v_grid[k] = _grid_value_at(s, t, horizon)
            v_dr[k] = _dr_value_at(s, t, horizon)

            s_next = s + dt * dynamics_fn(s, u)
            s_next[4] = (s_next[4] + np.pi) % (2 * np.pi) - np.pi
            states[k + 1] = s_next

        return {
            'trajectory': states,
            'controls': controls,
            'times': times,
            'v_grid': v_grid,
            'v_dr': v_dr,
        }

    # Grid controller
    combo.reset()
    result_grid = _run_one(combo.u_fn, comparison_horizon)
    result_grid['controller_type'] = 'grid'

    # DeepReach controller
    brat_ctrl.reset()
    result_dr = _run_one(brat_ctrl.u_fn, comparison_horizon)
    result_dr['controller_type'] = 'deepreach'

    return result_grid, result_dr


# ======================================================================
#  Metrics
# ======================================================================
def compute_metrics(result_grid, result_dr):
    """Compute comparison metrics between the two controllers."""
    tg = result_grid['trajectory']
    td = result_dr['trajectory']
    min_len = min(tg.shape[0], td.shape[0])

    # Position deviation
    pos_dev = np.linalg.norm(tg[:min_len, :2] - td[:min_len, :2], axis=1)

    # Value function error along grid trajectory
    v_err_grid_traj = np.abs(result_grid['v_dr'] - result_grid['v_grid'])

    # Value function error along DR trajectory
    v_err_dr_traj = np.abs(result_dr['v_dr'] - result_dr['v_grid'])

    # Control effort
    dt = result_grid['times'][1] - result_grid['times'][0] if len(result_grid['times']) > 1 else 0.1
    effort_grid = float(np.sum(np.linalg.norm(result_grid['controls'], axis=1)) * dt)
    effort_dr = float(np.sum(np.linalg.norm(result_dr['controls'], axis=1)) * dt)

    return {
        'mean_pos_deviation': float(np.mean(pos_dev)),
        'max_pos_deviation': float(np.max(pos_dev)),
        'mean_value_error_grid_traj': float(np.mean(v_err_grid_traj)),
        'mean_value_error_dr_traj': float(np.mean(v_err_dr_traj)),
        'max_value_error_grid_traj': float(np.max(v_err_grid_traj)),
        'max_value_error_dr_traj': float(np.max(v_err_dr_traj)),
        'control_effort_grid': effort_grid,
        'control_effort_dr': effort_dr,
    }


# ======================================================================
#  Plotting
# ======================================================================
def _draw_docking_scene(ax):
    from matplotlib.patches import Rectangle
    body = Rectangle((-3, 0), 6, 3, color='gray', alpha=0.3, label='Target body')
    ax.add_patch(body)
    post = Rectangle((-0.6, -0.2), 1.2, 0.2, color='gray', alpha=0.5, label='Docking post')
    ax.add_patch(post)
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
    ax.contour(PX, PY, v_grid, levels=[0], colors='purple', linewidths=1.5,
               linestyles='solid', label=f'Grid BRAT (t={grid_horizon}s)')

    # DeepReach BRT
    v_dr = deepreach_value_6D(brat_ctrl, flat, dr_horizon).reshape(PX.shape)
    ax.contour(PX, PY, v_dr, levels=[0], colors='green', linewidths=1.5,
               linestyles='dashed', label=f'DeepReach BRAT (t={dr_horizon}s)')


def plot_trajectory_overlay(result_grid, result_dr, ic_idx, save_path,
                            combo=None, brat_ctrl=None, grid_times=None,
                            grid_horizon=None, dr_horizon=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_docking_scene(ax)

    # Draw BRT zero level set contours if controllers are provided
    if combo is not None and brat_ctrl is not None:
        _add_brt_contours(ax, combo, brat_ctrl, grid_times,
                          grid_horizon, dr_horizon)

    tg = result_grid['trajectory']
    td = result_dr['trajectory']
    ax.plot(tg[:, 0], tg[:, 1], 'b-', lw=1.5, label='Grid')
    ax.plot(td[:, 0], td[:, 1], 'r--', lw=1.5, label='DeepReach')
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
    labels = ['px (m)', 'py (m)', 'vx (m/s)', 'vy (m/s)', 'θ (rad)', 'ω (rad/s)']
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    tg = result_grid['trajectory']
    td = result_dr['trajectory']
    dt = result_grid['times'][1] - result_grid['times'][0] if len(result_grid['times']) > 1 else 0.1
    times_g = np.arange(tg.shape[0]) * dt
    times_d = np.arange(td.shape[0]) * dt

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


def plot_value_comparison(result, traj_label, ic_idx, save_path):
    """Value functions along one trajectory (grid vs DR values)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    times = result['times']

    # Panel 1: Both value functions
    ax = axes[0]
    ax.plot(times, result['v_grid'], 'b-', lw=1.2, label='V_grid')
    ax.plot(times, result['v_dr'], 'r--', lw=1.2, label='V_DeepReach')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_ylabel('Value')
    ax.set_title(f'Value Functions Along {traj_label} Trajectory — IC {ic_idx}')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Absolute error
    ax = axes[1]
    err = np.abs(result['v_dr'] - result['v_grid'])
    ax.plot(times, err, 'k-', lw=1.0)
    ax.set_ylabel('|V_DR - V_grid|')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Value Error  |  Mean={np.mean(err):.4f}  Max={np.max(err):.4f}')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_aggregate(all_metrics, save_path):
    """Aggregated metrics across multiple ICs."""
    n = len(all_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Position deviation
    ax = axes[0]
    devs = [m['mean_pos_deviation'] for m in all_metrics]
    ax.bar(range(n), devs, color='steelblue')
    ax.axhline(np.mean(devs), color='red', ls='--',
               label=f'Mean={np.mean(devs):.3f}m')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Mean Position Deviation (m)')
    ax.set_title('Trajectory Deviation')
    ax.legend()

    # Value error along grid trajectory
    ax = axes[1]
    errs_g = [m['mean_value_error_grid_traj'] for m in all_metrics]
    errs_d = [m['mean_value_error_dr_traj'] for m in all_metrics]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, errs_g, w, label='Along grid traj', color='tab:blue')
    ax.bar(x + w/2, errs_d, w, label='Along DR traj', color='tab:red')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Mean |V_DR - V_grid|')
    ax.set_title('Value Function Error')
    ax.legend()

    # Control effort comparison
    ax = axes[2]
    eg = [m['control_effort_grid'] for m in all_metrics]
    ed = [m['control_effort_dr'] for m in all_metrics]
    ax.bar(x - w/2, eg, w, label='Grid', color='tab:blue')
    ax.bar(x + w/2, ed, w, label='DeepReach', color='tab:red')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Control Effort')
    ax.set_title('Control Effort')
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
        description='Value function comparison: grid-based vs DeepReach',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--tMax', type=float, default=10.0,
                        help='DeepReach tMax for value queries')
    parser.add_argument('--grid_final_time', type=float, default=-30.0,
                        help='Grid solver backward time horizon (negative)')
    parser.add_argument('--comparison_horizon', type=float, default=10.0,
                        help='Time horizon for IC selection and value comparison')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--max_sim_time', type=float, default=30.0)
    parser.add_argument('--n_ics', type=int, default=1)
    parser.add_argument('--n_candidates', type=int, default=500_000)
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/value_comparison')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--grid_cache_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.grid_cache_dir or os.path.join(args.output_dir, 'grid_cache')

    # ---- 1. Load controllers ----
    print('=' * 60)
    print('Loading DeepReach model ...')
    t0 = _time.time()
    brat_ctrl = BRATController(
        checkpoint_path=args.checkpoint_path,
        tMax=args.tMax,
        device=args.device,
    )
    print(f"  Loaded in {_time.time() - t0:.1f}s")

    print('\n' + '=' * 60)
    print('Loading grid-based controller (filter_mode=None) ...')
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

    # ---- 2. Select ICs ----
    print('\n' + '=' * 60)
    print(f'Selecting {args.n_ics} ICs (inside both BRATs at '
          f't={args.comparison_horizon}s) ...')
    ics = select_ics(combo, brat_ctrl, grid_times, args.comparison_horizon,
                     args.n_ics, args.n_candidates, args.seed)

    # ---- 3. Load shared dynamics ----
    dynamics_fn = _load_dynamics_fn()

    # ---- 4. Run comparisons ----
    print('\n' + '=' * 60)
    print(f'Running {len(ics)} comparison(s) ...')
    all_metrics = []

    for i, ic in enumerate(ics):
        print(f'\n--- IC {i}: {ic} ---')
        combo.reset()
        brat_ctrl.reset()

        result_grid, result_dr = simulate_both(
            ic, combo, brat_ctrl, dynamics_fn, args.dt, args.max_sim_time,
            grid_times, args.comparison_horizon)

        metrics = compute_metrics(result_grid, result_dr)
        all_metrics.append(metrics)

        print(f"  Pos deviation:     mean={metrics['mean_pos_deviation']:.3f}m  "
              f"max={metrics['max_pos_deviation']:.3f}m")
        print(f"  Value error (grid traj): mean={metrics['mean_value_error_grid_traj']:.4f}  "
              f"max={metrics['max_value_error_grid_traj']:.4f}")
        print(f"  Value error (DR traj):   mean={metrics['mean_value_error_dr_traj']:.4f}  "
              f"max={metrics['max_value_error_dr_traj']:.4f}")
        print(f"  Control effort:    grid={metrics['control_effort_grid']:.2f}  "
              f"DR={metrics['control_effort_dr']:.2f}")

        # Per-IC plots
        ic_dir = os.path.join(args.output_dir, f'ic_{i}')
        os.makedirs(ic_dir, exist_ok=True)
        plot_trajectory_overlay(result_grid, result_dr, i,
                                os.path.join(ic_dir, 'trajectory.png'),
                                combo=combo, brat_ctrl=brat_ctrl,
                                grid_times=grid_times,
                                grid_horizon=args.comparison_horizon,
                                dr_horizon=args.tMax)
        plot_states(result_grid, result_dr, i,
                    os.path.join(ic_dir, 'states.png'))
        plot_controls(result_grid, result_dr, i,
                      os.path.join(ic_dir, 'controls.png'))
        plot_value_comparison(result_grid, 'Grid', i,
                              os.path.join(ic_dir, 'values_along_grid_traj.png'))
        plot_value_comparison(result_dr, 'DeepReach', i,
                              os.path.join(ic_dir, 'values_along_dr_traj.png'))

    # ---- 5. Aggregate ----
    if len(ics) > 1:
        print('\n' + '=' * 60)
        print('Aggregate metrics:')
        devs = [m['mean_pos_deviation'] for m in all_metrics]
        errs_g = [m['mean_value_error_grid_traj'] for m in all_metrics]
        errs_d = [m['mean_value_error_dr_traj'] for m in all_metrics]
        print(f"  Pos deviation:      {np.mean(devs):.3f} ± {np.std(devs):.3f}m")
        print(f"  Value error (grid): {np.mean(errs_g):.4f} ± {np.std(errs_g):.4f}")
        print(f"  Value error (DR):   {np.mean(errs_d):.4f} ± {np.std(errs_d):.4f}")

        plot_aggregate(all_metrics,
                       os.path.join(args.output_dir, 'aggregate_metrics.png'))

    # ---- 6. Save JSON ----
    json_data = {
        '_metadata': {
            'checkpoint_path': args.checkpoint_path,
            'tMax': args.tMax,
            'grid_final_time': args.grid_final_time,
            'comparison_horizon': args.comparison_horizon,
            'dt': args.dt,
            'max_sim_time': args.max_sim_time,
            'n_ics': len(ics),
            'seed': args.seed,
        },
        'ics': ics.tolist(),
        'per_ic': all_metrics,
    }
    if len(ics) > 1:
        json_data['aggregate'] = {
            'mean_pos_deviation': float(np.mean(devs)),
            'std_pos_deviation': float(np.std(devs)),
            'mean_value_error_grid_traj': float(np.mean(errs_g)),
            'mean_value_error_dr_traj': float(np.mean(errs_d)),
        }

    json_path = os.path.join(args.output_dir, 'value_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f'All outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()
