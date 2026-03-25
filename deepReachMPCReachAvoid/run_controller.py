#!/usr/bin/env python3
"""
Unified controller runner for Docking6D.

Subcommands:
    single   -- Run one controller from a specified initial condition.
    compare  -- Run N rollouts per controller from random ICs and compare metrics.

Usage:
    python run_controller.py single  --controller brat   [options]
    python run_controller.py compare --controllers brat mpc mpc_terminal [options]
"""

import argparse
import inspect
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Controller classes (via convenience re-exports)
from utils.controllers import (
    BRATController, MPCController, MPCTerminalController,
    SafetyFilter, GridBasedController, RLController,
)

# Generic (multi-controller) visualisation
from utils.controllers.controller_animation import (
    animate_trajectories,
    plot_trajectories_static,
    plot_simulation_data_multi,
    CONTROLLER_LABELS,
    CONTROLLER_COLORS,
)

# BRAT-specific visualisation (value-function heatmap animation)
from utils.controllers.brat_animation import (
    create_brat_animation,
    create_mpc_terminal_animation,
    plot_trajectory_static as brat_plot_trajectory_static,
    plot_simulation_data   as brat_plot_simulation_data,
)

from dynamics import dynamics as dynamics_module



def _build_safety_filter(args):
    """Create a SafetyFilter from CLI args (shared across controllers)."""
    return SafetyFilter(
        mode=args.safety_filter_mode,
        checkpoint_path=args.safety_checkpoint_path,
        tMax=None,
        margin=args.safety_margin_phase1,
        gamma=args.safety_filter_gamma,
        device=args.device,
    )


def build_controller(name, args):
    """Instantiate a controller by name string."""
    # Safety filter (no-op when mode=0)
    sf = _build_safety_filter(args) if name in ('brat', 'mpc', 'mpc_terminal', 'rl') else None

    if name == 'brat':
        return BRATController(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
            safety_filter=sf,
            safety_margin_phase1=args.safety_margin_phase1,
            safety_margin_phase2=args.safety_margin_phase2,
            debug_phase2=args.debug_phase2,
            gradient_fallback=args.gradient_fallback,
            grad_threshold=args.grad_threshold,
            avoid_proximity_margin=args.avoid_proximity_margin,
        )
    elif name == 'mpc':
        return MPCController(
            checkpoint_path=args.checkpoint_path,
            planning_horizon_sec=args.planning_horizon,
            mpc_dt=args.mpc_dt,
            dt=args.dt,
            device=args.device,
            safety_filter=sf,
            gradient_lr=args.gradient_lr,
            gradient_iters=args.mpc_gradient_iters if args.mpc_gradient_iters is not None else args.gradient_iters,
            num_restarts=args.mpc_num_restarts if args.mpc_num_restarts is not None else args.num_restarts,
            goal_weight=args.goal_weight,
        )
    elif name == 'mpc_terminal':
        return MPCTerminalController(
            checkpoint_path=args.checkpoint_path,
            effective_horizon_sec=args.effective_horizon,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
            effort_weight=args.effort_weight,
            exploration_factor=args.exploration_factor,
            exploration_patience=args.exploration_patience,
            escape_thresh=args.escape_thresh,
            safety_filter=sf,
            safety_margin_phase1=args.safety_margin_phase1,
            safety_margin_phase2=args.safety_margin_phase2,
            debug_phase2=args.debug_phase2,
            gradient_lr=args.gradient_lr,
            gradient_iters=args.mpc_terminal_gradient_iters if args.mpc_terminal_gradient_iters is not None else args.gradient_iters,
            num_restarts=args.mpc_terminal_num_restarts if args.mpc_terminal_num_restarts is not None else args.num_restarts,
        )
    elif name == 'grid_based':
        fm = getattr(args, 'grid_filter_mode', 0)
        return GridBasedController(
            dt=args.dt,
            max_sim_time=args.max_sim_time,
            cache_dir=getattr(args, 'grid_cache_dir', None),
            filter_mode=None if fm == 0 else fm,
        )
    elif name == 'rl':
        return RLController(
            rl_checkpoint_path=args.rl_checkpoint_path,
            dt=args.dt,
            device=args.device,
            safety_filter=sf,
            architecture=args.rl_architecture,
            activation=args.rl_activation,
        )
    else:
        raise ValueError(f"Unknown controller: {name}")

SAMPLING_STATE_RANGE = np.array([
    [-13.0, 13.0],    # px  (m)
    [-13.0, 13.0],    # py  (m)
    [ -0.75,  0.75],  # vx  (m/s)  -- braking distance 2.81m keeps chaser in training domain
    [ -0.75,  0.75],  # vy  (m/s)
    [-np.pi, np.pi],  # theta (rad)
    [ -0.50,  0.50],  # omega (rad/s) -- braking from 0.50 takes 10s (2/3 of tMax=15)
])


def compute_feasibility_bounds(dynamics, tMax, budget_fraction=2.0 / 3.0):
    """Derive max feasible initial velocities
    """
    a_max = dynamics.u_bar / dynamics.mc
    alpha_max = dynamics.u_theta_bar / float(dynamics.jc)
    T = tMax * budget_fraction
    v_max = dynamics.eps_v + a_max * T
    omega_max = dynamics.eps_omega + alpha_max * T
    return v_max, omega_max


def sample_initial_conditions(dynamics, n, device='cuda', seed=42,
                              value_filter_fn=None, avoid_filter_fn=None):
    """
    Sample *n* valid initial conditions uniformly from a feasible
    sub-region of the state space.

    The sampling bounds are the intersection of ``SAMPLING_STATE_RANGE``
    (hard-coded above) and the dynamics' ``state_range_``, so we never
    exceed the model's training domain.

    Filtering pipeline (applied in order):
      1. Within sampling bounds.
      2. Not inside the failure set (avoid_fn > 0) and not already docked
         (reach_fn > 0).
      3. Outside the learned avoid BRT (avoid_filter_fn(states) > 0) -- not
         doomed to collide within the time horizon.
      4. (optional) Inside the learned BRAT (value_filter_fn(states) <= 0).

    Args:
        dynamics: Dynamics instance with avoid_fn() and reach_fn().
        n: Number of ICs to sample.
        device: Torch device for dynamics queries.
        seed: Random seed.
        value_filter_fn: Optional callable  (N,6) np.array -> (N,) np.array
            returning V(x, tMax) for each state.  States with V <= 0 are
            kept (inside the BRAT).  ``None`` disables this filter.
        avoid_filter_fn: Optional callable  (N,6) np.array -> (N,) np.array
            returning V_avoid(x, tMax_avoid) for each state.  States with
            V_avoid > 0 are kept (outside the avoid BRT).  ``None`` disables
            this filter.
    """
    rng = np.random.RandomState(seed)

    # Intersect SAMPLING_STATE_RANGE with the dynamics' state_range_
    dyn_lo = dynamics.state_range_[:, 0].cpu().numpy().astype(np.float64)
    dyn_hi = dynamics.state_range_[:, 1].cpu().numpy().astype(np.float64)
    state_lo = np.maximum(dyn_lo, SAMPLING_STATE_RANGE[:, 0])
    state_hi = np.minimum(dyn_hi, SAMPLING_STATE_RANGE[:, 1])

    samples = []
    attempts = 0
    max_attempts = n * 1000
    n_rejected_geom = 0
    n_rejected_avoid_brt = 0
    n_rejected_brat = 0

    while len(samples) < n and attempts < max_attempts:
        batch_size = min(n * 10, 5000)
        batch = rng.uniform(state_lo, state_hi, size=(batch_size, 6))
        batch_t = torch.tensor(batch, dtype=torch.float32, device=device)

        avoid_vals = dynamics.avoid_fn(batch_t).cpu().numpy()
        reach_vals = dynamics.reach_fn(batch_t).cpu().numpy()

        geom_valid = (avoid_vals > 0) & (reach_vals > 0)
        n_rejected_geom += int((~geom_valid).sum())

        candidates = batch[geom_valid]

        # Avoid-BRT filter: reject states doomed to collide even under
        # optimal avoidance control (V_avoid <= 0).
        if avoid_filter_fn is not None and len(candidates) > 0:
            V_avoid = avoid_filter_fn(candidates)
            avoid_brt_valid = V_avoid > 0
            n_rejected_avoid_brt += int((~avoid_brt_valid).sum())
            candidates = candidates[avoid_brt_valid]

        # Reach-avoid BRAT filter (optional, --sampling_method brat)
        if value_filter_fn is not None and len(candidates) > 0:
            values = value_filter_fn(candidates)
            brat_valid = values <= 0
            n_rejected_brat += int((~brat_valid).sum())
            candidates = candidates[brat_valid]

        for s in candidates:
            if len(samples) >= n:
                break
            samples.append(s)
        attempts += batch_size

    if len(samples) < n:
        print(f"WARNING: only sampled {len(samples)}/{n} valid ICs "
              f"after {attempts} attempts.")

    has_filters = (avoid_filter_fn is not None or value_filter_fn is not None)
    if has_filters:
        print(f"  IC sampling stats:  checked={attempts}  "
              f"rejected_geom={n_rejected_geom}  "
              f"rejected_avoid_brt={n_rejected_avoid_brt}  "
              f"rejected_brat={n_rejected_brat}  "
              f"accepted={len(samples)}")

    return np.array(samples[:n])


def compute_metrics(all_results):
    """Aggregate metrics from a list of result dicts.

    Docking  = reached goal without collision (docked & ~collision)
    Failure  = collision occurred
    Timeout  = never reached goal and no collision
    """
    n = len(all_results)
    if n == 0:
        return {}
    dockings   = sum(1 for r in all_results if r['success'])   # docked & not collided
    failures   = sum(1 for r in all_results if r['collision'])
    timeouts   = n - dockings - failures
    times      = [r['wall_time'] for r in all_results]

    # Control effort only for successful (docking) trajectories
    docking_efforts = [r['control_effort'] for r in all_results if r['success']]

    total_clipped = sum(r.get('n_clipped_steps', 0) for r in all_results)
    n_with_clipping = sum(1 for r in all_results
                          if r.get('n_clipped_steps', 0) > 0)

    return {
        'n': n,
        'docking_rate':         dockings / n,
        'failure_rate':         failures / n,
        'timeout_rate':         timeouts / n,
        'mean_control_effort':  float(np.mean(docking_efforts)) if docking_efforts else 0.0,
        'std_control_effort':   float(np.std(docking_efforts)) if docking_efforts else 0.0,
        'n_docking_effort':     len(docking_efforts),
        'mean_wall_time':       float(np.mean(times)),
        'std_wall_time':        float(np.std(times)),
        'total_clipped_steps':  total_clipped,
        'n_rollouts_with_clipping': n_with_clipping,
    }


def compute_docking_optimality(all_results, controller_names):
    """Paired docking-time comparison across controllers.

    Only considers the *common-success set*: ICs where every controller
    successfully docked.  Returns a dict with per-controller metrics and
    head-to-head win-rate matrix.

    Parameters
    ----------
    all_results : dict[str, list[dict]]
        {display_name: [result_dict_per_IC, ...]}.
    controller_names : list[str]
        Display names in the order they should appear.

    Returns
    -------
    dict  with keys:
        'common_n'          – size of the common-success set
        'total_n'           – total ICs
        'per_controller'    – {name: {median_dock_time, mean_dock_time,
                                      geo_mean_ratio, dock_times}}
        'baseline'          – name of the baseline controller (first in list)
        'head_to_head'      – {nameA: {nameB: win_fraction, ...}, ...}
    """
    names = controller_names
    n_ics = len(next(iter(all_results.values())))

    # Identify common-success set (ICs where ALL controllers docked)
    common_mask = np.ones(n_ics, dtype=bool)
    for name in names:
        for i, r in enumerate(all_results[name]):
            if not r['success']:
                common_mask[i] = False
    common_idxs = np.where(common_mask)[0]

    result = {
        'common_n': int(len(common_idxs)),
        'total_n': n_ics,
        'per_controller': {},
        'baseline': names[0],
        'head_to_head': {},
    }

    if len(common_idxs) == 0:
        return result

    # Gather docking times on the common set
    dock_times = {}
    for name in names:
        dock_times[name] = np.array([
            all_results[name][i]['times'][-1] for i in common_idxs
        ])

    # Baseline for time-ratio computation (first controller)
    baseline = names[0]
    baseline_times = dock_times[baseline]

    for name in names:
        t = dock_times[name]
        # Per-IC ratio relative to baseline
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(baseline_times > 0, t / baseline_times, 1.0)
        geo_mean_ratio = float(np.exp(np.mean(np.log(ratios))))

        result['per_controller'][name] = {
            'median_dock_time': float(np.median(t)),
            'mean_dock_time': float(np.mean(t)),
            'std_dock_time': float(np.std(t)),
            'geo_mean_ratio': geo_mean_ratio,
        }

    # Head-to-head win rates
    for a in names:
        result['head_to_head'][a] = {}
        for b in names:
            if a == b:
                result['head_to_head'][a][b] = 0.5
            else:
                wins = int(np.sum(dock_times[a] < dock_times[b]))
                result['head_to_head'][a][b] = round(
                    wins / len(common_idxs), 4)

    return result


def print_optimality_table(optimality):
    """Print a formatted docking-time optimality table."""
    cn = optimality['common_n']
    tn = optimality['total_n']
    baseline = optimality['baseline']

    header = (f"{'Controller':<22} {'Median(s)':>9} {'Mean(s)':>9} "
              f"{'Std(s)':>9} {'Ratio':>7}")
    sep = '-' * len(header)
    print('\n' + sep)
    print(f'DOCKING-TIME OPTIMALITY  (common-success set: '
          f'{cn}/{tn} ICs, baseline: {baseline})')
    print(sep)
    print(header)
    print(sep)
    for name, m in optimality['per_controller'].items():
        print(f"{name:<22} {m['median_dock_time']:>9.2f} "
              f"{m['mean_dock_time']:>9.2f} {m['std_dock_time']:>9.2f} "
              f"{m['geo_mean_ratio']:>7.3f}")
    print(sep)

    # Head-to-head
    names = list(optimality['per_controller'].keys())
    if len(names) > 1:
        h2h = optimality['head_to_head']
        col_w = max(len(n) for n in names) + 2
        hdr = ' ' * col_w + ''.join(f'{n:>{col_w}}' for n in names)
        print('\nHead-to-head win rate (row beats column):')
        print(hdr)
        for a in names:
            row = f'{a:<{col_w}}'
            for b in names:
                if a == b:
                    row += f'{"--":>{col_w}}'
                else:
                    row += f'{h2h[a][b]*100:>{col_w-1}.1f}%'
            print(row)
    print(sep + '\n')


def _to_jsonable(obj):
    """Convert numpy / torch-adjacent containers into JSON-safe objects."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def print_comparison_table(metrics_by_controller):
    """Print a formatted comparison table to stdout."""
    header = (f"{'Controller':<22} {'Dock%':>7} {'Fail%':>7} {'Time%':>7} "
              f"{'Effort (dock)':>18} {'Time (s)':>14} "
              f"{'Clip(tot)':>10} {'Clip(runs)':>11}")
    sep = '-' * len(header)
    print('\n' + sep)
    print('CONTROLLER COMPARISON')
    print(sep)
    print(header)
    print(sep)
    for name, m in metrics_by_controller.items():
        n_dock = m.get('n_docking_effort', 0)
        if n_dock > 0:
            effort_str = f"{m['mean_control_effort']:.1f} +/- {m['std_control_effort']:.1f} ({n_dock})"
        else:
            effort_str = "N/A (0)"
        time_str   = f"{m['mean_wall_time']:.2f} +/- {m['std_wall_time']:.2f}"
        clip_tot   = m.get('total_clipped_steps', 0)
        clip_runs  = m.get('n_rollouts_with_clipping', 0)
        print(f"{name:<22} {m['docking_rate']*100:>6.1f}% "
              f"{m['failure_rate']*100:>6.1f}% "
              f"{m['timeout_rate']*100:>6.1f}% "
              f"{effort_str:>18} {time_str:>14} "
              f"{clip_tot:>10} {clip_runs:>11}")
    print(sep + '\n')


def plot_metrics_bar(metrics_by_controller, save_path=None, optimality=None):
    """Grouped bar chart of comparison metrics."""
    names = list(metrics_by_controller.keys())
    n_ctrl = len(names)

    label_to_type = {v: k for k, v in CONTROLLER_LABELS.items()}
    colors = [CONTROLLER_COLORS.get(label_to_type.get(n, 'brat'), '#1f77b4')
              for n in names]

    n_panels = 4 if (optimality and optimality['common_n'] > 0) else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    x = np.arange(n_ctrl)

    # Rates
    w = 0.25
    axes[0].bar(x - w,
                [metrics_by_controller[n]['docking_rate'] * 100 for n in names],
                w, label='Docking %', color='#66c2a5')
    axes[0].bar(x,
                [metrics_by_controller[n]['failure_rate'] * 100 for n in names],
                w, label='Failure %', color='#fc8d62')
    axes[0].bar(x + w,
                [metrics_by_controller[n]['timeout_rate'] * 100 for n in names],
                w, label='Timeout %', color='#8da0cb')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title('Rates')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis='y', alpha=0.3)

    # Control effort (docking trajectories only)
    means = [metrics_by_controller[n]['mean_control_effort'] for n in names]
    stds  = [metrics_by_controller[n]['std_control_effort']  for n in names]
    bars = axes[1].bar(x, means, 0.5, yerr=stds, color=colors, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    axes[1].set_ylabel('Control Effort')
    axes[1].set_title('Mean Control Effort (Docking Only)')
    axes[1].grid(axis='y', alpha=0.3)
    # Annotate bars with count of docking runs
    for i, n in enumerate(names):
        n_dock = metrics_by_controller[n].get('n_docking_effort', 0)
        n_total = metrics_by_controller[n]['n']
        if n_dock > 0:
            axes[1].text(i, means[i] + stds[i] + 0.02 * max(means),
                         f'n={n_dock}/{n_total}', ha='center', va='bottom',
                         fontsize=7)

    # Runtime
    means = [metrics_by_controller[n]['mean_wall_time'] for n in names]
    stds  = [metrics_by_controller[n]['std_wall_time']  for n in names]
    axes[2].bar(x, means, 0.5, yerr=stds, color=colors, capsize=5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    axes[2].set_ylabel('Wall Time (s)')
    axes[2].set_title('Mean Computation Wall Time per Rollout')
    axes[2].grid(axis='y', alpha=0.3)

    # Docking-time optimality (common-success set)
    if n_panels == 4:
        pc = optimality['per_controller']
        opt_names = list(pc.keys())
        medians = [pc[n]['median_dock_time'] for n in opt_names]
        means   = [pc[n]['mean_dock_time']   for n in opt_names]
        stds    = [pc[n]['std_dock_time']     for n in opt_names]
        xo = np.arange(len(opt_names))
        w = 0.3
        axes[3].bar(xo - w/2, medians, w, label='Median', color='#66c2a5')
        axes[3].bar(xo + w/2, means, w, yerr=stds, label='Mean ± std',
                    color=colors[:len(opt_names)], capsize=5)
        axes[3].set_xticks(xo)
        axes[3].set_xticklabels(opt_names, fontsize=8, rotation=25, ha='right')
        axes[3].set_ylabel('Docking Time (s)')
        cn = optimality['common_n']
        tn = optimality['total_n']
        axes[3].set_title(f'Docking Time (common set: {cn}/{tn} ICs)')
        axes[3].legend(fontsize=8)
        axes[3].grid(axis='y', alpha=0.3)
        # Annotate with geo-mean ratio
        for i, n in enumerate(opt_names):
            r = pc[n]['geo_mean_ratio']
            axes[3].text(i, means[i] + stds[i] + 0.02 * max(means),
                         f'ratio={r:.3f}', ha='center', va='bottom',
                         fontsize=7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    return fig


def plot_docking_time_histogram(all_results_by_display, controller_names,
                                save_path=None,
                                grid_based_label='Grid-Based HJ'):
    """Two-panel docking-time histogram with grid-based as baseline.

    Panel 1: Histogram of grid-based absolute docking times.
    Panel 2: Overlaid histograms of docking-time ratios (other / grid)
             with a parity line at 1.0.

    Only ICs in the common success set (all controllers docked) are included.
    """
    names = controller_names
    n_ics = len(next(iter(all_results_by_display.values())))

    # Common success set
    common_mask = np.ones(n_ics, dtype=bool)
    for name in names:
        for i, r in enumerate(all_results_by_display[name]):
            if not r['success']:
                common_mask[i] = False
    common_idxs = np.where(common_mask)[0]

    if len(common_idxs) == 0 or grid_based_label not in all_results_by_display:
        print("Skipping docking-time histogram: no common-success ICs "
              "or grid-based controller not present.")
        return None

    # Gather docking times on the common set
    dock_times = {}
    for name in names:
        dock_times[name] = np.array([
            all_results_by_display[name][i]['times'][-1]
            for i in common_idxs
        ])

    grid_times = dock_times[grid_based_label]
    other_names = [n for n in names if n != grid_based_label]

    # Reverse-lookup for colours
    label_to_type = {v: k for k, v in CONTROLLER_LABELS.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1 — grid-based absolute docking times
    color_grid = CONTROLLER_COLORS.get(
        label_to_type.get(grid_based_label, 'grid_based'), '#d62728')
    ax1.hist(grid_times, bins='auto', color=color_grid,
             edgecolor='black', alpha=0.75)
    ax1.set_xlabel('Docking Time (s)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{grid_based_label} Docking Times\n'
                  f'(common success set: {len(common_idxs)} ICs)')
    median_t = np.median(grid_times)
    ax1.axvline(median_t, color='black', linestyle='--',
                label=f'Median = {median_t:.1f}s')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2 — time ratios (other / grid)
    if other_names:
        for name in other_names:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(grid_times > 0,
                                  dock_times[name] / grid_times, 1.0)
            color = CONTROLLER_COLORS.get(
                label_to_type.get(name, 'brat'), '#1f77b4')
            ax2.hist(ratios, bins='auto', color=color, edgecolor='black',
                     alpha=0.5, label=name)
        ax2.axvline(1.0, color='black', linestyle='-', linewidth=2,
                    label='Parity (ratio = 1)')
        ax2.set_xlabel(f'Docking Time Ratio (controller / {grid_based_label})')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Docking Time Ratios vs {grid_based_label}\n'
                      f'(common success set: {len(common_idxs)} ICs)')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Docking-time histogram saved to {save_path}")
    return fig


def load_dynamics(checkpoint_path):
    """Load a dynamics instance from the experiment's saved options."""
    experiment_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(checkpoint_path)))
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')
    with open(opt_path, 'rb') as f:
        orig_opt = pickle.load(f)

    dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
    sig = inspect.signature(dynamics_class)
    kwargs = {}
    for pn in sig.parameters.keys():
        if pn != 'self' and hasattr(orig_opt, pn):
            kwargs[pn] = getattr(orig_opt, pn)
    dynamics = dynamics_class(**kwargs)
    dynamics.set_model(orig_opt.deepReach_model)

    # Fallback: ensure normalization matches training
    if hasattr(orig_opt, 'state_range') and orig_opt.state_range is not None:
        dynamics.override_state_range(orig_opt.state_range)

    return dynamics

def run_single(args):
    """Run one controller from a specified initial condition."""
    os.makedirs(args.output_dir, exist_ok=True)

    ctrl_type = args.controller
    if ctrl_type != 'grid_based' and not os.path.exists(args.checkpoint_path):
        print(f"ERROR: checkpoint not found: {args.checkpoint_path}")
        return
    display = CONTROLLER_LABELS.get(ctrl_type, ctrl_type)

    print('=' * 60)
    print(f'{display} Controller -- Single Run')
    print('=' * 60)

    # Build controller
    controller = build_controller(ctrl_type, args)
    dynamics = controller.dynamics

    # Initial state (--initial_state takes priority over individual args)
    if args.initial_state is not None:
        initial_state = np.array(args.initial_state)
    else:
        initial_state = np.array([
            args.initial_px,  args.initial_py,
            args.initial_vx,  args.initial_vy,
            args.initial_theta, args.initial_omega,
        ])
    print(f"\nInitial state: px={initial_state[0]:.2f}, py={initial_state[1]:.2f}, "
          f"vx={initial_state[2]:.2f}, vy={initial_state[3]:.2f}, "
          f"theta={initial_state[4]:.2f}, omega={initial_state[5]:.2f}")

    # BRAT-specific pre-check
    if ctrl_type == 'brat':
        v0 = controller.get_value(initial_state, args.tMax)
        print(f"Initial V(x, tMax={args.tMax}s) = {v0:.4f}")
        print(f"State is {'INSIDE' if v0 <= 0 else 'OUTSIDE'} the BRAT")

    # Run simulation
    print(f"\nRunning simulation for {args.max_sim_time}s ...")
    result = controller.simulate_docking(initial_state, args.max_sim_time)

    # ---- Print results ----
    print('\n' + '=' * 60)
    print('SIMULATION RESULTS')
    print('=' * 60)
    print(f"Success:  {result['success']}")
    print(f"Docked:   {result.get('docked', 'N/A')}")
    print(f"Collision:{result.get('collision', 'N/A')}")
    fs = result['final_state']
    theta_err = np.arctan2(np.sin(fs[4] - np.pi/2), np.cos(fs[4] - np.pi/2))
    print(f"Final state: px={fs[0]:.4f}m  py={fs[1]:.4f}m  "
          f"vx={fs[2]:.4f}m/s  vy={fs[3]:.4f}m/s  "
          f"θ_err={theta_err:.4f}rad ({np.degrees(theta_err):.2f}°)  "
          f"ω={fs[5]:.4f}rad/s")
    print(f"Duration: {result['times'][-1]:.2f}s")
    print(f"Control effort: {result.get('control_effort', 0):.2f}")
    print(f"Wall time: {result.get('wall_time', 0):.2f}s")

    if ctrl_type == 'brat':
        if result.get('phase_transition_time') is not None:
            print(f"Entered BRAT (Phase 2) at t={result['phase_transition_time']:.2f}s")
        else:
            print("Never entered BRAT (stayed in Phase 1)")
        phases = result.get('phases', np.array([]))
        if len(phases) > 0:
            print(f"Time in Phase 1: {np.sum(phases == 1) * args.dt:.1f}s")
            print(f"Time in Phase 2: {np.sum(phases == 2) * args.dt:.1f}s")
        n_fb = result.get('n_fallback_steps', 0)
        n_p1 = int(np.sum(phases == 1)) if len(phases) > 0 else 0
        if n_fb > 0 or args.gradient_fallback:
            print(f"Gradient fallback: {n_fb}/{n_p1} Phase 1 steps")

    if ctrl_type == 'mpc_terminal':
        phases = result.get('phases', np.array([]))
        if len(phases) > 0:
            print(f"Time in Phase 1: {np.sum(phases == 1) * args.dt:.1f}s")
            print(f"Time in Phase 2: {np.sum(phases == 2) * args.dt:.1f}s")
        if result.get('brat_entry_time') is not None:
            print(f"Entered BRAT (Phase 2) at t={result['brat_entry_time']:.2f}s")
        else:
            print("Never entered BRAT (stayed in Phase 1)")

    phase2_debug = result.get('phase2_debug_log', [])
    if args.debug_phase2:
        print(f"Phase 2 debug entries captured: {len(phase2_debug)}")
        if phase2_debug and not result['success']:
            print("Last Phase 2 debug entries before failure:")
            for entry in phase2_debug[-5:]:
                if ctrl_type == 'brat':
                    print(
                        f"  t={entry['sim_time']:6.2f}s  "
                        f"t*={entry['selected_t_star']:5.2f}s ({entry['selected_status']})  "
                        f"V(t*)={entry['value_at_query']: .4f}  "
                        f"V(tMax)={entry['value_tmax']: .4f}  "
                        f"safety_override={entry['safety_filter_modified_control']}"
                    )
                elif ctrl_type == 'mpc_terminal':
                    terminal = entry.get('terminal_search', {})
                    print(
                        f"  t={entry['sim_time']:6.2f}s  "
                        f"current_t*={entry['current_t_star']:5.2f}s ({entry['current_status']})  "
                        f"best_terminal_t*={terminal.get('best_t_star', float('nan')):5.2f}s  "
                        f"best_combined={entry.get('best_combined_cost', float('nan')): .4f}"
                    )

    # Safety filter stats
    sf_mode = result.get('safety_filter_mode', 0)
    sf_log = result.get('safety_filter_log', [])
    if sf_mode > 0 and sf_log:
        mode_label = {1: 'Least-Restrictive', 2: 'CBF-QP'}[sf_mode]
        V_vals = [e['V_avoid'] for e in sf_log]
        print(f"\nSafety filter: Mode {sf_mode} ({mode_label})")
        print(f"  V_avoid range: [{min(V_vals):.4f}, {max(V_vals):.4f}]")
        if sf_mode == 1:
            n_active = sum(1 for e in sf_log if e.get('filter_active'))
            print(f"  Activated: {n_active}/{len(sf_log)} steps "
                  f"({100*n_active/len(sf_log):.1f}%)")
        elif sf_mode == 2:
            alphas = [e.get('alpha_effective', 0) for e in sf_log]
            n_active = sum(1 for a in alphas if a > 1e-6)
            print(f"  Intervened: {n_active}/{len(sf_log)} steps "
                  f"({100*n_active/len(sf_log):.1f}%)")
            if n_active > 0:
                active_alphas = [a for a in alphas if a > 1e-6]
                print(f"  alpha_eff range (when active): "
                      f"[{min(active_alphas):.4f}, {max(active_alphas):.4f}]")

    # ---- Generate outputs ----
    print('\n' + '=' * 60)
    print('GENERATING OUTPUTS')
    print('=' * 60)

    # Generic static plots (all controller types)
    result_dict = {display: result}

    json_path = os.path.join(args.output_dir, 'single_run_results.json')
    json_data = {
        'args': vars(args),
        'initial_state': initial_state.tolist(),
        'result': _to_jsonable(result),
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saving run summary: {json_path}")

    traj_path = os.path.join(args.output_dir, 'trajectory.png')
    print(f"Generating trajectory plot: {traj_path}")
    plot_trajectories_static(result_dict, dynamics, save_path=traj_path)

    data_path = os.path.join(args.output_dir, 'simulation_data.png')
    print(f"Generating data plots: {data_path}")
    plot_simulation_data_multi(result_dict, save_path=data_path)

    # Animation
    if not args.no_animation:
        if ctrl_type == 'brat' and not args.no_value_function:
            # BRAT-specific animation with value-function heatmap
            anim_path = os.path.join(args.output_dir, 'docking_animation.mp4')
            print(f"Generating BRAT animation (with value function): {anim_path}")
            create_brat_animation(
                controller, result, anim_path,
                skip_frames=args.skip_frames,
                resolution=args.resolution,
                show_value_function=True,
            )
        elif ctrl_type == 'mpc_terminal' and not args.no_value_function:
            # MPC+Terminal animation with phase-aware value function
            anim_path = os.path.join(args.output_dir, 'docking_animation.mp4')
            print(f"Generating MPC+Terminal animation (with value function): {anim_path}")
            create_mpc_terminal_animation(
                controller, result, anim_path,
                skip_frames=args.skip_frames,
                resolution=args.resolution,
                show_value_function=True,
            )
        else:
            # Generic animation (works for any controller)
            anim_path = os.path.join(args.output_dir, 'docking_animation.mp4')
            print(f"Generating animation: {anim_path}")
            animate_trajectories(
                result_dict, dynamics, anim_path,
                skip_frames=args.skip_frames, dt=args.dt,
            )

    print('\n' + '=' * 60)
    print('DONE')
    print('=' * 60)
    print(f"Outputs saved to: {args.output_dir}")

def run_compare(args):
    """Run N rollouts per controller and compare metrics."""
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: checkpoint not found: {args.checkpoint_path}")
        return

    # Load dynamics for IC sampling
    dynamics = load_dynamics(args.checkpoint_path)

    # Log feasibility bounds for diagnostics
    v_fb, omega_fb = compute_feasibility_bounds(dynamics, args.tMax)
    print(f"\nFeasibility bounds (2/3 tMax budget): "
          f"v_max={v_fb:.3f} m/s  omega_max={omega_fb:.4f} rad/s")
    if SAMPLING_STATE_RANGE[2, 1] > v_fb or SAMPLING_STATE_RANGE[5, 1] > omega_fb:
        print("  WARNING: SAMPLING_STATE_RANGE exceeds feasibility bounds")

    # Pre-build grid controller if present (needed for IC filtering)
    grid_controller = None
    if 'grid_based' in args.controllers:
        print("\nBuilding grid-based controller (for IC filtering & rollouts)...")
        grid_controller = build_controller('grid_based', args)

    # Build avoid-BRT filter: reject ICs doomed to collide even under
    # optimal avoidance control.
    avoid_filter_fn = None
    avoid_ckpt = getattr(args, 'safety_checkpoint_path', None)
    if avoid_ckpt and os.path.exists(avoid_ckpt):
        try:
            avoid_ckpt_resolved = SafetyFilter._resolve_checkpoint(avoid_ckpt)
            avoid_ctrl = BRATController(
                checkpoint_path=avoid_ckpt_resolved,
                device=args.device,
            )
            # Use the tMax from the trained model, not the BRATController default
            avoid_tMax = avoid_ctrl.orig_opt.tMax
            avoid_filter_fn = lambda states: avoid_ctrl.get_values_batch_states(
                states, avoid_tMax)
            print(f"  Avoid-BRT filter ready (tMax={avoid_tMax})")
        except Exception as e:
            print(f"  WARNING: could not load avoid model from "
                  f"{avoid_ckpt}: {e}\n  Skipping avoid-BRT filter.")
    else:
        print(f"  Avoid checkpoint not found at "
              f"{avoid_ckpt!r}, skipping avoid-BRT filter.")

    # BRAT filter for IC sampling: grid-based BRAT takes priority,
    # otherwise fall back to deepReach BRAT if sampling_method='brat'.
    value_filter_fn = None
    if grid_controller is not None:
        value_filter_fn = lambda states: grid_controller.get_brat_values_batch(
            states)
        print(f"  Grid-based BRAT filter ready — ICs will satisfy "
              f"max(V_4D, V_2D) <= 0 at t={args.max_sim_time}s horizon")
    elif getattr(args, 'sampling_method', 'uniform') == 'brat':
        print(f"Loading model for BRAT IC filtering (tMax={args.tMax}) ...")
        query_ctrl = BRATController(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            device=args.device,
        )
        value_filter_fn = lambda states: query_ctrl.get_values_batch_states(
            states, args.tMax)
        print(f"  BRAT filter ready — ICs will satisfy V(x, {args.tMax}) <= 0")

    # Sample initial conditions
    sampling_label = ('grid_brat' if grid_controller is not None
                      else getattr(args, 'sampling_method', 'uniform'))
    print(f"\nSampling {args.n_rollouts} initial conditions "
          f"(seed={args.seed}, method={sampling_label}) ...")
    ics = sample_initial_conditions(
        dynamics, args.n_rollouts, device=args.device, seed=args.seed,
        value_filter_fn=value_filter_fn, avoid_filter_fn=avoid_filter_fn)
    print(f"Sampled {len(ics)} valid ICs.\n")

    ic_path = os.path.join(args.output_dir, 'initial_conditions.npy')
    np.save(ic_path, ics)
    print(f"Initial conditions saved to {ic_path}")

    # Build controllers (reuse pre-built grid controller)
    controllers = {}
    for name in args.controllers:
        if name == 'grid_based' and grid_controller is not None:
            controllers[name] = grid_controller
        else:
            print(f"\nBuilding controller: {name}")
            controllers[name] = build_controller(name, args)

    # Run rollouts
    all_results = {}
    metrics_by_name = {}

    for ctrl_name, ctrl in controllers.items():
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        print(f"\n{'=' * 60}")
        print(f"Running {len(ics)} rollouts for {display}")
        print(f"{'=' * 60}")

        results = []
        idx_w = len(str(len(ics)))  # width for index formatting
        for i, ic in enumerate(ics):
            result = ctrl.simulate_docking(ic, args.max_sim_time)
            results.append(result)

            # Per-rollout summary
            fs = result['final_state']
            theta_err = np.arctan2(np.sin(fs[4] - np.pi/2),
                                   np.cos(fs[4] - np.pi/2))
            if result['collision']:
                outcome = 'COLLISION'
            elif result['docked']:
                outcome = 'DOCKED'
            else:
                outcome = 'TIMEOUT'
            t_final = result['times'][-1]
            print(f"  [{i+1:>{idx_w}}/{len(ics)}] {outcome:<10} "
                  f"t={t_final:6.1f}s  "
                  f"θ_err={np.degrees(theta_err):+7.2f}°  "
                  f"effort={result['control_effort']:7.1f}")

        all_results[ctrl_name] = results
        m = compute_metrics(results)
        metrics_by_name[display] = m
        print(f"\n{display}: dock={m['docking_rate']*100:.1f}%  "
              f"fail={m['failure_rate']*100:.1f}%  "
              f"timeout={m['timeout_rate']*100:.1f}%  "
              f"effort={m['mean_control_effort']:.1f}  "
              f"time={m['mean_wall_time']:.2f}s")

    # Print & save
    print_comparison_table(metrics_by_name)

    # ---- Docking-time optimality (paired comparison) ----
    display_names = [CONTROLLER_LABELS.get(c, c) for c in args.controllers]
    all_results_by_display = {CONTROLLER_LABELS.get(c, c): all_results[c]
                              for c in args.controllers}
    optimality = compute_docking_optimality(all_results_by_display,
                                            display_names)
    if optimality['common_n'] > 0:
        print_optimality_table(optimality)
    else:
        print("\nNo common-success ICs across all controllers; "
              "skipping docking-time optimality comparison.")

    # ---- Collect detailed per-rollout outcomes ----
    detailed_by_name = {}
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        results = all_results[ctrl_name]

        collisions = []
        docked_list = []
        timeouts = []

        for i, result in enumerate(results):
            fs = result['final_state']
            theta_err = np.arctan2(np.sin(fs[4] - np.pi/2),
                                   np.cos(fs[4] - np.pi/2))
            detail = {
                'rollout_idx': i,
                'initial_state': ics[i].tolist(),
                'final_state': fs.tolist(),
                'theta_err_deg': round(float(np.degrees(theta_err)), 4),
                'sim_time': round(float(result['times'][-1]), 4),
                'control_effort': round(float(result['control_effort']), 4),
            }

            if result['collision']:
                collisions.append(detail)
            elif result['docked']:
                docked_list.append(detail)
            else:
                timeouts.append(detail)

        detailed_by_name[display] = {
            'collisions': collisions,
            'docked': docked_list,
            'timeouts': timeouts,
        }

    # ---- Save collision ICs per controller as .npy ----
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        coll_details = detailed_by_name[display]['collisions']
        if coll_details:
            collision_ics = np.array([c['initial_state'] for c in coll_details])
            collision_path = os.path.join(
                args.output_dir, f'collision_ics_{ctrl_name}.npy')
            np.save(collision_path, collision_ics)
            print(f"  {display}: {len(coll_details)} collision IC(s) "
                  f"saved to {collision_path}")

    # ---- Print collision summary ----
    print('\n' + '-' * 60)
    print('COLLISION SUMMARY')
    print('-' * 60)
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        coll = detailed_by_name[display]['collisions']
        if coll:
            print(f"\n{display}  ({len(coll)} collision(s)):")
            for c in coll:
                ic = c['initial_state']
                print(f"  rollout {c['rollout_idx']:>3d}  "
                      f"IC=[{ic[0]:+7.2f}, {ic[1]:+7.2f}, {ic[2]:+5.2f}, "
                      f"{ic[3]:+5.2f}, {ic[4]:+5.2f}, {ic[5]:+5.2f}]  "
                      f"t_coll={c['sim_time']:.1f}s")
        else:
            print(f"\n{display}: no collisions")
    print('-' * 60)

    # ---- Build and save JSON ----
    json_path = os.path.join(args.output_dir, 'comparison_results.json')
    json_data = {
        '_metadata': {
            'sampling_method': getattr(args, 'sampling_method', 'uniform'),
            'n_rollouts': args.n_rollouts,
            'seed': args.seed,
            'tMax': args.tMax,
            'checkpoint_path': args.checkpoint_path,
        }
    }
    for k, v in metrics_by_name.items():
        json_data[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float))
                              else int(vv))
                         for kk, vv in v.items()}
        json_data[k].update(detailed_by_name[k])
    json_data['_docking_optimality'] = _to_jsonable(optimality)
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Results saved to {json_path}")

    bar_path = os.path.join(args.output_dir, 'metrics_comparison.png')
    plot_metrics_bar(metrics_by_name, save_path=bar_path,
                     optimality=optimality)

    # Docking-time histogram (grid-based baseline, if present)
    grid_label = CONTROLLER_LABELS.get('grid_based', 'Grid-Based HJ')
    if grid_label in all_results_by_display:
        hist_path = os.path.join(args.output_dir, 'docking_time_histogram.png')
        plot_docking_time_histogram(
            all_results_by_display, display_names,
            save_path=hist_path, grid_based_label=grid_label)

    # Trajectory comparison (first IC)
    first_results = {}
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        first_results[display] = all_results[ctrl_name][0]

    traj_path = os.path.join(args.output_dir, 'trajectory_comparison.png')
    plot_trajectories_static(first_results, dynamics, save_path=traj_path)

    data_path = os.path.join(args.output_dir, 'simulation_data_comparison.png')
    plot_simulation_data_multi(first_results, save_path=data_path)

    # Optional animation
    if args.animate:
        anim_path = os.path.join(args.output_dir, 'comparison_animation.mp4')
        print(f"\nGenerating comparison animation -> {anim_path}")
        animate_trajectories(
            first_results, dynamics, anim_path,
            skip_frames=args.skip_frames, dt=args.dt)

    print(f"\nAll outputs saved to {args.output_dir}")
    print("Done.")


def _add_shared_args(parser):
    """Arguments common to both subcommands."""
    parser.add_argument(
        '--checkpoint_path', type=str,
        default='runs/Docking6D_RA_14sec/training/checkpoints/model_epoch_145000.pth',
        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/controller',
                        help='Directory to save outputs')
    parser.add_argument('--tMax', type=float, default=15.0,
                        help='BRAT / terminal cost time horizon')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Control / integration timestep (s)')
    parser.add_argument('--max_sim_time', type=float, default=30.0,
                        help='Maximum simulation time (s)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device')
    # MPC-specific
    parser.add_argument('--planning_horizon', type=float, default=3.0,
                        help='MPC-only planning horizon (s)')
    parser.add_argument('--mpc_dt', type=float, default=0.5,
                        help='MPC planning timestep (s); simulation uses --dt')
    parser.add_argument('--effective_horizon', type=float, default=1.0,
                        help='MPC+Terminal effective horizon (s)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='(Legacy) MPC random-shooting samples')
    parser.add_argument('--num_refinement', type=int, default=10,
                        help='(Legacy) MPC iterative refinement passes')
    # Gradient MPC (shared defaults for mpc and mpc_terminal)
    parser.add_argument('--gradient_lr', type=float, default=1.0,
                        help='Adam learning rate for gradient MPC (default: 1.0)')
    parser.add_argument('--gradient_iters', type=int, default=50,
                        help='Adam iterations per MPC step (default: 50). '
                             'Overridden by --mpc_gradient_iters / '
                             '--mpc_terminal_gradient_iters if set.')
    parser.add_argument('--num_restarts', type=int, default=8,
                        help='Parallel restarts for gradient MPC (default: 8). '
                             'Overridden by --mpc_num_restarts / '
                             '--mpc_terminal_num_restarts if set.')
    # Per-controller overrides
    parser.add_argument('--mpc_gradient_iters', type=int, default=None,
                        help='Adam iterations for mpc controller '
                             '(overrides --gradient_iters)')
    parser.add_argument('--mpc_num_restarts', type=int, default=None,
                        help='Parallel restarts for mpc controller '
                             '(overrides --num_restarts)')
    parser.add_argument('--mpc_terminal_gradient_iters', type=int, default=None,
                        help='Adam iterations for mpc_terminal controller '
                             '(overrides --gradient_iters)')
    parser.add_argument('--mpc_terminal_num_restarts', type=int, default=None,
                        help='Parallel restarts for mpc_terminal controller '
                             '(overrides --num_restarts)')
    parser.add_argument('--goal_weight', type=float, default=0.01,
                        help='Goal-directed cost weight for baseline MPC '
                             '(default: 0.01)')
    parser.add_argument('--effort_weight', type=float, default=0.0,
                        help='Control effort penalty weight for MPC terminal '
                             'controllers (0.0 = disabled). Adds '
                             'effort_weight * sum(||u||*dt) to the combined '
                             'cost. Recommended range: 0.001 - 0.05.')
    # Graduated stagnation-escape tuning (MPC+Terminal controllers)
    parser.add_argument('--exploration_factor', type=float, default=3.0,
                        help='Multiplier for goal_weight escalation when in '
                             'EXPLORING mode (default 3.0)')
    parser.add_argument('--exploration_patience', type=int, default=1,
                        help='Number of stagnation windows (each ~5 s) in '
                             'EXPLORING mode before switching to BRAT '
                             'fallback (default 1)')
    parser.add_argument('--escape_thresh', type=float, default=0.5,
                        help='Distance improvement (m) from stagnation entry '
                             'required to declare local min escaped and '
                             'return to NORMAL mode (default 0.5)')
    # Animation
    parser.add_argument('--skip_frames', type=int, default=5,
                        help='Frames to skip in animation')
    # Safety filter
    parser.add_argument('--safety_filter_mode', type=int, default=0,
                        choices=[0, 1, 2],
                        help='Safety filter: 0=disabled, 1=least-restrictive, '
                             '2=CBF-QP (default: 0)')
    parser.add_argument('--safety_checkpoint_path', type=str,
                        default='runs/Docking6D_RA_avoid',
                        help='Path to avoid-only BRT checkpoint dir or .pth '
                             'file (default: runs/Docking6D_RA_avoid)')
    parser.add_argument('--safety_filter_margin', type=float, default=0.02,
                        help='Mode 1 activation margin delta (meters). '
                             'Safety overrides when V_avoid <= delta. '
                             'Same units as avoid_fn signed distance '
                             '(default: 0.02)')
    parser.add_argument('--safety_margin_phase1', type=float, default=0.1,
                        help='Safety filter margin when outside BRAT (Phase 1). '
                             'Higher value triggers filter earlier. '
                             '(default: 0.1)')
    parser.add_argument('--safety_margin_phase2', type=float, default=0.02,
                        help='Safety filter margin when inside BRAT (Phase 2). '
                             'Lower value for minimal intervention. '
                             '(default: 0.02)')
    parser.add_argument('--safety_filter_gamma', type=float, default=0.2,
                        help='Mode 2 CBF decay rate gamma '
                             '(default: 0.2, from ComboControl)')
    parser.add_argument('--debug_phase2', action='store_true',
                        help='Enable detailed Phase 2 search/control '
                             'diagnostics for single-run debugging')
    # Gradient fallback (BRAT Phase 1)
    parser.add_argument('--gradient_fallback', action='store_true',
                        default=True, dest='gradient_fallback',
                        help='Enable L2 goal-directed gradient fallback in '
                             'Phase 1 (default: enabled)')
    parser.add_argument('--no_gradient_fallback', action='store_false',
                        dest='gradient_fallback',
                        help='Disable L2 gradient fallback in Phase 1')
    parser.add_argument('--grad_threshold', type=float, default=0.01,
                        help='Gradient norm below which fallback activates '
                             '(default: 0.01)')
    parser.add_argument('--avoid_proximity_margin', type=float, default=1.0,
                        help='Obstacle SDF distance (m) below which fallback '
                             'is suppressed (default: 1.0)')
    # Grid-based controller
    parser.add_argument('--grid_cache_dir', type=str, default=None,
                        help='Cache directory for grid-based HJ value functions. '
                             'None = solve fresh each run.')
    parser.add_argument('--grid_filter_mode', type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help='Grid controller filter: 0=optimal bang-bang '
                             '(default), 1=least-restrictive, '
                             '2=smooth-blend QP, 3=nominal LQR')
    # RL controller
    parser.add_argument('--rl_checkpoint_path', type=str, default=None,
                        help='Path to trained RL Q-network .pth file')
    parser.add_argument('--rl_architecture', type=int, nargs='+',
                        default=[256, 256],
                        help='Hidden layer sizes for RL Q-network '
                             '(must match training)')
    parser.add_argument('--rl_activation', type=str, default='Tanh',
                        help='Activation for RL Q-network (must match training)')


def main():
    parser = argparse.ArgumentParser(
        description='Docking6D Controller Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_controller.py single  --controller brat\n"
            "  python run_controller.py compare --controllers brat mpc mpc_terminal\n"
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ---- single ----
    sp_single = subparsers.add_parser(
        'single', help='Run one controller from a specified initial condition')
    _add_shared_args(sp_single)
    sp_single.add_argument(
        '--controller', type=str, default='brat',
        choices=['brat', 'mpc', 'mpc_terminal', 'grid_based', 'rl'],
        help='Controller type to run')
    # Initial state — either as a single 6-element list or individual components
    sp_single.add_argument('--initial_state', type=float, nargs=6,
                           metavar=('PX', 'PY', 'VX', 'VY', 'THETA', 'OMEGA'),
                           default=None,
                           help='Initial state as 6 floats: px py vx vy theta omega. '
                                'Overrides individual --initial_* args if provided.')
    sp_single.add_argument('--initial_px',    type=float, default=2.0)
    sp_single.add_argument('--initial_py',    type=float, default=10.0)
    sp_single.add_argument('--initial_vx',    type=float, default=0.0)
    sp_single.add_argument('--initial_vy',    type=float, default=0.0)
    sp_single.add_argument('--initial_theta', type=float, default=-1.57)
    sp_single.add_argument('--initial_omega', type=float, default=0.0)
    # Animation
    sp_single.add_argument('--no_animation', action='store_true',
                           help='Skip animation generation')
    sp_single.add_argument('--no_value_function', action='store_true',
                           help='Skip value-function heatmap in BRAT animation')
    sp_single.add_argument('--resolution', type=int, default=40,
                           help='Value-function grid resolution (BRAT only)')

    # ---- compare ----
    sp_compare = subparsers.add_parser(
        'compare', help='Compare controllers over N rollouts from random ICs')
    _add_shared_args(sp_compare)
    sp_compare.add_argument(
        '--controllers', type=str, nargs='+',
        default=['brat', 'mpc', 'mpc_terminal'],
        choices=['brat', 'mpc', 'mpc_terminal', 'grid_based', 'rl'],
        help='Controllers to compare')
    sp_compare.add_argument('--n_rollouts', type=int, default=50,
                            help='Number of rollouts per controller')
    sp_compare.add_argument('--seed', type=int, default=1,
                            help='Random seed for IC sampling')
    sp_compare.add_argument('--sampling_method', type=str, default='uniform',
                            choices=['uniform', 'brat'],
                            help='IC sampling method: "uniform" = geometric '
                                 'constraints only; "brat" = additionally '
                                 'require V(x, tMax) <= 0 (inside learned BRAT)')
    sp_compare.add_argument('--animate', action='store_true',
                            help='Generate comparison animation for first IC')

    args = parser.parse_args()

    if args.command == 'single':
        run_single(args)
    elif args.command == 'compare':
        run_compare(args)


if __name__ == '__main__':
    main()
