#!/usr/bin/env python3
"""
Unified controller runner for 13-D spacecraft docking.

Subcommands
-----------
single   Run one controller from a single initial condition.
compare  Run N rollouts per controller from shared random ICs and show
         a side-by-side comparison table.

Examples
--------
  python run_controller_13d.py single  --controller brt_13d --checkpoint_path <CKPT> [opts]
  python run_controller_13d.py compare --controllers brt_13d mpc_terminal_13d  [opts]

See run_controller_13d.sh for ready-made command templates.
"""

import argparse
import inspect
import json
import os
import pickle
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.controllers import (
    BRTController13D, MPCController13D, MPCTerminalController13D,
    SafetyFilter, RLController13D, VanillaBRTController13D,
)
from utils.brt_visualization_13d import BRTVisualizer13D
from utils.controllers.controller_animation_13d import ControllerAnimation13D
from utils.controllers.trajectory_only_animation_13d import TrajectoryAnimation13D
from utils.controllers.slice_gif_13d import SliceGIF13D
from utils.controllers.combined_gif_13d import CombinedGIF13D
from utils.controllers.velocity_slice_gif_13d import VelocitySliceGIF13D
from utils.controllers.attitude_slice_gif_13d import AttitudeSliceGIF13D
from utils.controllers.docking13d_mixin import _quat_error_angle_np
from utils.controllers.static_plots_13d import (
    plot_trajectory_13d, plot_states_13d, plot_controls_13d,
)

from dynamics import dynamics as dynamics_module

# ------------------------------------------------------------------ #
#  Constants & helpers
# ------------------------------------------------------------------ #

CONTROLLER_LABELS = {
    'brt_13d':          'BRT 13D',
    'vanilla_brt_13d':  'Vanilla BRT 13D',
    'brt_safety_13d':   'BRT+Safety 13D',
    'brt_pd_hybrid':    'BRT+PD Hybrid 13D',
    'mpc_13d':          'MPC 13D',
    'mpc_terminal_13d': 'MPC+Terminal 13D',
    'rl_13d':           'RL 13D (DDQN)',
}

CONTROLLER_COLORS = {
    'brt_13d':          '#1f77b4',   # blue
    'vanilla_brt_13d':  '#17becf',   # cyan
    'brt_safety_13d':   '#2ca02c',   # green
    'mpc_13d':          '#ff7f0e',   # orange
    'mpc_terminal_13d': '#d62728',   # red
    'rl_13d':           '#9467bd',   # purple
}



def _fmt_state(state):
    """Return a compact, labelled one-liner for a 13-D state."""
    pos = f"pos=({state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f})"
    vel = f"vel=({state[3]:+.3f}, {state[4]:+.3f}, {state[5]:+.3f})"
    quat = (f"quat=({state[6]:.4f}, {state[7]:.4f}, "
            f"{state[8]:.4f}, {state[9]:.4f})")
    omg = f"omega=({state[10]:+.3f}, {state[11]:+.3f}, {state[12]:+.3f})"
    return f"{pos}  {vel}  {quat}  {omg}"


def _banner(text, width=60, char='='):
    """Print a centred banner line."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")

# ------------------------------------------------------------------ #
#  Builder
# ------------------------------------------------------------------ #

def _build_optional_safety_filter(args, margin_override=None):
    """Build a SafetyFilter if mode > 0 and checkpoint is provided.

    Args:
        args: parsed CLI arguments.
        margin_override: if provided, use this margin instead of
            ``args.safety_filter_margin``.  Used for single-phase controllers
            that need a docking-friendly margin.
    """
    sf_mode = getattr(args, 'safety_filter_mode', 0)
    sf_path = getattr(args, 'safety_checkpoint_path', None)
    margin = margin_override if margin_override is not None else args.safety_filter_margin
    if sf_mode > 0 and sf_path is not None:
        return SafetyFilter(
            mode=sf_mode,
            checkpoint_path=sf_path,
            tMax=args.safety_tMax,
            margin=margin,
            gamma=args.safety_filter_gamma,
            device=args.device,
        )
    if sf_mode > 0:
        print(f"[WARNING] safety_filter_mode={sf_mode} but no "
              f"--safety_checkpoint_path provided; disabling safety filter.")
    return None


def build_controller(name, args):
    """Instantiate a 13D controller by name string."""
    # Validate checkpoint_path for controllers that require it
    if name in ('brt_13d', 'brt_safety_13d', 'mpc_13d', 'mpc_terminal_13d'):
        if args.checkpoint_path is None:
            raise ValueError(f"--checkpoint_path is required for {name}")
    if name == 'vanilla_brt_13d':
        if not args.vanilla_checkpoint_path or not os.path.exists(args.vanilla_checkpoint_path):
            raise ValueError(
                f"--vanilla_checkpoint_path is required for {name} "
                f"(got: {getattr(args, 'vanilla_checkpoint_path', None)})")

    if name == 'brt_13d':
        return BRTController13D(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
        )
    elif name == 'brt_safety_13d':
        sf = SafetyFilter(
            mode=args.safety_filter_mode,
            checkpoint_path=args.safety_checkpoint_path,
            tMax=args.safety_tMax,
            margin=args.safety_filter_margin,
            gamma=args.safety_filter_gamma,
            device=args.device,
        )
        return BRTController13D(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
            safety_filter=sf,
            pd_torque_proximity=getattr(args, 'pd_torque_proximity', 2.0),
        )
    elif name == 'brt_pd_hybrid':
        sf = SafetyFilter(
            mode=args.safety_filter_mode,
            checkpoint_path=args.safety_checkpoint_path,
            tMax=args.safety_tMax,
            margin=args.safety_filter_margin,
            gamma=args.safety_filter_gamma,
            device=args.device,
        )
        return BRTController13D(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
            safety_filter=sf,
            pd_torque_proximity=getattr(args, 'pd_torque_proximity', 2.0),
        )
    elif name == 'mpc_13d':
        sf = _build_optional_safety_filter(args,
                                           margin_override=args.safety_filter_margin_docking)
        iters = getattr(args, 'mpc_gradient_iters', None) or args.gradient_iters
        restarts = getattr(args, 'mpc_num_restarts', None) or args.num_restarts
        return MPCController13D(
            checkpoint_path=args.checkpoint_path,
            planning_horizon_sec=args.planning_horizon,
            mpc_dt=args.mpc_dt,
            dt=args.dt,
            device=args.device,
            gradient_lr=args.gradient_lr,
            gradient_iters=iters,
            num_restarts=restarts,
            goal_weight=args.goal_weight,
            safety_filter=sf,
        )
    elif name == 'mpc_terminal_13d':
        sf = _build_optional_safety_filter(args)
        iters = getattr(args, 'mpc_terminal_gradient_iters', None) or args.gradient_iters
        restarts = getattr(args, 'mpc_terminal_num_restarts', None) or args.num_restarts
        return MPCTerminalController13D(
            checkpoint_path=args.checkpoint_path,
            effective_horizon_sec=args.effective_horizon,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
            effort_weight=args.effort_weight,
            exploration_factor=args.exploration_factor,
            exploration_patience=args.exploration_patience,
            escape_thresh=args.escape_thresh,
            gradient_lr=args.gradient_lr,
            gradient_iters=iters,
            num_restarts=restarts,
            goal_weight=args.goal_weight,
            safety_filter=sf,
        )
    elif name == 'vanilla_brt_13d':
        sf = _build_optional_safety_filter(args)
        return VanillaBRTController13D(
            checkpoint_path=args.vanilla_checkpoint_path,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
            safety_filter=sf,
        )
    elif name == 'rl_13d':
        sf = _build_optional_safety_filter(args,
                                           margin_override=args.safety_filter_margin_docking)
        return RLController13D(
            rl_checkpoint_path=args.rl_checkpoint_path,
            dt=args.dt,
            device=args.device,
            safety_filter=sf,
            architecture=args.rl_architecture,
            activation=args.rl_activation,
            pd_attitude=args.rl_pd_attitude,
        )
    else:
        raise ValueError(f"Unknown controller: {name}")

# ------------------------------------------------------------------ #
#  Initial-condition sampling
# ------------------------------------------------------------------ #

# Fixed IC that is always used as the first rollout in comparisons.
FIXED_IC_13D = np.array([
    10.0, -5.0, 2.0,          # position
    0.0,   0.0, 0.0,          # velocity
    0.7071, 0.0, 0.0, 0.7071, # quaternion (90° yaw)
    0.0,   0.0, 0.0,          # angular velocity
])


def sample_initial_conditions(dynamics, n, device='cuda', seed=42,
                              value_filter_fn=None):
    """Sample *n* valid 13D initial conditions.

    The first IC is always the fixed reference state
    (:data:`FIXED_IC_13D`).  The remaining *n − 1* are randomly sampled.

    Filters out states that are already docked (reach_fn <= 0) or
    inside the failure set (avoid_fn <= 0).

    Filtering:
      - Within sampling bounds.
      - Not inside the failure set (avoid_fn > 0).
      - Not already docked (reach_fn > 0).
      - (optional) Inside the learned BRAT (value_filter_fn(states) <= 0).

    Args:
        dynamics: Dynamics instance with avoid_fn() and reach_fn().
        n: Number of ICs to sample.
        device: Torch device for dynamics queries.
        seed: Random seed.
        value_filter_fn: Optional callable  (N,13) np.array -> (N,) np.array
            returning V(x, tMax) for each state.  States with V <= 0 are
            kept (inside the BRAT).  ``None`` disables this filter.
    """
    rng = np.random.RandomState(seed)

    # --- Fixed first IC ------------------------------------------------ #
    samples = [FIXED_IC_13D.copy()]
    n_random = n - 1
    if n_random <= 0:
        return np.array(samples[:n])

    # Sampling bounds: subset of training range
    dyn_lo = dynamics.state_range_[:, 0].cpu().numpy().astype(np.float64)
    dyn_hi = dynamics.state_range_[:, 1].cpu().numpy().astype(np.float64)

    # Tighten position range for feasibility
    sample_lo = dyn_lo.copy()
    sample_hi = dyn_hi.copy()

    # Goal quaternion for biased sampling (used when BRAT filtering)
    q_goal = dynamics.q_goal.cpu().numpy() if hasattr(dynamics, 'q_goal') else None

    if value_filter_fn is not None:
        # When filtering by BRAT, tighten bounds to improve acceptance
        # rate.  Wider bounds for longer-horizon models (bigger BRAT).
        sample_lo[:3] = np.maximum(sample_lo[:3], -4.0)
        sample_hi[:3] = np.minimum(sample_hi[:3],  4.0)
        sample_lo[3:6] = np.maximum(sample_lo[3:6], -0.3)
        sample_hi[3:6] = np.minimum(sample_hi[3:6],  0.3)
        sample_lo[10:13] = np.maximum(sample_lo[10:13], -0.3)
        sample_hi[10:13] = np.minimum(sample_hi[10:13],  0.3)
        # Quaternion perturbation scale (moderate = broader around goal)
        quat_sigma = 0.3
    else:
        sample_lo[:3] = np.maximum(sample_lo[:3], -13.0)
        sample_hi[:3] = np.minimum(sample_hi[:3],  13.0)
        sample_lo[3:6] = np.maximum(sample_lo[3:6], -0.15)
        sample_hi[3:6] = np.minimum(sample_hi[3:6],  0.15)
        sample_lo[10:13] = np.maximum(sample_lo[10:13], -0.3)
        sample_hi[10:13] = np.minimum(sample_hi[10:13],  0.3)
        quat_sigma = None  # uniform on S^3

    attempts = 0
    max_attempts = n_random * 5000 if value_filter_fn is not None else n_random * 500
    n_rejected_geom = 0
    n_rejected_brt  = 0

    while len(samples) < n and attempts < max_attempts:
        batch_size = min(n_random * 10, 5000)

        # Uniform sample for non-quaternion states
        batch = rng.uniform(sample_lo, sample_hi, size=(batch_size, 13))

        # Quaternion sampling
        if quat_sigma is not None and q_goal is not None:
            # Small perturbations around the goal quaternion
            q_rand = rng.randn(batch_size, 4) * quat_sigma + q_goal
        else:
            # Uniform on S^3
            q_rand = rng.randn(batch_size, 4)
        q_rand /= (np.linalg.norm(q_rand, axis=1, keepdims=True) + 1e-12)
        batch[:, 6:10] = q_rand

        batch_t = torch.tensor(batch, dtype=torch.float32, device=device)

        avoid_vals = dynamics.avoid_fn(batch_t).cpu().numpy()
        reach_vals = dynamics.reach_fn(batch_t).cpu().numpy()

        geom_valid = (avoid_vals > 0) & (reach_vals > 0)
        n_rejected_geom += int((~geom_valid).sum())

        if value_filter_fn is not None and geom_valid.any():
            # Apply BRAT filter only to geometrically valid candidates
            geom_batch = batch[geom_valid]
            values = value_filter_fn(geom_batch)
            brt_valid = values <= 0
            n_rejected_brt += int((~brt_valid).sum())
            accepted = geom_batch[brt_valid]
        else:
            accepted = batch[geom_valid]

        for s in accepted:
            if len(samples) >= n:
                break
            samples.append(s)
        attempts += batch_size

    if len(samples) < n:
        print(f"  WARNING: only found {len(samples)}/{n} valid ICs "
              f"(1 fixed + {len(samples)-1} random) "
              f"after {attempts:,} attempts.")

    total = attempts
    n_random_found = len(samples) - 1  # exclude the fixed IC
    if value_filter_fn is not None:
        accept_pct = 100 * n_random_found / max(total, 1)
        print(f"  IC sampling: 1 fixed + {total:,} checked | "
              f"{n_rejected_geom:,} reject(geom) | "
              f"{n_rejected_brt:,} reject(BRAT) | "
              f"{n_random_found} random accepted ({accept_pct:.2f}%)")
    else:
        print(f"  IC sampling: 1 fixed + {total:,} checked | "
              f"{n_rejected_geom:,} reject(geom) | "
              f"{n_random_found} random accepted")

    return np.array(samples[:n])

# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #

def compute_metrics(all_results):
    """Aggregate metrics from a list of result dicts.

    Docking  = reached goal without collision (docked & ~collision)
    Failure  = collision occurred
    Timeout  = never reached goal and no collision
    """
    n = len(all_results)
    if n == 0:
        return {}
    dockings   = sum(1 for r in all_results if r['success'])
    failures   = sum(1 for r in all_results if r['collision'])
    timeouts   = n - dockings - failures
    times      = [r['wall_time'] for r in all_results]

    # Control effort and docking time — only for successful (docking) runs
    docking_efforts = [r['control_effort'] for r in all_results if r['success']]
    docking_times = [r['times'][-1] for r in all_results if r['success']]

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
        'mean_dock_time':       float(np.mean(docking_times)) if docking_times else 0.0,
        'median_dock_time':     float(np.median(docking_times)) if docking_times else 0.0,
        'std_dock_time':        float(np.std(docking_times)) if docking_times else 0.0,
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
        'common_n'          -- size of the common-success set
        'total_n'           -- total ICs
        'per_controller'    -- {name: {median_dock_time, mean_dock_time,
                                      geo_mean_ratio, dock_times}}
        'baseline'          -- name of the baseline controller (first in list)
        'head_to_head'      -- {nameA: {nameB: win_fraction, ...}, ...}
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

    # Gather docking times and wall times on the common set
    dock_times = {}
    wall_times = {}
    for name in names:
        dock_times[name] = np.array([
            all_results[name][i]['times'][-1] for i in common_idxs
        ])
        wall_times[name] = np.array([
            all_results[name][i]['wall_time'] for i in common_idxs
        ])

    # Baseline for time-ratio computation (first controller)
    baseline = names[0]
    baseline_times = dock_times[baseline]

    for name in names:
        t = dock_times[name]
        w = wall_times[name]
        # Per-IC ratio relative to baseline
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(baseline_times > 0, t / baseline_times, 1.0)
        geo_mean_ratio = float(np.exp(np.mean(np.log(ratios))))

        result['per_controller'][name] = {
            'median_dock_time': float(np.median(t)),
            'mean_dock_time': float(np.mean(t)),
            'std_dock_time': float(np.std(t)),
            'geo_mean_ratio': geo_mean_ratio,
            'mean_wall_time': float(np.mean(w)),
            'std_wall_time': float(np.std(w)),
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
              f"{'Dock Time (s)':>18} {'Effort (dock)':>18} "
              f"{'Wall (s)':>14} "
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
            dock_t_str = f"{m['mean_dock_time']:.1f} +/- {m['std_dock_time']:.1f} ({n_dock})"
        else:
            effort_str = "N/A (0)"
            dock_t_str = "N/A (0)"
        wall_str   = f"{m['mean_wall_time']:.2f} +/- {m['std_wall_time']:.2f}"
        clip_tot   = m.get('total_clipped_steps', 0)
        clip_runs  = m.get('n_rollouts_with_clipping', 0)
        print(f"{name:<22} {m['docking_rate']*100:>6.1f}% "
              f"{m['failure_rate']*100:>6.1f}% "
              f"{m['timeout_rate']*100:>6.1f}% "
              f"{dock_t_str:>18} {effort_str:>18} "
              f"{wall_str:>14} "
              f"{clip_tot:>10} {clip_runs:>11}")
    print(sep + '\n')


def print_optimality_table(optimality):
    """Print a formatted docking-time optimality table."""
    cn = optimality['common_n']
    tn = optimality['total_n']
    baseline = optimality['baseline']

    header = (f"{'Controller':<22} {'Median(s)':>9} {'Mean(s)':>9} "
              f"{'Std(s)':>9} {'Ratio':>7} {'Wall(s)':>14}")
    sep = '-' * len(header)
    print('\n' + sep)
    print(f'DOCKING-TIME OPTIMALITY  (common-success set: '
          f'{cn}/{tn} ICs, baseline: {baseline})')
    print(sep)
    print(header)
    print(sep)
    for name, m in optimality['per_controller'].items():
        wall_str = f"{m['mean_wall_time']:.2f}+/-{m['std_wall_time']:.2f}"
        print(f"{name:<22} {m['median_dock_time']:>9.2f} "
              f"{m['mean_dock_time']:>9.2f} {m['std_dock_time']:>9.2f} "
              f"{m['geo_mean_ratio']:>7.3f} {wall_str:>14}")
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


def plot_metrics_bar(metrics_by_controller, save_path=None, optimality=None):
    """Grouped bar chart of comparison metrics (always 4 panels)."""
    import matplotlib.pyplot as plt

    names = list(metrics_by_controller.keys())
    n_ctrl = len(names)

    label_to_type = {v: k for k, v in CONTROLLER_LABELS.items()}
    colors = [CONTROLLER_COLORS.get(label_to_type.get(n, 'brt_13d'), '#1f77b4')
              for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    x = np.arange(n_ctrl)

    # Panel 1 — Rates
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

    # Panel 2 — Control effort (docking trajectories only)
    means = [metrics_by_controller[n]['mean_control_effort'] for n in names]
    stds  = [metrics_by_controller[n]['std_control_effort']  for n in names]
    axes[1].bar(x, means, 0.5, yerr=stds, color=colors, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    axes[1].set_ylabel('Control Effort')
    axes[1].set_title('Mean Control Effort (Docking Only)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, n in enumerate(names):
        n_dock = metrics_by_controller[n].get('n_docking_effort', 0)
        n_total = metrics_by_controller[n]['n']
        if n_dock > 0:
            axes[1].text(i, means[i] + stds[i] + 0.02 * max(max(means), 1),
                         f'n={n_dock}/{n_total}', ha='center', va='bottom',
                         fontsize=7)

    # Common-success set availability (used by Panels 3 & 4)
    has_opt = optimality and optimality['common_n'] > 0

    # Panel 3 — Computation wall time (common-success set when available)
    if has_opt:
        pc = optimality['per_controller']
        common_n = optimality['common_n']
        total_n = optimality['total_n']
        means = [pc[n]['mean_wall_time'] if n in pc else 0.0 for n in names]
        stds  = [pc[n]['std_wall_time']  if n in pc else 0.0 for n in names]
        axes[2].bar(x, means, 0.5, yerr=stds, color=colors, capsize=5)
        axes[2].set_title(f'Wall Time — Common Success Set (n={common_n}/{total_n})')
    else:
        means = [metrics_by_controller[n]['mean_wall_time'] for n in names]
        stds  = [metrics_by_controller[n]['std_wall_time']  for n in names]
        axes[2].bar(x, means, 0.5, yerr=stds, color=colors, capsize=5)
        axes[2].set_title('Mean Computation Wall Time per Rollout')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    axes[2].set_ylabel('Wall Time (s)')
    axes[2].grid(axis='y', alpha=0.3)

    # Panel 4 — Docking time (common-success set only)
    if has_opt:
        pc = optimality['per_controller']
        common_n = optimality['common_n']
        total_n = optimality['total_n']
        medians = [pc[n]['median_dock_time'] if n in pc else 0.0 for n in names]
        means   = [pc[n]['mean_dock_time']   if n in pc else 0.0 for n in names]
        stds    = [pc[n]['std_dock_time']     if n in pc else 0.0 for n in names]
        w = 0.3
        axes[3].bar(x - w/2, medians, w, label='Median', color='#66c2a5')
        axes[3].bar(x + w/2, means, w, yerr=stds, label='Mean ± std',
                    color=colors, capsize=5)
        axes[3].set_title(f'Docking Time — Common Success Set (n={common_n}/{total_n})')
        axes[3].legend(fontsize=8)
        # Annotate with geo-mean ratio
        for i, n in enumerate(names):
            if n in pc:
                ratio_str = f'ratio={pc[n]["geo_mean_ratio"]:.3f}'
                y_top = means[i] + stds[i]
                axes[3].text(i, y_top + 0.02 * max(max(means), 1),
                             ratio_str, ha='center', va='bottom', fontsize=7)
    else:
        # No common-success set — show per-controller info with "no data" markers
        axes[3].set_title('Docking Time — No Common Success Set')
        for i, n in enumerate(names):
            n_dock = metrics_by_controller[n].get('n_docking_effort', 0)
            n_total = metrics_by_controller[n]['n']
            if n_dock > 0:
                m = metrics_by_controller[n]
                axes[3].bar(i, m['mean_dock_time'], 0.5, yerr=m['std_dock_time'],
                            color=colors[i], capsize=5, alpha=0.4)
                axes[3].text(i, m['mean_dock_time'] + m['std_dock_time']
                             + 0.5, f'{n_dock}/{n_total} docked',
                             ha='center', va='bottom', fontsize=7)
            else:
                # No successes — draw a prominent "no docking" marker
                axes[3].bar(i, 0, 0.5, color='#d9d9d9', edgecolor='red',
                            linewidth=2, hatch='///')
                axes[3].text(i, 0.5, 'NO\nSUCCESS',
                             ha='center', va='bottom', fontsize=9,
                             fontweight='bold', color='red')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(names, fontsize=8, rotation=25, ha='right')
    axes[3].set_ylabel('Docking Time (s)')
    axes[3].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    return fig

# ------------------------------------------------------------------ #
#  Dynamics loader
# ------------------------------------------------------------------ #

def load_dynamics(checkpoint_path):
    experiment_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(checkpoint_path)))
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')
    with open(opt_path, 'rb') as f:
        orig_opt = pickle.load(f)
    dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
    sig = inspect.signature(dynamics_class)
    kwargs = {}
    for pn in sig.parameters:
        if pn != 'self' and hasattr(orig_opt, pn):
            kwargs[pn] = getattr(orig_opt, pn)
    dynamics = dynamics_class(**kwargs)
    dynamics.set_model(orig_opt.deepReach_model)
    if hasattr(orig_opt, 'state_range') and orig_opt.state_range is not None:
        dynamics.override_state_range(orig_opt.state_range)
    return dynamics

# ------------------------------------------------------------------ #
#  Sub-commands
# ------------------------------------------------------------------ #

def run_single(args):
    """Run one controller from a single initial condition."""
    os.makedirs(args.output_dir, exist_ok=True)

    ctrl_type = args.controller
    display = CONTROLLER_LABELS.get(ctrl_type, ctrl_type)
    _banner(f"{display} — Single Run")

    controller = build_controller(ctrl_type, args)
    dynamics = controller.dynamics

    # ---- Choose initial condition ------------------------------------
    ic_source = 'cli'
    ic_save_path = os.path.join(args.output_dir, 'last_ic.npy')
    sampling = getattr(args, 'sampling_method', 'default')
    if getattr(args, 'repeat', False):
        assert os.path.exists(ic_save_path), \
            f"--repeat: no saved IC found at {ic_save_path}"
        initial_state = np.load(ic_save_path)
        ic_source = 'repeat'
    elif args.initial_state is not None:
        initial_state = np.array(json.loads(args.initial_state), dtype=np.float64)
        assert len(initial_state) == 13, \
            f"initial_state must be length 13, got {len(initial_state)}"
    elif sampling == 'brt':
        ic_source = 'brt'
        print(f"\n  Sampling 1 IC from BRAT  (tMax={args.tMax}s) ...")
        # Use the controller's own checkpoint for value queries
        ckpt = (args.vanilla_checkpoint_path
                if ctrl_type == 'vanilla_brt_13d'
                else args.checkpoint_path)
        query_ctrl = BRTController13D(
            checkpoint_path=ckpt,
            tMax=args.tMax,
            device=args.device,
        )
        value_filter_fn = lambda states: query_ctrl.get_values_batch_states(
            states, args.tMax)
        seed = getattr(args, 'seed', 42)
        ics = sample_initial_conditions(
            dynamics, 1, device=args.device, seed=seed,
            value_filter_fn=value_filter_fn)
        assert len(ics) == 1, "Failed to find an IC inside the BRAT"
        initial_state = ics[0]
        v_val = value_filter_fn(initial_state[None])[0]
        print(f"  V(x, {args.tMax}) = {v_val:.4f}  (<= 0 ✓)")
    elif sampling == 'random':
        ic_source = 'random'
        seed = getattr(args, 'seed', 42)
        print(f"\n  Sampling 1 random IC outside failure set (seed={seed}) ...")
        ics = sample_initial_conditions(
            dynamics, 1, device=args.device, seed=seed,
            value_filter_fn=None)
        assert len(ics) == 1, "Failed to sample a valid IC"
        initial_state = ics[0]
    else:
        ic_source = 'default'
        q_goal = dynamics.q_goal.cpu().numpy()
        initial_state = np.array([
            10.0, -5.0, 2.0,
            0.0, 0.0, 0.0,
            q_goal[0], q_goal[1], q_goal[2], q_goal[3],
            0.0, 0.0, 0.0,
        ])

    print(f"\n  IC source : {ic_source}")
    print(f"  IC state  : {_fmt_state(initial_state)}")
    np.save(ic_save_path, initial_state)

    # ---- Run simulation ----------------------------------------------
    result = controller.simulate_docking(
        initial_state, max_sim_time=args.max_sim_time)

    # ---- Summary -----------------------------------------------------
    _banner("Result", char='-')
    outcome = ('DOCKED' if result['docked']
               else 'COLLISION' if result['collision']
               else 'TIMEOUT')
    traj = result['trajectory']
    # Distance to goal center (matches reach_fn / goal band definition)
    goal_center = np.array([0.0, dynamics.goal_y_center, 0.0])
    final_dist = float(np.linalg.norm(traj[-1, :3] - goal_center))
    final_qerr = float(_quat_error_angle_np(
        traj[-1, 6:10], dynamics.q_goal.cpu().numpy()))
    print(f"  Outcome       : {outcome}")
    print(f"  Sim time      : {result['times'][-1]:.1f} s  "
          f"({len(traj)} steps)")
    print(f"  Wall time     : {result['wall_time']:.1f} s")
    print(f"  Control effort: {result['control_effort']:.2f}")
    print(f"  Final dist    : {final_dist:.4f} m")
    print(f"  Final quat err: {np.degrees(final_qerr):.2f} deg")

    # PD torque summary
    n_pd = result.get('n_pd_torque_steps', 0)
    if n_pd > 0:
        print(f"  PD torque steps: {n_pd}/{len(traj)} "
              f"({100.0 * n_pd / max(len(traj), 1):.1f}%)")

    # Safety filter summary
    sf_log = result.get('safety_filter_log', [])
    if sf_log:
        active_steps = sum(1 for e in sf_log if e.get('filter_active', False))
        print(f"  Safety filter : {active_steps}/{len(sf_log)} steps active")

    # ---- Save --------------------------------------------------------
    np.save(os.path.join(args.output_dir, 'trajectory.npy'),
            result['trajectory'])
    np.save(os.path.join(args.output_dir, 'controls.npy'),
            result['controls'])
    print(f"\n  Saved to {args.output_dir}/")

    # ---- Static plots (always) ----------------------------------------
    print("  Generating static plots...")
    plot_trajectory_13d(
        result, dynamics,
        os.path.join(args.output_dir, 'trajectory.svg'))
    plot_states_13d(
        result, dynamics,
        os.path.join(args.output_dir, 'simulation_states.svg'))
    plot_controls_13d(
        result, dynamics,
        os.path.join(args.output_dir, 'simulation_controls.svg'))

    # ---- Animated visualisation (optional) ----------------------------
    if args.viz_html or args.viz_mp4:
        if hasattr(controller, 'model'):
            brt_viz = BRTVisualizer13D(controller, backend='plotly')
            anim = ControllerAnimation13D(
                brt_viz,
                grid_resolution=args.viz_resolution,
            )
            if args.viz_html:
                html_path = os.path.join(args.output_dir, 'animation.html')
                anim.generate_interactive_html(
                    result, html_path, max_frames=args.viz_max_frames)
            if args.viz_mp4:
                mp4_path = os.path.join(args.output_dir, 'animation.mp4')
                anim.generate_mp4(
                    result, mp4_path, fps=args.viz_fps,
                    max_frames=args.viz_max_frames)
        else:
            print("  (Pure MPC — using trajectory-only animation)")
            traj_anim = TrajectoryAnimation13D(dynamics)
            if args.viz_html:
                html_path = os.path.join(args.output_dir, 'animation.html')
                traj_anim.generate_interactive_html(
                    result, html_path, max_frames=args.viz_max_frames)
            if args.viz_mp4:
                mp4_path = os.path.join(args.output_dir, 'animation.mp4')
                traj_anim.generate_mp4(
                    result, mp4_path, fps=args.viz_fps,
                    max_frames=args.viz_max_frames)

    # ---- 2-D BRT slice GIFs (optional) --------------------------------
    if getattr(args, 'viz_gifs', False) and hasattr(controller, 'model'):
        brt_viz_for_gif = BRTVisualizer13D(controller, backend='plotly')
        slice_gif = SliceGIF13D(brt_viz_for_gif, dynamics)
        slice_gif.generate(
            result, args.output_dir,
            max_frames=args.viz_max_frames,
            resolution=getattr(args, 'viz_gif_resolution', 80),
            fps=getattr(args, 'viz_gif_fps', 5),
        )
        # Combined 4-panel GIF (3 slices + 3-D isometric)
        combined_gif = CombinedGIF13D(brt_viz_for_gif, dynamics)
        combined_gif.generate(
            result,
            os.path.join(args.output_dir, 'brt_combined.gif'),
            max_frames=args.viz_max_frames,
            resolution=getattr(args, 'viz_gif_resolution', 80),
            resolution_3d=min(30, getattr(args, 'viz_gif_resolution', 80) // 3),
            fps=getattr(args, 'viz_gif_fps', 5),
        )
        # Velocity-space 3-panel GIF
        vel_gif = VelocitySliceGIF13D(brt_viz_for_gif, dynamics)
        vel_gif.generate(
            result,
            os.path.join(args.output_dir, 'brt_velocity.gif'),
            max_frames=args.viz_max_frames,
            resolution=getattr(args, 'viz_gif_resolution', 80),
            fps=getattr(args, 'viz_gif_fps', 5),
        )
        # Attitude & angular-velocity 6-panel GIF
        att_gif = AttitudeSliceGIF13D(brt_viz_for_gif, dynamics)
        att_gif.generate(
            result,
            os.path.join(args.output_dir, 'brt_attitude.gif'),
            max_frames=args.viz_max_frames,
            resolution=getattr(args, 'viz_gif_resolution', 60),
            fps=getattr(args, 'viz_gif_fps', 5),
        )


def run_compare(args):
    """Run N rollouts per controller and compare."""
    os.makedirs(args.output_dir, exist_ok=True)

    sampling_label = getattr(args, 'sampling_method', 'uniform')
    _banner(f"13D Comparison  |  {len(args.controllers)} controllers  |  "
            f"{args.num_rollouts} rollouts  |  IC={sampling_label}")

    # ---- Load dynamics for IC sampling --------------------------------
    ckpt_for_dynamics = args.checkpoint_path or args.vanilla_checkpoint_path
    if ckpt_for_dynamics is None:
        raise ValueError("At least one of --checkpoint_path or "
                         "--vanilla_checkpoint_path is required")
    dynamics = load_dynamics(ckpt_for_dynamics)

    # ---- (Optional) BRAT value-function filter ------------------------
    value_filter_fn = None
    if sampling_label == 'brt':
        print(f"\n  Loading BRAT filter  (tMax={args.tMax}s) ...")
        query_ctrl = BRTController13D(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            device=args.device,
        )
        value_filter_fn = lambda states: query_ctrl.get_values_batch_states(
            states, args.tMax)
        print(f"  Filter ready — ICs will satisfy V(x, {args.tMax}) <= 0")

    # ---- Sample ICs ---------------------------------------------------
    print(f"\n  Sampling {args.num_rollouts} ICs  "
          f"(seed={args.seed}, method={sampling_label}) ...")
    ics = sample_initial_conditions(
        dynamics, args.num_rollouts, device=args.device, seed=args.seed,
        value_filter_fn=value_filter_fn)
    print(f"  {len(ics)} valid ICs obtained.")
    np.save(os.path.join(args.output_dir, 'initial_conditions.npy'), ics)

    # ---- Helper: build detail record for one rollout -------------------
    q_goal_np = (dynamics.q_goal.cpu().numpy()
                 if hasattr(dynamics, 'q_goal')
                 else np.array([1, 0, 0, 0]))
    goal_center = np.array([0.0, dynamics.goal_y_center, 0.0])

    def _detail(i, result):
        fs = result['final_state']
        q = fs[6:10] / (np.linalg.norm(fs[6:10]) + 1e-12)
        dot = np.clip(np.abs(np.dot(q, q_goal_np)), 0, 1)
        return {
            'rollout_idx': i,
            'initial_state': ics[i].tolist(),
            'final_state': fs.tolist(),
            'quat_err_deg': round(float(np.degrees(2 * np.arccos(dot))), 4),
            'final_dist': round(float(np.linalg.norm(fs[:3] - goal_center)), 4),
            'sim_time': round(float(result['times'][-1]), 4),
            'control_effort': round(float(result['control_effort']), 4),
            'wall_time': round(float(result['wall_time']), 4),
        }

    def _save_checkpoint(all_results_so_far, tag='checkpoint'):
        """Write comparison_results.json with whatever rollouts are done."""
        json_path = os.path.join(args.output_dir, 'comparison_results.json')
        json_data = {
            '_metadata': {
                'sampling_method': sampling_label,
                'num_rollouts': args.num_rollouts,
                'seed': args.seed,
                'tMax': args.tMax,
                'max_sim_time': args.max_sim_time,
                'checkpoint_path': args.checkpoint_path,
                'status': tag,
            }
        }
        for cn in all_results_so_far:
            disp = CONTROLLER_LABELS.get(cn, cn)
            results = all_results_so_far[cn]
            m = compute_metrics(results)
            entry = {k: (float(v) if isinstance(v, (np.floating, float))
                         else int(v)) for k, v in m.items()}
            colls, docks, touts = [], [], []
            for j, r in enumerate(results):
                d = _detail(j, r)
                if r['collision']:
                    colls.append(d)
                elif r['docked']:
                    docks.append(d)
                else:
                    touts.append(d)
            entry['collisions'] = colls
            entry['docked'] = docks
            entry['timeouts'] = touts
            json_data[disp] = entry
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

    # ---- Run each controller ------------------------------------------
    CHECKPOINT_INTERVAL = 100

    metrics_all = {}
    all_results = {}
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        _banner(f"{display}  ({len(ics)} rollouts)", char='-')

        controller = build_controller(ctrl_name, args)

        # Use batch simulation for MPC controllers when available
        if hasattr(controller, 'simulate_docking_batch'):
            results = controller.simulate_docking_batch(
                ics, max_sim_time=args.max_sim_time)
            for i, res in enumerate(results):
                tag = ('DOCK' if res['docked']
                       else 'COLL' if res['collision']
                       else 'TOUT')
                print(f"  [{i+1:>{len(str(len(ics)))}}/{len(ics)}] "
                      f"{tag}  t={res['times'][-1]:5.1f}s  "
                      f"effort={res['control_effort']:.1f}")
        else:
            results = []
            for i, ic in enumerate(ics):
                res = controller.simulate_docking(
                    ic, max_sim_time=args.max_sim_time)
                tag = ('DOCK' if res['docked']
                       else 'COLL' if res['collision']
                       else 'TOUT')
                print(f"  [{i+1:>{len(str(len(ics)))}}/{len(ics)}] "
                      f"{tag}  t={res['times'][-1]:5.1f}s  "
                      f"effort={res['control_effort']:.1f}  "
                      f"wall={res['wall_time']:.1f}s")
                results.append(res)

                # Periodic checkpoint every CHECKPOINT_INTERVAL rollouts
                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    all_results[ctrl_name] = results
                    _save_checkpoint(all_results,
                                     tag=f'checkpoint_{ctrl_name}_{i+1}')
                    print(f"  ** Checkpoint saved "
                          f"({i+1}/{len(ics)} rollouts)")

        all_results[ctrl_name] = results
        m = compute_metrics(results)
        metrics_all[display] = m
        print(f"\n{display}: dock={m['docking_rate']*100:.1f}%  "
              f"fail={m['failure_rate']*100:.1f}%  "
              f"timeout={m['timeout_rate']*100:.1f}%  "
              f"effort={m['mean_control_effort']:.1f}  "
              f"time={m['mean_wall_time']:.2f}s")

    # ---- Summary table ------------------------------------------------
    print_comparison_table(metrics_all)

    # ---- Docking-time optimality (paired comparison) ------------------
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

    # ---- Collect detailed per-rollout outcomes ------------------------
    detailed_by_name = {}
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        results = all_results[ctrl_name]

        collisions = []
        docked_list = []
        timeouts = []

        for i, result in enumerate(results):
            detail = _detail(i, result)

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

    # ---- Save collision ICs per controller as .npy --------------------
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

    # ---- Save timeout ICs per controller as .npy ----------------------
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        tout_details = detailed_by_name[display]['timeouts']
        if tout_details:
            timeout_ics = np.array([c['initial_state'] for c in tout_details])
            timeout_path = os.path.join(
                args.output_dir, f'timeout_ics_{ctrl_name}.npy')
            np.save(timeout_path, timeout_ics)
            print(f"  {display}: {len(tout_details)} timeout IC(s) "
                  f"saved to {timeout_path}")

    # ---- Print failure summary ----------------------------------------
    print('\n' + '-' * 60)
    print('  FAILURE SUMMARY')
    print('-' * 60)
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        coll = detailed_by_name[display]['collisions']
        tout = detailed_by_name[display]['timeouts']
        if coll:
            print(f"\n  {display}  ({len(coll)} collision(s)):")
            for c in coll:
                ic = c['initial_state']
                print(f"    rollout {c['rollout_idx']:>3d}  "
                      f"pos=({ic[0]:+.2f},{ic[1]:+.2f},{ic[2]:+.2f})  "
                      f"t_coll={c['sim_time']:.1f}s")
        if tout:
            print(f"\n  {display}  ({len(tout)} timeout(s)):")
            for c in tout:
                ic = c['initial_state']
                print(f"    rollout {c['rollout_idx']:>3d}  "
                      f"pos=({ic[0]:+.2f},{ic[1]:+.2f},{ic[2]:+.2f})  "
                      f"dist={c['final_dist']:.3f}m  "
                      f"q_err={c['quat_err_deg']:.1f}°")
        if not coll and not tout:
            print(f"\n  {display}: no failures")
    print('-' * 60)

    # ---- Build and save JSON --------------------------------------------
    json_path = os.path.join(args.output_dir, 'comparison_results.json')
    json_data = {
        '_metadata': {
            'sampling_method': sampling_label,
            'num_rollouts': args.num_rollouts,
            'seed': args.seed,
            'tMax': args.tMax,
            'max_sim_time': args.max_sim_time,
            'checkpoint_path': args.checkpoint_path,
        }
    }
    for k, v in metrics_all.items():
        json_data[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float))
                              else int(vv))
                         for kk, vv in v.items()}
        json_data[k].update(detailed_by_name[k])
    json_data['_docking_optimality'] = _to_jsonable(optimality)
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ---- Plots -----------------------------------------------------------
    bar_path = os.path.join(args.output_dir, 'metrics_comparison.png')
    plot_metrics_bar(metrics_all, save_path=bar_path, optimality=optimality)

    print(f"\nAll outputs saved to {args.output_dir}")
    print("Done.")

# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='Run 13D docking controllers.')
    subparsers = parser.add_subparsers(dest='mode')

    # --- Shared arguments -------------------------------------------- #
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model_final.pth (required for brt/mpc controllers)')
    parent.add_argument('--tMax', type=float, default=14.0)
    parent.add_argument('--dt', type=float, default=0.1)
    parent.add_argument('--device', type=str, default='cuda')
    parent.add_argument('--max_sim_time', type=float, default=60.0)
    parent.add_argument('--output_dir', type=str, default='./outputs/single_13d')

    # Safety filter arguments (for brt_safety_13d)
    parent.add_argument('--safety_filter_mode', type=int, default=1,
                        help='Safety filter mode: 0=off, 1=least-restrictive, 2=CBF-QP')
    parent.add_argument('--safety_checkpoint_path', type=str, default=None,
                        help='Path to avoid-only BRT checkpoint for safety filter')
    parent.add_argument('--safety_tMax', type=float, default=None,
                        help='tMax for safety BRT queries (None = use model default)')
    parent.add_argument('--safety_filter_margin', type=float, default=0.1,
                        help='Phase-1 activation threshold for BRAT controllers '
                             '(also used as the sole margin for single-phase '
                             'controllers unless --safety_filter_margin_docking '
                             'is set)')
    parent.add_argument('--safety_filter_margin_docking', type=float, default=0.02,
                        help='Safety margin for single-phase controllers '
                             '(mpc_13d, rl_13d) that lack a phase-2 transition. '
                             'Lower than phase-1 to allow docking while still '
                             'providing collision protection.')
    parent.add_argument('--safety_filter_gamma', type=float, default=0.2,
                        help='CBF decay rate for mode 2')

    # PD hybrid arguments
    parent.add_argument('--pd_torque_proximity', type=float, default=2.0,
                        help='Proximity multiplier for PD torque activation '
                             '(brt_pd_hybrid only). Default: 2.0')

    # MPC arguments 
    parent.add_argument('--planning_horizon', type=float, default=2.0)
    parent.add_argument('--mpc_dt', type=float, default=0.5)
    parent.add_argument('--effective_horizon', type=float, default=1.0)
    parent.add_argument('--gradient_iters', type=int, default=50,
                        help='Adam iterations per MPC step (default for both controllers)')
    parent.add_argument('--num_restarts', type=int, default=8,
                        help='Parallel random restarts (default for both controllers)')
    parent.add_argument('--gradient_lr', type=float, default=1.0,
                        help='Adam learning rate for gradient MPC')
    parent.add_argument('--goal_weight', type=float, default=0.01,
                        help='Weight for goal-directed regularisation')
    parent.add_argument('--effort_weight', type=float, default=0.0)
    parent.add_argument('--exploration_factor', type=float, default=3.0)
    parent.add_argument('--exploration_patience', type=int, default=2)
    parent.add_argument('--escape_thresh', type=float, default=0.5)

    # Per-controller MPC overrides (None = use shared defaults above)
    parent.add_argument('--mpc_gradient_iters', type=int, default=None,
                        help='Override gradient_iters for mpc_13d only')
    parent.add_argument('--mpc_num_restarts', type=int, default=None,
                        help='Override num_restarts for mpc_13d only')
    parent.add_argument('--mpc_terminal_gradient_iters', type=int, default=20,
                        help='Override gradient_iters for mpc_terminal_13d only')
    parent.add_argument('--mpc_terminal_num_restarts', type=int, default=1,
                        help='Override num_restarts for mpc_terminal_13d only')

    # Vanilla BRT arguments
    parent.add_argument('--vanilla_checkpoint_path', type=str, default=None,
                        help='Path to vanilla DeepReach checkpoint for '
                             'vanilla_brt_13d controller (no MPC supervision, '
                             'no gradient refinement).')

    # RL arguments
    parent.add_argument('--rl_checkpoint_path', type=str, default=None,
                        help='Path to trained RL Q-network .pth checkpoint.')
    parent.add_argument('--rl_architecture', type=int, nargs='+',
                        default=[256, 256],
                        help='Hidden layer dims for RL Q-network (must match training).')
    parent.add_argument('--rl_activation', type=str, default='Tanh',
                        help='Activation function for RL Q-network (must match training).')
    parent.add_argument('--rl_pd_attitude', action='store_true',
                        help='Use PD attitude controller for torques (27 force-only actions). '
                             'Must match training configuration.')

    # Viz arguments
    parent.add_argument('--viz_html', action='store_true',
                        help='Generate interactive HTML visualisation.')
    parent.add_argument('--viz_mp4', action='store_true',
                        help='Generate MP4 animation.')
    parent.add_argument('--viz_gifs', action='store_true',
                        help='Generate 2-D BRT slice GIFs (XY, XZ, YZ).')
    parent.add_argument('--viz_gif_resolution', type=int, default=80,
                        help='Grid resolution for slice GIFs.')
    parent.add_argument('--viz_gif_fps', type=int, default=5,
                        help='Frames per second for slice GIFs (default: 3).')
    parent.add_argument('--viz_resolution', type=int, default=40)
    parent.add_argument('--viz_max_frames', type=int, default=50)
    parent.add_argument('--viz_fps', type=int, default=10)

    # --- single ------------------------------------------------------ #
    sp_single = subparsers.add_parser('single', parents=[parent])
    sp_single.add_argument('--controller', type=str, required=True,
                           choices=['brt_13d', 'vanilla_brt_13d',
                                    'brt_safety_13d',
                                    'brt_pd_hybrid',
                                    'mpc_13d', 'mpc_terminal_13d',
                                    'rl_13d'])
    sp_single.add_argument('--initial_state', type=str, default=None,
                           help='JSON array of 13 floats, e.g. "[10,0,0,...]"')
    sp_single.add_argument('--sampling_method', type=str, default='default',
                           choices=['default', 'random', 'brt'],
                           help='IC sampling method when --initial_state is '
                                'not provided: "default" uses a hardcoded IC, '
                                '"random" samples outside failure set, '
                                '"brt" ensures V(x,tMax)<=0')
    sp_single.add_argument('--seed', type=int, default=42,
                           help='Random seed for IC sampling')
    sp_single.add_argument('--repeat', action='store_true',
                           help='Reuse the initial condition from the last '
                                'single run (saved in output_dir/last_ic.npy)')

    # --- compare ----------------------------------------------------- #
    sp_compare = subparsers.add_parser('compare', parents=[parent])
    sp_compare.add_argument('--controllers', nargs='+', required=True,
                            choices=['brt_13d', 'vanilla_brt_13d',
                                     'brt_safety_13d',
                                     'brt_pd_hybrid',
                                     'mpc_13d', 'mpc_terminal_13d',
                                     'rl_13d'])
    sp_compare.add_argument('--num_rollouts', type=int, default=20)
    sp_compare.add_argument('--seed', type=int, default=42)
    sp_compare.add_argument('--sampling_method', type=str, default='uniform',
                            choices=['uniform', 'brt'],
                            help='IC sampling method: "uniform" = geometric '
                                 'constraints only; "brt" = additionally '
                                 'require V(x, tMax) <= 0 (inside learned BRAT)')

    args = parser.parse_args()
    if args.mode == 'single':
        run_single(args)
    elif args.mode == 'compare':
        run_compare(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
