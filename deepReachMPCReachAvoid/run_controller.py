#!/usr/bin/env python3
"""
Unified controller runner for Docking6D.

Subcommands:
    single   -- Run one controller from a specified initial condition.
    compare  -- Run N rollouts per controller from random ICs and compare metrics.

Usage:
    python run_controller.py single  --controller brt   [options]
    python run_controller.py compare --controllers brt mpc mpc_terminal [options]
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
    BRTController, MPCController, MPCTerminalController,
    CascadedBRTController, CascadedMPCTerminalController,
)

# Generic (multi-controller) visualisation
from utils.controllers.controller_animation import (
    animate_trajectories,
    plot_trajectories_static,
    plot_simulation_data_multi,
    CONTROLLER_LABELS,
    CONTROLLER_COLORS,
)

# BRT-specific visualisation (value-function heatmap animation)
from utils.controllers.brt_animation import (
    create_deepreach_animation,
    create_cascaded_deepreach_animation,
    create_mpc_terminal_animation,
    create_cascaded_mpc_terminal_animation,
    plot_trajectory_static as brt_plot_trajectory_static,
    plot_simulation_data   as brt_plot_simulation_data,
)

from dynamics import dynamics as dynamics_module



def build_controller(name, args):
    """Instantiate a controller by name string."""
    # Validate inner checkpoint for cascaded controllers
    if name.startswith('cascaded') and not args.inner_checkpoint_path:
        raise ValueError(
            f"Controller '{name}' requires --inner_checkpoint_path but none was provided.")

    if name == 'brt':
        return BRTController(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
        )
    elif name == 'mpc':
        return MPCController(
            checkpoint_path=args.checkpoint_path,
            planning_horizon_sec=args.planning_horizon,
            mpc_dt=args.mpc_dt,
            dt=args.dt,
            num_samples=args.num_samples,
            num_refinement=args.num_refinement,
            device=args.device,
        )
    elif name == 'mpc_terminal':
        return MPCTerminalController(
            checkpoint_path=args.checkpoint_path,
            effective_horizon_sec=args.effective_horizon,
            tMax=args.tMax,
            dt=args.dt,
            num_samples=args.num_samples,
            num_refinement=args.num_refinement,
            device=args.device,
            effort_weight=args.effort_weight,
            exploration_factor=args.exploration_factor,
            exploration_patience=args.exploration_patience,
            escape_thresh=args.escape_thresh,
        )
    elif name == 'cascaded_brt':
        return CascadedBRTController(
            outer_checkpoint=args.checkpoint_path,
            inner_checkpoint=args.inner_checkpoint_path,
            outer_tMax=args.tMax,
            inner_tMax=args.inner_tMax,
            dt=args.dt,
            device=args.device,
        )
    elif name == 'cascaded_mpc_terminal':
        return CascadedMPCTerminalController(
            outer_checkpoint=args.checkpoint_path,
            inner_checkpoint=args.inner_checkpoint_path,
            effective_horizon_sec=args.effective_horizon,
            outer_tMax=args.tMax,
            inner_tMax=args.inner_tMax,
            dt=args.dt,
            num_samples=args.num_samples,
            num_refinement=args.num_refinement,
            device=args.device,
            effort_weight=args.effort_weight,
            exploration_factor=args.exploration_factor,
            exploration_patience=args.exploration_patience,
            escape_thresh=args.escape_thresh,
        )
    else:
        raise ValueError(f"Unknown controller: {name}")

SAMPLING_STATE_RANGE = np.array([
    [-13.0, 13.0],   # px  (m)
    [-13.0, 13.0],   # py  (m)
    [ -1.0,  1.0],   # vx  (m/s)
    [ -1.0,  1.0],   # vy  (m/s)
    [-np.pi, np.pi],  # theta (rad)  -- full range is fine
    [ -0.75,  0.75],   # omega (rad/s)
])

def sample_initial_conditions(dynamics, n, device='cuda', seed=42,
                              value_filter_fn=None):
    """
    Sample *n* valid initial conditions uniformly from a feasible
    sub-region of the state space.

    The sampling bounds are the intersection of ``SAMPLING_STATE_RANGE``
    (hard-coded above) and the dynamics' ``state_range_``, so we never
    exceed the model's training domain.

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
        value_filter_fn: Optional callable  (N,6) np.array -> (N,) np.array
            returning V(x, tMax) for each state.  States with V <= 0 are
            kept (inside the BRAT).  ``None`` disables this filter.
    """
    rng = np.random.RandomState(seed)

    # Intersect SAMPLING_STATE_RANGE with the dynamics' state_range_
    dyn_lo = dynamics.state_range_[:, 0].cpu().numpy().astype(np.float64)
    dyn_hi = dynamics.state_range_[:, 1].cpu().numpy().astype(np.float64)
    state_lo = np.maximum(dyn_lo, SAMPLING_STATE_RANGE[:, 0])
    state_hi = np.minimum(dyn_hi, SAMPLING_STATE_RANGE[:, 1])

    samples = []
    attempts = 0
    max_attempts = n * 500  # increase headroom for stricter BRT filter
    n_rejected_geom = 0
    n_rejected_brt  = 0

    while len(samples) < n and attempts < max_attempts:
        batch_size = min(n * 10, 5000)
        batch = rng.uniform(state_lo, state_hi, size=(batch_size, 6))
        batch_t = torch.tensor(batch, dtype=torch.float32, device=device)

        avoid_vals = dynamics.avoid_fn(batch_t).cpu().numpy()
        reach_vals = dynamics.reach_fn(batch_t).cpu().numpy()

        geom_valid = (avoid_vals > 0) & (reach_vals > 0)
        n_rejected_geom += int((~geom_valid).sum())

        if value_filter_fn is not None and geom_valid.any():
            # Apply BRT filter only to geometrically valid candidates
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
        print(f"WARNING: only sampled {len(samples)}/{n} valid ICs "
              f"after {attempts} attempts.")

    if value_filter_fn is not None:
        total_checked = attempts
        print(f"  IC sampling stats:  checked={total_checked}  "
              f"rejected_geom={n_rejected_geom}  rejected_brt={n_rejected_brt}  "
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
    }


def print_comparison_table(metrics_by_controller):
    """Print a formatted comparison table to stdout."""
    header = (f"{'Controller':<22} {'Dock%':>7} {'Fail%':>7} {'Time%':>7} "
              f"{'Effort (dock)':>18} {'Time (s)':>14}")
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
        print(f"{name:<22} {m['docking_rate']*100:>6.1f}% "
              f"{m['failure_rate']*100:>6.1f}% "
              f"{m['timeout_rate']*100:>6.1f}% "
              f"{effort_str:>18} {time_str:>14}")
    print(sep + '\n')


def plot_metrics_bar(metrics_by_controller, save_path=None):
    """Grouped bar chart of comparison metrics."""
    names = list(metrics_by_controller.keys())
    n_ctrl = len(names)

    label_to_type = {v: k for k, v in CONTROLLER_LABELS.items()}
    colors = [CONTROLLER_COLORS.get(label_to_type.get(n, 'brt'), '#1f77b4')
              for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
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
    axes[2].set_title('Mean Runtime per Rollout')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
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

    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: checkpoint not found: {args.checkpoint_path}")
        return

    ctrl_type = args.controller
    display = CONTROLLER_LABELS.get(ctrl_type, ctrl_type)

    print('=' * 60)
    print(f'{display} Controller -- Single Run')
    print('=' * 60)

    # Build controller
    controller = build_controller(ctrl_type, args)
    dynamics = controller.dynamics

    # Initial state
    initial_state = np.array([
        args.initial_px,  args.initial_py,
        args.initial_vx,  args.initial_vy,
        args.initial_theta, args.initial_omega,
    ])
    print(f"\nInitial state: px={initial_state[0]:.2f}, py={initial_state[1]:.2f}, "
          f"vx={initial_state[2]:.2f}, vy={initial_state[3]:.2f}, "
          f"theta={initial_state[4]:.2f}, omega={initial_state[5]:.2f}")

    # BRT-specific pre-check
    if ctrl_type == 'brt':
        v0 = controller.get_value(initial_state, args.tMax)
        print(f"Initial V(x, tMax={args.tMax}s) = {v0:.4f}")
        print(f"State is {'INSIDE' if v0 <= 0 else 'OUTSIDE'} the BRT")

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

    if ctrl_type == 'brt':
        if result.get('phase_transition_time') is not None:
            print(f"Entered BRT (Phase 2) at t={result['phase_transition_time']:.2f}s")
        else:
            print("Never entered BRT (stayed in Phase 1)")
        phases = result.get('phases', np.array([]))
        if len(phases) > 0:
            print(f"Time in Phase 1: {np.sum(phases == 1) * args.dt:.1f}s")
            print(f"Time in Phase 2: {np.sum(phases == 2) * args.dt:.1f}s")

    if ctrl_type == 'mpc_terminal':
        phases = result.get('phases', np.array([]))
        if len(phases) > 0:
            print(f"Time in Phase 1: {np.sum(phases == 1) * args.dt:.1f}s")
            print(f"Time in Phase 2: {np.sum(phases == 2) * args.dt:.1f}s")
        if result.get('brt_entry_time') is not None:
            print(f"Entered BRT (Phase 2) at t={result['brt_entry_time']:.2f}s")
        else:
            print("Never entered BRT (stayed in Phase 1)")

    if ctrl_type == 'cascaded_mpc_terminal':
        phases = result.get('phases', np.array([]))
        if len(phases) > 0:
            print(f"Time in Phase 1: {np.sum(phases == 1) * args.dt:.1f}s")
            print(f"Time in Phase 2: {np.sum(phases == 2) * args.dt:.1f}s")
            print(f"Time in Phase 3: {np.sum(phases == 3) * args.dt:.1f}s")
        if result.get('outer_entry_time') is not None:
            print(f"Entered outer BRT (Phase 2) at t={result['outer_entry_time']:.2f}s")
        if result.get('inner_entry_time') is not None:
            print(f"Entered inner BRT (Phase 3) at t={result['inner_entry_time']:.2f}s")

    # ---- Generate outputs ----
    print('\n' + '=' * 60)
    print('GENERATING OUTPUTS')
    print('=' * 60)

    # Generic static plots (all controller types)
    result_dict = {display: result}

    traj_path = os.path.join(args.output_dir, 'trajectory.png')
    print(f"Generating trajectory plot: {traj_path}")
    plot_trajectories_static(result_dict, dynamics, save_path=traj_path)

    data_path = os.path.join(args.output_dir, 'simulation_data.png')
    print(f"Generating data plots: {data_path}")
    plot_simulation_data_multi(result_dict, save_path=data_path)

    # Animation
    if not args.no_animation:
        if ctrl_type == 'brt' and not args.no_value_function:
            # BRT-specific animation with value-function heatmap
            anim_path = os.path.join(args.output_dir, 'docking_animation.mp4')
            print(f"Generating BRT animation (with value function): {anim_path}")
            create_deepreach_animation(
                controller, result, anim_path,
                skip_frames=args.skip_frames,
                resolution=args.resolution,
                show_value_function=True,
            )
        elif ctrl_type == 'cascaded_brt' and not args.no_value_function:
            # Cascaded BRT animation with phase-aware value function
            anim_path = os.path.join(args.output_dir, 'docking_animation.mp4')
            print(f"Generating Cascaded BRT animation (with value function): {anim_path}")
            create_cascaded_deepreach_animation(
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
        elif ctrl_type == 'cascaded_mpc_terminal' and not args.no_value_function:
            # Cascaded MPC+Terminal animation with 3-phase value function
            anim_path = os.path.join(args.output_dir, 'docking_animation.mp4')
            print(f"Generating Cascaded MPC+Terminal animation (with value function): {anim_path}")
            create_cascaded_mpc_terminal_animation(
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

    # Optionally build a value-function filter for BRT-based IC sampling
    value_filter_fn = None
    if getattr(args, 'sampling_method', 'uniform') == 'brt':
        print(f"\nLoading model for BRT IC filtering (tMax={args.tMax}) ...")
        query_ctrl = BRTController(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            device=args.device,
        )
        value_filter_fn = lambda states: query_ctrl.get_values_batch_states(
            states, args.tMax)
        print(f"  BRT filter ready — ICs will satisfy V(x, {args.tMax}) <= 0")

    # Sample initial conditions
    sampling_label = getattr(args, 'sampling_method', 'uniform')
    print(f"\nSampling {args.n_rollouts} initial conditions "
          f"(seed={args.seed}, method={sampling_label}) ...")
    ics = sample_initial_conditions(
        dynamics, args.n_rollouts, device=args.device, seed=args.seed,
        value_filter_fn=value_filter_fn)
    print(f"Sampled {len(ics)} valid ICs.\n")

    ic_path = os.path.join(args.output_dir, 'initial_conditions.npy')
    np.save(ic_path, ics)
    print(f"Initial conditions saved to {ic_path}")

    # Build controllers
    controllers = {}
    for name in args.controllers:
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
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Results saved to {json_path}")

    bar_path = os.path.join(args.output_dir, 'metrics_comparison.png')
    plot_metrics_bar(metrics_by_name, save_path=bar_path)

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
                        help='BRT / terminal cost time horizon')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Control / integration timestep (s)')
    parser.add_argument('--max_sim_time', type=float, default=30.0,
                        help='Maximum simulation time (s)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device')
    # Cascaded controller args
    parser.add_argument('--inner_checkpoint_path', type=str, default=None,
                        help='Path to inner (short-horizon) model checkpoint (cascaded controllers)')
    parser.add_argument('--inner_tMax', type=float, default=3.0,
                        help='Inner BRT time horizon (cascaded controllers)')
    # MPC-specific
    parser.add_argument('--planning_horizon', type=float, default=3.0,
                        help='MPC-only planning horizon (s)')
    parser.add_argument('--mpc_dt', type=float, default=0.5,
                        help='MPC planning timestep (s); simulation uses --dt')
    parser.add_argument('--effective_horizon', type=float, default=3.0,
                        help='MPC+Terminal effective horizon (s)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='MPC random-shooting samples')
    parser.add_argument('--num_refinement', type=int, default=10,
                        help='MPC iterative refinement passes')
    parser.add_argument('--effort_weight', type=float, default=0.0,
                        help='Control effort penalty weight for MPC terminal '
                             'controllers (0.0 = disabled). Adds '
                             'effort_weight * sum(||u||*dt) to the combined '
                             'cost. Recommended range: 0.001 - 0.05.')
    # Graduated stagnation-escape tuning (MPC+Terminal controllers)
    parser.add_argument('--exploration_factor', type=float, default=3.0,
                        help='Multiplier for MPC eps_var when in EXPLORING '
                             'mode (default 3.0)')
    parser.add_argument('--exploration_patience', type=int, default=1,
                        help='Number of stagnation windows (each ~5 s) in '
                             'EXPLORING mode before switching to BRT '
                             'fallback (default 1)')
    parser.add_argument('--escape_thresh', type=float, default=0.5,
                        help='Distance improvement (m) from stagnation entry '
                             'required to declare local min escaped and '
                             'return to NORMAL mode (default 0.5)')
    # Animation
    parser.add_argument('--skip_frames', type=int, default=5,
                        help='Frames to skip in animation')


def main():
    parser = argparse.ArgumentParser(
        description='Docking6D Controller Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_controller.py single  --controller brt\n"
            "  python run_controller.py compare --controllers brt mpc mpc_terminal\n"
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ---- single ----
    sp_single = subparsers.add_parser(
        'single', help='Run one controller from a specified initial condition')
    _add_shared_args(sp_single)
    sp_single.add_argument(
        '--controller', type=str, default='brt',
        choices=['brt', 'mpc', 'mpc_terminal', 'cascaded_brt', 'cascaded_mpc_terminal'],
        help='Controller type to run')
    # Initial state
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
                           help='Skip value-function heatmap in BRT animation')
    sp_single.add_argument('--resolution', type=int, default=40,
                           help='Value-function grid resolution (BRT only)')

    # ---- compare ----
    sp_compare = subparsers.add_parser(
        'compare', help='Compare controllers over N rollouts from random ICs')
    _add_shared_args(sp_compare)
    sp_compare.add_argument(
        '--controllers', type=str, nargs='+',
        default=['brt', 'mpc', 'mpc_terminal'],
        choices=['brt', 'mpc', 'mpc_terminal', 'cascaded_brt', 'cascaded_mpc_terminal'],
        help='Controllers to compare')
    sp_compare.add_argument('--n_rollouts', type=int, default=50,
                            help='Number of rollouts per controller')
    sp_compare.add_argument('--seed', type=int, default=1,
                            help='Random seed for IC sampling')
    sp_compare.add_argument('--sampling_method', type=str, default='uniform',
                            choices=['uniform', 'brt'],
                            help='IC sampling method: "uniform" = geometric '
                                 'constraints only; "brt" = additionally '
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
