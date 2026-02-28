#!/usr/bin/env python3
"""
Unified controller runner for Docking13D.

Subcommands:
    single   -- Run one controller from a specified initial condition.
    compare  -- Run N rollouts per controller from random ICs and compare.

Usage:
    python run_controller_13d.py single  --controller brt_13d   [options]
    python run_controller_13d.py compare --controllers brt_13d mpc_terminal_13d [options]
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.controllers import (
    BRTController13D, MPCController13D, MPCTerminalController13D,
)
from utils.brt_visualization_13d import BRTVisualizer13D
from utils.controllers.controller_animation_13d import ControllerAnimation13D
from utils.controllers.trajectory_only_animation_13d import TrajectoryAnimation13D
from utils.controllers.docking13d_mixin import _quat_error_angle_np
from utils.controllers.static_plots_13d import (
    plot_trajectory_13d, plot_states_13d, plot_controls_13d,
)

from dynamics import dynamics as dynamics_module

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

CONTROLLER_LABELS = {
    'brt_13d':          'BRT 13D',
    'mpc_13d':          'MPC 13D',
    'mpc_terminal_13d': 'MPC+Terminal 13D',
}

CONTROLLER_COLORS = {
    'brt_13d':          '#1f77b4',
    'mpc_13d':          '#ff7f0e',
    'mpc_terminal_13d': '#2ca02c',
}

# ------------------------------------------------------------------ #
#  Builder
# ------------------------------------------------------------------ #

def build_controller(name, args):
    """Instantiate a 13D controller by name string."""
    if name == 'brt_13d':
        return BRTController13D(
            checkpoint_path=args.checkpoint_path,
            tMax=args.tMax,
            dt=args.dt,
            device=args.device,
        )
    elif name == 'mpc_13d':
        return MPCController13D(
            checkpoint_path=args.checkpoint_path,
            planning_horizon_sec=args.planning_horizon,
            mpc_dt=args.mpc_dt,
            dt=args.dt,
            num_samples=args.num_samples,
            num_refinement=args.num_refinement,
            device=args.device,
        )
    elif name == 'mpc_terminal_13d':
        return MPCTerminalController13D(
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
    else:
        raise ValueError(f"Unknown controller: {name}")

# ------------------------------------------------------------------ #
#  Initial-condition sampling
# ------------------------------------------------------------------ #

def sample_initial_conditions(dynamics, n, device='cuda', seed=42):
    """Sample *n* valid 13D initial conditions.

    Filters out states that are already docked (reach_fn <= 0) or
    inside the failure set (avoid_fn <= 0).
    """
    rng = np.random.RandomState(seed)

    # Sampling bounds: subset of training range
    dyn_lo = dynamics.state_range_[:, 0].cpu().numpy().astype(np.float64)
    dyn_hi = dynamics.state_range_[:, 1].cpu().numpy().astype(np.float64)

    # Tighten position range for feasibility
    sample_lo = dyn_lo.copy()
    sample_hi = dyn_hi.copy()
    sample_lo[:3] = np.maximum(sample_lo[:3], -13.0)
    sample_hi[:3] = np.minimum(sample_hi[:3],  13.0)
    # Tighten velocity
    sample_lo[3:6] = np.maximum(sample_lo[3:6], -0.15)
    sample_hi[3:6] = np.minimum(sample_hi[3:6],  0.15)
    # Tighten omega
    sample_lo[10:13] = np.maximum(sample_lo[10:13], -0.3)
    sample_hi[10:13] = np.minimum(sample_hi[10:13],  0.3)

    samples = []
    attempts = 0
    max_attempts = n * 500

    while len(samples) < n and attempts < max_attempts:
        batch_size = min(n * 10, 5000)

        # Uniform sample for non-quaternion states
        batch = rng.uniform(sample_lo, sample_hi, size=(batch_size, 13))

        # Quaternion: sample uniformly on S^3 then normalize
        q_rand = rng.randn(batch_size, 4)
        q_rand /= (np.linalg.norm(q_rand, axis=1, keepdims=True) + 1e-12)
        batch[:, 6:10] = q_rand

        batch_t = torch.tensor(batch, dtype=torch.float32, device=device)

        avoid_vals = dynamics.avoid_fn(batch_t).cpu().numpy()
        reach_vals = dynamics.reach_fn(batch_t).cpu().numpy()

        valid = (avoid_vals > 0) & (reach_vals > 0)
        for s in batch[valid]:
            if len(samples) >= n:
                break
            samples.append(s)
        attempts += batch_size

    if len(samples) < n:
        print(f"WARNING: only sampled {len(samples)}/{n} valid ICs "
              f"after {attempts} attempts.")

    return np.array(samples[:n])

# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #

def compute_metrics(all_results):
    n = len(all_results)
    if n == 0:
        return {}
    goals      = sum(1 for r in all_results if r['docked'])
    collisions = sum(1 for r in all_results if r['collision'])
    successes  = sum(1 for r in all_results if r['success'])
    times_w    = [r['wall_time'] for r in all_results]
    success_efforts = [r['control_effort'] for r in all_results if r['success']]
    return {
        'n':                   n,
        'goal_rate':           goals / n,
        'collision_rate':      collisions / n,
        'success_rate':        successes / n,
        'mean_control_effort': float(np.mean(success_efforts)) if success_efforts else 0.0,
        'std_control_effort':  float(np.std(success_efforts))  if success_efforts else 0.0,
        'n_success_effort':    len(success_efforts),
        'mean_wall_time':      float(np.mean(times_w)),
        'std_wall_time':       float(np.std(times_w)),
    }


def print_comparison_table(metrics_by_controller):
    header = (f"{'Controller':<22} {'Goal%':>7} {'Coll%':>7} {'Succ%':>7} "
              f"{'Effort (succ)':>18} {'Time (s)':>14}")
    sep = '-' * len(header)
    print('\n' + sep)
    print('13D CONTROLLER COMPARISON')
    print(sep)
    print(header)
    print(sep)
    for name, m in metrics_by_controller.items():
        n_s = m.get('n_success_effort', 0)
        effort_str = (f"{m['mean_control_effort']:.1f}"
                      f" +/- {m['std_control_effort']:.1f} ({n_s})"
                      if n_s > 0 else "N/A (0)")
        time_str = f"{m['mean_wall_time']:.2f} +/- {m['std_wall_time']:.2f}"
        print(f"{name:<22} {m['goal_rate']*100:>6.1f}% "
              f"{m['collision_rate']*100:>6.1f}% "
              f"{m['success_rate']*100:>6.1f}% "
              f"{effort_str:>18} {time_str:>14}")
    print(sep + '\n')

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
    """Run one controller from a specified initial condition."""
    os.makedirs(args.output_dir, exist_ok=True)

    ctrl_type = args.controller
    display = CONTROLLER_LABELS.get(ctrl_type, ctrl_type)
    print('=' * 60)
    print(f'{display} Controller -- Single Run (13D)')
    print('=' * 60)

    controller = build_controller(ctrl_type, args)
    dynamics = controller.dynamics

    # Initial state: either supplied via CLI or a default
    if args.initial_state is not None:
        initial_state = np.array(json.loads(args.initial_state), dtype=np.float64)
        assert len(initial_state) == 13, \
            f"initial_state must be length 13, got {len(initial_state)}"
    else:
        # Default: 10m away in x, at goal attitude, at rest
        q_goal = dynamics.q_goal.cpu().numpy()
        initial_state = np.array([
            10.0, -5.0, 2.0,      # position
            0.0, 0.0, 0.0,        # velocity
            q_goal[0], q_goal[1], q_goal[2], q_goal[3],  # quaternion
            0.0, 0.0, 0.0,        # omega
        ])

    print(f"\nInitial state (13D): {initial_state}")

    # Run simulation
    result = controller.simulate_docking(
        initial_state, max_sim_time=args.max_sim_time)

    # Print summary
    print(f"\nResult: docked={result['docked']}, "
          f"collision={result['collision']}, "
          f"effort={result['control_effort']:.2f}, "
          f"wall_time={result['wall_time']:.2f}s")

    # Save trajectory
    np.save(os.path.join(args.output_dir, 'trajectory.npy'),
            result['trajectory'])
    np.save(os.path.join(args.output_dir, 'controls.npy'),
            result['controls'])

    # --- Static plots (always generated) ------------------------------ #
    print("\n[Viz] Generating static plots...")
    plot_trajectory_13d(
        result, dynamics,
        os.path.join(args.output_dir, 'trajectory.png'))
    plot_states_13d(
        result, dynamics,
        os.path.join(args.output_dir, 'simulation_states.png'))
    plot_controls_13d(
        result, dynamics,
        os.path.join(args.output_dir, 'simulation_controls.png'))

    # --- Animated visualisation (optional) ----------------------------- #
    if args.viz_html or args.viz_mp4:
        # BRT and MPC-terminal controllers have a model; MPC-only does not
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
            # MPC-only: lightweight trajectory animation (no BRT)
            print("[Viz] Pure MPC controller — using trajectory-only animation.")
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


def run_compare(args):
    """Run N rollouts per controller and compare."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dynamics for initial-condition sampling
    dynamics = load_dynamics(args.checkpoint_path)
    ics = sample_initial_conditions(
        dynamics, args.num_rollouts, device=args.device, seed=args.seed)
    np.save(os.path.join(args.output_dir, 'initial_conditions.npy'), ics)
    print(f"Sampled {len(ics)} initial conditions  (seed={args.seed})")

    metrics_all = {}
    for ctrl_name in args.controllers:
        display = CONTROLLER_LABELS.get(ctrl_name, ctrl_name)
        print(f"\n{'='*50}")
        print(f"Running: {display}  ({len(ics)} rollouts)")
        print('='*50)

        controller = build_controller(ctrl_name, args)
        results = []
        for i, ic in enumerate(ics):
            print(f"\n--- Rollout {i+1}/{len(ics)} ---")
            res = controller.simulate_docking(
                ic, max_sim_time=args.max_sim_time)
            results.append(res)

        m = compute_metrics(results)
        metrics_all[display] = m

    print_comparison_table(metrics_all)

    # Save
    json_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"Metrics saved to {json_path}")

# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='Run 13D docking controllers.')
    subparsers = parser.add_subparsers(dest='mode')

    # --- Shared arguments -------------------------------------------- #
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model_final.pth')
    parent.add_argument('--tMax', type=float, default=14.0)
    parent.add_argument('--dt', type=float, default=0.1)
    parent.add_argument('--device', type=str, default='cuda')
    parent.add_argument('--max_sim_time', type=float, default=60.0)
    parent.add_argument('--output_dir', type=str, default='./outputs/single_13d')

    # MPC arguments
    parent.add_argument('--planning_horizon', type=float, default=20.0)
    parent.add_argument('--mpc_dt', type=float, default=0.5)
    parent.add_argument('--effective_horizon', type=float, default=2.0)
    parent.add_argument('--num_samples', type=int, default=500)
    parent.add_argument('--num_refinement', type=int, default=10)
    parent.add_argument('--effort_weight', type=float, default=0.0)
    parent.add_argument('--exploration_factor', type=float, default=3.0)
    parent.add_argument('--exploration_patience', type=int, default=2)
    parent.add_argument('--escape_thresh', type=float, default=0.5)

    # Viz arguments
    parent.add_argument('--viz_html', action='store_true',
                        help='Generate interactive HTML visualisation.')
    parent.add_argument('--viz_mp4', action='store_true',
                        help='Generate MP4 animation.')
    parent.add_argument('--viz_resolution', type=int, default=40)
    parent.add_argument('--viz_max_frames', type=int, default=50)
    parent.add_argument('--viz_fps', type=int, default=10)

    # --- single ------------------------------------------------------ #
    sp_single = subparsers.add_parser('single', parents=[parent])
    sp_single.add_argument('--controller', type=str, required=True,
                           choices=['brt_13d', 'mpc_13d', 'mpc_terminal_13d'])
    sp_single.add_argument('--initial_state', type=str, default=None,
                           help='JSON array of 13 floats, e.g. "[10,0,0,...]"')

    # --- compare ----------------------------------------------------- #
    sp_compare = subparsers.add_parser('compare', parents=[parent])
    sp_compare.add_argument('--controllers', nargs='+', required=True,
                            choices=['brt_13d', 'mpc_13d', 'mpc_terminal_13d'])
    sp_compare.add_argument('--num_rollouts', type=int, default=20)
    sp_compare.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if args.mode == 'single':
        run_single(args)
    elif args.mode == 'compare':
        run_compare(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
