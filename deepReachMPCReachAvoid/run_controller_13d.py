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
    SafetyFilter,
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
    'brt_safety_13d':   'BRT+Safety 13D',
    'brt_pd_hybrid':    'BRT+PD Hybrid 13D',
    'mpc_13d':          'MPC 13D',
    'mpc_terminal_13d': 'MPC+Terminal 13D',
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

def build_controller(name, args):
    """Instantiate a 13D controller by name string."""
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

def sample_initial_conditions(dynamics, n, device='cuda', seed=42,
                              value_filter_fn=None):
    """Sample *n* valid 13D initial conditions.

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

    samples = []
    attempts = 0
    max_attempts = n * 5000 if value_filter_fn is not None else n * 500
    n_rejected_geom = 0
    n_rejected_brt  = 0

    while len(samples) < n and attempts < max_attempts:
        batch_size = min(n * 10, 5000)

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
              f"after {attempts:,} attempts.")

    total = attempts
    if value_filter_fn is not None:
        accept_pct = 100 * len(samples) / max(total, 1)
        print(f"  IC sampling: {total:,} checked | "
              f"{n_rejected_geom:,} reject(geom) | "
              f"{n_rejected_brt:,} reject(BRAT) | "
              f"{len(samples)} accepted ({accept_pct:.2f}%)")
    else:
        print(f"  IC sampling: {total:,} checked | "
              f"{n_rejected_geom:,} reject(geom) | "
              f"{len(samples)} accepted")

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
    """Print a nicely formatted comparison table to stdout."""
    col_w = 80
    sep = '-' * col_w

    print(f"\n{sep}")
    print(f"  {'13D CONTROLLER COMPARISON':^{col_w - 4}}")
    print(sep)
    print(f"  {'Controller':<20} {'Goal':>6} {'Coll':>6} {'Succ':>6} "
          f"{'Effort (succ)':>18} {'Wall time':>16}")
    print(sep)

    for name, m in metrics_by_controller.items():
        if not m:  # empty dict when n=0
            print(f"  {name:<20} {'--':>6} {'--':>6} {'--':>6} "
                  f"{'-- (0 runs)':>18} {'--':>16}")
            continue
        n_s = m.get('n_success_effort', 0)
        effort_str = (f"{m['mean_control_effort']:.1f}"
                      f" +/- {m['std_control_effort']:.1f} ({n_s})"
                      if n_s > 0 else f"-- ({n_s})")
        time_str = (f"{m['mean_wall_time']:.1f}"
                    f" +/- {m['std_wall_time']:.1f} s")
        print(f"  {name:<20} "
              f"{m['goal_rate']*100:>5.0f}% "
              f"{m['collision_rate']*100:>5.0f}% "
              f"{m['success_rate']*100:>5.0f}% "
              f"{effort_str:>18} {time_str:>16}")

    print(sep)

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
        query_ctrl = BRTController13D(
            checkpoint_path=args.checkpoint_path,
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
    dynamics = load_dynamics(args.checkpoint_path)

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
                print(f"  ** Checkpoint saved ({i+1}/{len(ics)} rollouts)")

        all_results[ctrl_name] = results
        m = compute_metrics(results)
        metrics_all[display] = m

    # ---- Summary table ------------------------------------------------
    print_comparison_table(metrics_all)

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

    # ---- Final JSON save (reuses checkpoint helper) --------------------
    _save_checkpoint(all_results, tag='final')
    json_path = os.path.join(args.output_dir, 'comparison_results.json')
    print(f"\n  Results saved to: {json_path}")

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

    # Safety filter arguments (for brt_safety_13d)
    parent.add_argument('--safety_filter_mode', type=int, default=1,
                        help='Safety filter mode: 0=off, 1=least-restrictive, 2=CBF-QP')
    parent.add_argument('--safety_checkpoint_path', type=str, default=None,
                        help='Path to avoid-only BRT checkpoint for safety filter')
    parent.add_argument('--safety_tMax', type=float, default=None,
                        help='tMax for safety BRT queries (None = use model default)')
    parent.add_argument('--safety_filter_margin', type=float, default=0.1,
                        help='Activation threshold delta for mode 1')
    parent.add_argument('--safety_filter_gamma', type=float, default=0.2,
                        help='CBF decay rate for mode 2')

    # PD hybrid arguments
    parent.add_argument('--pd_torque_proximity', type=float, default=2.0,
                        help='Proximity multiplier for PD torque activation '
                             '(brt_pd_hybrid only). Default: 2.0')

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
                           choices=['brt_13d', 'brt_safety_13d',
                                    'brt_pd_hybrid',
                                    'mpc_13d', 'mpc_terminal_13d'])
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
                            choices=['brt_13d', 'brt_safety_13d',
                                     'brt_pd_hybrid',
                                     'mpc_13d', 'mpc_terminal_13d'])
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
