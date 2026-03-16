#!/usr/bin/env python3
"""
Gradient Quality Comparison: Grid-Based (25s) vs DeepReach (10s)

Picks ICs that lie INSIDE the 25s grid-based BRAT but OUTSIDE the 10s
learned BRAT.  At these states the DeepReach controller operates in
Phase 1 (static gradient at tMax).  We compare:
  - Trajectories and control inputs from both controllers
  - Direct gradient cosine similarity and sign agreement on the
    control-relevant components (vx, vy, omega)

ICs are selected near the outer edge of the 25s grid BRAT to showcase
gradient quality far from the goal.

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
    physical_time_to_index, grid_value_6D, deepreach_value_6D,
)

# Lazy import for grid-based controller
_combo_module = None
def _get_combo_module():
    global _combo_module
    if _combo_module is None:
        import importlib
        _combo_module = importlib.import_module('ComboControl')
    return _combo_module


# First-principles IC sampling bounds (matches run_controller.py reasoning):
#   - Position: ±13m (within dynamics ±15m with margin)
#   - Velocity: ±0.75 m/s (braking distance ~2.8m keeps chaser in domain)
#   - Theta: full range
#   - Omega: ±0.50 rad/s (braking from 0.50 takes ~10s at alpha_max)
GRADIENT_IC_BOUNDS = np.array([
    [-13.0, 13.0],     # px (m)
    [-13.0, 13.0],     # py (m)
    [ -0.75,  0.75],   # vx (m/s)
    [ -0.75,  0.75],   # vy (m/s)
    [-np.pi, np.pi],   # theta (rad)
    [ -0.50,  0.50],   # omega (rad/s)
])


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


def _in_goal_set(state, dynamics):
    """Check if a 6D state is inside the docking goal set."""
    px, py, vx, vy, theta, omega = state
    x_ok = np.abs(px) <= dynamics.eps_p
    y_ok = dynamics.goal_y_min <= py <= dynamics.goal_y_max
    vel_ok = np.sqrt(vx**2 + vy**2) <= dynamics.eps_v
    theta_goal = float(dynamics.goal_state[4])
    omega_goal = float(dynamics.goal_state[5])
    theta_diff = np.abs(np.arctan2(
        np.sin(theta - theta_goal), np.cos(theta - theta_goal)))
    theta_ok = theta_diff <= dynamics.eps_theta
    omega_ok = np.abs(omega - omega_goal) <= dynamics.eps_omega
    return x_ok and y_ok and vel_ok and theta_ok and omega_ok


# ======================================================================
#  IC selection
# ======================================================================
def select_ics(combo, brat_ctrl, grid_times, grid_horizon, n_ics,
               n_candidates=500_000, seed=42):
    """Sample ICs near the outer edge of the grid BRAT, outside the DeepReach BRAT.

    Uses first-principles sampling bounds (GRADIENT_IC_BOUNDS) and selects
    candidates whose grid value is closest to zero from below (outer boundary).
    """
    rng = np.random.RandomState(seed)
    states = rng.uniform(GRADIENT_IC_BOUNDS[:, 0], GRADIENT_IC_BOUNDS[:, 1],
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

    # Sort by grid value closest to 0 from below -> near outer BRAT boundary
    gv_cand = gv[mask]
    order = np.argsort(-gv_cand)  # least negative first = closest to boundary
    selected = candidates[order[:n_ics]]
    print(f"  IC selection: {mask.sum()} candidates found, selected {len(selected)}")
    print(f"  Grid values of selected ICs: "
          f"[{gv_cand[order[:n_ics]].min():.4f}, {gv_cand[order[:n_ics]].max():.4f}]")
    return selected


# ======================================================================
#  Joint simulation (both controllers advance in lockstep)
# ======================================================================
def simulate_both(ic, combo, brat_ctrl, dynamics_fn, dt, max_sim_time,
                  grid_times):
    """Simulate both controllers from the same IC using shared dynamics.

    Both controllers advance in lockstep. The loop continues until BOTH
    controllers have docked or max_sim_time is reached.

    Returns (result_grid, result_dr) dicts with keys:
        trajectory, controls, times, values, gradients,
        goal_reached, goal_time, controller_type
    """
    num_steps = int(max_sim_time / dt)

    # Allocate arrays for both controllers
    states_g = np.zeros((num_steps + 1, 6))
    states_d = np.zeros((num_steps + 1, 6))
    controls_g = np.zeros((num_steps, 3))
    controls_d = np.zeros((num_steps, 3))
    values_g = np.zeros(num_steps)
    values_d = np.zeros(num_steps)
    grads_g = np.zeros((num_steps, 6))
    grads_d = np.zeros((num_steps, 6))
    sim_times = np.zeros(num_steps)

    states_g[0] = states_d[0] = ic.copy()

    # Reset controllers
    combo.reset()
    brat_ctrl.reset()

    # Grid value/gradient helpers
    def grid_value(s, t):
        brt_time = max(combo.final_time, min(0, combo.final_time + t))
        tidx = int(np.argmin(np.abs(np.array(combo.times) - brt_time)))
        v4 = combo.value_at_state_4D(s[:4], tidx)
        v2 = combo.value_at_state_2D(s[4:], tidx)
        return max(v4, v2)

    def grid_gradient(s, t):
        brt_time = max(combo.final_time, min(0, combo.final_time + t))
        tidx = int(np.argmin(np.abs(np.array(combo.times) - brt_time)))
        g4 = combo.grad_at_state_4D(s[:4], tidx)
        g2 = combo.grad_at_state_2D(s[4:], tidx)
        return np.concatenate([g4, g2])

    grid_docked = False
    dr_docked = False
    grid_dock_time = None
    dr_dock_time = None
    n_steps = num_steps

    for k in range(num_steps):
        t = k * dt
        sim_times[k] = t

        # Grid controller step
        controls_g[k] = combo.u_fn(states_g[k], t)
        values_g[k] = grid_value(states_g[k], t)
        grads_g[k] = grid_gradient(states_g[k], t)

        # DeepReach controller step
        controls_d[k] = brat_ctrl.u_fn(states_d[k], t)
        values_d[k] = brat_ctrl.get_value(states_d[k], brat_ctrl.tMax)
        grads_d[k] = brat_ctrl.get_gradient(states_d[k], brat_ctrl.tMax)

        # Euler integration (independent trajectories, shared dynamics)
        s_next_g = states_g[k] + dt * dynamics_fn(states_g[k], controls_g[k])
        s_next_d = states_d[k] + dt * dynamics_fn(states_d[k], controls_d[k])
        s_next_g[4] = (s_next_g[4] + np.pi) % (2 * np.pi) - np.pi
        s_next_d[4] = (s_next_d[4] + np.pi) % (2 * np.pi) - np.pi
        states_g[k + 1] = s_next_g
        states_d[k + 1] = s_next_d

        # Track docking (record first time each controller enters goal set)
        if not grid_docked and _in_goal_set(s_next_g, brat_ctrl.dynamics):
            grid_docked = True
            grid_dock_time = (k + 1) * dt
        if not dr_docked and _in_goal_set(s_next_d, brat_ctrl.dynamics):
            dr_docked = True
            dr_dock_time = (k + 1) * dt

        # Only stop when BOTH have docked
        if grid_docked and dr_docked:
            n_steps = k + 1
            break

    # Report docking status
    for label, docked, dock_time in [('Grid', grid_docked, grid_dock_time),
                                      ('DeepReach', dr_docked, dr_dock_time)]:
        if docked:
            print(f"  {label} reached goal at t={dock_time:.2f}s")
        else:
            print(f"  {label} did NOT reach goal within {max_sim_time:.1f}s")

    result_grid = {
        'trajectory': states_g[:n_steps + 1],
        'controls': controls_g[:n_steps],
        'times': sim_times[:n_steps],
        'values': values_g[:n_steps],
        'gradients': grads_g[:n_steps],
        'goal_reached': grid_docked,
        'goal_time': grid_dock_time,
        'controller_type': 'grid',
    }
    result_dr = {
        'trajectory': states_d[:n_steps + 1],
        'controls': controls_d[:n_steps],
        'times': sim_times[:n_steps],
        'values': values_d[:n_steps],
        'gradients': grads_d[:n_steps],
        'goal_reached': dr_docked,
        'goal_time': dr_dock_time,
        'controller_type': 'deepreach',
    }
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
    return np.linalg.norm(pos_grid - pos_dr, axis=1)


# ======================================================================
#  Plotting helpers
# ======================================================================
def _draw_docking_scene(ax):
    """Draw target body, docking post, and goal region on ax."""
    from matplotlib.patches import Rectangle
    body = Rectangle((-3, 0), 6, 3, color='gray', alpha=0.3, label='Target body')
    ax.add_patch(body)
    post = Rectangle((-0.6, -0.2), 1.2, 0.2, color='gray', alpha=0.5, label='Docking post')
    ax.add_patch(post)
    goal = Rectangle((-0.1, -1.4), 0.2, 0.5, color='green', alpha=0.3, label='Goal')
    ax.add_patch(goal)


def _add_dock_markers(ax, result_grid, result_dr):
    """Add vertical docking-time lines to a time-series axis."""
    if result_grid.get('goal_time') is not None:
        ax.axvline(result_grid['goal_time'], color='blue', ls=':', alpha=0.6, lw=1)
    if result_dr.get('goal_time') is not None:
        ax.axvline(result_dr['goal_time'], color='red', ls=':', alpha=0.6, lw=1)


def _add_brt_contours(ax, combo, brat_ctrl, grid_times, grid_horizon, dr_horizon,
                       ic=None, px_range=(-15, 15), py_range=(-15, 15), resolution=120):
    """Draw zero level set contours of grid and DeepReach BRTs on a px-py plot.

    When ic is provided, the non-position state components (vx, vy, theta, omega)
    are taken from the IC. Otherwise defaults to nominal docking state.
    """
    if ic is not None:
        vx_fixed, vy_fixed = ic[2], ic[3]
        theta_fixed, omega_fixed = ic[4], ic[5]
    else:
        vx_fixed, vy_fixed = 0.0, 0.0
        theta_fixed, omega_fixed = np.pi / 2, 0.0

    px = np.linspace(*px_range, resolution)
    py = np.linspace(*py_range, resolution)
    PX, PY = np.meshgrid(px, py)
    flat = np.column_stack([
        PX.ravel(), PY.ravel(),
        np.full(PX.size, vx_fixed),
        np.full(PX.size, vy_fixed),
        np.full(PX.size, theta_fixed),
        np.full(PX.size, omega_fixed),
    ])

    # Grid BRT
    grid_idx = physical_time_to_index(grid_horizon, grid_times)
    v_grid = grid_value_6D(combo, flat, grid_idx).reshape(PX.shape)
    ax.contour(PX, PY, v_grid, levels=[0], colors='purple', linewidths=1,
               linestyles='dashed')

    # DeepReach BRT
    v_dr = deepreach_value_6D(brat_ctrl, flat, dr_horizon).reshape(PX.shape)
    ax.contour(PX, PY, v_dr, levels=[0], colors='green', linewidths=1,
               linestyles='dashed')


# ======================================================================
#  Plotting
# ======================================================================
def plot_trajectory_overlay(result_grid, result_dr, ic, ic_idx, save_path,
                            combo=None, brat_ctrl=None, grid_times=None,
                            grid_horizon=None, dr_horizon=None):
    """2D position trajectory overlay with docking scene and IC-dependent BRAT contours."""
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_docking_scene(ax)

    if combo is not None and brat_ctrl is not None:
        _add_brt_contours(ax, combo, brat_ctrl, grid_times,
                          grid_horizon, dr_horizon, ic=ic)

    tg = result_grid['trajectory']
    td = result_dr['trajectory']
    ax.plot(tg[:, 0], tg[:, 1], 'b-', lw=1.5, label='Grid (25s)')
    ax.plot(td[:, 0], td[:, 1], 'r--', lw=1.5, label='DeepReach (10s)')
    ax.plot(tg[0, 0], tg[0, 1], 'ko', ms=8, label='Start')
    ax.plot(tg[-1, 0], tg[-1, 1], 'bs', ms=7)
    ax.plot(td[-1, 0], td[-1, 1], 'r^', ms=7)

    # Add legend entries for BRAT contours
    ax.plot([], [], color='purple', ls='--', lw=1, label=f'Grid BRAT ({grid_horizon}s)')
    ax.plot([], [], color='green', ls='--', lw=1, label=f'DR BRAT ({dr_horizon}s)')

    ax.set_xlabel('px (m)')
    ax.set_ylabel('py (m)')
    title = f'Trajectory Comparison — IC {ic_idx}'
    if ic is not None:
        title += (f'\nSlice: vx={ic[2]:.2f}, vy={ic[3]:.2f}, '
                  f'\u03b8={ic[4]:.2f}, \u03c9={ic[5]:.2f}')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_states(result_grid, result_dr, ic_idx, save_path, dynamics=None):
    """6 states vs time (3x2 subplots) with goal tolerance bands."""
    labels = ['px (m)', 'py (m)', 'vx (m/s)', 'vy (m/s)', '\u03b8 (rad)', '\u03c9 (rad/s)']
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    tg = result_grid['trajectory']
    td = result_dr['trajectory']
    dt = result_grid['times'][1] - result_grid['times'][0] if len(result_grid['times']) > 1 else 0.1
    times_g = np.arange(tg.shape[0]) * dt
    times_d = np.arange(td.shape[0]) * dt

    # Goal targets and tolerance bands per state
    goal_bands = None
    if dynamics is not None:
        goal = dynamics.goal_state.cpu().numpy()
        goal_bands = [
            (0.0, dynamics.eps_p),                         # px: center=0, half=eps_p
            (None, None),                                  # py: asymmetric band
            (0.0, dynamics.eps_v),                         # vx: center=0, half=eps_v
            (0.0, dynamics.eps_v),                         # vy: center=0, half=eps_v
            (float(goal[4]), dynamics.eps_theta),           # theta: center=pi/2
            (float(goal[5]), dynamics.eps_omega),           # omega: center=0
        ]

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(times_g, tg[:, i], 'b-', lw=1.2, label='Grid')
        ax.plot(times_d, td[:, i], 'r--', lw=1.2, label='DeepReach')

        # Goal tolerance bands
        if goal_bands is not None:
            if i == 1:  # py has asymmetric goal band
                ax.axhspan(dynamics.goal_y_min, dynamics.goal_y_max,
                           alpha=0.15, color='green')
                ax.axhline(dynamics.goal_y_center, color='green', ls=':', alpha=0.5)
            else:
                center, half = goal_bands[i]
                ax.axhspan(center - half, center + half, alpha=0.15, color='green')
                ax.axhline(center, color='green', ls=':', alpha=0.5)

        # Docking time markers
        _add_dock_markers(ax, result_grid, result_dr)

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
    """3 controls vs time with docking markers."""
    labels = ['u_x (N)', 'u_y (N)', 'u_\u03b8 (N\u00b7m)']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.step(result_grid['times'], result_grid['controls'][:, i],
                'b-', lw=1.2, where='post', label='Grid')
        ax.step(result_dr['times'], result_dr['controls'][:, i],
                'r--', lw=1.2, where='post', label='DeepReach')
        _add_dock_markers(ax, result_grid, result_dr)
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Control Comparison — IC {ic_idx}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_gradient_metrics(grad_metrics, times, ic_idx, save_path,
                          result_grid=None, result_dr=None):
    """Cosine similarity and sign agreement vs time with docking markers."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Cosine similarity
    ax = axes[0]
    ax.plot(times, grad_metrics['cosine_sim'], 'k-', lw=1.0)
    ax.axhline(1.0, color='green', ls=':', alpha=0.5)
    ax.axhline(0.0, color='gray', ls=':', alpha=0.3)
    if result_grid is not None and result_dr is not None:
        _add_dock_markers(ax, result_grid, result_dr)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Gradient Comparison — IC {ic_idx}  |  '
                 f'Mean cos-sim = {grad_metrics["mean_cosine_sim"]:.4f}')
    ax.set_ylim(-0.2, 1.1)
    ax.grid(alpha=0.3)

    # Per-component sign agreement
    ax = axes[1]
    ctrl_labels = ['vx', 'vy', '\u03c9']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for j, (lbl, c) in enumerate(zip(ctrl_labels, colors)):
        agree = grad_metrics['sign_agreement'][:, j].astype(float)
        window = min(20, len(agree) // 5) if len(agree) > 20 else 1
        if window > 1:
            kernel = np.ones(window) / window
            smooth = np.convolve(agree, kernel, mode='same')
        else:
            smooth = agree
        ax.plot(times, smooth, color=c, lw=1.2,
                label=f'{lbl} ({grad_metrics["sign_agreement_rate"][j]*100:.1f}%)')
    if result_grid is not None and result_dr is not None:
        _add_dock_markers(ax, result_grid, result_dr)
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

    ax = axes[0]
    ax.bar(range(n), cos_sims, color='steelblue')
    ax.axhline(np.mean(cos_sims), color='red', ls='--',
               label=f'Mean={np.mean(cos_sims):.4f}')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Gradient Cosine Similarity')
    ax.legend()

    ax = axes[1]
    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, sign_rates[:, 0], w, label='vx', color='tab:blue')
    ax.bar(x, sign_rates[:, 1], w, label='vy', color='tab:orange')
    ax.bar(x + w, sign_rates[:, 2], w, label='\u03c9', color='tab:green')
    ax.set_xlabel('IC index')
    ax.set_ylabel('Sign Agreement Rate')
    ax.set_title('Control Sign Agreement')
    ax.legend()

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

        result_grid, result_dr = simulate_both(
            ic, combo, brat_ctrl, dynamics_fn, args.dt, args.max_sim_time,
            grid_times)

        # Gradient metrics (trajectories are same length from joint sim)
        grad_met = compute_gradient_metrics(
            result_grid['gradients'], result_dr['gradients'])

        # Trajectory deviation
        tdev = trajectory_deviation(result_grid['trajectory'],
                                    result_dr['trajectory'])
        grad_met['mean_traj_deviation'] = float(np.mean(tdev))
        grad_met['max_traj_deviation'] = float(np.max(tdev))
        grad_met['goal_reached_grid'] = result_grid['goal_reached']
        grad_met['goal_reached_dr'] = result_dr['goal_reached']
        grad_met['goal_time_grid'] = result_grid['goal_time']
        grad_met['goal_time_dr'] = result_dr['goal_time']

        all_metrics.append(grad_met)
        all_results.append((result_grid, result_dr))

        # Print per-IC summary
        print(f"  Cosine similarity:  {grad_met['mean_cosine_sim']:.4f}")
        print(f"  Sign agreement:     vx={grad_met['sign_agreement_rate'][0]*100:.1f}%  "
              f"vy={grad_met['sign_agreement_rate'][1]*100:.1f}%  "
              f"\u03c9={grad_met['sign_agreement_rate'][2]*100:.1f}%")
        print(f"  Trajectory dev:     mean={grad_met['mean_traj_deviation']:.3f}m  "
              f"max={grad_met['max_traj_deviation']:.3f}m")

        # Per-IC plots
        ic_dir = os.path.join(args.output_dir, f'ic_{i}')
        os.makedirs(ic_dir, exist_ok=True)
        plot_trajectory_overlay(result_grid, result_dr, ic, i,
                                os.path.join(ic_dir, 'trajectory.png'),
                                combo=combo, brat_ctrl=brat_ctrl,
                                grid_times=grid_times,
                                grid_horizon=args.grid_time_horizon,
                                dr_horizon=args.tMax)
        plot_states(result_grid, result_dr, i,
                    os.path.join(ic_dir, 'states.png'),
                    dynamics=brat_ctrl.dynamics)
        plot_controls(result_grid, result_dr, i,
                      os.path.join(ic_dir, 'controls.png'))
        plot_gradient_metrics(grad_met, result_grid['times'], i,
                              os.path.join(ic_dir, 'gradient_metrics.png'),
                              result_grid=result_grid, result_dr=result_dr)

    # ---- 6. Aggregate results ----
    if len(ics) > 1:
        print('\n' + '=' * 60)
        print('Aggregate metrics:')
        cos_sims = [m['mean_cosine_sim'] for m in all_metrics]
        sign_rates = np.array([m['sign_agreement_rate'] for m in all_metrics])
        tdevs = [m['mean_traj_deviation'] for m in all_metrics]
        print(f"  Cosine similarity:  {np.mean(cos_sims):.4f} \u00b1 {np.std(cos_sims):.4f}")
        print(f"  Sign agreement vx:  {np.mean(sign_rates[:,0])*100:.1f}% \u00b1 {np.std(sign_rates[:,0])*100:.1f}%")
        print(f"  Sign agreement vy:  {np.mean(sign_rates[:,1])*100:.1f}% \u00b1 {np.std(sign_rates[:,1])*100:.1f}%")
        print(f"  Sign agreement \u03c9:   {np.mean(sign_rates[:,2])*100:.1f}% \u00b1 {np.std(sign_rates[:,2])*100:.1f}%")
        print(f"  Trajectory dev:     {np.mean(tdevs):.3f} \u00b1 {np.std(tdevs):.3f}m")

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
            'ic_bounds': GRADIENT_IC_BOUNDS.tolist(),
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
                'goal_reached_grid': m['goal_reached_grid'],
                'goal_reached_dr': m['goal_reached_dr'],
                'goal_time_grid': m['goal_time_grid'],
                'goal_time_dr': m['goal_time_dr'],
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
