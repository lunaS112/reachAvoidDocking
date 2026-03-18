#!/usr/bin/env python3
"""Visualize reach_fn slices to verify goal set geometry before retraining.

Generates three PNG files:
  1. reach_fn_xy.png   — X vs Y position slice
  2. reach_fn_vxvy.png — Vx vs Vy velocity slice
  3. reach_fn_quat.png — Yaw-error vs Roll-error quaternion slice

Usage:
    python visualize_reach_fn.py [--outdir DIR]
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# Add parent to path so we can import dynamics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics.dynamics import Docking13D


def make_dynamics():
    """Instantiate dynamics (no checkpoint needed — just reach_fn)."""
    return Docking13D(set_mode='reach_avoid')


def build_goal_state(dyn):
    """Return the goal-center state as a 1D numpy array (13,)."""
    gs = dyn.goal_state.detach().cpu().numpy().copy()
    # Set vy to midpoint of axial band
    gs[4] = (dyn.eps_v_axial_lo + dyn.eps_v_axial_hi) / 2.0  # +0.065
    return gs


def eval_reach_fn_grid(dyn, base_state, axis0_idx, axis0_vals,
                       axis1_idx, axis1_vals):
    """Evaluate reach_fn on a 2D grid, varying two state indices.

    Returns (N0, N1) array of reach_fn values where
    axis0 is rows (vertical) and axis1 is columns (horizontal).
    """
    n0, n1 = len(axis0_vals), len(axis1_vals)
    # Build (n0*n1, 13) state tensor
    base = torch.tensor(base_state, dtype=torch.float32).unsqueeze(0)
    states = base.expand(n0 * n1, -1).clone()

    grid0, grid1 = np.meshgrid(axis0_vals, axis1_vals, indexing='ij')
    states[:, axis0_idx] = torch.tensor(grid0.ravel(), dtype=torch.float32)
    states[:, axis1_idx] = torch.tensor(grid1.ravel(), dtype=torch.float32)

    with torch.no_grad():
        vals = dyn.reach_fn(states)

    return vals.numpy().reshape(n0, n1)


def plot_slice(values, x_vals, y_vals, xlabel, ylabel, title, save_path,
               overlay_rect=None):
    """Plot a 2D reach_fn slice with diverging colormap and zero-contour."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Symmetric color limits
    vmax = max(abs(values.min()), abs(values.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(x_vals, y_vals, values, cmap='RdBu_r', norm=norm,
                       shading='auto')
    cb = fig.colorbar(im, ax=ax, label='reach_fn value')

    # Zero contour
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    cs = ax.contour(X, Y, values, levels=[0.0], colors='black', linewidths=2)
    ax.clabel(cs, fmt='0', fontsize=10)

    # Optional rectangle overlay showing expected goal bounds
    if overlay_rect is not None:
        x0, x1, y0, y1 = overlay_rect
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=1.5, edgecolor='lime',
                              facecolor='none', linestyle='--',
                              label='Expected goal')
        ax.add_patch(rect)
        ax.legend(fontsize=10)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def quat_from_axis_angle_np(axis, angle):
    """axis-angle → scalar-first quaternion (numpy)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def quat_mul_np(q, p):
    """Hamilton product q ⊗ p, scalar-first."""
    q0, q1, q2, q3 = q
    p0, p1, p2, p3 = p
    return np.array([
        q0*p0 - q1*p1 - q2*p2 - q3*p3,
        q0*p1 + q1*p0 + q2*p3 - q3*p2,
        q0*p2 - q1*p3 + q2*p0 + q3*p1,
        q0*p3 + q1*p2 - q2*p1 + q3*p0,
    ], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='Visualize reach_fn slices')
    parser.add_argument('--outdir', default='./outputs/reach_fn_slices',
                        help='Output directory for PNGs')
    parser.add_argument('--resolution', type=int, default=300,
                        help='Grid resolution per axis')
    args = parser.parse_args()

    dyn = make_dynamics()
    base = build_goal_state(dyn)
    N = args.resolution

    print(f"Goal state: {base}")
    print(f"goal_y_min={dyn.goal_y_min:.4f}, goal_y_max={dyn.goal_y_max:.4f}")
    print(f"eps_p={dyn.eps_p}, eps_v_lateral={dyn.eps_v_lateral}")
    print(f"eps_v_axial_lo={dyn.eps_v_axial_lo}, eps_v_axial_hi={dyn.eps_v_axial_hi}")
    print(f"eps_q={dyn.eps_q}")
    print(f"eps_omega_pitchyaw={dyn.eps_omega_pitchyaw}, eps_omega_roll={dyn.eps_omega_roll}")

    # ------------------------------------------------------------------ #
    # Slice 1: X vs Y position
    # ------------------------------------------------------------------ #
    print("\n--- Slice 1: X-Y position ---")
    x_range = np.linspace(-0.3, 0.3, N)
    y_range = np.linspace(dyn.goal_y_center - 0.6, dyn.goal_y_center + 0.6, N)

    # axes: axis0 = x (state idx 0), axis1 = y (state idx 1)
    # We want x horizontal, y vertical → eval with axis0=y, axis1=x
    vals_xy = eval_reach_fn_grid(dyn, base,
                                 axis0_idx=1, axis0_vals=y_range,
                                 axis1_idx=0, axis1_vals=x_range)

    plot_slice(vals_xy, x_range, y_range,
               xlabel='x (m)', ylabel='y (m)',
               title='reach_fn: X-Y position slice\n'
                     f'(all other states at goal, vy={base[4]:.3f})',
               save_path=os.path.join(args.outdir, 'reach_fn_xy.png'),
               overlay_rect=(-dyn.eps_p, dyn.eps_p,
                             dyn.goal_y_min, dyn.goal_y_max))

    # ------------------------------------------------------------------ #
    # Slice 2: Vx vs Vy velocity
    # ------------------------------------------------------------------ #
    print("\n--- Slice 2: Vx-Vy velocity ---")
    # Use goal-center position for base (not the vy midpoint — we sweep vy)
    base_vel = base.copy()
    base_vel[4] = 0.0  # will be overwritten by grid

    vx_range = np.linspace(-0.15, 0.15, N)
    vy_range = np.linspace(-0.15, 0.20, N)

    # axis0 = vy (vertical), axis1 = vx (horizontal)
    vals_vxvy = eval_reach_fn_grid(dyn, base_vel,
                                   axis0_idx=4, axis0_vals=vy_range,
                                   axis1_idx=3, axis1_vals=vx_range)

    plot_slice(vals_vxvy, vx_range, vy_range,
               xlabel='vx (m/s)', ylabel='vy (m/s)',
               title='reach_fn: Vx-Vy velocity slice\n'
                     '(pos=goal, vz=0, q=q_goal, ω=0)',
               save_path=os.path.join(args.outdir, 'reach_fn_vxvy.png'),
               overlay_rect=(-dyn.eps_v_lateral, dyn.eps_v_lateral,
                             dyn.eps_v_axial_lo, dyn.eps_v_axial_hi))

    # ------------------------------------------------------------------ #
    # Slice 3: Quaternion — yaw error vs roll error
    # ------------------------------------------------------------------ #
    print("\n--- Slice 3: Quaternion (yaw-error vs roll-error) ---")
    q_goal_np = dyn.q_goal.detach().cpu().numpy()
    max_angle = np.radians(15)
    yaw_errors = np.linspace(-max_angle, max_angle, N)
    roll_errors = np.linspace(-max_angle, max_angle, N)

    # Build (N*N, 13) states with perturbed quaternions
    base_q = base.copy()
    n_total = N * N
    states_q = np.tile(base_q, (n_total, 1))

    idx = 0
    for i, yaw_err in enumerate(yaw_errors):
        for j, roll_err in enumerate(roll_errors):
            # q_error = q_yaw ⊗ q_roll (small-angle composition)
            q_yaw = quat_from_axis_angle_np([0, 0, 1], yaw_err)
            q_roll = quat_from_axis_angle_np([1, 0, 0], roll_err)
            q_err = quat_mul_np(q_yaw, q_roll)
            # q = q_err ⊗ q_goal
            q_total = quat_mul_np(q_err, q_goal_np)
            # Normalize
            q_total = q_total / (np.linalg.norm(q_total) + 1e-12)
            states_q[idx, 6:10] = q_total
            idx += 1

    states_t = torch.tensor(states_q, dtype=torch.float32)
    with torch.no_grad():
        vals_q = dyn.reach_fn(states_t).numpy().reshape(N, N)

    # Plot: axis0 = yaw_errors (vertical), axis1 = roll_errors (horizontal)
    yaw_deg = np.degrees(yaw_errors)
    roll_deg = np.degrees(roll_errors)

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(vals_q.min()), abs(vals_q.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(roll_deg, yaw_deg, vals_q, cmap='RdBu_r', norm=norm,
                       shading='auto')
    fig.colorbar(im, ax=ax, label='reach_fn value')

    # Zero contour
    R_deg, Y_deg = np.meshgrid(roll_deg, yaw_deg, indexing='ij')
    # vals_q is indexed [yaw, roll], meshgrid with ij gives [roll, yaw]
    # so transpose for contour
    cs = ax.contour(roll_deg, yaw_deg, vals_q, levels=[0.0],
                    colors='black', linewidths=2)
    ax.clabel(cs, fmt='0', fontsize=10)

    # Reference circle at eps_q radius
    eps_q_deg = np.degrees(dyn.eps_q)
    circle = plt.Circle((0, 0), eps_q_deg, fill=False, edgecolor='lime',
                         linewidth=1.5, linestyle='--',
                         label=f'eps_q = {eps_q_deg:.1f}°')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)

    ax.set_xlabel('Roll error (deg)', fontsize=12)
    ax.set_ylabel('Yaw error (deg)', fontsize=12)
    ax.set_title('reach_fn: Quaternion slice (yaw vs roll error)\n'
                 f'(pos=goal, vel=goal with vy={base[4]:.3f}, ω=0)',
                 fontsize=14)
    ax.grid(True, alpha=0.2)

    quat_path = os.path.join(args.outdir, 'reach_fn_quat.png')
    os.makedirs(os.path.dirname(quat_path) or '.', exist_ok=True)
    plt.savefig(quat_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {quat_path}")

    print(f"\nAll slices saved to {args.outdir}/")


if __name__ == '__main__':
    main()
