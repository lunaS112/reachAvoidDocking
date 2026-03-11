#!/usr/bin/env python3
"""
Standalone visualization of goal set vs failure set geometry for Docking13D.

Shows YZ-plane cross-section (x=0) with:
  - Target body and docking post (physical geometry)
  - Inflated collision envelope (failure set boundary)
  - Goal set boundary
  - Labeled gap between goal and failure sets
  - Chaser silhouette and safety margins

Usage:
    python3 -m utils.goal_geometry_viz          # from deepReachMPCReachAvoid/
    python3 utils/goal_geometry_viz.py           # also works
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def main():
    # ── Physical dimensions (from Docking13D.__init__) ──
    w_c, h_c, d_c = 1.0, 1.0, 1.0
    cb = math.sqrt(w_c**2 + h_c**2 + d_c**2) / 2.0  # ≈ 0.866

    w_t, h_t, d_t = 6.0, 3.0, 3.0
    post_hw_z = 0.6
    post_length = 0.2

    # ── Goal set parameters ──
    eps_p = 0.30
    eps_q = 0.20  # rad
    goal_clearance = 0.134
    goal_band_height = 0.4

    goal_y_max = -(post_length + cb + goal_clearance)
    goal_y_min = goal_y_max - goal_band_height

    # ── Inflated bounds (full cb everywhere) ──
    post_z_lo = -(post_hw_z + cb)
    post_z_hi = (post_hw_z + cb)
    post_y_lo = -(post_length + cb)
    post_y_hi = cb

    body_z_lo = -(d_t / 2.0 + cb)
    body_z_hi = (d_t / 2.0 + cb)
    body_y_lo = -cb
    body_y_hi = h_t + cb

    # ── Safety analysis ──
    gap = abs(goal_y_max - post_y_lo)
    phys_clearance = abs(goal_y_max) - post_length - 0.5
    corner_dy = 0.707 * eps_q
    safety_margin = phys_clearance - corner_dy

    # ── Figure 1: Overview ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlabel('Z (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Docking13D Geometry — YZ Plane (x = 0)\n'
                 'Goal Set vs Failure Set (full chaser_buffer)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Target body
    ax.add_patch(patches.Rectangle(
        (-d_t/2, 0), d_t, h_t,
        linewidth=2, edgecolor='black', facecolor='lightgray',
        label='Target body', zorder=2))

    # Docking post
    ax.add_patch(patches.Rectangle(
        (-post_hw_z, -post_length), 2*post_hw_z, post_length,
        linewidth=2, edgecolor='black', facecolor='gray',
        label='Docking post', zorder=2))

    # Inflated body
    ax.add_patch(patches.Rectangle(
        (body_z_lo, body_y_lo), body_z_hi - body_z_lo, body_y_hi - body_y_lo,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.08,
        label='Failure set (body)', zorder=1))

    # Inflated post
    ax.add_patch(patches.Rectangle(
        (post_z_lo, post_y_lo), post_z_hi - post_z_lo, post_y_hi - post_y_lo,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.08,
        label='Failure set (post)', zorder=1))

    # Goal set
    ax.add_patch(patches.Rectangle(
        (-eps_p, goal_y_min), 2 * eps_p, goal_y_max - goal_y_min,
        linewidth=2.5, edgecolor='green', facecolor='green', alpha=0.20,
        label=f'Goal set (eps_p={eps_p}, gap={goal_clearance}m)', zorder=3))

    # Gap annotation
    gap_z = eps_p + 0.15
    gap_mid = (post_y_lo + goal_y_max) / 2.0
    ax.annotate('', xy=(gap_z, post_y_lo), xytext=(gap_z, goal_y_max),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(gap_z + 0.08, gap_mid,
            f'{gap:.3f} m\n(gap)',
            fontsize=11, color='blue', fontweight='bold', va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.9))

    # Reference lines
    ax.axhline(y=post_y_lo, color='red', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axhline(y=goal_y_max, color='green', linewidth=0.8, linestyle=':', alpha=0.6)

    # Info box
    info = (
        f"chaser_buffer = {cb:.3f} m\n"
        f"post_length   = {post_length} m\n"
        f"\n"
        f"Inflated post tip  y = {post_y_lo:.3f} m\n"
        f"Goal top           y = {goal_y_max:.3f} m\n"
        f"Goal bottom        y = {goal_y_min:.3f} m\n"
        f"\n"
        f"eps_p = {eps_p} m,  eps_q = {eps_q} rad ({math.degrees(eps_q):.1f}°)\n"
        f"Physical hull clearance = {phys_clearance:.3f} m\n"
        f"Max corner Δy at eps_q  = {corner_dy:.4f} m\n"
        f"Safety margin           = {safety_margin:.4f} m"
    )
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.5, 4.5)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    out_dir = Path(__file__).parent.parent / 'data'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'goal_geometry_yz.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # ── Figure 2: Zoomed gap detail ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
    ax2.set_aspect('equal')
    ax2.set_xlabel('Z (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Goal–Failure Gap Detail — YZ Plane (x = 0)\n'
                  f'eps_q = {eps_q} rad ({math.degrees(eps_q):.1f}°), '
                  f'safety margin = {safety_margin*1000:.0f} mm', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Post
    ax2.add_patch(patches.Rectangle(
        (-post_hw_z, -post_length), 2*post_hw_z, post_length,
        linewidth=2, edgecolor='black', facecolor='gray', alpha=0.4,
        label='Post (physical)'))

    # Inflated post
    ax2.add_patch(patches.Rectangle(
        (post_z_lo, post_y_lo), post_z_hi - post_z_lo, post_y_hi - post_y_lo,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.10,
        label=f'Failure (cb={cb:.2f})'))

    # Goal
    ax2.add_patch(patches.Rectangle(
        (-eps_p, goal_y_min), 2*eps_p, goal_y_max - goal_y_min,
        linewidth=2.5, edgecolor='green', facecolor='green', alpha=0.20,
        label='Goal set'))

    # Chaser silhouette at goal_y_max
    chaser_y = goal_y_max
    ax2.add_patch(patches.Rectangle(
        (-0.5, chaser_y - 0.5), 1.0, 1.0,
        linewidth=1.5, edgecolor='blue', facecolor='blue', alpha=0.08,
        linestyle='-.', label='Chaser at goal top'))

    # Hull gap annotation
    chaser_top = chaser_y + 0.5
    ax2.annotate('', xy=(0.6, -post_length), xytext=(0.6, chaser_top),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    hull_gap = abs(-post_length - chaser_top)
    ax2.text(0.65, (-post_length + chaser_top) / 2,
             f'{hull_gap*1000:.0f} mm\nhull gap',
             fontsize=11, color='purple', fontweight='bold', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.9))

    # SDF gap annotation
    ax2.annotate('', xy=(eps_p + 0.1, post_y_lo), xytext=(eps_p + 0.1, goal_y_max),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax2.text(eps_p + 0.15, gap_mid,
             f'{gap*1000:.0f} mm SDF',
             fontsize=10, color='blue', fontweight='bold', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='blue', alpha=0.9))

    # Corner sweep
    corner_y_displaced = chaser_top + corner_dy
    ax2.plot([0.5, 0.5], [chaser_top, corner_y_displaced],
             'o-', color='orange', markersize=4, linewidth=2, zorder=5)
    ax2.annotate(f'corner Δy\n{corner_dy*1000:.1f} mm',
                 xy=(0.5, corner_y_displaced), xytext=(0.85, corner_y_displaced),
                 fontsize=9, color='orange', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='orange'),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='orange', alpha=0.9))

    # Reference lines
    for y_val, color, label in [
        (post_y_lo, 'red', f'y={post_y_lo:.3f} (inflated tip)'),
        (-post_length, 'black', f'y={-post_length:.1f} (physical tip)'),
        (goal_y_max, 'green', f'y={goal_y_max:.3f} (goal top)'),
        (goal_y_min, 'green', f'y={goal_y_min:.3f} (goal bot)'),
    ]:
        ax2.axhline(y=y_val, color=color, linewidth=0.8, linestyle=':', alpha=0.5)
        ax2.text(-1.9, y_val + 0.012, label, fontsize=8, color=color)

    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(goal_y_min - 0.4, 0.3)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    out_path2 = out_dir / 'goal_geometry_yz_zoom.png'
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")

    plt.close('all')


if __name__ == '__main__':
    main()
