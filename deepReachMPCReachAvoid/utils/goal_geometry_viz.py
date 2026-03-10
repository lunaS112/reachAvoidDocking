#!/usr/bin/env python3
"""
Standalone visualization of goal set vs failure set geometry for Docking13D.

Shows YZ-plane cross-section (x=0) with:
  - Target body and docking post (physical geometry)
  - Inflated collision envelope (failure set boundary)
  - Current goal set boundary (solid green) — uses cb_tip for post tip
  - Previous goal set boundary (dashed gray) — used full chaser_buffer
  - Labeled gaps and safety margins

Usage:
    python3 -m utils.goal_geometry_viz          # from deepReachMPCReachAvoid/
    python3 utils/goal_geometry_viz.py           # also works
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def main():
    # ── Physical dimensions (from Docking13D.__init__) ──
    w_c, h_c, d_c = 1.0, 1.0, 1.0          # chaser dims
    cb = math.sqrt(w_c**2 + h_c**2 + d_c**2) / 2.0  # bounding sphere ≈ 0.866
    cb_tip = 0.55                            # aligned-approach inflation for post tip

    w_t, h_t, d_t = 6.0, 3.0, 3.0           # target dims
    post_hw_z = 0.6                           # post half-width in z
    post_length = 0.2                         # post extent in -y

    # ── Current values (cb_tip approach) ──
    eps_p_cur = 0.10
    eps_q_cur = 0.065  # rad
    goal_clearance_cur = 0.01
    goal_band_height = 0.10

    goal_y_max_cur = -(post_length + cb_tip + goal_clearance_cur)
    goal_y_min_cur = goal_y_max_cur - goal_band_height

    # ── Previous values (full chaser_buffer, for comparison) ──
    eps_p_prev = 0.10
    goal_clearance_prev = 0.05

    goal_y_max_prev = -(post_length + cb + goal_clearance_prev)
    goal_y_min_prev = goal_y_max_prev - goal_band_height

    # ── Inflated bounds (failure set boundary) ──
    # Post: sides use full cb, tip (-Y face) uses cb_tip
    post_z_lo_inf = -(post_hw_z + cb)
    post_z_hi_inf =  (post_hw_z + cb)
    post_y_lo_inf = -(post_length + cb_tip)   # tip uses cb_tip
    post_y_hi_inf =  cb                       # +Y face uses full cb

    # Old inflated tip (for comparison)
    post_y_lo_inf_old = -(post_length + cb)

    # Body inflated in YZ plane (-Y face uses cb_tip, same as post tip)
    body_z_lo_inf = -(d_t / 2.0 + cb)
    body_z_hi_inf =  (d_t / 2.0 + cb)
    body_y_lo_inf = -cb_tip
    body_y_hi_inf =  h_t + cb

    # ── Safety analysis ──
    # Physical clearance: chaser center at goal_y_max, top face at +0.5
    phys_clearance = abs(goal_y_max_cur + 0.5) - post_length  # distance hull-to-tip
    # Worst-case corner displacement at eps_q
    corner_dy = 0.707 * eps_q_cur
    safety_margin = (abs(goal_y_max_cur) - post_length - 0.5) - corner_dy

    # ── Figure 1: Overview ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlabel('Z (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Docking13D Geometry — YZ Plane (x = 0)\nGoal Set vs Failure Set (cb_tip approach)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Target body (physical)
    ax.add_patch(patches.Rectangle(
        (-d_t/2, 0), d_t, h_t,
        linewidth=2, edgecolor='black', facecolor='lightgray', label='Target body', zorder=2))

    # Docking post (physical)
    ax.add_patch(patches.Rectangle(
        (-post_hw_z, -post_length), 2*post_hw_z, post_length,
        linewidth=2, edgecolor='black', facecolor='gray', label='Docking post', zorder=2))

    # Inflated body (failure set)
    ax.add_patch(patches.Rectangle(
        (body_z_lo_inf, body_y_lo_inf),
        body_z_hi_inf - body_z_lo_inf, body_y_hi_inf - body_y_lo_inf,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.08,
        label='Failure set (body, full cb)', zorder=1))

    # Inflated post — current (cb_tip for tip)
    ax.add_patch(patches.Rectangle(
        (post_z_lo_inf, post_y_lo_inf),
        post_z_hi_inf - post_z_lo_inf, post_y_hi_inf - post_y_lo_inf,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.08,
        label=f'Failure set (post, tip={cb_tip}m)', zorder=1))

    # Old inflated post tip (dashed red)
    ax.axhline(y=post_y_lo_inf_old, color='red', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.text(1.6, post_y_lo_inf_old + 0.03,
            f'Old inflated tip (cb={cb:.3f})',
            fontsize=9, color='red', alpha=0.7)

    # Current goal set
    ax.add_patch(patches.Rectangle(
        (-eps_p_cur, goal_y_min_cur), 2 * eps_p_cur, goal_y_max_cur - goal_y_min_cur,
        linewidth=2.5, edgecolor='green', facecolor='green', alpha=0.20,
        label=f'Current goal (cb_tip={cb_tip}, gap={goal_clearance_cur}m)', zorder=3))

    # Previous goal set (dashed)
    ax.add_patch(patches.Rectangle(
        (-eps_p_prev, goal_y_min_prev), 2 * eps_p_prev, goal_y_max_prev - goal_y_min_prev,
        linewidth=2, edgecolor='gray', facecolor='none', linestyle='--',
        label=f'Previous goal (cb={cb:.2f}, gap={goal_clearance_prev}m)', zorder=3))

    # Annotate current gap (inflated tip to goal top)
    gap_top = post_y_lo_inf
    gap_bot = goal_y_max_cur
    gap_z = eps_p_cur + 0.15
    gap_val = abs(gap_top - gap_bot)
    gap_mid = (gap_top + gap_bot) / 2.0
    ax.annotate('', xy=(gap_z, gap_top), xytext=(gap_z, gap_bot),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(gap_z + 0.08, gap_mid,
            f'{gap_val:.3f} m\n(SDF gap)',
            fontsize=11, color='blue', fontweight='bold', va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.9))

    # Reference lines
    ax.axhline(y=post_y_lo_inf, color='red', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axhline(y=goal_y_max_cur, color='green', linewidth=0.8, linestyle=':', alpha=0.6)

    # Key values text box
    info = (
        f"chaser_buffer (sphere) = {cb:.3f} m\n"
        f"cb_tip (aligned)       = {cb_tip:.3f} m\n"
        f"post_length            = {post_length} m\n"
        f"\n"
        f"Inflated post tip  y = {post_y_lo_inf:.3f} m  (was {post_y_lo_inf_old:.3f})\n"
        f"Goal top           y = {goal_y_max_cur:.3f} m  (was {goal_y_max_prev:.3f})\n"
        f"Goal bottom        y = {goal_y_min_cur:.3f} m\n"
        f"\n"
        f"eps_p = {eps_p_cur} m,  eps_q = {eps_q_cur} rad ({math.degrees(eps_q_cur):.1f}°)\n"
        f"Physical hull clearance = {phys_clearance:.3f} m\n"
        f"Max corner Δy at eps_q  = {corner_dy:.4f} m\n"
        f"Safety margin           = {safety_margin:.4f} m"
    )
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.0, 4.5)
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
                  f'eps_q = {eps_q_cur} rad ({math.degrees(eps_q_cur):.1f}°), '
                  f'safety margin = {safety_margin*1000:.1f} mm', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Post (physical)
    ax2.add_patch(patches.Rectangle(
        (-post_hw_z, -post_length), 2*post_hw_z, post_length,
        linewidth=2, edgecolor='black', facecolor='gray', alpha=0.4, label='Post (physical)'))

    # Inflated post (current, cb_tip for tip)
    ax2.add_patch(patches.Rectangle(
        (post_z_lo_inf, post_y_lo_inf),
        post_z_hi_inf - post_z_lo_inf, post_y_hi_inf - post_y_lo_inf,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.10,
        label=f'Failure (tip cb_tip={cb_tip})'))

    # Old inflated post (dashed, full cb for tip)
    ax2.add_patch(patches.Rectangle(
        (post_z_lo_inf, post_y_lo_inf_old),
        post_z_hi_inf - post_z_lo_inf, post_y_hi_inf - post_y_lo_inf_old,
        linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.4,
        linestyle='--', label=f'Old failure (tip cb={cb:.2f})'))

    # Current goal
    ax2.add_patch(patches.Rectangle(
        (-eps_p_cur, goal_y_min_cur), 2*eps_p_cur, goal_y_max_cur - goal_y_min_cur,
        linewidth=2.5, edgecolor='green', facecolor='green', alpha=0.20,
        label='Current goal'))

    # Previous goal (dashed)
    ax2.add_patch(patches.Rectangle(
        (-eps_p_prev, goal_y_min_prev), 2*eps_p_prev, goal_y_max_prev - goal_y_min_prev,
        linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--',
        label='Previous goal'))

    # Chaser silhouette at goal_y_max (aligned, showing physical extent)
    chaser_y_center = goal_y_max_cur
    ax2.add_patch(patches.Rectangle(
        (-0.5, chaser_y_center - 0.5), 1.0, 1.0,
        linewidth=1.5, edgecolor='blue', facecolor='blue', alpha=0.08,
        linestyle='-.', label='Chaser at goal top'))

    # Chaser top face to post tip distance
    chaser_top = chaser_y_center + 0.5
    ax2.annotate('', xy=(0.6, -post_length), xytext=(0.6, chaser_top),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    hull_gap = abs(-post_length - chaser_top)
    ax2.text(0.65, (-post_length + chaser_top) / 2,
             f'{hull_gap*1000:.0f} mm\nhull gap',
             fontsize=11, color='purple', fontweight='bold', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.9))

    # SDF gap annotation
    ax2.annotate('', xy=(0.18, gap_top), xytext=(0.18, gap_bot),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax2.text(0.22, gap_mid,
             f'{gap_val*1000:.0f} mm SDF',
             fontsize=10, color='blue', fontweight='bold', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='blue', alpha=0.9))

    # Corner sweep arc (show worst-case rotation displacement)
    # Corner at (0.5, 0.5) relative to chaser center → world (0.5, chaser_top)
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
        (post_y_lo_inf, 'red', f'y={post_y_lo_inf:.3f} (inflated tip)'),
        (-post_length, 'black', f'y={-post_length:.1f} (physical tip)'),
        (goal_y_max_cur, 'green', f'y={goal_y_max_cur:.3f} (goal top)'),
        (goal_y_min_cur, 'green', f'y={goal_y_min_cur:.3f} (goal bot)'),
    ]:
        ax2.axhline(y=y_val, color=color, linewidth=0.8, linestyle=':', alpha=0.5)
        ax2.text(-1.9, y_val + 0.012, label, fontsize=8, color=color)

    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(goal_y_min_cur - 0.4, 0.3)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    out_path2 = out_dir / 'goal_geometry_yz_zoom.png'
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")

    plt.close('all')


if __name__ == '__main__':
    main()
