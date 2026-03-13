#!/usr/bin/env python3
"""
Standalone visualization of goal set vs failure set geometry for Docking13D.

Shows YZ-plane cross-section (x=0) with:
  - Target body and docking post (physical geometry)
  - OLD failure set envelope (full bounding sphere cb=0.866)
  - NEW failure set envelope (orientation-dependent cb_y at eps_q boundary)
  - Old and new goal set positions
  - Labeled gaps

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

    eps_p = 0.10
    eps_q = 0.151  # rad (~5° per axis worst-case)

    # ── Orientation-dependent Y-inflation ──
    # cb_y_worst = max Y-extent of chaser at eps_q from goal quat
    # = 0.5 * (cos(eps_q) + sqrt(2)*sin(eps_q))
    cb_y_worst = 0.5 * (math.cos(eps_q) + math.sqrt(2) * math.sin(eps_q))
    cb_y_aligned = 0.5  # at goal quaternion (one face aligned)

    # ── OLD goal set (full cb everywhere) ──
    old_goal_clearance = 0.134
    old_goal_band_height = 0.4
    old_post_tip = -(post_length + cb)
    old_goal_y_max = -(post_length + cb + old_goal_clearance)
    old_goal_y_min = old_goal_y_max - old_goal_band_height

    # ── NEW goal set (orientation-dependent cb_y) ──
    new_goal_clearance = 0.07
    new_goal_band_height = 0.4
    new_post_tip = -(post_length + cb_y_worst)     # worst-case within eps_q
    new_goal_y_max = -(post_length + cb_y_worst + new_goal_clearance)
    new_goal_y_min = new_goal_y_max - new_goal_band_height

    # Post tip at aligned quaternion (tightest possible)
    aligned_post_tip = -(post_length + cb_y_aligned)

    # ── Inflated bounds for drawing ──
    # OLD (full cb on all faces)
    old_post_z_lo = -(post_hw_z + cb)
    old_post_z_hi = (post_hw_z + cb)
    old_post_y_lo = old_post_tip
    old_post_y_hi = cb

    old_body_z_lo = -(d_t / 2.0 + cb)
    old_body_z_hi = (d_t / 2.0 + cb)
    old_body_y_lo = -cb
    old_body_y_hi = h_t + cb

    # NEW (cb_y_worst on Y faces, full cb on Z faces)
    new_post_z_lo = -(post_hw_z + cb)
    new_post_z_hi = (post_hw_z + cb)
    new_post_y_lo = new_post_tip
    new_post_y_hi = cb_y_worst

    new_body_z_lo = -(d_t / 2.0 + cb)
    new_body_z_hi = (d_t / 2.0 + cb)
    new_body_y_lo = -cb_y_worst
    new_body_y_hi = h_t + cb

    # ── Gaps ──
    old_gap = abs(old_goal_y_max - old_post_tip)
    new_gap = abs(new_goal_y_max - new_post_tip)
    goal_shift = old_goal_y_max - new_goal_y_max  # positive = moved closer (less negative)

    # ══════════════════════════════════════════════════════════════════
    #  Figure 1: Overview comparison
    # ══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlabel('Z (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Docking13D Geometry — Old vs New Failure Set\n'
                 'Orientation-dependent cb_y replaces full bounding sphere on Y faces',
                 fontsize=14)
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

    # OLD inflated post (full cb) — dashed outline only
    ax.add_patch(patches.Rectangle(
        (old_post_z_lo, old_post_y_lo),
        old_post_z_hi - old_post_z_lo, old_post_y_hi - old_post_y_lo,
        linewidth=1.5, edgecolor='red', facecolor='none',
        linestyle='--', label=f'OLD failure (cb={cb:.3f})', zorder=1))

    # OLD inflated body — dashed outline only
    ax.add_patch(patches.Rectangle(
        (old_body_z_lo, old_body_y_lo),
        old_body_z_hi - old_body_z_lo, old_body_y_hi - old_body_y_lo,
        linewidth=1.5, edgecolor='red', facecolor='none',
        linestyle='--', zorder=1))

    # NEW inflated post (cb_y_worst) — solid fill
    ax.add_patch(patches.Rectangle(
        (new_post_z_lo, new_post_y_lo),
        new_post_z_hi - new_post_z_lo, new_post_y_hi - new_post_y_lo,
        linewidth=2, edgecolor='orangered', facecolor='orangered', alpha=0.10,
        label=f'NEW failure (cb_y={cb_y_worst:.3f} at eps_q)', zorder=1))

    # NEW inflated body (cb_y_worst)
    ax.add_patch(patches.Rectangle(
        (new_body_z_lo, new_body_y_lo),
        new_body_z_hi - new_body_z_lo, new_body_y_hi - new_body_y_lo,
        linewidth=2, edgecolor='orangered', facecolor='orangered', alpha=0.10,
        zorder=1))

    # OLD goal set — dashed green outline
    ax.add_patch(patches.Rectangle(
        (-eps_p, old_goal_y_min), 2*eps_p, old_goal_y_max - old_goal_y_min,
        linewidth=2, edgecolor='green', facecolor='none',
        linestyle='--', label=f'OLD goal (gap={old_goal_clearance}m)', zorder=3))

    # NEW goal set — solid green fill
    ax.add_patch(patches.Rectangle(
        (-eps_p, new_goal_y_min), 2*eps_p, new_goal_y_max - new_goal_y_min,
        linewidth=2.5, edgecolor='limegreen', facecolor='limegreen', alpha=0.25,
        label=f'NEW goal (gap={new_goal_clearance}m)', zorder=3))

    # Reference lines
    ax.axhline(y=old_post_tip, color='red', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(y=new_post_tip, color='orangered', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axhline(y=old_goal_y_max, color='green', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(y=new_goal_y_max, color='limegreen', linewidth=0.8, linestyle=':', alpha=0.6)

    # Info box
    info = (
        f"OLD: cb = {cb:.3f}m (full sphere)\n"
        f"     post tip = {old_post_tip:.3f}m\n"
        f"     goal top = {old_goal_y_max:.3f}m  (gap={old_gap*1000:.0f}mm)\n"
        f"\n"
        f"NEW: cb_y(eps_q={eps_q}) = {cb_y_worst:.3f}m\n"
        f"     cb_y(aligned)    = {cb_y_aligned:.3f}m\n"
        f"     post tip = {new_post_tip:.3f}m\n"
        f"     goal top = {new_goal_y_max:.3f}m  (gap={new_gap*1000:.0f}mm)\n"
        f"\n"
        f"Goal moved {abs(goal_shift)*1000:.0f}mm closer\n"
        f"eps_p = {eps_p}m,  eps_q = {eps_q} rad ({math.degrees(eps_q):.1f}deg)"
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

    # ══════════════════════════════════════════════════════════════════
    #  Figure 2: Zoomed gap detail — NEW geometry
    # ══════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    ax2.set_aspect('equal')
    ax2.set_xlabel('Z (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('NEW Goal–Failure Gap Detail — YZ Plane (x = 0)\n'
                  f'cb_y(eps_q={eps_q}) = {cb_y_worst:.3f}m, '
                  f'goal_clearance = {new_goal_clearance}m', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Physical post
    ax2.add_patch(patches.Rectangle(
        (-post_hw_z, -post_length), 2*post_hw_z, post_length,
        linewidth=2, edgecolor='black', facecolor='gray', alpha=0.4,
        label='Post (physical)'))

    # OLD inflated post — dashed
    ax2.add_patch(patches.Rectangle(
        (old_post_z_lo, old_post_y_lo),
        old_post_z_hi - old_post_z_lo, old_post_y_hi - old_post_y_lo,
        linewidth=1.5, edgecolor='red', facecolor='none',
        linestyle='--', label=f'OLD failure (cb={cb:.2f})'))

    # NEW inflated post — solid
    ax2.add_patch(patches.Rectangle(
        (new_post_z_lo, new_post_y_lo),
        new_post_z_hi - new_post_z_lo, new_post_y_hi - new_post_y_lo,
        linewidth=2, edgecolor='orangered', facecolor='orangered', alpha=0.12,
        label=f'NEW failure (cb_y={cb_y_worst:.3f})'))

    # Aligned inflation (best case) — thin dotted
    aligned_post_y_lo = aligned_post_tip
    aligned_post_y_hi = cb_y_aligned
    ax2.add_patch(patches.Rectangle(
        (new_post_z_lo, aligned_post_y_lo),
        new_post_z_hi - new_post_z_lo, aligned_post_y_hi - aligned_post_y_lo,
        linewidth=1, edgecolor='blue', facecolor='none',
        linestyle=':', label=f'At goal quat (cb_y={cb_y_aligned:.2f})'))

    # OLD goal — dashed
    ax2.add_patch(patches.Rectangle(
        (-eps_p, old_goal_y_min), 2*eps_p, old_goal_y_max - old_goal_y_min,
        linewidth=1.5, edgecolor='green', facecolor='none',
        linestyle='--', label='OLD goal'))

    # NEW goal — solid
    ax2.add_patch(patches.Rectangle(
        (-eps_p, new_goal_y_min), 2*eps_p, new_goal_y_max - new_goal_y_min,
        linewidth=2.5, edgecolor='limegreen', facecolor='limegreen', alpha=0.25,
        label='NEW goal'))

    # Chaser silhouette at new goal_y_max (aligned orientation)
    chaser_y = new_goal_y_max
    ax2.add_patch(patches.Rectangle(
        (-0.5, chaser_y - 0.5), 1.0, 1.0,
        linewidth=1.5, edgecolor='blue', facecolor='blue', alpha=0.08,
        linestyle='-.', label='Chaser at goal top (aligned)'))

    # Hull gap (physical clearance: chaser top at goal_y_max + 0.5 vs post tip)
    chaser_top = chaser_y + 0.5
    ax2.annotate('', xy=(0.65, -post_length), xytext=(0.65, chaser_top),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    hull_gap = abs(-post_length - chaser_top)
    ax2.text(0.70, (-post_length + chaser_top) / 2,
             f'{hull_gap*1000:.0f} mm\nhull gap',
             fontsize=11, color='purple', fontweight='bold', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='purple', alpha=0.9))

    # NEW SDF gap annotation
    new_gap_mid = (new_post_tip + new_goal_y_max) / 2.0
    ax2.annotate('', xy=(eps_p + 0.12, new_post_tip),
                 xytext=(eps_p + 0.12, new_goal_y_max),
                 arrowprops=dict(arrowstyle='<->', color='orangered', lw=2))
    ax2.text(eps_p + 0.17, new_gap_mid,
             f'{new_gap*1000:.0f} mm\nNEW gap',
             fontsize=10, color='orangered', fontweight='bold',
             va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='orangered', alpha=0.9))

    # OLD SDF gap annotation (farther right)
    old_gap_mid = (old_post_tip + old_goal_y_max) / 2.0
    ax2.annotate('', xy=(eps_p + 0.55, old_post_tip),
                 xytext=(eps_p + 0.55, old_goal_y_max),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=1.5,
                                 linestyle='--'))
    ax2.text(eps_p + 0.60, old_gap_mid,
             f'{old_gap*1000:.0f} mm\nOLD gap',
             fontsize=9, color='red', va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='red', alpha=0.8))

    # Reference lines with labels
    for y_val, color, lbl in [
        (old_post_tip, 'red',
         f'y={old_post_tip:.3f} (OLD tip, cb={cb:.3f})'),
        (new_post_tip, 'orangered',
         f'y={new_post_tip:.3f} (NEW tip, cb_y={cb_y_worst:.3f})'),
        (aligned_post_tip, 'blue',
         f'y={aligned_post_tip:.3f} (aligned tip, cb_y={cb_y_aligned:.2f})'),
        (-post_length, 'black',
         f'y={-post_length:.1f} (physical tip)'),
        (new_goal_y_max, 'limegreen',
         f'y={new_goal_y_max:.3f} (NEW goal top)'),
        (old_goal_y_max, 'green',
         f'y={old_goal_y_max:.3f} (OLD goal top)'),
        (new_goal_y_min, 'limegreen',
         f'y={new_goal_y_min:.3f} (NEW goal bot)'),
    ]:
        ax2.axhline(y=y_val, color=color, linewidth=0.8, linestyle=':', alpha=0.5)
        ax2.text(-2.4, y_val + 0.012, lbl, fontsize=7, color=color)

    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(new_goal_y_min - 0.4, 0.4)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    out_path2 = out_dir / 'goal_geometry_yz_zoom.png'
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")

    plt.close('all')

    # ── Summary to console ──
    print(f"\n{'='*50}")
    print(f"  GEOMETRY SUMMARY")
    print(f"{'='*50}")
    print(f"  cb (full sphere)     = {cb:.4f} m")
    print(f"  cb_y (at goal quat)  = {cb_y_aligned:.4f} m")
    print(f"  cb_y (worst at eps_q)= {cb_y_worst:.4f} m")
    print(f"")
    print(f"  OLD post tip (full cb)    = {old_post_tip:.4f} m")
    print(f"  NEW post tip (cb_y worst) = {new_post_tip:.4f} m")
    print(f"  Aligned post tip (cb_y=0.5) = {aligned_post_tip:.4f} m")
    print(f"")
    print(f"  OLD goal_y_max = {old_goal_y_max:.4f} m  (gap = {old_gap*1000:.0f} mm)")
    print(f"  NEW goal_y_max = {new_goal_y_max:.4f} m  (gap = {new_gap*1000:.0f} mm)")
    print(f"  Goal moved {abs(goal_shift)*1000:.0f} mm closer to target")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
