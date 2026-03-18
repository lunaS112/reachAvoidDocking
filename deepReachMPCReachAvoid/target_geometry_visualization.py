#!/usr/bin/env python3
"""Standalone visualization of the 6D target geometry (body + docking post).

Plots:
  - Target spacecraft body (rectangle)
  - Docking post (small rectangle protruding in -y)
  - Inflated obstacle boundary (chaser buffer)
  - Goal band (reach set in position space)
  - Coordinate annotations

Run:  python target_geometry_visualization.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patches as mpath_patches

# ─── geometry constants (must match dynamics.py Docking6D) ───
W_T = 6.0            # target body width (x)
H_T = 3.0            # target body height (y)
POST_HW_X = 0.6      # post half-width in x
POST_LENGTH = 0.2    # post extent in -y
W_C, H_C = 1.0, 1.0  # chaser dimensions
CHASER_BUFFER = np.sqrt(W_C**2 + H_C**2) / 2  # ≈ 0.707

GOAL_CLEARANCE = -0.007
GOAL_BAND_HEIGHT = 0.5
GOAL_Y_MAX = -(POST_LENGTH + CHASER_BUFFER + GOAL_CLEARANCE)   # ≈ -1.0
GOAL_Y_MIN = GOAL_Y_MAX - GOAL_BAND_HEIGHT                     # ≈ -1.4
GOAL_Y_CENTER = (GOAL_Y_MIN + GOAL_Y_MAX) / 2.0                # ≈ -1.2
EPS_P = 0.1  # position tolerance


def _rect_outline(cx, y_lo, half_w, y_hi):
    """Return (xs, ys) for a closed rectangle centered at cx."""
    return ([cx - half_w, cx - half_w, cx + half_w, cx + half_w, cx - half_w],
            [y_lo, y_hi, y_hi, y_lo, y_lo])


def main():
    fig, ax = plt.subplots(figsize=(8, 10))

    # --- Target body (filled gray) ---
    body = mpatches.Rectangle((-W_T / 2, 0), W_T, H_T,
                              facecolor='lightgray', edgecolor='black',
                              linewidth=1.5, label='Target body')
    ax.add_patch(body)

    # --- Docking post (filled gray) ---
    post = mpatches.Rectangle((-POST_HW_X, -POST_LENGTH), 2 * POST_HW_X, POST_LENGTH,
                              facecolor='lightgray', edgecolor='black', linewidth=1.5,
                              label='Docking post')
    ax.add_patch(post)

    # --- Inflated obstacle boundary (dashed) ---
    cb = CHASER_BUFFER
    inflated_verts = [
        (-(W_T / 2 + cb), -cb),
        (-(W_T / 2 + cb), H_T + cb),
        (W_T / 2 + cb, H_T + cb),
        (W_T / 2 + cb, -cb),
        (POST_HW_X + cb, -cb),
        (POST_HW_X + cb, -(POST_LENGTH + cb)),
        (-(POST_HW_X + cb), -(POST_LENGTH + cb)),
        (-(POST_HW_X + cb), -cb),
        (-(W_T / 2 + cb), -cb),
    ]
    xs, ys = zip(*inflated_verts)
    ax.plot(xs, ys, 'r--', linewidth=1.5, label=f'Inflated boundary (buf={cb:.3f})')

    # --- Goal band ---
    goal_rect = mpatches.Rectangle((-EPS_P, GOAL_Y_MIN), 2 * EPS_P,
                                   GOAL_Y_MAX - GOAL_Y_MIN,
                                   facecolor='lime', edgecolor='darkgreen',
                                   alpha=0.5, linewidth=1.5, label='Goal band')
    ax.add_patch(goal_rect)
    ax.plot(0, GOAL_Y_CENTER, 'g*', markersize=14, zorder=30, label=f'Goal center (0, {GOAL_Y_CENTER:.1f})')

    # --- Annotations ---
    ax.annotate(f'w_t = {W_T}', xy=(0, H_T), xytext=(0, H_T + 0.5),
                ha='center', fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'h_t = {H_T}', xy=(W_T / 2 + 0.3, H_T / 2), ha='left', fontsize=9)
    ax.annotate(f'post_hw_x = {POST_HW_X}', xy=(POST_HW_X, -POST_LENGTH / 2),
                xytext=(1.5, -POST_LENGTH / 2), ha='left', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'post_length = {POST_LENGTH}', xy=(0, -POST_LENGTH),
                xytext=(0, -POST_LENGTH - 0.4), ha='center', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'goal_y_max = {GOAL_Y_MAX:.3f}', xy=(EPS_P + 0.1, GOAL_Y_MAX),
                fontsize=7, color='darkgreen')
    ax.annotate(f'goal_y_min = {GOAL_Y_MIN:.3f}', xy=(EPS_P + 0.1, GOAL_Y_MIN),
                fontsize=7, color='darkgreen')

    # --- Formatting ---
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.5, 5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Docking6D Target Geometry (2D simplification of 13D)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('target_geometry_6D.png', dpi=150)
    print('Saved target_geometry_6D.png')
    plt.show()


if __name__ == '__main__':
    main()
