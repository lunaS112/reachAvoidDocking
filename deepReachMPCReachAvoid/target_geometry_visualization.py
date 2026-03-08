#!/usr/bin/env python3
"""Standalone visualization of the 6D target geometry (body + docking post).

Plots:
  - Target spacecraft body (rectangle)
  - Docking post (small rectangle protruding in -y)
  - Old inflated obstacle boundary (circular chaser buffer)
  - New inflated obstacle boundary (AABB at goal theta)
  - Old and new goal bands
  - Chaser at goal orientation (rotated by theta_goal)
  - Coordinate annotations

Run:  python target_geometry_visualization.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D

# ─── geometry constants (must match dynamics.py Docking6D) ───
W_T = 6.0            # target body width (x)
H_T = 3.0            # target body height (y)
POST_HW_X = 0.6      # post half-width in x
POST_LENGTH = 0.2     # post extent in -y
W_C, H_C = 1.0, 1.0  # chaser dimensions
CHASER_BUFFER = np.sqrt(W_C**2 + H_C**2) / 2  # ≈ 0.707  (old circular)

GOAL_CLEARANCE = 0.1
GOAL_BAND_HEIGHT = 0
THETA_GOAL = np.pi / 2  # goal orientation
EPS_P = 0.05  # position tolerance

# AABB half-extents at goal theta
H_X_GOAL = (W_C / 2) * abs(np.cos(THETA_GOAL)) + (H_C / 2) * abs(np.sin(THETA_GOAL))
H_Y_GOAL = (W_C / 2) * abs(np.sin(THETA_GOAL)) + (H_C / 2) * abs(np.cos(THETA_GOAL))

# Old goal (circular buffer)
OLD_GOAL_Y_MAX = -(POST_LENGTH + CHASER_BUFFER + GOAL_CLEARANCE)
OLD_GOAL_Y_MIN = OLD_GOAL_Y_MAX - GOAL_BAND_HEIGHT
OLD_GOAL_Y_CENTER = (OLD_GOAL_Y_MIN + OLD_GOAL_Y_MAX) / 2.0

# New goal (AABB at goal theta)
NEW_GOAL_Y_MAX = -(POST_LENGTH + H_Y_GOAL + GOAL_CLEARANCE)
NEW_GOAL_Y_MIN = NEW_GOAL_Y_MAX - GOAL_BAND_HEIGHT
NEW_GOAL_Y_CENTER = (NEW_GOAL_Y_MIN + NEW_GOAL_Y_MAX) / 2.0


def _inflated_boundary(h_x, h_y):
    """Return vertices for the inflated body+post outline with given half-extents."""
    return [
        (-(W_T / 2 + h_x), -h_y),
        (-(W_T / 2 + h_x), H_T + h_y),
        (W_T / 2 + h_x, H_T + h_y),
        (W_T / 2 + h_x, -h_y),
        (POST_HW_X + h_x, -h_y),
        (POST_HW_X + h_x, -(POST_LENGTH + h_y)),
        (-(POST_HW_X + h_x), -(POST_LENGTH + h_y)),
        (-(POST_HW_X + h_x), -h_y),
        (-(W_T / 2 + h_x), -h_y),
    ]


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

    # --- Old inflated boundary (circular buffer, dashed red) ---
    old_verts = _inflated_boundary(CHASER_BUFFER, CHASER_BUFFER)
    xs, ys = zip(*old_verts)
    ax.plot(xs, ys, 'r--', linewidth=1.2, alpha=0.6,
            label=f'Old circular buffer ({CHASER_BUFFER:.3f}m)')

    # --- New inflated boundary at goal theta (AABB, solid orange) ---
    new_verts = _inflated_boundary(H_X_GOAL, H_Y_GOAL)
    xs, ys = zip(*new_verts)
    ax.plot(xs, ys, color='darkorange', linewidth=1.8,
            label=f'AABB at θ={np.degrees(THETA_GOAL):.0f}° (h_x={H_X_GOAL:.3f}, h_y={H_Y_GOAL:.3f})')

    # --- Old goal band (dashed outline) ---
    old_goal = mpatches.Rectangle((-EPS_P, OLD_GOAL_Y_MIN), 2 * EPS_P,
                                   OLD_GOAL_Y_MAX - OLD_GOAL_Y_MIN if GOAL_BAND_HEIGHT > 0 else 0.02,
                                   facecolor='none', edgecolor='red',
                                   alpha=0.5, linewidth=1.2, linestyle='--',
                                   label=f'Old goal (y={OLD_GOAL_Y_CENTER:.3f})')
    ax.add_patch(old_goal)
    ax.plot(0, OLD_GOAL_Y_CENTER, 'r+', markersize=10, zorder=30)

    # --- New goal band (filled green) ---
    new_goal = mpatches.Rectangle((-EPS_P, NEW_GOAL_Y_MIN), 2 * EPS_P,
                                   NEW_GOAL_Y_MAX - NEW_GOAL_Y_MIN if GOAL_BAND_HEIGHT > 0 else 0.02,
                                   facecolor='lime', edgecolor='darkgreen',
                                   alpha=0.5, linewidth=1.5,
                                   label=f'New goal (y={NEW_GOAL_Y_CENTER:.3f})')
    ax.add_patch(new_goal)
    ax.plot(0, NEW_GOAL_Y_CENTER, 'g*', markersize=14, zorder=30,
            label=f'New goal center (0, {NEW_GOAL_Y_CENTER:.3f})')

    # --- Chaser at new goal, rotated by theta_goal ---
    chaser = mpatches.Rectangle((-W_C / 2, -H_C / 2), W_C, H_C,
                                facecolor='lightskyblue', edgecolor='blue',
                                linewidth=1.5, alpha=0.4, zorder=25)
    t = Affine2D().rotate(THETA_GOAL).translate(0, NEW_GOAL_Y_CENTER) + ax.transData
    chaser.set_transform(t)
    ax.add_patch(chaser)
    ax.plot([], [], 's', color='lightskyblue', markeredgecolor='blue',
            label=f'Chaser at goal (θ={np.degrees(THETA_GOAL):.0f}°)')

    # --- Chaser at old goal (outline only, for comparison) ---
    chaser_old = mpatches.Rectangle((-W_C / 2, -H_C / 2), W_C, H_C,
                                    facecolor='none', edgecolor='red',
                                    linewidth=1.0, linestyle='--', alpha=0.4, zorder=24)
    t_old = Affine2D().rotate(THETA_GOAL).translate(0, OLD_GOAL_Y_CENTER) + ax.transData
    chaser_old.set_transform(t_old)
    ax.add_patch(chaser_old)

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
    # Show the improvement
    dy = OLD_GOAL_Y_CENTER - NEW_GOAL_Y_CENTER
    ax.annotate('', xy=(0.3, NEW_GOAL_Y_CENTER), xytext=(0.3, OLD_GOAL_Y_CENTER),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    ax.text(0.4, (OLD_GOAL_Y_CENTER + NEW_GOAL_Y_CENTER) / 2,
            f'Δy = {dy:.3f}m\ncloser', fontsize=8, color='purple', va='center')

    # --- Formatting ---
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.5, 5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Docking6D Target Geometry: Old Circular vs New AABB Buffer')
    ax.legend(loc='upper left', fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('target_geometry_6D.png', dpi=150)
    print('Saved target_geometry_6D.png')
    plt.show()


if __name__ == '__main__':
    main()
