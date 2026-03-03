#!/usr/bin/env python3
"""
Visualize the target satellite geometry for the 13D docking problem.

Generates an interactive 3D Plotly figure showing:
  • Target body (silver rectangular prism)
  • Docking post / peg (green, protruding from the -y face)
  • Goal band (translucent yellow slab at post tip)
  • Chaser spacecraft cube (orange wireframe, for scale)
  • Approach direction arrow
  • Dimension annotations

Usage:
    python visualize_target.py                       # opens in browser
    python visualize_target.py --save target.html    # save to file
    python visualize_target.py --backend mpl         # matplotlib static fig
"""

from __future__ import annotations

import argparse
import math
import numpy as np

# ──────────────────────────────────────────────────────────────────────────── #
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────── #

def _box_mesh(center, half_extents):
    """Return (vertices, i/j/k triangle index lists) for a Plotly Mesh3d box."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    verts_x = [cx + sx * hx for sx in (-1, 1, 1, -1, -1, 1, 1, -1)]
    verts_y = [cy + sy * hy for sy in (-1, -1, 1, 1, -1, -1, 1, 1)]
    verts_z = [cz + sz * hz for sz in (-1, -1, -1, -1, 1, 1, 1, 1)]

    quad_idx = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    i_list, j_list, k_list = [], [], []
    for qi in quad_idx:
        i_list += [qi[0], qi[0]]
        j_list += [qi[1], qi[2]]
        k_list += [qi[2], qi[3]]
    return verts_x, verts_y, verts_z, i_list, j_list, k_list


def _box_wireframe(center, half_extents):
    """Return (xe, ye, ze) line segments for a wireframe box."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    corners = np.array([
        [cx + sx * hx, cy + sy * hy, cz + sz * hz]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])  # 8 corners, ordered: ---,--+,-+-,-++,+--,+-+,++-,+++
    #   0=---, 1=--+, 2=-+-, 3=-++, 4=+--, 5=+-+, 6=++-, 7=+++
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # z-parallel
        (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel
        (0, 4), (1, 5), (2, 6), (3, 7),  # x-parallel
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe += [corners[a, 0], corners[b, 0], None]
        ye += [corners[a, 1], corners[b, 1], None]
        ze += [corners[a, 2], corners[b, 2], None]
    return xe, ye, ze


def _rotated_box_wireframe(center, half_extents, quaternion):
    """Wireframe box rotated by a scalar-first quaternion [q0,q1,q2,q3]."""
    hx, hy, hz = half_extents
    body_corners = np.array([
        [sx * hx, sy * hy, sz * hz]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    q0, q1, q2, q3 = quaternion
    R = np.array([
        [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)],
    ])
    # R is LVLH→Body, so R^T maps Body→LVLH
    lvlh_corners = (R.T @ body_corners.T).T + np.array(center)

    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe += [lvlh_corners[a, 0], lvlh_corners[b, 0], None]
        ye += [lvlh_corners[a, 1], lvlh_corners[b, 1], None]
        ze += [lvlh_corners[a, 2], lvlh_corners[b, 2], None]
    return xe, ye, ze


# ──────────────────────────────────────────────────────────────────────────── #
#  Plotly visualisation
# ──────────────────────────────────────────────────────────────────────────── #

def build_plotly_figure():
    """Build and return a Plotly Figure of the target satellite geometry."""
    import plotly.graph_objects as go

    # ── Spacecraft parameters (match Docking13D) ──────────────────────────
    w_t, h_t, d_t = 6.0, 3.0, 3.0        # target dimensions
    w_c, h_c, d_c = 1.0, 1.0, 1.0        # chaser dimensions
    post_hw_x  = 0.60                      # post half-width x  (1.2× chaser)
    post_hw_z  = 0.60                      # post half-width z  (1.2× chaser)
    post_length = 0.20                      # outdent (0.2× chaser)
    h_c_half   = 0.50                      # chaser half-height (y)
    goal_y_min = -post_length - 0.10 - h_c_half  # -0.80  (face 0.1m past tip)
    goal_y_max = -post_length - h_c_half          # -0.70  (face at tip)
    q_goal = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]  # 90° yaw

    fig = go.Figure()

    # 1. Target body (silver box, y ∈ [0, h_t]) ─────────────────────────────
    bx, by, bz, bi, bj, bk = _box_mesh(
        center=(0, h_t / 2, 0),
        half_extents=(w_t / 2, h_t / 2, d_t / 2))
    fig.add_trace(go.Mesh3d(
        x=bx, y=by, z=bz, i=bi, j=bj, k=bk,
        color='silver', opacity=0.40, flatshading=True,
        name='Target body',
    ))
    ex, ey, ez = _box_wireframe(
        center=(0, h_t / 2, 0),
        half_extents=(w_t / 2, h_t / 2, d_t / 2))
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color='gray', width=2),
        name='Target edges', showlegend=False,
    ))

    # 2. Docking post (green peg, y ∈ [-post_length, 0]) ───────────────────
    px, py, pz, pi, pj, pk = _box_mesh(
        center=(0, -post_length / 2, 0),
        half_extents=(post_hw_x, post_length / 2, post_hw_z))
    fig.add_trace(go.Mesh3d(
        x=px, y=py, z=pz, i=pi, j=pj, k=pk,
        color='limegreen', opacity=0.55, flatshading=True,
        name='Docking post',
    ))
    # Post base rim at y=0
    rim_x = [-post_hw_x, post_hw_x, post_hw_x, -post_hw_x, -post_hw_x]
    rim_y = [0, 0, 0, 0, 0]
    rim_z = [-post_hw_z, -post_hw_z, post_hw_z, post_hw_z, -post_hw_z]
    fig.add_trace(go.Scatter3d(
        x=rim_x, y=rim_y, z=rim_z, mode='lines',
        line=dict(color='limegreen', width=5),
        name='Post base rim',
    ))
    # Post wireframe edges
    pxe, pye, pze = _box_wireframe(
        center=(0, -post_length / 2, 0),
        half_extents=(post_hw_x, post_length / 2, post_hw_z))
    fig.add_trace(go.Scatter3d(
        x=pxe, y=pye, z=pze, mode='lines',
        line=dict(color='green', width=2),
        name='Post edges', showlegend=False,
    ))

    # 2b. Goal band (translucent yellow slab, y ∈ [goal_y_min, goal_y_max]) ─
    eps_p = 0.1
    goal_hw_x = post_hw_x + eps_p
    goal_hw_z = post_hw_z + eps_p
    goal_cy = (goal_y_min + goal_y_max) / 2
    goal_hy = (goal_y_max - goal_y_min) / 2
    gx, gy, gz, gi, gj, gk = _box_mesh(
        center=(0, goal_cy, 0),
        half_extents=(goal_hw_x, goal_hy, goal_hw_z))
    fig.add_trace(go.Mesh3d(
        x=gx, y=gy, z=gz, i=gi, j=gj, k=gk,
        color='gold', opacity=0.25, flatshading=True,
        name='Goal band',
    ))

    # 3. Chaser ghost at goal center (orange wireframe, rotated to goal) ───
    goal_y_center = (goal_y_min + goal_y_max) / 2
    cxe, cye, cze = _rotated_box_wireframe(
        center=(0, goal_y_center, 0),
        half_extents=(w_c / 2, h_c / 2, d_c / 2),
        quaternion=q_goal)
    fig.add_trace(go.Scatter3d(
        x=cxe, y=cye, z=cze, mode='lines',
        line=dict(color='orange', width=3),
        name='Chaser (goal pose)',
    ))

    # 5. Approach arrow (pointing in -y) ────────────────────────────────────
    arrow_y0 = -3.0
    arrow_y1 = -0.5
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[arrow_y0, arrow_y1], z=[0, 0],
        mode='lines',
        line=dict(color='red', width=4),
        name='Approach dir', showlegend=False,
    ))
    fig.add_trace(go.Cone(
        x=[0], y=[arrow_y1], z=[0],
        u=[0], v=[0.6], w=[0],
        sizemode='absolute', sizeref=0.3,
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False, name='Approach arrow',
    ))

    # 6. Dimension annotations ──────────────────────────────────────────────
    annots = [
        # Target width
        dict(x=-w_t/2, y=-0.4, z=d_t/2 + 0.3,
             text=f"w_t = {w_t}m", font=dict(size=11, color='gray')),
        # Target height
        dict(x=w_t/2 + 0.3, y=h_t/2, z=d_t/2 + 0.3,
             text=f"h_t = {h_t}m", font=dict(size=11, color='gray')),
        # Target depth
        dict(x=w_t/2 + 0.3, y=-0.4, z=0,
             text=f"d_t = {d_t}m", font=dict(size=11, color='gray')),
        # Post cross-section
        dict(x=post_hw_x + 0.15, y=-post_length/2, z=-post_hw_z - 0.2,
             text=f"post = {2*post_hw_x:.1f}×{2*post_hw_z:.1f}m",
             font=dict(size=10, color='green')),
        # Post length
        dict(x=post_hw_x + 0.2, y=-post_length/2, z=0,
             text=f"length = {post_length:.2f}m",
             font=dict(size=10, color='green')),
        # Goal band
        dict(x=0, y=goal_y_center, z=goal_hw_z + 0.3,
             text=f"goal band y∈[{goal_y_min},{goal_y_max}]",
             font=dict(size=10, color='goldenrod')),
        # Chaser
        dict(x=0, y=-1.5, z=d_c/2 + 0.3,
             text=f"chaser = {w_c}×{h_c}×{d_c}m",
             font=dict(size=10, color='orange')),
    ]

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f"Target Satellite Geometry — "
                  f"Docking Post ({2*post_hw_x:.1f}×{2*post_hw_z:.1f}m, "
                  f"length {post_length:.2f}m)"),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title='X  (m)',
            yaxis_title='Y  (m)  — approach from -Y',
            zaxis_title='Z  (m)',
            aspectmode='data',
            annotations=[
                dict(showarrow=False, **a) for a in annots
            ],
            camera=dict(
                eye=dict(x=1.8, y=-1.4, z=0.9),
                up=dict(x=0, y=1, z=0),
            ),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────── #
#  Matplotlib fallback
# ──────────────────────────────────────────────────────────────────────────── #

def build_mpl_figure():
    """Build a matplotlib 3D figure of the target satellite."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    w_t, h_t, d_t = 6.0, 3.0, 3.0
    w_c, h_c, d_c = 1.0, 1.0, 1.0
    post_hw_x  = 0.60
    post_hw_z  = 0.60
    post_length = 0.20
    h_c_half   = h_c / 2  # 0.5
    goal_y_min = -post_length - 0.10 - h_c_half  # -0.80
    goal_y_max = -post_length - h_c_half          # -0.70
    goal_y_center = (goal_y_min + goal_y_max) / 2
    q_goal = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Helper: polygon faces of an axis-aligned box
    def _box_faces(center, hx, hy, hz):
        cx, cy, cz = center
        c = np.array([
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ])
        faces = [
            [c[0], c[1], c[2], c[3]],
            [c[4], c[5], c[6], c[7]],
            [c[0], c[1], c[5], c[4]],
            [c[2], c[3], c[7], c[6]],
            [c[0], c[3], c[7], c[4]],
            [c[1], c[2], c[6], c[5]],
        ]
        return faces

    # Target body
    faces_t = _box_faces((0, h_t / 2, 0), w_t / 2, h_t / 2, d_t / 2)
    mesh_t = Poly3DCollection(faces_t, alpha=0.25, linewidths=0.5)
    mesh_t.set_facecolor('silver')
    mesh_t.set_edgecolor('gray')
    ax.add_collection3d(mesh_t)

    # Docking post (peg)
    faces_p = _box_faces((0, -post_length / 2, 0),
                         post_hw_x, post_length / 2, post_hw_z)
    mesh_p = Poly3DCollection(faces_p, alpha=0.55, linewidths=1)
    mesh_p.set_facecolor('limegreen')
    mesh_p.set_edgecolor('green')
    ax.add_collection3d(mesh_p)

    # Goal band
    goal_hy = (goal_y_max - goal_y_min) / 2
    eps_p = 0.1
    faces_g = _box_faces((0, goal_y_center, 0),
                         post_hw_x + eps_p, goal_hy, post_hw_z + eps_p)
    mesh_g = Poly3DCollection(faces_g, alpha=0.2, linewidths=0.5)
    mesh_g.set_facecolor('gold')
    mesh_g.set_edgecolor('goldenrod')
    ax.add_collection3d(mesh_g)

    # Chaser at goal center (rotated wireframe)
    q0, q1, q2, q3 = q_goal
    R = np.array([
        [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)],
    ])
    body_c = np.array([
        [sx * w_c / 2, sy * h_c / 2, sz * d_c / 2]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    world_c = (R.T @ body_c.T).T + np.array([0, goal_y_center, 0])
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot([world_c[a, 0], world_c[b, 0]],
                [world_c[a, 1], world_c[b, 1]],
                [world_c[a, 2], world_c[b, 2]],
                color='orange', linewidth=2)

    # Approach arrow
    ax.quiver(0, -3, 0, 0, 2.5, 0, color='red',
              arrow_length_ratio=0.15, linewidth=2)
    ax.text(0.2, -2, 0, 'Approach', color='red', fontsize=9)

    # Labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(
        f'Target Satellite — Docking Post '
        f'({2*post_hw_x:.1f}×{2*post_hw_z:.1f}m, length {post_length:.2f}m)')

    # Equal aspect ratio
    max_range = max(w_t, h_t, d_t) / 2 + 1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-2, h_t + 1)
    ax.set_zlim(-max_range, max_range)

    return fig


# ──────────────────────────────────────────────────────────────────────────── #
#  CLI
# ──────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description='Visualize 13D docking target satellite geometry.')
    parser.add_argument('--backend', choices=['plotly', 'mpl'],
                        default='plotly',
                        help='Rendering backend (default: plotly)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save to file instead of showing '
                             '(e.g. target.html or target.png)')
    args = parser.parse_args()

    if args.backend == 'plotly':
        fig = build_plotly_figure()
        if args.save:
            fig.write_html(args.save)
            print(f'Saved → {args.save}')
        else:
            fig.show()
    else:
        fig = build_mpl_figure()
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f'Saved → {args.save}')
        else:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':
    main()
