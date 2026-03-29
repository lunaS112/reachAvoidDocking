#!/usr/bin/env python3
"""Paper-quality two-panel docking geometry figure.

Panel (a): 2D cross-section (x–y) for the 6-state planar model.
Panel (b): 3D isometric view for the 13-state model.

Elements shown:
  - Target body (rectangle / prism)
  - Docking post (protruding in -y)
  - Failure-set boundary (red dashed) — body + post inflated by collision buffer
  - Goal set (green band below docking-post tip)
  - Collision-buffer region (light red fill / annotation)
  - LVLH coordinate axes

Run:  python target_geometry_visualization.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ─── Paper-quality matplotlib style ──────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Palatino Linotype', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ─── Consistent colour palette ───────────────────────────────────────────────
C_BODY   = '#D0D0D0'       # target body / post fill
C_EDGE   = '#404040'       # body / post edge
C_GOAL   = '#2ca02c'       # goal-set fill
C_GOAL_E = '#1a6d1a'       # goal-set edge
C_FAIL   = '#d62728'       # failure-set boundary
C_BUF    = '#FFDDDD'       # collision-buffer fill (very light red)
C_CHASER = '#ff9500'       # chaser body (BRAT highlight orange)
C_CHASER_E = '#cc7700'    # chaser edge
C_APPR   = '#0048a6'       # approach-direction arrow (guide dark blue)

# ─── Shared target / chaser dimensions ───────────────────────────────────────
W_T   = 6.0                # target body width  (x)   [m]
H_T   = 3.0                # target body height (y)   [m]
P_HWX = 0.6                # post half-width    (x)   [m]
P_LEN = 0.2                # post length (-y extent)  [m]
W_C   = 1.0                # chaser width             [m]
H_C   = 1.0                # chaser height            [m]

# ─── 6-state (6D) planar model ───────────────────────────────────────────────
CB6   = math.sqrt(W_C**2 + H_C**2) / 2             # 0.707 m  (2-D half-diagonal)
GC6   = 0.143                                       # goal clearance
GBH6  = 0.2                                         # goal-band height
GYX6  = -(P_LEN + CB6 + GC6)                       # goal y_max  (-1.050)
GYN6  = GYX6 - GBH6                                # goal y_min  (-1.250)
EPS_P = 0.1                                         # position tolerance

# ─── 13-state (13D) model ────────────────────────────────────────────────────
D_T   = 3.0;  D_C = 1.0;  P_HWZ = 0.6
CB13  = math.sqrt(W_C**2 + H_C**2 + D_C**2) / 2   # 0.866 m  (3-D bounding sphere)
EPS_Q = 0.151                                       # quaternion angle tolerance [rad]
CBY_W = 0.5 * (math.cos(EPS_Q)
              + math.sqrt(2) * math.sin(EPS_Q))     # 0.601 m  (orientation-dep. buf.)
GC13  = 0.07                                        # goal clearance
GBH13 = 0.4                                         # goal-band height
GYX13 = -(P_LEN + CBY_W + GC13)                    # goal y_max  (-0.871)
GYN13 = GYX13 - GBH13                              # goal y_min  (-1.271)


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _inflated_verts(cb):
    """Closed polygon for the inflated L-shape obstacle boundary in x-y."""
    return [
        (-(W_T / 2 + cb),         -cb),
        (-(W_T / 2 + cb),    H_T + cb),
        (  W_T / 2 + cb,     H_T + cb),
        (  W_T / 2 + cb,         -cb),
        (  P_HWX   + cb,         -cb),
        (  P_HWX   + cb, -(P_LEN + cb)),
        (-(P_HWX   + cb), -(P_LEN + cb)),
        (-(P_HWX   + cb),        -cb),
        (-(W_T / 2 + cb),         -cb),
    ]


def _box_faces(c, hx, hy, hz):
    """Six quadrilateral faces of an axis-aligned box centred at *c*."""
    cx, cy, cz = c
    v = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ])
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
    return [[v[i] for i in q] for q in quads]


def _box_edges(c, hx, hy, hz):
    """Return list of ((x0,x1),(y0,y1),(z0,z1)) pairs for each edge of an AA box."""
    cx, cy, cz = c
    v = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    idx = [(0, 1), (2, 3), (4, 5), (6, 7),
           (0, 2), (1, 3), (4, 6), (5, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [([v[a, 0], v[b, 0]], [v[a, 1], v[b, 1]], [v[a, 2], v[b, 2]])
            for a, b in idx]


def _lshape_faces(cb):
    """14 quad faces for the L-shaped failure-set prism (body + post union).

    Body part uses z-extent ±(D_T/2 + cb), post protrusion uses the smaller
    z-extent ±(P_HWZ + cb).  The body bottom face has a rectangular opening
    where the post continues downward.  No internal surfaces.
    """
    bx = W_T / 2 + cb              # body half-width  (x)
    yt = H_T + cb                   # body top         (y)
    yb = -cb                        # body bottom      (y)  — transition level
    zb = D_T / 2 + cb               # body half-depth  (z)

    px = P_HWX + cb                 # post half-width  (x)
    yp = -(P_LEN + cb)              # post bottom      (y)
    zp = P_HWZ + cb                 # post half-depth  (z)  — smaller than zb

    V = np.array
    return [
        # ── body outer walls ─────────────────────────────────────────────
        V([[-bx,yt,-zb],[ bx,yt,-zb],[ bx,yt, zb],[-bx,yt, zb]]),  # top
        V([[-bx,yb,-zb],[ bx,yb,-zb],[ bx,yt,-zb],[-bx,yt,-zb]]),  # front
        V([[-bx,yb, zb],[ bx,yb, zb],[ bx,yt, zb],[-bx,yt, zb]]),  # back
        V([[-bx,yb,-zb],[-bx,yt,-zb],[-bx,yt, zb],[-bx,yb, zb]]),  # left
        V([[ bx,yb,-zb],[ bx,yt,-zb],[ bx,yt, zb],[ bx,yb, zb]]),  # right
        # ── body bottom (y = yb) with opening for post ──────────────────
        V([[-bx,yb,-zb],[-px,yb,-zb],[-px,yb, zb],[-bx,yb, zb]]),  # left strip
        V([[ px,yb,-zb],[ bx,yb,-zb],[ bx,yb, zb],[ px,yb, zb]]),  # right strip
        V([[-px,yb,-zb],[ px,yb,-zb],[ px,yb,-zp],[-px,yb,-zp]]),   # front strip
        V([[-px,yb, zp],[ px,yb, zp],[ px,yb, zb],[-px,yb, zb]]),   # back strip
        # ── post protrusion (smaller z-extent) ──────────────────────────
        V([[-px,yp,-zp],[ px,yp,-zp],[ px,yb,-zp],[-px,yb,-zp]]),   # front
        V([[-px,yp, zp],[ px,yp, zp],[ px,yb, zp],[-px,yb, zp]]),   # back
        V([[-px,yp,-zp],[-px,yb,-zp],[-px,yb, zp],[-px,yp, zp]]),   # left
        V([[ px,yp,-zp],[ px,yb,-zp],[ px,yb, zp],[ px,yp, zp]]),   # right
        V([[-px,yp,-zp],[ px,yp,-zp],[ px,yp, zp],[-px,yp, zp]]),   # bottom
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Panel (a) — 6-state planar model  (2-D cross-section, x–y plane)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_6d(ax):
    # ── Collision-buffer fill (lowest z-order so body covers the interior) ───
    ax.add_patch(mpatches.Polygon(
        _inflated_verts(CB6), closed=True,
        fc=C_BUF, ec='none', zorder=0))

    # ── Target body ──────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (-W_T / 2, 0), W_T, H_T,
        fc=C_BODY, ec=C_EDGE, lw=1.2, zorder=1))

    # ── Docking post ─────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (-P_HWX, -P_LEN), 2 * P_HWX, P_LEN,
        fc=C_BODY, ec=C_EDGE, lw=1.2, zorder=1))

    # ── Failure-set boundary (red dashed outline) ────────────────────────────
    xs, ys = zip(*_inflated_verts(CB6))
    ax.plot(xs, ys, color=C_FAIL, ls='--', lw=1.4, zorder=2)

    # ── Goal band ────────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (-EPS_P, GYN6), 2 * EPS_P, GYX6 - GYN6,
        fc=C_GOAL, ec=C_GOAL_E, alpha=0.55, lw=1.2, zorder=3))
    ax.plot(0, (GYN6 + GYX6) / 2, 'o', color=C_GOAL_E, ms=3, zorder=10)

    # ── Dimension: collision buffer (horizontal, right side at mid-height) ───
    yb = H_T / 2
    ax.annotate('', xy=(W_T / 2, yb), xytext=(W_T / 2 + CB6, yb),
                arrowprops=dict(arrowstyle='<->', color=C_FAIL, lw=0.9,
                                shrinkA=0, shrinkB=0))
    ax.text(W_T / 2 + CB6 + 0.1, yb,
            r'$r_{\mathrm{buf}}$' + f' = {CB6:.2f} m',
            fontsize=8, color=C_FAIL, va='center', ha='left')

    # ── Dimension: body width (horizontal above inflated boundary) ───────────
    yd = H_T + CB6 + 0.3
    ax.annotate('', xy=(-W_T / 2, yd), xytext=(W_T / 2, yd),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.7,
                                shrinkA=0, shrinkB=0))
    ax.text(0, yd + 0.15, f'$w_t = {W_T:.0f}$ m', fontsize=8,
            color='gray', ha='center')

    # ── Dimension: body height (vertical, far right — clear of buffer) ───────
    xd2 = W_T / 2 + CB6 - 7.9
    ax.annotate('', xy=(xd2, 0), xytext=(xd2, H_T),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.7,
                                shrinkA=0, shrinkB=0))
    ax.text(xd2 - 0.4, H_T / 2, f'$h_t = {H_T:.0f}$ m', fontsize=8,
            color='gray', va='center', rotation=90)

    # ── Chaser (at goal set — docked state) ─────────────────────────────────
    gc6 = (GYN6 + GYX6) / 2
    ax.add_patch(mpatches.Rectangle(
        (-W_C / 2, gc6 - H_C / 2), W_C, H_C,
        fc=C_CHASER, ec=C_CHASER_E, alpha=0.85, lw=1.2, zorder=4))

    # ── Approach direction ───────────────────────────────────────────────────
    ax.annotate('', xy=(0, GYX6 + 0.05), xytext=(0, gc6 - H_C / 2 - 0.7),
                arrowprops=dict(arrowstyle='->', color=C_APPR, lw=1.8))
    ax.text(0.18, gc6 - H_C / 2 - 0.5, 'Approach', fontsize=8, color=C_APPR)

    # ── Axes formatting ─────────────────────────────────────────────────────
    ax.set_xlim(-5.5, 7.0)
    ax.set_ylim(-2.8, 5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.grid(True, alpha=0.3, linestyle='--')


# ═══════════════════════════════════════════════════════════════════════════════
#  Panel (b) — 13-state model  (3-D isometric view)
#
#  Strategy: pack *every* face (failure boundary, body, post, goal) into a
#  SINGLE Poly3DCollection so matplotlib's per-face z-sort renders them in the
#  correct depth order.  Per-face RGBA colours distinguish the elements.
# ═══════════════════════════════════════════════════════════════════════════════

def draw_13d(ax):
    cb = CB13

    # ── Parse hex colours to (R, G, B) floats ────────────────────────────────
    def _rgb(h):
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    rf, rb, re = _rgb(C_FAIL), _rgb(C_BODY), _rgb(C_EDGE)
    rg, rge = _rgb(C_GOAL), _rgb(C_GOAL_E)

    faces, fc, ec = [], [], []          # geometry, face-colour, edge-colour

    # 1. Failure-set boundary — L-shaped union of inflated body + post ───────
    #    Faces 5-8 are the body-bottom strips that fill the opening around
    #    the post.  Their internal shared edges are artifacts — hide them by
    #    setting edge alpha to 0 (adjacent wall faces supply the real edges).
    lshape = _lshape_faces(cb)
    for i, f in enumerate(lshape):
        faces.append(f)
        fc.append((*rf, 0.09))
        ec.append((*rf, 0.0) if 5 <= i <= 8 else (*rf, 0.55))

    # 2. Target body (nearly opaque grey) ─────────────────────────────────────
    for f in _box_faces((0, H_T / 2, 0), W_T / 2, H_T / 2, D_T / 2):
        faces.append(f)
        fc.append((*rb, 0.92))
        ec.append((*re, 1.0))

    # 3. Docking post (nearly opaque grey) ────────────────────────────────────
    for f in _box_faces((0, -P_LEN / 2, 0), P_HWX, P_LEN / 2, P_HWZ):
        faces.append(f)
        fc.append((*rb, 0.92))
        ec.append((*re, 1.0))

    # 4. Goal band (green slab, slightly enlarged for visibility) ─────────────
    gcy = (GYN13 + GYX13) / 2
    ghy = (GYX13 - GYN13) / 2
    eps_vis = 0.25                      # visual half-width in x & z
    for f in _box_faces((0, gcy, 0), eps_vis, ghy, eps_vis):
        faces.append(f)
        fc.append((*rg, 0.60))
        ec.append((*rge, 1.0))

    # 5. Chaser (at goal set — docked state, orange) ───────────────────────────
    rc = _rgb(C_CHASER)
    rce = _rgb(C_CHASER_E)
    for f in _box_faces((0, gcy, 0), W_C / 2, H_C / 2, D_C / 2):
        faces.append(f)
        fc.append((*rc, 0.85))
        ec.append((*rce, 1.0))

    # ── Single combined mesh — matplotlib sorts faces by depth ───────────────
    mesh = Poly3DCollection(faces, linewidths=0.8)
    mesh.set_facecolors(fc)
    mesh.set_edgecolors(ec)
    ax.add_collection3d(mesh)

    # ── Approach direction arrow ─────────────────────────────────────────────
    ax.quiver(0, -2.5, 0, 0, 1.0, 0, color=C_APPR,
              arrow_length_ratio=0.18, linewidth=1.5)
    ax.text2D(0.32, 0.33, 'Approach', color=C_APPR, fontsize=8,
              transform=ax.transAxes)

    # ── d_t dimension (2-D screen text — always readable) ────────────────────
    ax.text2D(0.78, 0.52, f'$d_h$ = 3m', color='black', fontsize=8,
              transform=ax.transAxes)
    ax.text2D(0.71, 0.41, f'$d_w$ = 3m', color='black', fontsize=8,
              transform=ax.transAxes)
    ax.text2D(0.59, 0.65, f'$d_l$ = 6m', color='black', fontsize=8,
              transform=ax.transAxes)

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlabel(r'$x$ (m)', labelpad=2)
    ax.set_ylabel(r'$y$ (m)', labelpad=2)
    ax.set_zlabel(r'$z$ (m)', labelpad=2)
    rng = 5
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-2.5, H_T + 1.5)
    ax.set_zlim(-rng, rng)
    ax.view_init(elev=25, azim=-60)
    ax.tick_params(labelsize=10, pad=1)

    # ── Clean panes / grids ──────────────────────────────────────────────────
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#E0E0E0')
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo['grid'].update(color='#E8E8E8', linewidth=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — compose the two-panel figure
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(7.16, 3.5))

    # ── Panel (a): 6-state planar model ──────────────────────────────────────
    ax1 = fig.add_subplot(121)
    draw_6d(ax1)

    # ── Panel (b): 13-state model ────────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection='3d')
    draw_13d(ax2)

    # ── Shared legend — one row below both panels ─────────────────────────────
    legend_handles = [
        mpatches.Patch(fc=C_BODY,   ec=C_EDGE,    label='Target body'),
        mpatches.Patch(fc=C_BUF,   ec='none',     label='Collision buffer'),
        Line2D([0], [0], color=C_FAIL, ls='--', lw=1.5, label='Failure set bdry.'),
        mpatches.Patch(fc=C_GOAL,   ec=C_GOAL_E,  alpha=0.55, label='Goal set'),
        mpatches.Patch(fc=C_CHASER, ec=C_CHASER_E, alpha=0.85, label='Chaser'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               frameon=False, columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(w_pad=0.5, rect=[0, 0.13, 1, 1])

    # ── Panel labels — placed in figure coords so they are vertically aligned
    label_y = 0.96
    fig.text(0.02, label_y, '(a)', fontsize=11, fontweight='bold', va='top')
    fig.text(0.50, label_y, '(b)', fontsize=11, fontweight='bold', va='top')
    plt.savefig('target_geometry_paper.png', dpi=300)
    plt.savefig('target_geometry_paper.pdf')
    print('Saved target_geometry_paper.{png,pdf}')
    plt.show()


if __name__ == '__main__':
    main()
