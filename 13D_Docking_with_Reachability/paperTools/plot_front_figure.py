#!/usr/bin/env python3
"""
Publication-quality front figure for the CDC 2026 BRAT spacecraft docking paper.

Three-panel layout (7.16 × 5.5 in, double-column):
  Top-left:    Offline training pipeline (MPC guidance → losses → curriculum → V_θ)
  Right:       Online control (BRAT Controller + Terminal MPC + Safety Filter)
  Bottom-left: 3D trajectory placeholder (image slot)

Output: figs/front_figure.pdf
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg
import numpy as np

# ── Path to trajectory image (swap in your render later) ───────────
TRAJECTORY_IMG_PATH = None  # e.g. 'figs/trajectory_render.png'

# ── Style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 8,
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ── Colors (SIA Lab palette from figureGuide.md) ──────────────────
ORANGE   = '#ff9500'
TEAL     = '#0d948f'
MAGENTA  = '#850f67'
DBLUE    = '#0048a6'
BLACK    = '#000000'
GRAY     = '#6b6b6b'
WHITE    = '#ffffff'

ORANGE_L = '#fff3e0'
TEAL_L   = '#e0f2f1'
MAG_L    = '#f3e5f5'
BLUE_L   = '#e3f2fd'
GRAY_L   = '#eeeeee'
OFF_BG   = '#d6e4f5'
ON_BG    = '#e8f5e9'


# ── Utility functions ─────────────────────────────────────────────
def rbox(ax, x, y, w, h, label, fc, ec=None, fs=7.5, tc=BLACK,
         lw=0.8, sub=None, sfs=6, sc=None, z=3, bold=False):
    """Draw a rounded box with an optional subtitle line."""
    ec = ec or fc
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.04',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    cx, cy = x + w / 2, y + h / 2
    wt = 'bold' if bold else 'normal'
    if sub:
        ax.text(cx, cy + 0.07, label, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=wt, zorder=z + 1)
        ax.text(cx, cy - 0.07, sub, ha='center', va='center',
                fontsize=sfs, color=sc or tc, zorder=z + 1)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=wt,
                zorder=z + 1, linespacing=1.1)
    return cx, cy


def arw(ax, x0, y0, x1, y1, c=BLACK, lw=0.7, sty='->', z=5, cs=None):
    """Draw an arrow between two points."""
    kw = dict(arrowstyle=sty, color=c, lw=lw,
              shrinkA=2, shrinkB=2, mutation_scale=10, zorder=z)
    if cs:
        kw['connectionstyle'] = cs
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), **kw))


def right_angle_arw(ax, x0, y0, x1, y1, c=BLACK, lw=0.7, sty='->', z=5,
                    via='horizontal_first'):
    """Draw an L-shaped (right-angle routed) arrow."""
    if via == 'horizontal_first':
        mid_x, mid_y = x1, y0
    else:  # vertical_first
        mid_x, mid_y = x0, y1
    # Draw line segment (no arrowhead)
    ax.plot([x0, mid_x], [y0, mid_y], color=c, lw=lw, zorder=z,
            solid_capstyle='butt')
    # Draw final segment with arrowhead
    arw(ax, mid_x, mid_y, x1, y1, c=c, lw=lw, sty=sty, z=z)


def pbg(ax, x, y, w, h, fc, ec, title, tc, title_inside=False):
    """Draw a panel background with a title."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.06',
        facecolor=fc, edgecolor=ec, linewidth=1.4, alpha=0.55, zorder=0))
    if title_inside:
        ax.text(x + w / 2, y + h - 0.12, title,
                ha='center', va='top', fontsize=10, fontweight='bold',
                color=tc, zorder=1)
    else:
        ax.text(x + w / 2, y + h + 0.10, title,
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=tc, zorder=1)


# ═══════════════════════════════════════════════════════════════════
#  CANVAS
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.16, 5.8))
ax.set_xlim(-0.20, 7.36)
ax.set_ylim(-0.40, 5.90)
ax.set_aspect('equal')
ax.axis('off')

# ── Panel geometry ─────────────────────────────────────────────────
GAP = 0.15          # gap between panels
LEFT_W = 2.40       # width of left column
RIGHT_W = 4.46      # width of right column
TOTAL_H = 5.60      # total panel height

# Offline panel (top-left)
OFF_H = 3.80
OFF_X, OFF_Y = 0.0, TOTAL_H - OFF_H
OFF_W = LEFT_W

# Trajectory placeholder (bottom-left)
TRAJ_H = TOTAL_H - OFF_H - GAP
TRAJ_X, TRAJ_Y = 0.0, 0.0
TRAJ_W = LEFT_W

# Online panel (right, full height)
ON_X = LEFT_W + GAP
ON_Y = 0.0
ON_W = RIGHT_W
ON_H = TOTAL_H

# Draw panel backgrounds
pbg(ax, OFF_X, OFF_Y, OFF_W, OFF_H, OFF_BG, DBLUE, 'Offline', DBLUE)
pbg(ax, ON_X, ON_Y, ON_W, ON_H, ON_BG, TEAL, 'Online', TEAL)


# ═══════════════════════════════════════════════════════════════════
#  OFFLINE PANEL (top-left)
# ═══════════════════════════════════════════════════════════════════
ocx = OFF_X + OFF_W / 2

# ── MPC Guidance ─────────────────────────────────────────────────
mw, mh = 1.90, 0.60
mx = ocx - mw / 2
my = OFF_Y + OFF_H - 0.35 - mh

ax.add_patch(FancyBboxPatch(
    (mx, my), mw, mh, boxstyle='round,pad=0.04',
    facecolor=WHITE, edgecolor=DBLUE, linewidth=1.0, zorder=3))
ax.text(mx + mw / 2, my + mh - 0.10,
        'MPC-Based Guidance', ha='center', va='center',
        fontsize=7.5, fontweight='bold', color=DBLUE, zorder=4)
ax.text(mx + mw / 2, my + mh / 2 - 0.05,
        r'$\hat{V}(x, t) = \min_{u}\; J(x, u, t)$',
        ha='center', va='center', fontsize=7.5, color=BLACK, zorder=4)
ax.text(mx + mw / 2, my + 0.10,
        r's.t.  $\dot{\xi}\!=\!f(\xi,u)$,   $u \!\in\! \mathcal{U}$',
        ha='center', va='center', fontsize=5.5, color=GRAY, zorder=4)

# ── Loss boxes ───────────────────────────────────────────────────
lbw, lbh = 0.90, 0.34
lgap_box = 0.16
ltot = 2 * lbw + lgap_box
lx0 = ocx - ltot / 2
ly = my - 0.36 - lbh

pcx, pcy = rbox(ax, lx0, ly, lbw, lbh,
                r'$\mathcal{L}_{\mathrm{pde}}$',
                fc=BLUE_L, ec=DBLUE, fs=9,
                sub='VI residual', sfs=5.5, sc=GRAY)
mlx = lx0 + lbw + lgap_box
mcx, mcy = rbox(ax, mlx, ly, lbw, lbh,
                r'$\mathcal{L}_{\mathrm{mpc}}$',
                fc=BLUE_L, ec=DBLUE, fs=9,
                sub=r'$|V_\theta \!-\! \hat{V}|$', sfs=5.5, sc=GRAY)

# Arrows: MPC → losses (straight down, slight offset)
arw(ax, ocx - 0.22, my, pcx, ly + lbh, c=DBLUE, lw=0.7)
arw(ax, ocx + 0.22, my, mcx, ly + lbh, c=DBLUE, lw=0.7)

# ── Summation ────────────────────────────────────────────────────
sy = ly - 0.30
ax.text(ocx, sy, r'$\Sigma$', ha='center', va='center',
        fontsize=13, color=BLACK, zorder=4,
        bbox=dict(fc=WHITE, ec=BLACK, boxstyle='circle,pad=0.05', lw=0.8))

arw(ax, pcx, ly, ocx - 0.07, sy + 0.11, c=DBLUE)
arw(ax, mcx, ly, ocx + 0.07, sy + 0.11, c=DBLUE)

# ── Curriculum ───────────────────────────────────────────────────
cw, ch = 1.30, 0.28
cbx = ocx - cw / 2
cby = sy - 0.46
rbox(ax, cbx, cby, cw, ch,
     'Curriculum Training', fc=TEAL_L, ec=TEAL,
     fs=7, tc=TEAL, bold=True)
arw(ax, ocx, sy - 0.11, ocx, cby + ch, c=BLACK, lw=0.9)
ax.text(cbx + cw + 0.03, cby + ch / 2,
        r'$t_{\max}\!\uparrow\!T$',
        ha='left', va='center', fontsize=5.5, color=TEAL, zorder=4)

# ── V_theta ──────────────────────────────────────────────────────
vw, vh = 1.40, 0.48
vbx = ocx - vw / 2
vby = cby - 0.42 - vh
vcx, vcy = rbox(ax, vbx, vby, vw, vh,
                r'$V_\theta(x, t)$',
                fc=ORANGE_L, ec=ORANGE, fs=12,
                tc=ORANGE, bold=True, lw=1.4,
                sub='SIREN Network', sfs=6.5, sc=GRAY)
arw(ax, ocx, cby, ocx, vby + vh, c=BLACK, lw=0.9)

# ── Refinement loop (right side, dashed, vertical) ───────────────
loop_x = vbx + vw + 0.06
loop_top = my + mh * 0.30
loop_bot = vby + vh * 0.65

# Vertical dashed line up
ax.annotate('',
    xy=(mx + mw - 0.04, loop_top),
    xytext=(vbx + vw - 0.04, loop_bot),
    arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8,
                    connectionstyle='arc3,rad=0.50',
                    linestyle=(0, (4, 3))))
ax.text(OFF_X + OFF_W - 0.08, (loop_top + loop_bot) / 2,
        'refine', ha='center', va='center', fontsize=5, color=GRAY,
        fontstyle='italic', rotation=90, zorder=4,
        bbox=dict(fc=OFF_BG, ec='none', alpha=0.95, pad=1.0))

# Feedback arrow (V_θ → L_pde, for gradient updates)
ax.annotate('',
    xy=(pcx - 0.15, ly),
    xytext=(vbx + 0.15, vby + vh),
    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.6,
                    connectionstyle='arc3,rad=0.28'))


# ═══════════════════════════════════════════════════════════════════
#  ARROW: Offline → Online (V_θ crosses the gap)
# ═══════════════════════════════════════════════════════════════════
# Horizontal arrow from V_θ box right edge → Online panel left edge
transfer_y = vcy
arw(ax, vbx + vw, transfer_y, ON_X, transfer_y,
    c=ORANGE, lw=1.6, sty='-|>')
gap_label_x = (vbx + vw + ON_X) / 2
ax.text(gap_label_x, transfer_y + 0.14,
        r'$V_\theta,\;\nabla_x V_\theta$',
        ha='center', va='bottom', fontsize=8.5, color=ORANGE,
        fontweight='bold', zorder=6,
        bbox=dict(fc=WHITE, ec='none', alpha=0.9, pad=1.2))


# ═══════════════════════════════════════════════════════════════════
#  TRAJECTORY PLACEHOLDER (bottom-left)
# ═══════════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch(
    (TRAJ_X, TRAJ_Y), TRAJ_W, TRAJ_H, boxstyle='round,pad=0.06',
    facecolor=GRAY_L, edgecolor='#999999', linewidth=1.0,
    alpha=0.6, zorder=0))

if TRAJECTORY_IMG_PATH and os.path.isfile(TRAJECTORY_IMG_PATH):
    img = mpimg.imread(TRAJECTORY_IMG_PATH)
    img_pad = 0.08
    ax.imshow(img, aspect='auto', zorder=2,
              extent=[TRAJ_X + img_pad, TRAJ_X + TRAJ_W - img_pad,
                      TRAJ_Y + img_pad, TRAJ_Y + TRAJ_H - img_pad])
else:
    ax.text(TRAJ_X + TRAJ_W / 2, TRAJ_Y + TRAJ_H / 2,
            'trajectory placeholder',
            ha='center', va='center', fontsize=9, color='#888888',
            fontstyle='italic', zorder=2)


# ═══════════════════════════════════════════════════════════════════
#  ONLINE PANEL (right, full height)
# ═══════════════════════════════════════════════════════════════════
on_cx = ON_X + ON_W / 2

# ── State x ──────────────────────────────────────────────────────
st_y = ON_Y + ON_H - 0.35
ax.text(on_cx, st_y, r'State $x$', ha='center', va='center',
        fontsize=9, color=BLACK, zorder=4,
        bbox=dict(fc=WHITE, ec=BLACK, boxstyle='round,pad=0.06', lw=0.6))

# ── Controller column geometry ───────────────────────────────────
col_w = 1.70           # width of each controller sub-panel
col_gap = 0.22         # gap between columns
col_total = 2 * col_w + col_gap
col1_x = on_cx - col_total / 2
col2_x = col1_x + col_w + col_gap
col1_cx = col1_x + col_w / 2
col2_cx = col2_x + col_w / 2

sub_top = st_y - 0.32  # top of sub-panel backgrounds
sub_bot = 1.45          # bottom of sub-panel backgrounds
sub_h = sub_top - sub_bot

# BRAT controller sub-background
ax.add_patch(FancyBboxPatch(
    (col1_x, sub_bot), col_w, sub_h, boxstyle='round,pad=0.04',
    facecolor=ORANGE_L, edgecolor=ORANGE, linewidth=0.6,
    alpha=0.35, zorder=1))
ax.text(col1_cx, sub_top - 0.06, 'BRAT Controller',
        ha='center', va='top', fontsize=7.5, fontweight='bold',
        color=ORANGE, zorder=2)

# Terminal MPC sub-background
ax.add_patch(FancyBboxPatch(
    (col2_x, sub_bot), col_w, sub_h, boxstyle='round,pad=0.04',
    facecolor=TEAL_L, edgecolor=TEAL, linewidth=0.6,
    alpha=0.35, zorder=1))
ax.text(col2_cx, sub_top - 0.06, 'Terminal MPC',
        ha='center', va='top', fontsize=7.5, fontweight='bold',
        color=TEAL, zorder=2)

# State → column tops (right-angle: vertical then horizontal)
arw(ax, on_cx - 0.30, st_y - 0.12, col1_cx, sub_top, c=BLACK, lw=0.6)
arw(ax, on_cx + 0.30, st_y - 0.12, col2_cx, sub_top, c=BLACK, lw=0.6)

# ── BRAT column boxes ───────────────────────────────────────────
bw, bh = 1.45, 0.40
b1x = col1_cx - bw / 2

p1y = sub_top - 0.70
p1cx, p1cy = rbox(ax, b1x, p1y, bw, bh,
                  'Phase 1: Converge', fc=WHITE, ec=ORANGE,
                  fs=7, tc=ORANGE, bold=True,
                  sub=r'$u\!=\!-\bar{u}\,\mathrm{sign}(\nabla_v V|_{t=T})$',
                  sfs=5.5, sc=BLACK)

# Transition condition between phases
cond1_y = p1y - 0.34
ax.text(col1_cx, cond1_y, r'$V(x,T)\!\leq\!0$',
        ha='center', va='center', fontsize=6, color=BLACK, zorder=4,
        bbox=dict(fc=WHITE, ec=ORANGE, boxstyle='round,pad=0.04',
                  lw=0.5, alpha=0.9))
arw(ax, col1_cx, p1y, col1_cx, cond1_y + 0.12, c=BLACK, lw=0.7)

p2y = cond1_y - 0.44
p2cx, p2cy = rbox(ax, b1x, p2y, bw, bh,
                  'Phase 2: Precision', fc=WHITE, ec=ORANGE,
                  fs=7, tc=ORANGE, bold=True,
                  sub=r'$u\!=\!-\bar{u}\,\mathrm{sign}(\nabla_v V|_{t=t^*})$',
                  sfs=5.5, sc=BLACK)
arw(ax, col1_cx, cond1_y - 0.11, col1_cx, p2y + bh, c=BLACK, lw=0.7)

# ── Terminal MPC column boxes ────────────────────────────────────
b2x = col2_cx - bw / 2

mpc_y = sub_top - 0.70
mpc_cx, mpc_cy = rbox(ax, b2x, mpc_y, bw, bh,
                      'Short-horizon MPC', fc=WHITE, ec=TEAL,
                      fs=7, tc=TEAL, bold=True,
                      sub=r'Terminal cost: $V_\theta(x, t)$',
                      sfs=5.5, sc=BLACK)

# Transition condition
cond2_y = mpc_y - 0.34
ax.text(col2_cx, cond2_y, r'$V(x,T)\!\leq\!0$',
        ha='center', va='center', fontsize=6, color=BLACK, zorder=4,
        bbox=dict(fc=WHITE, ec=TEAL, boxstyle='round,pad=0.04',
                  lw=0.5, alpha=0.9))
arw(ax, col2_cx, mpc_y, col2_cx, cond2_y + 0.12, c=BLACK, lw=0.7)

mt_y = cond2_y - 0.44
mt_cx, mt_cy = rbox(ax, b2x, mt_y, bw, bh,
                    'Min-time search', fc=WHITE, ec=TEAL,
                    fs=7, tc=TEAL, bold=True,
                    sub=r'$t^*\!=\!\min\,t : V(x,t)\!\leq\!0$',
                    sfs=5.5, sc=BLACK)
arw(ax, col2_cx, cond2_y - 0.11, col2_cx, mt_y + bh, c=BLACK, lw=0.7)

# ── Safety filter (shared, below both columns) ──────────────────
sf_w = col_total + 0.10
sf_x = on_cx - sf_w / 2
sf_y = 0.50
sf_h = 0.52
sfcx, sfcy = rbox(ax, sf_x, sf_y, sf_w, sf_h,
                  'Least-Restrictive Safety Filter',
                  fc=MAG_L, ec=MAGENTA, fs=7.5,
                  tc=MAGENTA, bold=True, lw=1.0,
                  sub=r'Avoid-only BRT:  override when $V_{\mathrm{avoid}}(x) < \gamma$',
                  sfs=5.5, sc=GRAY)

# Arrows: columns → safety filter (straight down from bottom of
# each column, right-angle into filter top)
sf_top = sf_y + sf_h

# Left column → safety filter
arw(ax, col1_cx, sub_bot, col1_cx, sf_top, c=MAGENTA, lw=0.7)
# Right column → safety filter
arw(ax, col2_cx, sub_bot, col2_cx, sf_top, c=MAGENTA, lw=0.7)

# "if unsafe" label
ax.text(on_cx, (sub_bot + sf_top) / 2,
        'if unsafe', ha='center', va='center', fontsize=5.5,
        color=MAGENTA, fontstyle='italic', zorder=4,
        bbox=dict(fc=ON_BG, ec='none', alpha=0.85, pad=0.8))

# Output: u_safe
arw(ax, sfcx, sf_y, sfcx, sf_y - 0.14, c=BLACK, lw=1.0)
ax.text(sfcx, sf_y - 0.18, r'$u_{\mathrm{safe}}$',
        ha='center', va='top', fontsize=9, color=BLACK, zorder=4)


# ═══════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════
out_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'figs')
os.makedirs(out_dir, exist_ok=True)

for ext in ('pdf', 'png'):
    p = os.path.join(out_dir, f'front_figure.{ext}')
    fig.savefig(p)
    print(f'Saved: {p}')

plt.close(fig)
