#!/usr/bin/env python3
"""
Publication-quality front figure for the CDC 2026 BRAT spacecraft docking paper.

Two-panel layout: Offline (training) | Online (two novel controllers).
  - Offline: MPC guidance + PDE/MPC losses + curriculum -> V_theta
  - Online:  BRAT Controller (bang-bang) and Terminal MPC Controller,
             both with shared safety filter

Double-column figure (7.16 in wide).
Output: figs/front_figure.pdf
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Polygon)
import numpy as np

# ── Style ───────────────────────────────────────────────────────────
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

# ── Colors ──────────────────────────────────────────────────────────
ORANGE   = '#ff9500'
TEAL     = '#0d948f'
MAGENTA  = '#850f67'
DBLUE    = '#0048a6'
BLACK    = '#000000'
GRAY     = '#6b6b6b'
GREEN    = '#2ca02c'
RED      = '#d62728'
WHITE    = '#ffffff'

ORANGE_L = '#fff3e0'
TEAL_L   = '#e0f2f1'
MAG_L    = '#f3e5f5'
BLUE_L   = '#e3f2fd'
GRAY_L   = '#eeeeee'
OFF_BG   = '#d6e4f5'
ON_BG    = '#e8f5e9'


def rbox(ax, x, y, w, h, label, fc, ec=None, fs=7.5, tc=BLACK,
         lw=0.8, sub=None, sfs=6, sc=None, z=3, bold=False):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.04',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    cx, cy = x + w/2, y + h/2
    wt = 'bold' if bold else 'normal'
    if sub:
        ax.text(cx, cy + 0.07, label, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=wt, zorder=z+1)
        ax.text(cx, cy - 0.07, sub, ha='center', va='center',
                fontsize=sfs, color=sc or tc, zorder=z+1)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=wt,
                zorder=z+1, linespacing=1.1)
    return cx, cy


def arw(ax, x0, y0, x1, y1, c=BLACK, lw=0.7, sty='->', z=5, cs=None):
    kw = dict(arrowstyle=sty, color=c, lw=lw,
              shrinkA=2, shrinkB=2, mutation_scale=9, zorder=z)
    if cs:
        kw['connectionstyle'] = cs
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), **kw))


def pbg(ax, x, y, w, h, fc, ec, title, tc):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.06',
        facecolor=fc, edgecolor=ec, linewidth=1.4, alpha=0.55, zorder=0))
    ax.text(x + w/2, y + h + 0.10, title,
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=tc, zorder=1)


# ═══════════════════════════════════════════════════════════════════
#  CANVAS
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.16, 4.2))
ax.set_xlim(-0.35, 7.40)
ax.set_ylim(-0.25, 4.30)
ax.set_aspect('equal')
ax.axis('off')

# Panel geometry — taller to give more vertical room
OX, OY, OW, OH = 0.0, 0.0, 2.90, 3.80       # Offline (narrower)
NX, NY, NW, NH = 3.20, 0.0, 4.05, 3.80       # Online (wider)

pbg(ax, OX, OY, OW, OH, OFF_BG, DBLUE, 'Offline', DBLUE)
pbg(ax, NX, NY, NW, NH, ON_BG, TEAL, 'Online', TEAL)


# ═══════════════════════════════════════════════════════════════════
#  OFFLINE PANEL
# ═══════════════════════════════════════════════════════════════════
ocx = OX + OW / 2

# ── MPC Guidance ──────────────────────────────────────────────────
mw, mh = 1.95, 0.65
mx = ocx - mw/2
my = 2.85
ax.add_patch(FancyBboxPatch(
    (mx, my), mw, mh, boxstyle='round,pad=0.05',
    facecolor=WHITE, edgecolor=DBLUE, linewidth=1.0, zorder=3))
ax.text(mx + mw/2, my + mh - 0.10,
        'MPC-Based Guidance', ha='center', va='center',
        fontsize=7.5, fontweight='bold', color=DBLUE, zorder=4)
ax.text(mx + mw/2, my + mh/2 - 0.05,
        r'$\hat{V}(x, t) = \min_{u}\; J(x, u, t)$',
        ha='center', va='center', fontsize=7.5, color=BLACK, zorder=4)
ax.text(mx + mw/2, my + 0.10,
        r's.t.  $\dot{\xi}\!=\!f(\xi,u)$,   $u \!\in\! \mathcal{U}$',
        ha='center', va='center', fontsize=6, color=GRAY, zorder=4)

# ── Loss boxes ────────────────────────────────────────────────────
lbw, lbh = 1.00, 0.36
lgap = 0.20
ltot = 2*lbw + lgap
lx0 = ocx - ltot/2
ly = 2.15

pcx, _ = rbox(ax, lx0, ly, lbw, lbh,
              r'$\mathcal{L}_{\mathrm{pde}}$',
              fc=BLUE_L, ec=DBLUE, fs=9,
              sub='VI residual', sfs=5.5, sc=GRAY)
mlx = lx0 + lbw + lgap
mcx, _ = rbox(ax, mlx, ly, lbw, lbh,
              r'$\mathcal{L}_{\mathrm{mpc}}$',
              fc=BLUE_L, ec=DBLUE, fs=9,
              sub=r'$|V_\theta \!-\! \hat{V}|$', sfs=5.5, sc=GRAY)

arw(ax, ocx - 0.25, my, pcx, ly + lbh, c=DBLUE, lw=0.7)
arw(ax, ocx + 0.25, my, mcx, ly + lbh, c=DBLUE, lw=0.7)

# ── Summation ─────────────────────────────────────────────────────
sy = 1.72
ax.text(ocx, sy, r'$\Sigma$', ha='center', va='center',
        fontsize=14, color=BLACK, zorder=4,
        bbox=dict(fc=WHITE, ec=BLACK, boxstyle='circle,pad=0.06', lw=0.8))
arw(ax, pcx, ly, ocx - 0.08, sy + 0.11, c=DBLUE)
arw(ax, mcx, ly, ocx + 0.08, sy + 0.11, c=DBLUE)

# ── Curriculum ────────────────────────────────────────────────────
cw, ch = 1.35, 0.28
cbx = ocx - cw/2
cby = 1.15
rbox(ax, cbx, cby, cw, ch,
     'Curriculum Training', fc=TEAL_L, ec=TEAL,
     fs=7, tc=TEAL, bold=True)
arw(ax, ocx, sy - 0.11, ocx, cby + ch, c=BLACK, lw=0.9)
ax.text(cbx + cw + 0.04, cby + ch/2,
        r'$t_{\max}\!\uparrow\!T$',
        ha='left', va='center', fontsize=6, color=TEAL, zorder=4)

# ── V_theta ──────────────────────────────────────────────────────
vw, vh = 1.50, 0.50
vbx = ocx - vw/2
vby = 0.30
vcx, vcy = rbox(ax, vbx, vby, vw, vh,
                r'$V_\theta(x, t)$',
                fc=ORANGE_L, ec=ORANGE, fs=13,
                tc=ORANGE, bold=True, lw=1.4,
                sub='SIREN Network', sfs=6.5, sc=GRAY)
arw(ax, ocx, cby, ocx, vby + vh, c=BLACK, lw=0.9)

# Loop
ax.annotate('',
    xy=(mx + 0.06, my + mh * 0.30),
    xytext=(vbx + 0.06, vby + vh * 0.65),
    arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8,
                    connectionstyle='arc3,rad=0.55',
                    linestyle=(0, (4, 3))))
ax.text(-0.18, 1.75, 'train-verify\n   -refine',
        ha='center', va='center', fontsize=5, color=GRAY,
        fontstyle='italic', rotation=90, zorder=4,
        bbox=dict(fc=OFF_BG, ec='none', alpha=0.95, pad=1.5))

# Feedback
ax.annotate('',
    xy=(pcx - 0.18, ly),
    xytext=(vbx + 0.15, vby + vh),
    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.6,
                    connectionstyle='arc3,rad=0.30'))


# ═══════════════════════════════════════════════════════════════════
#  ARROW: Offline -> Online
# ═══════════════════════════════════════════════════════════════════
gap_cx = (OX + OW + NX) / 2
arw(ax, OX + OW, vcy, NX, NY + NH/2,
    c=ORANGE, lw=1.8, sty='-|>')
ax.text(gap_cx, NY + NH/2 + 0.12,
        r'$V_\theta,\;\nabla_x V_\theta$',
        ha='center', va='bottom', fontsize=9.5, color=ORANGE,
        fontweight='bold', zorder=6,
        bbox=dict(fc=WHITE, ec='none', alpha=0.9, pad=1.5))


# ═══════════════════════════════════════════════════════════════════
#  ONLINE PANEL
# ═══════════════════════════════════════════════════════════════════
nmx = NX + NW / 2

# ── State ─────────────────────────────────────────────────────────
st_y = NY + NH - 0.25
ax.text(nmx, st_y, r'State $x$', ha='center', va='center',
        fontsize=9, color=BLACK, zorder=4,
        bbox=dict(fc=WHITE, ec=BLACK, boxstyle='round,pad=0.06', lw=0.6))

# ── Controller column layout ─────────────────────────────────────
# Light sub-panel backgrounds for each controller
col_w = 1.65
col_gap = 0.25
col1_x = NX + 0.15
col2_x = col1_x + col_w + col_gap
col1_cx = col1_x + col_w / 2
col2_cx = col2_x + col_w / 2

sub_top = NY + NH - 0.55
sub_bot = NY + 0.72
sub_h = sub_top - sub_bot

# BRAT controller sub-background
ax.add_patch(FancyBboxPatch(
    (col1_x, sub_bot), col_w, sub_h, boxstyle='round,pad=0.04',
    facecolor=ORANGE_L, edgecolor=ORANGE, linewidth=0.6,
    alpha=0.35, zorder=1))
ax.text(col1_cx, sub_top - 0.04, 'BRAT Controller',
        ha='center', va='top', fontsize=8, fontweight='bold',
        color=ORANGE, zorder=2)

# Terminal MPC sub-background
ax.add_patch(FancyBboxPatch(
    (col2_x, sub_bot), col_w, sub_h, boxstyle='round,pad=0.04',
    facecolor=TEAL_L, edgecolor=TEAL, linewidth=0.6,
    alpha=0.35, zorder=1))
ax.text(col2_cx, sub_top - 0.04, 'Terminal MPC',
        ha='center', va='top', fontsize=8, fontweight='bold',
        color=TEAL, zorder=2)

# State -> columns
arw(ax, nmx - 0.50, st_y - 0.10, col1_cx, sub_top, c=BLACK, lw=0.6)
arw(ax, nmx + 0.50, st_y - 0.10, col2_cx, sub_top, c=BLACK, lw=0.6)

# ── BRAT column boxes ────────────────────────────────────────────
bw, bh = 1.40, 0.40
b1x = col1_cx - bw/2

p1y = sub_top - 0.60
p1cx, p1cy = rbox(ax, b1x, p1y, bw, bh,
                  'Phase 1: Converge', fc=WHITE, ec=ORANGE,
                  fs=7, tc=ORANGE, bold=True,
                  sub=r'$u\!=\!-\bar{u}\,\mathrm{sign}(\nabla_v V|_{t=T})$',
                  sfs=5.5, sc=BLACK)

p2y = p1y - bh - 0.28
p2cx, p2cy = rbox(ax, b1x, p2y, bw, bh,
                  'Phase 2: Precision', fc=WHITE, ec=ORANGE,
                  fs=7, tc=ORANGE, bold=True,
                  sub=r'$u\!=\!-\bar{u}\,\mathrm{sign}(\nabla_v V|_{t=t^*})$',
                  sfs=5.5, sc=BLACK)

arw(ax, col1_cx, p1y, col1_cx, p2y + bh, c=BLACK, lw=0.8)
ax.text(col1_cx + 0.12, (p1y + p2y + bh) / 2,
        r'$V(x,T)\!\leq\!0$',
        ha='left', va='center', fontsize=5.5, color=BLACK, zorder=4)

# ── Terminal MPC column boxes ─────────────────────────────────────
b2x = col2_cx - bw/2

mpc_y = sub_top - 0.60
mpc_cx, mpc_cy = rbox(ax, b2x, mpc_y, bw, bh,
                      'Short-horizon MPC', fc=WHITE, ec=TEAL,
                      fs=7, tc=TEAL, bold=True,
                      sub=r'Terminal cost: $V_\theta(x, t)$',
                      sfs=5.5, sc=BLACK)

mt_y = mpc_y - bh - 0.28
mt_cx, mt_cy = rbox(ax, b2x, mt_y, bw, bh,
                    'Min-time search', fc=WHITE, ec=TEAL,
                    fs=7, tc=TEAL, bold=True,
                    sub=r'$t^*\!=\!\min\,t : V(x,t)\!\leq\!0$',
                    sfs=5.5, sc=BLACK)

arw(ax, col2_cx, mpc_y, col2_cx, mt_y + bh, c=BLACK, lw=0.8)
ax.text(col2_cx + 0.12, (mpc_y + mt_y + bh) / 2,
        r'$V(x,T)\!\leq\!0$',
        ha='left', va='center', fontsize=5.5, color=BLACK, zorder=4)

# ── Safety filter (shared) ───────────────────────────────────────
sf_w = 2 * col_w + col_gap
sf_x = col1_x
sf_y = NY + 0.18
sf_h = 0.42
sfcx, sfcy = rbox(ax, sf_x, sf_y, sf_w, sf_h,
                  'Least-Restrictive Safety Filter',
                  fc=MAG_L, ec=MAGENTA, fs=8,
                  tc=MAGENTA, bold=True, lw=1.0,
                  sub=r'Avoid-only BRT:  override when $V_{\mathrm{avoid}}(x) < \gamma$',
                  sfs=5.5, sc=GRAY)

arw(ax, p2cx, p2y, p2cx, sf_y + sf_h, c=MAGENTA, lw=0.7)
arw(ax, mt_cx, mt_y, mt_cx, sf_y + sf_h, c=MAGENTA, lw=0.7)

# "if unsafe"
ax.text((p2cx + mt_cx) / 2, (min(p2y, mt_y) + sf_y + sf_h) / 2,
        'if unsafe', ha='center', va='center', fontsize=5.5,
        color=MAGENTA, fontstyle='italic', zorder=4,
        bbox=dict(fc=ON_BG, ec='none', alpha=0.8, pad=0.5))

# Output
arw(ax, sfcx, sf_y, sfcx, sf_y - 0.12, c=BLACK, lw=1.0)
ax.text(sfcx, sf_y - 0.16, r'$u_{\mathrm{safe}}$',
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
