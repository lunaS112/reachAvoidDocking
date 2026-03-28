#!/usr/bin/env python3
"""
Publication-quality block diagram of the two-phase BRAT controller (CDC 2026).

Double-column figure (7.16 in wide).
Output: figs/controller_architecture.pdf
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── CDC figure style ────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 9,
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ── Colors from figureGuide.md ──────────────────────────────────────
ORANGE   = '#ff9500'
TEAL     = '#0d948f'
MAGENTA  = '#850f67'
BLACK    = '#000000'

# Lighter tints for box fills
ORANGE_LIGHT  = '#ffe0b2'
TEAL_LIGHT    = '#b2dfdb'
MAGENTA_LIGHT = '#e1bee7'
GRAY_LIGHT    = '#e0e0e0'

# Very light tints for phase backgrounds
ORANGE_BG = '#fff3e0'
TEAL_BG   = '#e0f2f1'


def box(ax, x, y, w, h, label, fc, ec=None, fontcolor=BLACK,
        fontsize=9, lw=1.0, sublabel=None, sublabel_size=7,
        sublabel_color=None):
    """Draw a rounded box with optional smaller subtitle line."""
    if ec is None:
        ec = fc
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.04',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.08, label,
                ha='center', va='center', fontsize=fontsize,
                color=fontcolor, zorder=4)
        ax.text(cx, cy - 0.08, sublabel,
                ha='center', va='center', fontsize=sublabel_size,
                color=sublabel_color or fontcolor, zorder=4)
    else:
        ax.text(cx, cy, label,
                ha='center', va='center', fontsize=fontsize,
                color=fontcolor, zorder=4, linespacing=1.2)
    return cx, cy


def arr(ax, x0, y0, x1, y1, color=BLACK, lw=0.8, style='->'):
    """Draw an arrow."""
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, color=color,
        lw=lw, shrinkA=3, shrinkB=3, mutation_scale=10, zorder=5))


# ── Layout ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.16, 3.8))
ax.set_xlim(-0.10, 7.30)
ax.set_ylim(-0.55, 3.85)
ax.set_aspect('equal')
ax.axis('off')

# Box dimensions — taller to fit sublabels
BW = 1.30
BH = 0.48
VGAP = 0.25

# Phase column centers
P1_CX = 1.50
P2_CX = 5.55

# Row y-positions — 4 rows
R1 = 2.55
R2 = R1 - BH - VGAP
R3 = R2 - BH - VGAP
R4 = R3 - BH - VGAP

# Background bounds
BG_TOP = R1 + BH + 0.10
BG_BOT = R4 - 0.10
BG_H = BG_TOP - BG_BOT

# ── Phase 1 background (wider to include PD fallback) ─────────────
P1_BG_LEFT = P1_CX - BW / 2 - 0.15
P1_BG_W = 2.55
ax.add_patch(FancyBboxPatch(
    (P1_BG_LEFT, BG_BOT), P1_BG_W, BG_H,
    boxstyle='round,pad=0.06',
    facecolor=ORANGE_BG, edgecolor=ORANGE, linewidth=1.0,
    alpha=0.55, zorder=0))
ax.text(P1_BG_LEFT + P1_BG_W / 2, BG_TOP + 0.06,
        'Phase 1: Convergence',
        ha='center', va='bottom', fontsize=10,
        fontweight='bold', color=ORANGE)

# ── Phase 2 background ───────────────────────────────────────────
P2_BG_W = 2.10
p2_bg_x = P2_CX - P2_BG_W / 2
ax.add_patch(FancyBboxPatch(
    (p2_bg_x, BG_BOT), P2_BG_W, BG_H,
    boxstyle='round,pad=0.06',
    facecolor=TEAL_BG, edgecolor=TEAL, linewidth=1.0,
    alpha=0.55, zorder=0))
ax.text(P2_CX, BG_TOP + 0.06, 'Phase 2: Precision',
        ha='center', va='bottom', fontsize=10,
        fontweight='bold', color=TEAL)

# ═══════════════════════════════════════════════════════════════════
#  PHASE 1 BOXES  (rows 1–3)
# ═══════════════════════════════════════════════════════════════════
p1_left = P1_CX - BW / 2

box(ax, p1_left, R1, BW, BH,
    r'Value function',
    fc=ORANGE_LIGHT, ec=ORANGE, fontsize=8.5,
    sublabel=r'$V(x,\, T)$, $\nabla_x V$',
    sublabel_size=7, sublabel_color=BLACK)

box(ax, p1_left, R2, BW, BH,
    'Bang-bang',
    fc=ORANGE_LIGHT, ec=ORANGE, fontsize=8.5,
    sublabel=r'$u = -\bar{u}\,\mathrm{sign}(\nabla_v V)$',
    sublabel_size=6.5, sublabel_color=ORANGE)

box(ax, p1_left, R3, BW, BH,
    'Safety filter',
    fc=MAGENTA_LIGHT, ec=MAGENTA, fontsize=8.5)

# Phase 1 vertical arrows
arr(ax, P1_CX, R1, P1_CX, R2 + BH, lw=0.9)
arr(ax, P1_CX, R2, P1_CX, R3 + BH, lw=0.9)

ax.text(P1_CX + 0.10, (R2 + R3 + BH) / 2,
        'if unsafe', ha='left', va='center', fontsize=6.5,
        color=MAGENTA, fontstyle='italic')

# Phase 1 output
arr(ax, P1_CX, R3, P1_CX, R3 - 0.20, color=BLACK, lw=0.9)
ax.text(P1_CX, R3 - 0.25, '$u$', ha='center', va='top',
        fontsize=10, color=BLACK)

# ── PD fallback (right of bang-bang, inside Phase 1 bg) ───────────
PD_W = 0.72
PD_H = 0.30
pd_x = p1_left + BW + 0.22
pd_y = R2 + (BH - PD_H) / 2
box(ax, pd_x, pd_y, PD_W, PD_H,
    'PD fallback',
    fc=GRAY_LIGHT, ec='#888888', fontsize=7)
ax.annotate('', xy=(pd_x, pd_y + PD_H / 2),
            xytext=(p1_left + BW, R2 + BH / 2),
            arrowprops=dict(arrowstyle='->', color='#888888',
                            lw=0.8, linestyle=(0, (4, 3))))
ax.text(pd_x + PD_W / 2, pd_y - 0.05,
        r'if $\|\nabla V\| \approx 0$',
        ha='center', va='top', fontsize=5.5, color='#666666',
        fontstyle='italic')

# ═══════════════════════════════════════════════════════════════════
#  PHASE 2 BOXES  (rows 1–4)
# ═══════════════════════════════════════════════════════════════════
p2_left = P2_CX - BW / 2

box(ax, p2_left, R1, BW, BH,
    r'Min-time $t^*$',
    fc=TEAL_LIGHT, ec=TEAL, fontsize=8.5,
    sublabel=r'$\min\; t : V(x, t) \leq 0$',
    sublabel_size=6.5, sublabel_color=TEAL)

box(ax, p2_left, R2, BW, BH,
    'Value function',
    fc=ORANGE_LIGHT, ec=ORANGE, fontsize=8.5,
    sublabel=r'$V(x,\, t^*)$, $\nabla_x V$',
    sublabel_size=7, sublabel_color=BLACK)

box(ax, p2_left, R3, BW, BH,
    'Bang-bang',
    fc=ORANGE_LIGHT, ec=ORANGE, fontsize=8.5,
    sublabel=r'$u = -\bar{u}\,\mathrm{sign}(\nabla_v V)$',
    sublabel_size=6.5, sublabel_color=ORANGE)

box(ax, p2_left, R4, BW, BH,
    'Safety filter',
    fc=MAGENTA_LIGHT, ec=MAGENTA, fontsize=8.5)

# Phase 2 vertical arrows
arr(ax, P2_CX, R1, P2_CX, R2 + BH, lw=0.9)
arr(ax, P2_CX, R2, P2_CX, R3 + BH, lw=0.9)
arr(ax, P2_CX, R3, P2_CX, R4 + BH, lw=0.9)

ax.text(P2_CX + 0.10, (R3 + R4 + BH) / 2,
        'if unsafe', ha='left', va='center', fontsize=6.5,
        color=MAGENTA, fontstyle='italic')

# Phase 2 output
arr(ax, P2_CX, R4, P2_CX, R4 - 0.20, color=BLACK, lw=0.9)
ax.text(P2_CX, R4 - 0.25, '$u$', ha='center', va='top',
        fontsize=10, color=BLACK)

# ═══════════════════════════════════════════════════════════════════
#  STATE INPUT (well above phase titles)
# ═══════════════════════════════════════════════════════════════════
mid_x = (P1_CX + P2_CX) / 2
state_y = BG_TOP + 0.60
ax.text(mid_x, state_y, r'State $x$', ha='center', va='center',
        fontsize=10, color=BLACK)

arr(ax, mid_x - 0.15, state_y - 0.14, P1_CX, R1 + BH,
    color=BLACK, lw=0.7)
arr(ax, mid_x + 0.15, state_y - 0.14, P2_CX, R1 + BH,
    color=BLACK, lw=0.7)

# ═══════════════════════════════════════════════════════════════════
#  TRANSITION ARROW  (Phase 1 → Phase 2)
# ═══════════════════════════════════════════════════════════════════
trans_y = R1 + BH / 2
arr(ax, p1_left + BW, trans_y, p2_left, trans_y,
    color=BLACK, lw=1.0, style='-|>')
ax.text((p1_left + BW + p2_left) / 2, trans_y + 0.10,
        r'$V(x, T) \leq 0$',
        ha='center', va='bottom', fontsize=8, color=BLACK,
        bbox=dict(fc='white', ec='none', pad=1.0, alpha=0.85))

# ── Save ────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'figs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'controller_architecture.pdf')
fig.savefig(out_path)
print(f'Saved: {out_path}')
plt.close(fig)
