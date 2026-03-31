#!/usr/bin/env python3
"""Paper-quality comparison figures from comparison_results.json.

Figure 1 (controller_comparison_13d): 2×2 bar chart of controller metrics.
Figure 2 (docking_time_histogram):   Overlaid docking-time distributions for
                                      controllers that achieved docking, vs the
                                      grid-based reference.

Run:
    python comparisons/paper_figures_comparison_13d.py \
        --results_path outputs/13d_compare_all/comparison_results.json \
        --output_dir figs/
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paper-quality matplotlib style (figureGuide.md) ───────────────────────────
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

# ── Color palette (figureGuide.md) ────────────────────────────────────────────
CTRL_COLORS = {
    'BRT+Safety 13D':  '#ff9500',  # orange
    'MPC+Terminal 13D': '#0d948f',  # teal
    'MPC 13D':          '#b3b3b3',  # gray
    'Vanilla BRT 13D':  '#0048a6',  # dark blue
    'RL 13D (DDQN)':    '#0091ff',  # bright blue
}

CTRL_DISPLAY = {
    'BRT+Safety 13D':          'BRT+Safety',
    'MPC+Terminal 13D':  'T-MPC',
    'MPC 13D':           'MPC',
    'Vanilla BRT 13D':  'V-DR',
    'RL 13D (DDQN)':     'RL',
}

# Fixed ordering:  our method  baselines by performance
CTRL_ORDER = ['BRT+Safety 13D', 'MPC+Terminal 13D', 'MPC 13D', 'Vanilla BRT 13D', 'RL 13D (DDQN)']


def load_results(path):
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 1 — 1×4 Metrics Bar Chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_metrics_comparison(data, output_dir):
    """1×4 grouped bar chart of controller metrics."""
    ctrls = [c for c in CTRL_ORDER if c in data]
    n_c = len(ctrls)
    x = np.arange(n_c)
    labels = [CTRL_DISPLAY[c] for c in ctrls]
    colors = [CTRL_COLORS[c] for c in ctrls]

    # INCREASE FIGURE SIZE: 14x4 or 12x4 is better for a 1x4 grid
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    (ax_rates, ax_effort, ax_wall, ax_dock) = axes

    # ── (a) Outcome Rates ─────────────────────────────────────────────────
    w = 0.25
    dock_pct = [data[c]['docking_rate'] * 100 for c in ctrls]
    fail_pct  = [data[c]['failure_rate']  * 100 for c in ctrls]
    time_pct  = [data[c]['timeout_rate']  * 100 for c in ctrls]

    ax_rates.bar(x - w, dock_pct, w, color='#2ca02c', label='Docking', zorder=3, alpha=0.7)
    ax_rates.bar(x,     fail_pct, w, color='#d62728', label='Failure',  zorder=3, alpha=0.7)
    ax_rates.bar(x + w, time_pct, w, color='#b3b3b3', label='Timeout',  zorder=3, alpha=0.7)

    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(labels, rotation=45, ha='right')
    ax_rates.set_ylabel('Rate (%)')
    ax_rates.set_title('Outcome Rates')
    ax_rates.set_ylim(0, 110) # Reduced from 118 to keep bars substantial
    ax_rates.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Move legend outside or to a better spot to prevent compression
    ax_rates.legend(
        frameon=False, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.9), 
        ncol=3, 
        fontsize=9,
        columnspacing=0.8,  # Adjust this value to bring entries closer
        handletextpad=0.4   # Adjust this to bring the color box closer to its text
    )

    # ── (b) Mean Control Effort ───────────────────────────────────────────
    means_e, stds_e = [], []
    for c in ctrls:
        if data[c].get('n_docking_effort', 0) > 0:
            means_e.append(data[c]['mean_control_effort'])
            stds_e.append(data[c]['std_control_effort'])
        else:
            means_e.append(0.0)
            stds_e.append(0.0)

    bars_e = ax_effort.bar(x, means_e, 0.6, yerr=stds_e, color=colors,
                            capsize=4, error_kw={'linewidth': 0.8}, zorder=3, alpha=0.7)
    
    for i, c in enumerate(ctrls):
        if data[c].get('n_docking_effort', 0) == 0:
            bars_e[i].set_hatch('///')
            bars_e[i].set_facecolor('#e0e0e0')
            ax_effort.text(i, 5, 'N/A', ha='center', va='bottom',
                           fontsize=8, color='#999999', fontweight='bold')


    ax_effort.set_xticks(x)
    ax_effort.set_xticklabels(labels, rotation=45, ha='right')
    ax_effort.set_ylabel('Control Effort (N·s)')
    ax_effort.set_title('Mean Control Effort')
    ax_effort.grid(axis='y', alpha=0.3, linestyle='--')

    # ── (c) Mean Computation Wall Time ────────────────────────────────────
    means_w = [data[c]['mean_wall_time'] for c in ctrls]
    stds_w  = [data[c]['std_wall_time']  for c in ctrls]

    ax_wall.bar(x, means_w, 0.6, yerr=stds_w, color=colors,
                capsize=4, error_kw={'linewidth': 0.8}, zorder=3, alpha=0.7)
    ax_wall.set_xticks(x)
    ax_wall.set_xticklabels(labels, rotation=45, ha='right')
    ax_wall.set_ylabel('Wall Time (s)')
    ax_wall.set_title('Mean Comp. Time') # Shortened title for space
    ax_wall.grid(axis='y', alpha=0.3, linestyle='--')

    # ── (d) Mean Docking Time ─────────────────────────────────────────────
    means_d, stds_d = [], []
    for c in ctrls:
        if data[c].get('n_docking_effort', 0) > 0:
            means_d.append(data[c]['mean_dock_time'])
            stds_d.append(data[c]['std_dock_time'])
        else:
            means_d.append(0.0)
            stds_d.append(0.0)

    bars_d = ax_dock.bar(x, means_d, 0.6, yerr=stds_d, color=colors,
                         capsize=4, error_kw={'linewidth': 0.8}, zorder=3, alpha=0.7)
    
    for i, c in enumerate(ctrls):
        if data[c].get('n_docking_effort', 0) == 0:
            bars_d[i].set_hatch('///')
            bars_d[i].set_facecolor('#e0e0e0')
            ax_dock.text(i, 0.5, 'N/A', ha='center', va='bottom',
                         fontsize=8, color='#999999', fontweight='bold')

    ax_dock.set_xticks(x)
    ax_dock.set_xticklabels(labels, rotation=45, ha='right')
    ax_dock.set_ylabel('Docking Time (s)')
    ax_dock.set_title('Mean Docking Time')
    ax_dock.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for legend and x-labels
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'controller_comparison_13D.png'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-quality comparison figures from results JSON.')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to comparison_results.json')
    parser.add_argument('--output_dir', type=str, default='figs/',
                        help='Directory to save figures (default: figs/)')
    args = parser.parse_args()

    print(f"Loading results from {args.results_path} ...")
    data = load_results(args.results_path)

    # Strip non-controller keys
    skip = {'_metadata', '_docking_optimality'}
    ctrl_data = {k: v for k, v in data.items() if k not in skip}

    print(f"  Controllers found: {list(ctrl_data.keys())}")
    for c, v in ctrl_data.items():
        print(f"    {c}: dock={v['docking_rate']:.1%}  "
              f"fail={v['failure_rate']:.1%}  "
              f"n_docked={len(v.get('docked', []))}")

    print("\nGenerating Figure 1 — Metrics comparison ...")
    plot_metrics_comparison(ctrl_data, args.output_dir)

    print(f"\nDone. Figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
