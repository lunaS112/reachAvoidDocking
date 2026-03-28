# Figure & Plot Style Guide for CDC 2026 Paper

> **Purpose**: Reference document for generating publication-quality figures and tables for the spacecraft docking BRAT paper. Follow these rules for every matplotlib plot, system diagram, and LaTeX table.

---

## 1. Color Palette

Use the SIA Lab color scheme consistently across all figures and tables.

### Primary Colors

| Role | Hex | Use |
|------|-----|-----|
| **Your method (highlight)** | `#ff9500` | BRAT controller, your value function, your trajectories |
| **Teal accent** | `#0d948f` | Terminal MPC or secondary novel contribution |
| **Medium gray** | `#b3b3b3` | Obvious/weak baselines (e.g., vanilla MPC) |
| **Dark gray** | `#6b6b6b` | Secondary baselines |
| **Black** | `#000000` | Grid-based ground truth, axes, text |
| **Dark blue** | `#0048a6` | Base DeepReach baseline |
| **Bright blue** | `#0091ff` | RL reach-avoid baseline |
| **Magenta** | `#850f67` | Safety filter interventions |
| **Purple** | `#740cad` | Alternative method variant if needed |
| **Light purple** | `#cdb3ff` | Shaded confidence bands |

### Set Colors (for geometry figures)

| Element | Color | Notes |
|---------|-------|-------|
| Goal set | Green (`#2ca02c` or similar) | Consistent across all geometry figures |
| Failure set | Red (`#d62728` or similar) | Target body + docking port |
| BRAT zero-level-set | Orange `#ff9500` | Your method's boundary |
| Grid-based BRAT | Black dashed | Ground truth reference |
| Collision buffer | Light red / pink with alpha | Inflated obstacle region |

### Rules

- **One highlight color for your method** — use `#ff9500` (orange) every time you show your BRAT controller results.
- **Dull colors for obvious baselines** — grays and black for MPC, grid-based.
- **Brighter colors for interesting baselines** — blues and purples for DeepReach, RL reach-avoid.
- **Shades of the same base color** for variants of the same method. Use a shade picker like [maketintsandshades.com](https://maketintsandshades.com/).
- **Be consistent** — if the BRAT controller is orange in Figure 2, it must be orange in Figure 5 and in the tables.

---

## 2. Matplotlib Plot Settings

Apply these settings globally at the top of every plotting script:

```python
import matplotlib.pyplot as plt
import matplotlib

# Font: use serif to match the LaTeX manuscript
matplotlib.rcParams.update({
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
    'axes.spines.top': False,     # Remove top border
    'axes.spines.right': False,   # Remove right border
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'pdf.fonttype': 42,           # Editable text in PDFs
    'ps.fonttype': 42,
})
```

### Axis Formatting Rules

- **Title case for all axis labels**: "Docking Time (s)", not "docking time (s)".
- **Include units** in parentheses: "Position (m)", "Angular Rate (rad/s)".
- **Tight axis limits** — do not leave large empty margins. Use `ax.set_xlim()` and `ax.set_ylim()` explicitly.
- **No bold axis labels** by default. Only bold if the figure is very small and readability suffers.
- **Grid lines**: use `ax.grid(True, alpha=0.3, linestyle='--')` if needed, otherwise omit.

### Legend Rules

- If two subplots share the same methods, use **one shared legend** placed between or below them, not duplicate legends.
- Place legends outside the plot area when possible: `ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`.
- If space is tight, use `ax.legend(frameon=False)` for a cleaner look.

---

## 3. Figure Types for This Paper

### 3.1 System / Geometry Diagram (Fig. 1)

- Shows chaser-target docking geometry for 6D and 13D.
- Goal set in green, failure set in red, collision buffer in light red.
- Label coordinate axes (LVLH frame: +x radial, +y along-track, +z completing).
- Use consistent line styles: solid for boundaries, dashed for inflated regions.
- Add dimension annotations (e.g., "3 m", "0.6 m") with leader lines.
- This is likely made in a vector tool (Inkscape, PowerPoint, Keynote) or matplotlib with patches.

### 3.2 BRAT Slice Comparison (Fig. 2 — validation)

- 2D positional slice (x vs. y) of the value function zero-level-set.
- Learned BRAT: solid orange line.
- Grid-based BRAT: black dashed line.
- Goal and failure sets shown as filled regions (green, red).
- Use identical axis limits for both plots.
- Multiple time horizons can be shown as subplots or overlaid with decreasing alpha.

### 3.3 Trajectory Comparison (Fig. 3)

- Position-space trajectory (x vs. y or 3D) from a shared initial condition.
- Color-code by controller: orange for BRAT, black for grid, gray for MPC, blue for DeepReach.
- Mark start with a circle (`o`) and end with an `x`.
- Show goal set and failure set as background patches.
- Optionally overlay time-spaced dots along the trajectory to show speed.

### 3.4 Controller Comparison Bar Chart (Fig. 4)

- Grouped bar chart: x-axis = metric (success rate, collision rate, etc.), grouped bars = controllers.
- Use the color palette consistently.
- Add value labels on top of bars if space permits.
- Error bars for metrics with variance (mean ± std).

### 3.5 Time-Series Plots (optional)

- State components over time for a representative trajectory.
- Subplots: position, velocity, attitude, angular rate.
- Shade tolerance bands (goal tolerances) in light green.
- Mark Phase 1 → Phase 2 transition with a vertical dashed line.

### 3.6 Controller Architecture Block Diagram (Fig. 5)

- Shows Phase 1 (convergence) and Phase 2 (precision) logic.
- Color-code novel components in orange/teal, off-the-shelf in gray.
- Show the safety filter as a separate block with a distinct color (magenta).
- Annotate with key mathematical symbols used in the text ($V(x,T)$, $\nabla_x V$, $t^*$).
- Mark offline vs. online components with background shading or labeled regions.

---

## 4. Figure Sizing

### IEEE Two-Column Format

| Figure Type | Width | LaTeX Command |
|-------------|-------|---------------|
| Single-column figure | 3.5 in (88 mm) | `\begin{figure}[t]` |
| Double-column figure | 7.16 in (181 mm) | `\begin{figure*}[t]` |

### Matplotlib Export

```python
# Single-column figure
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Double-column figure
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

# Save as PDF for vector quality
fig.savefig('figure_name.pdf', bbox_inches='tight', pad_inches=0.05)
```

- Always save as **PDF** for vector graphics (plots, diagrams).
- Save as **PNG at 300 DPI** only for rasterized content (renderings, photos).
- **Check legibility** at the final printed size — fonts should be at least 8pt when the figure is scaled into the column.

---

## 5. Figure Captions

- **3–6 lines minimum.** Captions should be self-contained.
- Explain what the reader is looking at, what the colors/markers mean, and what the takeaway is.
- Reference relevant equations or sections.
- Example: "Positional slice of the learned BRAT zero-level-set (solid orange) compared with the grid-based ground truth (black dashed) at $t = 15$\,s. The goal set (green) and failure set (red) are shown for reference. The learned boundary closely matches the grid-based solution, confirming high-fidelity value function approximation."

---

## 6. Tables

### Style Rules

- Use `booktabs` package: `\toprule`, `\midrule`, `\bottomrule`. No vertical lines.
- Bold your method name in the leftmost column: `\textbf{BRAT (ours)}`.
- Bold the best result in each column.
- Report mean ± std where applicable.
- Mark infeasible entries with "N/A" or "—", not blank cells.
- Use `\footnotesize` or `\small` if the table is too wide for the column.

### Example Template

```latex
\begin{table}[t]
\caption{Controller Comparison ($N = 500$ Rollouts)}
\label{tab:comparison}
\centering
\small
\begin{tabular}{lccccc}
\toprule
Metric & \textbf{BRAT} & T-MPC & MPC & V-DR & RL \\
\midrule
Success (\%) & \textbf{98.2} & 95.1 & 82.4 & 61.3 & 74.8 \\
Collision (\%) & \textbf{0.4} & 1.2 & 5.8 & 12.1 & 8.6 \\
Timeout (\%) & 1.4 & 3.7 & 11.8 & 26.6 & 16.6 \\
Effort (N$\cdot$s) & 142 & 158 & 201 & 189 & 176 \\
Wall time (ms) & \textbf{0.8} & 12.4 & 85.2 & 0.9 & 0.7 \\
\bottomrule
\end{tabular}
\end{table>
```

### Color-Coding in Tables (optional)

- Use `\cellcolor` from the `colortbl` package to highlight your method's row.
- Use sparingly — one highlight color max. Standard CDC papers rarely color tables.

---

## 7. Before/After Comparison Figures

When showing before/after (e.g., untrained vs. trained value function, or Phase 1 vs. Phase 2):

1. **Use identical axis limits** for both panels.
2. **Use identical color scales** if showing heatmaps/contours.
3. **Label panels** as (a) and (b) in the top-left corner.
4. **Use a shared colorbar** placed to the right of the rightmost panel.

---

## 8. Saving and File Organization

```
figs/
├── geometry_6d_13d.pdf          # System diagram (Fig. 1)
├── brat_overlap_6d.pdf          # BRAT slice comparison (Fig. 2)
├── trajectory_comparison.pdf    # Trajectory comparison (Fig. 3)
├── controller_comparison_6d.pdf # Bar chart (Fig. 4)
├── controller_comparison_13d.pdf
├── controller_architecture.pdf  # Block diagram (Fig. 5)
├── trajectory_13d_example.pdf   # 13D trajectory example
└── docking_time_histogram.pdf   # Docking time distribution
```

- Name files descriptively.
- Keep one script per figure for reproducibility.
- Store plotting scripts alongside the figure outputs.

---

## 9. Quick Checklist Before Submission

For every figure:
- [ ] Uses the consistent color palette from Section 1
- [ ] Serif font (Times New Roman) for all text
- [ ] Top and right spines removed
- [ ] Axis labels in Title Case with units
- [ ] Tight axis limits, no wasted whitespace
- [ ] Legend is not duplicated across subplots
- [ ] Caption is 3–6 lines and self-contained
- [ ] Saved as PDF (vector) at appropriate size
- [ ] Legible at final printed column width (fonts ≥ 8pt)
- [ ] Before/after panels share axis limits and color scales

For every table:
- [ ] Uses `booktabs` (no vertical lines)
- [ ] Your method is bolded in the name column
- [ ] Best results are bolded per column
- [ ] Mean ± std reported where applicable
- [ ] Fits within the column width
