"""
Feature ablation curves – 3-column grid, Cell journal style.
X-axis: 1–40 features (ablation), then a break, then 92 (full-feature AUC).
One panel per drug; each panel shows all retained models for that drug.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import os

os.chdir(r'e:\RS\铜绿')

# ── Global style ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':   'Arial',
    'font.size':     7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size':  3,   'ytick.major.size':  3,
    'xtick.minor.size':  0,   'ytick.minor.size':  0,
    'pdf.fonttype':  42,
    'ps.fonttype':   42,
})

# ── Colour palette (Cell-friendly) ───────────────────────────────
MODEL_COLOR = {
    'LGBM': '#2166AC',   # deep blue
    'GB':   '#D6604D',   # brick red
    'XGB':  '#4DAC26',   # green
    'RF':   '#8073AC',   # purple
}
MODEL_MARKER = {'LGBM': 'o', 'GB': 's', 'XGB': '^', 'RF': 'D'}

# ── Drug info ────────────────────────────────────────────────────
drugs_cn = ['环丙沙星','左氧氟沙星','氨曲南','庆大霉素',
            '妥布霉素','头孢吡肟','头孢他啶','哌拉西林']
drugs_en = ['Ciprofloxacin','Levofloxacin','Aztreonam','Gentamicin',
            'Tobramycin','Cefepime','Ceftazidime','Piperacillin']
PANEL_LABELS = list('ABCDEFGH')

FULL_FEAT = 92    # full-feature x-position label
X_FULL    = 45    # visual x position for the full-feature point (after gap)
X_MAX_PLOT = 46   # right edge of x-axis

# ── Figure layout: 3 cols × 3 rows ───────────────────────────────
NCOLS, NROWS = 3, 3
fig_w, fig_h = 8.5, 8.0      # inches
fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(fig_w, fig_h),
                         constrained_layout=True)
axes = axes.flatten()

# ── Helper: draw break marks on x-axis ───────────────────────────
def draw_break(ax, x_break=41.5, y_frac=-0.04, height=0.04):
    """Draw // break marks just below x-axis at x_break."""
    trans = ax.get_xaxis_transform()
    for dx in (-0.4, 0.4):
        ax.plot([x_break + dx - 0.5, x_break + dx + 0.5],
                [y_frac - height, y_frac + height],
                transform=trans, color='#666666',
                linewidth=0.9, clip_on=False)

# ── Plot each drug ────────────────────────────────────────────────
for idx, (drug_cn, drug_en) in enumerate(zip(drugs_cn, drugs_en)):
    ax = axes[idx]
    xl = pd.ExcelFile(f'建模结果/{drug_cn}/results.xlsx')

    abl_sheets = [s for s in xl.sheet_names if s.startswith('Ablation_')]
    models_in_drug = [s.replace('Ablation_', '') for s in abl_sheets]

    all_aucs = []   # collect for y-axis range

    for model in models_in_drug:
        color  = MODEL_COLOR.get(model, '#333333')
        marker = MODEL_MARKER.get(model, 'o')

        abl_df = xl.parse(f'Ablation_{model}')
        delong = xl.parse(f'AblaDeLong_{model}')

        x_data  = abl_df['N_Features'].values          # 1..40
        y_data  = abl_df['AUC'].values
        p_vals  = delong['P_value'].values
        full_auc = delong['Full_AUC'].iloc[0]

        # Optimal feature count (first n where P >= 0.05)
        opt_mask = delong['P_value'] >= 0.05
        opt_n    = delong.loc[opt_mask, 'N_Features'].iloc[0] if opt_mask.any() else 40

        all_aucs.extend(y_data.tolist())
        all_aucs.append(full_auc)

        # ── Split line: significant (p<0.05) vs retained (p>=0.05) ──
        sig_mask = p_vals < 0.05   # excluded region

        # Plot full ablation line (thin, muted)
        ax.plot(x_data, y_data,
                color=color, linewidth=1.0, alpha=0.35, zorder=2)

        # Overlay retained segment (p>=0.05) with bold line
        ret_x = x_data[~sig_mask]
        ret_y = y_data[~sig_mask]
        if len(ret_x):
            ax.plot(ret_x, ret_y,
                    color=color, linewidth=1.8, alpha=0.95, zorder=3)

        # Markers: filled=retained, open=excluded
        for xi, yi, sig in zip(x_data, y_data, sig_mask):
            if sig:
                ax.scatter(xi, yi, s=12, color='white',
                           edgecolors=color, linewidths=0.7,
                           zorder=4, alpha=0.6)
            else:
                ax.scatter(xi, yi, s=14, color=color,
                           linewidths=0, zorder=4)

        # Vertical dashed line at optimal n
        ax.axvline(opt_n, color=color, linewidth=0.9,
                   linestyle='--', alpha=0.7, zorder=1)

        # Full-feature point (star marker at X_FULL)
        ax.scatter(X_FULL, full_auc,
                   s=55, marker='*', color=color,
                   linewidths=0, zorder=5)
        # Connect last ablation point to full-feature with dotted line
        ax.plot([40, X_FULL], [y_data[-1], full_auc],
                color=color, linewidth=0.7,
                linestyle=':', alpha=0.5, zorder=2)

        # Model label at right end
        ax.text(X_FULL + 0.3, full_auc,
                f'{model}\n({full_auc:.3f})',
                va='center', ha='left', fontsize=5.0,
                color=color, zorder=6)

    # ── Shaded "retained" region background ──────────────────────
    ax.axvspan(0.5, opt_n + 0.5, alpha=0.04, color='#2166AC', zorder=0)

    # ── Axes formatting ───────────────────────────────────────────
    y_min = max(0.45, min(all_aucs) - 0.04)
    y_max = min(1.01, max(all_aucs) + 0.04)
    ax.set_xlim(0.5, X_MAX_PLOT)
    ax.set_ylim(y_min, y_max)

    # X-ticks: 10, 20, 30, 40, then gap, then X_FULL labelled as "92\n(All)"
    ax.set_xticks([10, 20, 30, 40, X_FULL])
    ax.set_xticklabels(['10', '20', '30', '40', f'All\n({FULL_FEAT})'],
                       fontsize=6.2)

    # Y-ticks
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.tick_params(axis='y', which='minor', length=2, width=0.5)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Light reference line at y = full_auc of first model (visual guide)
    ax.axhline(0.90, color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)

    # Axis labels
    ax.set_xlabel('Number of features', fontsize=7, labelpad=3)
    if idx % NCOLS == 0:
        ax.set_ylabel('5-fold CV AUC', fontsize=7, labelpad=3)

    # Panel title + letter
    ax.set_title(f'{PANEL_LABELS[idx]}  {drug_en}',
                 fontsize=8.0, fontweight='bold', loc='left', pad=4)

    # Break marks between 40 and X_FULL
    draw_break(ax, x_break=42.5)

    # Subtle grid
    ax.yaxis.grid(True, linewidth=0.4, color='#e8e8e8', zorder=0)
    ax.set_axisbelow(True)

# ── Hide unused panel (index 8) ───────────────────────────────────
axes[8].set_visible(False)

# ── Shared legend ─────────────────────────────────────────────────
legend_elements = []
for m in ['LGBM', 'GB', 'XGB', 'RF']:
    legend_elements.append(
        plt.Line2D([0], [0], color=MODEL_COLOR[m], linewidth=1.5,
                   marker=MODEL_MARKER[m], markersize=4,
                   label=m))
legend_elements += [
    plt.Line2D([0], [0], color='gray', linewidth=1.5,
               label='Retained region (DeLong P ≥ 0.05)'),
    plt.Line2D([0], [0], color='gray', linewidth=1.0, alpha=0.4,
               label='Excluded region (DeLong P < 0.05)'),
    plt.scatter([], [], s=40, marker='*', color='gray',
                label='Full-feature AUC (All features)'),
]

fig.legend(handles=legend_elements,
           loc='lower center',
           bbox_to_anchor=(0.36, -0.01),
           ncol=4, fontsize=6.5, frameon=False,
           handlelength=1.5, columnspacing=1.2)

plt.savefig('ablation_grid.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
print('Saved: ablation_grid.pdf')
