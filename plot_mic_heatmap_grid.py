"""
Combined MIC correlation heatmap – two versions:
  mic_heatmap_noborder.pdf  – no cell borders (clean gradient look)
  mic_heatmap_border.pdf    – uniform thin borders on every cell

8 drugs (rows) × union of selected m/z features (cols).
Color  = Spearman r vs log2(MIC);  grey = not selected by model.
*      = BH-FDR Padj < 0.05.
Top panel = feature sharing frequency bar.
Cell journal style, pdf.fonttype=42, Arial.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import numpy as np
import pandas as pd
import re
import os

os.chdir(r'e:\RS\铜绿')

plt.rcParams.update({
    'font.family':  'Arial',
    'font.size':    7,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})

# ── Drug / model info ─────────────────────────────────────────────
drugs_cn = ['环丙沙星','左氧氟沙星','氨曲南','庆大霉素',
            '妥布霉素','头孢吡肟','头孢他啶','哌拉西林']
drugs_en = ['Ciprofloxacin','Levofloxacin','Aztreonam','Gentamicin',
            'Tobramycin','Cefepime','Ceftazidime','Piperacillin']
final_models = ['LGBM','LGBM','XGB','RF','LGBM','XGB','LGBM','LGBM']

MODEL_COLOR = {
    'LGBM':'#2166AC','GB':'#D6604D','XGB':'#4DAC26','RF':'#8073AC'}

# ── Load MIC correlation data ─────────────────────────────────────
drug_data = {}
for cn, en, fm in zip(drugs_cn, drugs_en, final_models):
    xl = pd.ExcelFile(f'建模结果/{cn}/results.xlsx')
    mic_sheet = next(s for s in xl.sheet_names if 'MIC' in s)
    drug_data[en] = xl.parse(mic_sheet)

# ── Build sorted union feature list ──────────────────────────────
all_feats = sorted(
    {f for df in drug_data.values() for f in df['Feature']},
    key=lambda x: int(re.search(r'\d+', x).group()))
n_feats  = len(all_feats)
feat_idx = {f: i for i, f in enumerate(all_feats)}
n_drugs  = len(drugs_en)

# ── Build r-matrix (NaN = not selected) and sig-mask ─────────────
r_mat   = np.full((n_drugs, n_feats), np.nan)
sig_mat = np.zeros((n_drugs, n_feats), dtype=bool)
for ri, en in enumerate(drugs_en):
    for _, rec in drug_data[en].iterrows():
        ci = feat_idx[rec['Feature']]
        r_mat[ri, ci]   = rec['Spearman_r']
        sig_mat[ri, ci] = bool(rec['Significant'])

nan_mask = np.isnan(r_mat)
freq     = (~nan_mask).sum(axis=0)      # 0..8 – feature sharing count

# Symmetric colour limit
CLIM = np.ceil(np.nanmax(np.abs(r_mat)) * 20) / 20   # round to 0.05
cmap_div = plt.cm.RdBu_r
norm     = TwoSlopeNorm(vmin=-CLIM, vcenter=0, vmax=CLIM)
GREY_CLR = '#e4e4e4'

# pcolormesh mesh edges
X_mesh = np.arange(n_feats + 1) - 0.5
Y_mesh = np.arange(n_drugs  + 1) - 0.5

# ── Shared draw function ──────────────────────────────────────────
def build_figure(edge_color, edge_lw, out_fname):
    FIG_W, FIG_H = 14.5, 7.2
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(
        2, 1,
        height_ratios=[1, 5],
        hspace=0.05,
        left=0.115, right=0.925,
        top=0.91,   bottom=0.15)

    ax_freq = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # ── Top: feature frequency bar ───────────────────────────────
    bar_c = plt.cm.YlOrRd(freq / 8)
    # base bars (yellow-orange gradient)
    ax_freq.bar(np.arange(n_feats), freq, width=0.85,
                color=bar_c, edgecolor='none')
    # red overlay for shared ≥ 3
    for ci in np.where(freq >= 3)[0]:
        ax_freq.bar(ci, freq[ci], width=0.85,
                    color='#d73027', edgecolor='none')
    ax_freq.set_xlim(-0.5, n_feats - 0.5)
    ax_freq.set_ylim(0, 9)
    ax_freq.set_yticks([2, 4, 6, 8])
    ax_freq.set_yticklabels(['2','4','6','8'], fontsize=6)
    ax_freq.set_ylabel('# Drugs\nsharing peak', fontsize=6, labelpad=3,
                       multialignment='center')
    ax_freq.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_freq.tick_params(axis='y', length=2, width=0.5)
    ax_freq.spines['top'].set_visible(False)
    ax_freq.spines['right'].set_visible(False)
    ax_freq.spines['bottom'].set_linewidth(0.4)
    ax_freq.spines['left'].set_linewidth(0.4)
    ax_freq.yaxis.grid(True, linewidth=0.3, color='#e8e8e8', zorder=0)
    ax_freq.set_axisbelow(True)

    # ── Main heatmap via pcolormesh ───────────────────────────────
    # 1. Grey layer for NaN cells
    grey_data = np.ma.masked_where(~nan_mask, np.ones((n_drugs, n_feats)))
    ax_heat.pcolormesh(X_mesh, Y_mesh, grey_data,
                       cmap=ListedColormap([GREY_CLR]),
                       edgecolors=edge_color, linewidth=edge_lw,
                       zorder=2)

    # 2. Coloured layer for data cells
    r_masked = np.ma.masked_where(nan_mask, r_mat)
    im = ax_heat.pcolormesh(X_mesh, Y_mesh, r_masked,
                            cmap=cmap_div, norm=norm,
                            edgecolors=edge_color, linewidth=edge_lw,
                            zorder=3)

    ax_heat.invert_yaxis()   # row 0 (CIP) at top

    # 3. Significance stars
    for ri in range(n_drugs):
        for ci in range(n_feats):
            if sig_mat[ri, ci]:
                rv = r_mat[ri, ci]
                fc = 'white' if abs(rv) > 0.20 else '#2d2d2d'
                ax_heat.text(ci, ri, '*',
                             ha='center', va='center',
                             fontsize=7, color=fc,
                             fontweight='bold', zorder=5)

    # ── Axes formatting ───────────────────────────────────────────
    ax_heat.set_xlim(-0.5, n_feats + 3.0)
    ax_heat.set_ylim(n_drugs - 0.5, -0.5)   # top=0, bottom=n_drugs

    # X ticks: m/z values
    ax_heat.set_xticks(np.arange(n_feats))
    ax_heat.set_xticklabels(
        [f.replace('mz_', '') for f in all_feats],
        rotation=90, fontsize=5.5, ha='center')
    ax_heat.tick_params(axis='x', length=2, pad=1, width=0.4)

    # Y labels: coloured model badge + drug name
    ax_heat.set_yticks(np.arange(n_drugs))
    ax_heat.set_yticklabels([])
    ax_heat.tick_params(axis='y', left=False)

    for ri, (en, fm) in enumerate(zip(drugs_en, final_models)):
        n_sel = len(drug_data[en])
        n_sig = int(drug_data[en]['Significant'].sum())
        mc    = MODEL_COLOR.get(fm, '#555')
        # model badge
        ax_heat.text(-5.0, ri, fm,
                     ha='right', va='center', fontsize=6.0,
                     color=mc, fontweight='bold',
                     transform=ax_heat.transData)
        # drug name
        ax_heat.text(-0.7, ri, en,
                     ha='right', va='center', fontsize=7.0,
                     color='#111',
                     transform=ax_heat.transData)
        # sig count on right
        ax_heat.text(n_feats - 0.1, ri, f'{n_sig}/{n_sel}',
                     ha='left', va='center', fontsize=6.0,
                     color='#555',
                     transform=ax_heat.transData)

    # Sig./Total header
    ax_heat.text(n_feats - 0.1, -0.75, 'Sig/Total',
                 ha='left', va='center', fontsize=6.0,
                 color='#444', style='italic',
                 transform=ax_heat.transData)

    # Thin outer spines
    for sp in ax_heat.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor('#aaaaaa')

    ax_heat.set_xlabel('m/z peak (Da)', fontsize=7.5, labelpad=7)

    # ── Colorbar ──────────────────────────────────────────────────
    cax = fig.add_axes([0.932, 0.15, 0.011, 0.59])
    cb  = fig.colorbar(im, cax=cax, extend='neither')
    cb.set_label('Spearman r\nvs log2(MIC)',
                 fontsize=6.5, labelpad=7, rotation=270, va='bottom')
    ticks = [-CLIM, -CLIM/2, 0, CLIM/2, CLIM]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{v:.2f}' for v in ticks], fontsize=6)
    cb.outline.set_linewidth(0.4)
    # Add midpoint labels for clarity
    cb.ax.axhline(0, color='#888', linewidth=0.5, linestyle='--')

    # ── Align freq bar x-axis with heatmap ────────────────────────
    ax_freq.set_xlim(ax_heat.get_xlim())

    # ── Legend ────────────────────────────────────────────────────
    leg_items = [
        mpatches.Patch(fc=GREY_CLR, ec='#aaa', lw=0.6,
                       label='Not selected by model'),
        plt.Line2D([0],[0], marker='*', color='#555',
                   lw=0, markersize=8,
                   label='BH-FDR Padj < 0.05'),
        mpatches.Patch(fc='#d73027',
                       label='Shared by >= 3 drugs (top bar)'),
    ]
    for mn, mc in MODEL_COLOR.items():
        leg_items.append(
            mpatches.Patch(fc=mc, label=f'{mn}'))

    fig.legend(handles=leg_items,
               loc='lower center',
               bbox_to_anchor=(0.50, -0.01),
               ncol=4, fontsize=6.5,
               frameon=False, handlelength=1.3,
               columnspacing=1.5, handletextpad=0.5)

    # ── Title ─────────────────────────────────────────────────────
    fig.suptitle(
        'Selected Feature Peaks vs log2(MIC) — Spearman Correlation  '
        '(Training Set, 8 Antibiotics)\n'
        'Rows: drug / final model;   Columns: m/z peaks (Da, sorted);   '
        'Grey: not selected;   * BH-FDR Padj < 0.05',
        fontsize=8, fontweight='bold', y=0.99,
        color='#1a1a1a')

    plt.savefig(out_fname, format='pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_fname}')


# ── Version 1: no borders ─────────────────────────────────────────
build_figure(edge_color='none', edge_lw=0,
             out_fname='mic_heatmap_noborder.pdf')

# ── Version 2: uniform thin borders ──────────────────────────────
build_figure(edge_color='white', edge_lw=0.55,
             out_fname='mic_heatmap_border.pdf')
