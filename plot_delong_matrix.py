import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import os

os.chdir(r'e:\RS\铜绿')

drugs_cn = ['环丙沙星','左氧氟沙星','氨曲南','庆大霉素','妥布霉素','头孢吡肟','头孢他啶','哌拉西林']
drugs_en = ['Ciprofloxacin','Levofloxacin','Aztreonam','Gentamicin',
            'Tobramycin','Cefepime','Ceftazidime','Piperacillin']
ALL_MODELS = ['LGBM','GB','XGB','RF','AdaBoost','NB','DT','LR']

# ── 1. Load data ─────────────────────────────────────────────────
auc_table  = {}
pval_table = {}
ref_table  = {}

for drug_cn, drug_en in zip(drugs_cn, drugs_en):
    xl      = pd.ExcelFile(f'建模结果/{drug_cn}/results.xlsx')
    bstrap  = xl.parse('CV_Bootstrap_CI')
    auc_map = dict(zip(bstrap['Model'], bstrap['AUC']))

    delong    = xl.parse('DeLong_全特征')
    ref_model = delong['Model1'].iloc[0]

    pval_map = {ref_model: 1.0}
    for _, row in delong.iterrows():
        pval_map[row['Model2']] = row['P_value']

    auc_table[drug_en]  = auc_map
    pval_table[drug_en] = pval_map
    ref_table[drug_en]  = ref_model

# ── 2. Sort models by mean AUC ───────────────────────────────────
mean_auc = {m: np.nanmean([auc_table[d].get(m, np.nan) for d in drugs_en])
            for m in ALL_MODELS}
models_sorted = sorted(ALL_MODELS, key=lambda m: -mean_auc[m])

n_models = len(models_sorted)
n_drugs  = len(drugs_en)

AUC_MAT  = np.zeros((n_models, n_drugs))
PVAL_MAT = np.full((n_models, n_drugs), np.nan)
for j, drug_en in enumerate(drugs_en):
    for i, model in enumerate(models_sorted):
        AUC_MAT[i, j]  = auc_table[drug_en].get(model, np.nan)
        PVAL_MAT[i, j] = pval_table[drug_en].get(model, np.nan)

# ── 3. Color scheme (p-value, no special gold for reference) ─────
BOUNDS = [0, 0.001, 0.01, 0.05, 0.5, 1.001]
COLORS = ['#1a3d5c', '#2b6b7f', '#5da8a2', '#c8e8e4', '#f0f8f7']

def get_bg(p):
    if np.isnan(p): return '#eeeeee'
    for i in range(len(BOUNDS)-1):
        if BOUNDS[i] <= p < BOUNDS[i+1]:
            return COLORS[i]
    return COLORS[-1]

def get_fg(p):
    if np.isnan(p): return '#888888'
    return '#ffffff' if p < 0.05 else '#1a1a1a'

def fmt_cell(p, is_ref):
    """Single-line label: AUC handled separately; this returns p-value string."""
    if is_ref:    return 'p = 1.000  Ref'
    if np.isnan(p): return 'N/A'
    if p < 0.0001: return 'p < 0.0001 ***'
    if p < 0.001:  return f'p = {p:.4f} ***'
    if p < 0.01:   return f'p = {p:.3f} **'
    if p < 0.05:   return f'p = {p:.3f} *'
    return f'p = {p:.3f}  ns'

# ── 4. Figure: width : height = 2 : 1 ───────────────────────────
# Each cell: cell_w x cell_h, with cell_w / cell_h driving 2:1 overall ratio
# Plot area: 8 cols x 8 rows  → need overall_w / overall_h = 2
# Fix cell_h, derive cell_w:  (8*cell_w) / (8*cell_h) = 2  → cell_w = 2*cell_h
cell_h   = 0.62   # inches
cell_w   = 1.24   # inches  (2 × cell_h)

left_mar  = 1.15
right_mar = 1.10
top_mar   = 0.70
bot_mar   = 1.05

fig_w = left_mar + n_drugs  * cell_w + right_mar   # ≈ 12.1"
fig_h = top_mar  + n_models * cell_h + bot_mar     # ≈ 6.7" → ratio ≈ 1.8

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'pdf.fonttype': 42,    # embed TrueType fonts → real selectable text in PDF
    'ps.fonttype': 42,
})

fig = plt.figure(figsize=(fig_w, fig_h))
ax_l = left_mar / fig_w
ax_b = bot_mar  / fig_h
ax_w = (n_drugs  * cell_w) / fig_w
ax_h = (n_models * cell_h) / fig_h
ax   = fig.add_axes([ax_l, ax_b, ax_w, ax_h])

for i, model in enumerate(models_sorted):
    for j, drug_en in enumerate(drugs_en):
        auc    = AUC_MAT[i, j]
        p      = PVAL_MAT[i, j]
        is_ref = (model == ref_table[drug_en])

        bg = get_bg(p)
        fg = get_fg(p)
        row_y = n_models - 1 - i   # y-coordinate of cell bottom

        # Background
        ax.add_patch(plt.Rectangle(
            [j, row_y], 1, 1,
            facecolor=bg, edgecolor='white', linewidth=1.0, zorder=1))

        # Single line: "AUC  |  p = x.xxx  **"
        auc_str  = f'AUC {auc:.3f}'
        p_str    = fmt_cell(p, is_ref)
        full_str = f'{auc_str}    {p_str}'

        ax.text(j + 0.5, row_y + 0.50,
                full_str,
                ha='center', va='center',
                fontsize=6.0, color=fg, zorder=2,
                fontweight='semibold')


# Separator after top retained block (visual guide only — omit if messy)
for i in range(n_models + 1):
    ax.axhline(i, color='white', linewidth=1.0, zorder=3)
for j in range(n_drugs + 1):
    ax.axvline(j, color='white', linewidth=1.0, zorder=3)

ax.set_xlim(0, n_drugs)
ax.set_ylim(0, n_models)
ax.spines[:].set_visible(False)
ax.tick_params(length=0)
ax.set_aspect('auto')   # cells are 2:1, not square

# X-axis
ax.set_xticks([j + 0.5 for j in range(n_drugs)])
ax.set_xticklabels(drugs_en, rotation=35, ha='right',
                   fontsize=8.5, fontweight='bold')

# Y-axis: model + mean AUC
ax.set_yticks([n_models - 1 - i + 0.5 for i in range(n_models)])
ax.set_yticklabels(
    [f'{m}  (mean {mean_auc[m]:.3f})' for m in models_sorted],
    fontsize=7.5)

ax.set_title(
    'Full-Feature Model Performance and DeLong Test vs Reference Model\n'
    '(each cell: CV AUC  |  DeLong p-value vs highest-AUC model per antibiotic)',
    fontsize=9, fontweight='bold', pad=10)

# ── 5. Colorbar ──────────────────────────────────────────────────
cmap = ListedColormap(COLORS)
norm = BoundaryNorm(BOUNDS, cmap.N)
cax_x = (left_mar + n_drugs * cell_w + 0.15) / fig_w
cax_y = bot_mar / fig_h
cax_w = 0.20 / fig_w
cax_h = (n_models * cell_h) / fig_h
cax   = fig.add_axes([cax_x, cax_y, cax_w, cax_h])
sm    = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar  = fig.colorbar(sm, cax=cax)
cbar.set_ticks(BOUNDS[:-1])
cbar.set_ticklabels(['<0.001','0.001–0.01','0.01–0.05','0.05–0.5','0.5–1.0'],
                    fontsize=6.2)
cbar.set_label('DeLong P value vs reference', fontsize=6.5,
               rotation=270, labelpad=13)
cbar.outline.set_linewidth(0.4)

# ── 6. Legend ────────────────────────────────────────────────────
ret_patch  = mpatches.Patch(facecolor=COLORS[-1], edgecolor='#ccc',
                             linewidth=0.6, label='Retained  (P ≥ 0.05,  ns)')
excl_patch = mpatches.Patch(facecolor=COLORS[0],
                             label='Excluded  (P < 0.001,  ***)')
fig.legend(handles=[ret_patch, excl_patch],
           loc='lower center', bbox_to_anchor=(0.44, 0.01),
           ncol=3, fontsize=6.5, frameon=False, handlelength=1.2)

plt.savefig('delong_matrix_heatmap.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
print(f'Saved: delong_matrix_heatmap.pdf  ({fig_w:.1f}" x {fig_h:.1f}")')
