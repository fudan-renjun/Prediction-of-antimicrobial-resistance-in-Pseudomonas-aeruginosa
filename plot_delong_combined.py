import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from scipy import stats
import os

os.chdir(r'e:\RS\铜绿')

drugs_cn = ['环丙沙星','左氧氟沙星','氨曲南','庆大霉素','妥布霉素','头孢吡肟','头孢他啶','哌拉西林']
drugs_en = ['Ciprofloxacin','Levofloxacin','Aztreonam','Gentamicin',
            'Tobramycin','Cefepime','Ceftazidime','Piperacillin']
CV_SHEETS = {'RF':'CV_RF','DT':'CV_DT','GB':'CV_GB','XGB':'CV_XGB',
             'AdaBoost':'CV_AdaBoost','LR':'CV_LR','NB':'CV_NB','LGBM':'CV_LGBM'}
MODEL_ABBREV = {'RF':'RF','DT':'DT','GB':'GB','XGB':'XGB',
                'AdaBoost':'Ada','LR':'LR','NB':'NB','LGBM':'LGBM'}

BOUNDS = [0, 0.01, 0.05, 0.5, 0.8, 0.95, 1.001]
COLORS = ['#1a3d5c','#2b6b7f','#5da8a2','#aad4cc','#d4ece9','#f0f8f7']

def fmt_p(p, diag=False):
    if diag: return '1.000'
    if p < 0.0001: return '<0.0001\n***'
    if p < 0.001:  return f'{p:.4f}\n***'
    if p < 0.01:   return f'{p:.3f}\n**'
    if p < 0.05:   return f'{p:.3f}\n*'
    return f'{p:.3f}'

def cell_color(p, diag=False):
    if diag: return COLORS[-1]
    for i in range(len(BOUNDS)-1):
        if BOUNDS[i] <= p < BOUNDS[i+1]:
            return COLORS[i]
    return COLORS[-1]

def text_color(p, diag=False):
    return '#333333' if (diag or p >= 0.5) else 'white'

# ── Collect data ─────────────────────────────────────────────────
all_data = []
for drug_cn, drug_en in zip(drugs_cn, drugs_en):
    xl = pd.ExcelFile(f'建模结果/{drug_cn}/results.xlsx')
    delong = xl.parse('DeLong_全特征')
    ref_model = delong['Model1'].iloc[0]
    ref_auc   = delong['AUC1'].iloc[0]

    delong_pvals = {}
    retained = {ref_model}
    for _, row in delong.iterrows():
        delong_pvals[(row['Model1'], row['Model2'])] = row['P_value']
        delong_pvals[(row['Model2'], row['Model1'])] = row['P_value']
        if '保留' in str(row['Decision']):
            retained.add(row['Model2'])

    fold_aucs = {}
    for m, sh in CV_SHEETS.items():
        df = xl.parse(sh)
        fold_aucs[m] = df['AUC'].values[:5]

    bstrap  = xl.parse('CV_Bootstrap_CI')
    auc_map = dict(zip(bstrap['Model'], bstrap['AUC']))
    models  = sorted(auc_map.keys(), key=lambda m: -auc_map[m])
    n       = len(models)

    pmat = np.ones((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                pmat[i,j] = 1.0
            elif (m1, m2) in delong_pvals:
                pmat[i,j] = delong_pvals[(m1, m2)]
            else:
                a1 = fold_aucs.get(m1, np.zeros(5))
                a2 = fold_aucs.get(m2, np.zeros(5))
                _, p = stats.ttest_rel(a1, a2)
                pmat[i,j] = p

    all_data.append(dict(drug_en=drug_en, ref_model=ref_model, ref_auc=ref_auc,
                         retained=retained, models=models, auc_map=auc_map, pmat=pmat))

# ── Layout: 1 row x 8 columns ────────────────────────────────────
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6})

n_drugs   = 8
n_models  = 8
cell_size = 0.38        # inches per cell
panel_w   = cell_size * n_models   # 3.04"
panel_h   = cell_size * n_models   # 3.04"
gap_x     = 0.50
cbar_w    = 0.45
left_mar  = 0.82
top_mar   = 0.55
bot_mar   = 1.10

fig_w = left_mar + n_drugs * panel_w + (n_drugs - 1) * gap_x + cbar_w + 0.4
fig_h = top_mar + panel_h + bot_mar

fig = plt.figure(figsize=(fig_w, fig_h))

for col, d in enumerate(all_data):
    models   = d['models']
    pmat     = d['pmat']
    auc_map  = d['auc_map']
    retained = d['retained']
    n        = len(models)

    x0 = (left_mar + col * (panel_w + gap_x)) / fig_w
    y0 = bot_mar / fig_h
    w  = panel_w / fig_w
    h  = panel_h / fig_h
    ax = fig.add_axes([x0, y0, w, h])

    for i in range(n):
        for j in range(n):
            p    = pmat[i, j]
            diag = (i == j)
            rect = plt.Rectangle([j-0.5, n-1-i-0.5], 1, 1,
                                  facecolor=cell_color(p, diag),
                                  edgecolor='white', linewidth=0.5, zorder=1)
            ax.add_patch(rect)
            ax.text(j, n-1-i, fmt_p(p, diag),
                    ha='center', va='center', fontsize=4.0,
                    color=text_color(p, diag),
                    fontweight='bold' if diag else 'normal',
                    linespacing=1.1, zorder=2)

    # Red box around retained models
    ret_idx = sorted([i for i, m in enumerate(models) if m in retained])
    if ret_idx:
        lo, hi = min(ret_idx), max(ret_idx)
        ax.add_patch(plt.Rectangle(
            [lo-0.5, n-1-hi-0.5], hi-lo+1, hi-lo+1,
            fill=False, edgecolor='#C0392B', linewidth=1.4, zorder=5))

    abbrevs = [f'{MODEL_ABBREV[m]}\n({auc_map[m]:.3f})' for m in models]
    ax.set_xticks(range(n))
    ax.set_xticklabels(abbrevs, rotation=45, ha='right', fontsize=4.8)
    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.5, n-0.5)
    ax.set_aspect('equal')
    ax.tick_params(length=0, pad=1.5)
    ax.spines[:].set_visible(False)

    # Y-axis labels only on leftmost panel
    if col == 0:
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f'{MODEL_ABBREV[models[n-1-i]]}\n({auc_map[models[n-1-i]]:.3f})'
             for i in range(n)], fontsize=4.8)
    else:
        ax.set_yticks([])

    ax.set_title(f'{d["drug_en"]}\nRef: {d["ref_model"]}',
                 fontsize=6.0, fontweight='bold', pad=4)

    # Panel letter A-H
    ax.text(-0.48, n - 0.38, chr(65 + col),
            fontsize=8, fontweight='bold', ha='left', va='bottom',
            transform=ax.transData)

# ── Shared colorbar ──────────────────────────────────────────────
cmap = ListedColormap(COLORS)
norm = BoundaryNorm(BOUNDS, cmap.N)
cax_x = (left_mar + n_drugs * (panel_w + gap_x) - gap_x + 0.12) / fig_w
cax_y = bot_mar / fig_h
cax_w = 0.18 / fig_w
cax_h = panel_h / fig_h
cax   = fig.add_axes([cax_x, cax_y, cax_w, cax_h])
sm    = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar  = fig.colorbar(sm, cax=cax)
cbar.set_ticks(BOUNDS[:-1])
cbar.set_ticklabels(['<0.01','0.01-0.05','0.05-0.5','0.5-0.8','0.8-0.95','0.95-1'],
                    fontsize=5.2)
cbar.set_label('P value', fontsize=6.0, rotation=270, labelpad=10)
cbar.outline.set_linewidth(0.4)

# ── Legend ───────────────────────────────────────────────────────
kept = mpatches.Patch(facecolor='white', edgecolor='#C0392B', linewidth=1.2,
                      label='Retained (DeLong P >= 0.05 vs reference)')
excl = mpatches.Patch(facecolor='#1a3d5c',
                      label='Excluded (DeLong P < 0.05 vs reference)')
fig.legend(handles=[kept, excl], loc='lower center',
           bbox_to_anchor=(0.46, 0.005), ncol=2,
           fontsize=5.8, frameon=False, handlelength=1.2)

plt.savefig('delong_fullfeature_combined.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
print(f'Saved: delong_fullfeature_combined.pdf  ({fig_w:.1f} x {fig_h:.1f} inches)')
