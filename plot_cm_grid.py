"""
2×4 grid of confusion matrices – training set, final optimised model per drug.
Cell journal style, A4 landscape, pdf.fonttype=42.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os

os.chdir(r'e:\RS\铜绿')

plt.rcParams.update({
    'font.family':  'Arial',
    'font.size':    7,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})

# ── Drug info ─────────────────────────────────────────────────────
drugs_cn = ['环丙沙星','左氧氟沙星','氨曲南','庆大霉素',
            '妥布霉素','头孢吡肟','头孢他啶','哌拉西林']
drugs_en = ['Ciprofloxacin','Levofloxacin','Aztreonam','Gentamicin',
            'Tobramycin','Cefepime','Ceftazidime','Piperacillin']
PANEL_LABELS = list('ABCDEFGH')

# Custom blue-white colormap (Cell-friendly)
CM_CMAP = LinearSegmentedColormap.from_list(
    'cm_blue', ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])

# ── Load confusion matrix data ────────────────────────────────────
records = []
for drug_cn, drug_en in zip(drugs_cn, drugs_en):
    xl    = pd.ExcelFile(f'建模结果/{drug_cn}/results.xlsx')
    fm    = xl.parse('最终模型')
    bs    = xl.parse('CV_Bootstrap_CI')
    model = fm['Final_Model'].iloc[0]
    n_feat= int(fm['N_Features'].iloc[0])
    row   = bs[bs['Model'] == model].iloc[0]
    records.append(dict(
        drug_en=drug_en, model=model, n_feat=n_feat,
        TP=int(row['TP']), TN=int(row['TN']),
        FP=int(row['FP']), FN=int(row['FN']),
        AUC =float(row['AUC']),
        Sens=float(row['Sensitivity']),
        Spec=float(row['Specificity']),
        Acc =float(row['Accuracy']),
    ))

# ── Figure: A4 landscape ──────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(11.69, 8.27))
fig.subplots_adjust(left=0.04, right=0.97,
                    top=0.93,  bottom=0.08,
                    wspace=0.38, hspace=0.48)

for ax, rec, panel in zip(axes.flatten(), records, PANEL_LABELS):
    TP, TN, FP, FN = rec['TP'], rec['TN'], rec['FP'], rec['FN']
    total = TP + TN + FP + FN

    # 2×2 matrix: rows = Actual (NS top, S bottom), cols = Pred (NS left, S right)
    cm = np.array([[TP, FN],
                   [FP, TN]], dtype=float)

    # Normalise per actual class (row-wise) for colour intensity
    cm_norm = cm.copy()
    cm_norm[0] /= (TP + FN) if (TP + FN) > 0 else 1
    cm_norm[1] /= (FP + TN) if (FP + TN) > 0 else 1

    im = ax.imshow(cm_norm, cmap=CM_CMAP, vmin=0, vmax=1,
                   aspect='equal')

    # Cell text: count (%) – white on dark cells, dark on light cells
    labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            val   = int(cm[i, j])
            pct   = cm_norm[i, j] * 100
            lbl   = labels[i][j]
            bright = cm_norm[i, j] < 0.55
            fc    = '#1a1a1a' if bright else 'white'

            ax.text(j, i - 0.12, f'{val}',
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color=fc)
            ax.text(j, i + 0.18, f'({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=8, color=fc, alpha=0.85)
            ax.text(j, i + 0.40, lbl,
                    ha='center', va='center',
                    fontsize=7, color=fc, alpha=0.65,
                    fontstyle='italic')

    # Axes labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred NS', 'Pred S'], fontsize=7.5)
    ax.set_yticklabels(['Actual NS', 'Actual S'], fontsize=7.5)
    ax.tick_params(length=0)

    # Move x-tick labels to top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Spine style
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#aaaaaa')

    # Title: panel letter + drug name + model
    ax.set_title(
        f'{panel}  {rec["drug_en"]}\n'
        f'{rec["model"]}  |  n={rec["n_feat"]} features',
        fontsize=8.0, fontweight='bold', pad=6, loc='left')

    # Bottom metrics bar
    metrics_txt = (f'AUC {rec["AUC"]:.3f}   '
                   f'Sens {rec["Sens"]:.3f}   '
                   f'Spec {rec["Spec"]:.3f}   '
                   f'Acc {rec["Acc"]:.3f}   '
                   f'N={total:,}')
    ax.set_xlabel(metrics_txt, fontsize=6.5, labelpad=6,
                  color='#444444')

    # Colourbar per panel (thin, right side)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        shrink=0.85)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0%', '50%', '100%'], fontsize=6)
    cbar.set_label('Row %', fontsize=6, rotation=270, labelpad=8)
    cbar.outline.set_linewidth(0.4)

# ── Figure title ─────────────────────────────────────────────────
fig.suptitle(
    'Confusion Matrices – Training Set (5-Fold CV Bootstrap)\n'
    'Final Optimised Model per Antibiotic',
    fontsize=10, fontweight='bold', y=0.99)

plt.savefig('cm_training_grid.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
print('Saved: cm_training_grid.pdf')
