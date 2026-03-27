"""
Butterfly (diverging) bar chart – 8 confusion matrices in one figure.
Each horizontal row = one drug.
Left  side: Actual NS class → TP (solid, Sensitivity) + FN (hatched, 1-Sens)
Right side: Actual S  class → TN (solid, Specificity) + FP (hatched, 1-Spec)
Bars normalised 0–1 within each actual class.
Cell journal style, A4 landscape, pdf.fonttype=42.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.chdir(r'e:\RS\铜绿')

plt.rcParams.update({
    'font.family':  'Arial',
    'font.size':    7,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})

# ── Data ─────────────────────────────────────────────────────────
records = [
    dict(drug='Ciprofloxacin (LGBM)',  TP=550, TN=648, FP=91,  FN=73,  AUC=0.9428),
    dict(drug='Levofloxacin (LGBM)',   TP=710, TN=491, FP=68,  FN=93,  AUC=0.9420),
    dict(drug='Aztreonam (XGB)',        TP=918, TN=246, FP=41,  FN=157, AUC=0.9233),
    dict(drug='Gentamicin (RF)',        TP=176, TN=720, FP=97,  FN=18,  AUC=0.9551),
    dict(drug='Tobramycin (LGBM)',      TP=161, TN=908, FP=100, FN=9,   AUC=0.9758),
    dict(drug='Cefepime (XGB)',         TP=417, TN=766, FP=87,  FN=92,  AUC=0.9340),
    dict(drug='Ceftazidime (LGBM)',     TP=400, TN=758, FP=122, FN=58,  AUC=0.9429),
    dict(drug='Piperacillin (LGBM)',    TP=513, TN=679, FP=69,  FN=61,  AUC=0.9559),
]
PANEL_LABELS = list('ABCDEFGH')

# Colours
C_TP = '#2171b5'   # deep blue    – correct NS
C_FN = '#9ecae1'   # light blue   – missed NS
C_TN = '#a63603'   # deep amber   – correct S
C_FP = '#fdbe85'   # light amber  – false alarm

BAR_H   = 0.55    # bar height (y units per row)
Y_GAP   = 1.0     # y-spacing between drug rows
n       = len(records)

fig, ax = plt.subplots(figsize=(11.69, 6.0))
y_ticks = []
y_labels = []

for idx, (rec, panel) in enumerate(zip(records, PANEL_LABELS)):
    TP, TN, FP, FN = rec['TP'], rec['TN'], rec['FP'], rec['FN']
    NS_tot = TP + FN
    S_tot  = FP + TN
    Sens   = TP / NS_tot
    Spec   = TN / S_tot
    y = idx * Y_GAP

    # ── Left side: NS class ──────────────────────────────────────
    # TP (correct) extends from 0 to -Sens
    ax.barh(y, -Sens, height=BAR_H,
            color=C_TP, align='center', zorder=3)
    # FN (missed) stacks further left  (-Sens to -1.0)
    ax.barh(y, -(1 - Sens), left=-Sens, height=BAR_H,
            color=C_FN, align='center',
            hatch='////', edgecolor='white', linewidth=0.3, zorder=3)

    # ── Right side: S class ──────────────────────────────────────
    # TN (correct) extends from 0 to +Spec
    ax.barh(y, Spec, height=BAR_H,
            color=C_TN, align='center', zorder=3)
    # FP (false alarm) stacks further right (Spec to 1.0)
    ax.barh(y, (1 - Spec), left=Spec, height=BAR_H,
            color=C_FP, align='center',
            hatch='////', edgecolor='white', linewidth=0.3, zorder=3)

    # ── Annotations ──────────────────────────────────────────────
    fs = 6.5
    # Sensitivity value (inside TP bar)
    ax.text(-Sens / 2, y, f'{Sens:.3f}',
            ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=5)
    # 1-Sens (inside FN bar, only if wide enough)
    fn_w = 1 - Sens
    if fn_w > 0.06:
        ax.text(-Sens - fn_w / 2, y, f'{FN}',
                ha='center', va='center', fontsize=fs - 0.5,
                color='#1a1a1a', zorder=5)
    # Specificity value (inside TN bar)
    ax.text(Spec / 2, y, f'{Spec:.3f}',
            ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=5)
    # 1-Spec (inside FP bar)
    fp_w = 1 - Spec
    if fp_w > 0.06:
        ax.text(Spec + fp_w / 2, y, f'{FP}',
                ha='center', va='center', fontsize=fs - 0.5,
                color='#1a1a1a', zorder=5)

    # AUC label on the right margin
    ax.text(1.06, y, f'AUC {rec["AUC"]:.3f}',
            ha='left', va='center', fontsize=6.5,
            color='#333333', zorder=5,
            transform=ax.get_yaxis_transform())

    # Panel letter on left margin
    ax.text(-1.10, y, panel,
            ha='center', va='center', fontsize=7.5,
            color='#111111', fontweight='bold', zorder=5)

    # NS / S sample sizes above/below bar ends
    ax.text(-1.02, y + BAR_H / 2 + 0.07,
            f'NS  n={NS_tot:,}',
            ha='right', va='bottom', fontsize=5.8, color='#2171b5')
    ax.text(1.02, y + BAR_H / 2 + 0.07,
            f'S  n={S_tot:,}',
            ha='left', va='bottom', fontsize=5.8, color='#a63603')

    y_ticks.append(y)
    y_labels.append(rec['drug'])

# ── Centre line ───────────────────────────────────────────────────
ax.axvline(0, color='#555555', linewidth=0.8, zorder=4)

# ── Axes ─────────────────────────────────────────────────────────
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-0.75, (n - 1) * Y_GAP + 0.75)

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=7.5)
ax.tick_params(axis='y', length=0, pad=5)

# X ticks: symmetric 0–100%
x_ticks = np.array([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'{abs(v)*100:.0f}%' for v in x_ticks], fontsize=6.5)

# Subtle grid
ax.xaxis.grid(True, linewidth=0.4, color='#eeeeee', zorder=0)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.5)
    ax.spines[spine].set_edgecolor('#aaaaaa')

# ── Axis labels ───────────────────────────────────────────────────
ax.set_xlabel('Proportion within actual class', fontsize=7.5, labelpad=6)

# Secondary x-axis labels for NS / S sides
ax.text(-0.5, (n - 1) * Y_GAP + 0.88,
        'Actual Non-Susceptible (NS)',
        ha='center', va='bottom', fontsize=8, color='#2171b5',
        fontweight='bold', transform=ax.transData)
ax.text(0.5, (n - 1) * Y_GAP + 0.88,
        'Actual Susceptible (S)',
        ha='center', va='bottom', fontsize=8, color='#a63603',
        fontweight='bold', transform=ax.transData)

# ── Legend ───────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc=C_TP, label='TP – Predicted NS, correct (Sensitivity)'),
    mpatches.Patch(fc=C_FN, hatch='////', ec='gray', lw=0.5,
                   label='FN – Predicted S,  wrong  (1 − Sensitivity)'),
    mpatches.Patch(fc=C_TN, label='TN – Predicted S,  correct (Specificity)'),
    mpatches.Patch(fc=C_FP, hatch='////', ec='gray', lw=0.5,
                   label='FP – Predicted NS, wrong  (1 − Specificity)'),
]
fig.legend(handles=handles,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.04),
           ncol=4, fontsize=6.5,
           frameon=False, handlelength=1.6,
           columnspacing=1.2)

# ── Title ─────────────────────────────────────────────────────────
ax.set_title(
    'Butterfly Confusion Chart – Training Set (5-Fold CV Bootstrap)  '
    '|  Bar length = proportion within actual class  |  Bold value = Sensitivity / Specificity',
    fontsize=8, fontweight='bold', pad=10)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('cm_butterfly.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
print('Saved: cm_butterfly.pdf')
