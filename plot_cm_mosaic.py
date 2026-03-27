"""
Mosaic / Marimekko chart – 8 confusion matrices in one figure.
Column width  ∝  total bootstrap sample size per drug.
Within each column:
  left  sub-column = Actual NS  (width ∝ NS prevalence)
  right sub-column = Actual S   (width ∝ S  prevalence)
Height segments (bottom → top):
  Actual NS:  TP (predicted NS, correct) then FN (predicted S, wrong)
  Actual S:   FP (predicted NS, wrong)   then TN (predicted S, correct)
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

# ── Confusion matrix data (CV Bootstrap, final model per drug) ────
records = [
    dict(drug='CIP\n(LGBM)', TP=550, TN=648, FP=91,  FN=73,  AUC=0.9428),
    dict(drug='LVX\n(LGBM)', TP=710, TN=491, FP=68,  FN=93,  AUC=0.9420),
    dict(drug='ATM\n(XGB)',  TP=918, TN=246, FP=41,  FN=157, AUC=0.9233),
    dict(drug='GEN\n(RF)',   TP=176, TN=720, FP=97,  FN=18,  AUC=0.9551),
    dict(drug='TOB\n(LGBM)', TP=161, TN=908, FP=100, FN=9,   AUC=0.9758),
    dict(drug='FEP\n(XGB)',  TP=417, TN=766, FP=87,  FN=92,  AUC=0.9340),
    dict(drug='CAZ\n(LGBM)', TP=400, TN=758, FP=122, FN=58,  AUC=0.9429),
    dict(drug='PIP\n(LGBM)', TP=513, TN=679, FP=69,  FN=61,  AUC=0.9559),
]

# ── Colour scheme ─────────────────────────────────────────────────
C_TP = '#2171b5'   # deep blue   – Actual NS, Predicted NS (correct)
C_FN = '#9ecae1'   # light blue  – Actual NS, Predicted S  (wrong)
C_TN = '#a63603'   # deep amber  – Actual S,  Predicted S  (correct)
C_FP = '#fdbe85'   # light amber – Actual S,  Predicted NS (wrong)

PANEL_LABELS = list('ABCDEFGH')
GAP = 0.006   # fractional gap between drug columns

totals    = [r['TP']+r['TN']+r['FP']+r['FN'] for r in records]
total_all = sum(totals)
usable_w  = 1.0 - GAP * len(records)   # total width for bars

# ── Figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.69, 5.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('auto')

x_cursor   = 0.0
x_mids     = []        # centre x per drug  (for bottom tick)
x_ns_mids  = []        # centre of NS sub-col (for inner tick)
x_s_mids   = []        # centre of S  sub-col
col_widths  = []

for idx, (rec, panel) in enumerate(zip(records, PANEL_LABELS)):
    TP, TN, FP, FN = rec['TP'], rec['TN'], rec['FP'], rec['FN']
    total    = TP + TN + FP + FN
    NS_total = TP + FN        # actual NS count
    S_total  = FP + TN        # actual S  count

    col_w = total / total_all * usable_w
    ns_w  = col_w * NS_total / total
    s_w   = col_w * S_total  / total

    # --- Actual NS sub-column (left) --------------------------------
    tp_h = TP / NS_total      # correct (bottom)
    fn_h = FN / NS_total      # wrong   (top)

    ax.add_patch(mpatches.FancyArrow(0, 0, 0, 0))   # dummy (unused)
    ax.add_patch(mpatches.Rectangle(
        (x_cursor, 0), ns_w, tp_h, fc=C_TP, ec='white', lw=0.5, zorder=2))
    ax.add_patch(mpatches.Rectangle(
        (x_cursor, tp_h), ns_w, fn_h, fc=C_FN, ec='white', lw=0.5, zorder=2))

    # --- Actual S sub-column (right) --------------------------------
    fp_h = FP / S_total       # wrong   (bottom)
    tn_h = TN / S_total       # correct (top)

    ax.add_patch(mpatches.Rectangle(
        (x_cursor + ns_w, 0), s_w, fp_h, fc=C_FP, ec='white', lw=0.5, zorder=2))
    ax.add_patch(mpatches.Rectangle(
        (x_cursor + ns_w, fp_h), s_w, tn_h, fc=C_TN, ec='white', lw=0.5, zorder=2))

    # Thin inner divider (NS | S boundary)
    ax.plot([x_cursor + ns_w, x_cursor + ns_w], [0, 1],
            color='white', lw=1.2, zorder=3)

    # --- Cell annotations -------------------------------------------
    min_h_txt = 0.055   # minimum height to print text
    fs_main, fs_sub = 6.0, 5.5

    def cell_txt(cx, cy, label, count, pct, fc):
        ax.text(cx, cy + 0.030, f'{label}',
                ha='center', va='center', fontsize=fs_sub,
                color=fc, fontstyle='italic', alpha=0.80, zorder=4)
        ax.text(cx, cy,         f'{count}',
                ha='center', va='center', fontsize=fs_main,
                color=fc, fontweight='bold', zorder=4)
        ax.text(cx, cy - 0.028, f'({pct:.1f}%)',
                ha='center', va='center', fontsize=5.5,
                color=fc, alpha=0.85, zorder=4)

    cx_ns = x_cursor + ns_w / 2
    cx_s  = x_cursor + ns_w + s_w / 2

    if tp_h > min_h_txt:
        cell_txt(cx_ns, tp_h / 2,        'TP', TP, tp_h*100, 'white')
    if fn_h > min_h_txt:
        cell_txt(cx_ns, tp_h + fn_h / 2, 'FN', FN, fn_h*100, '#1a1a1a')
    if fp_h > min_h_txt:
        cell_txt(cx_s,  fp_h / 2,        'FP', FP, fp_h*100, '#1a1a1a')
    if tn_h > min_h_txt:
        cell_txt(cx_s,  fp_h + tn_h / 2, 'TN', TN, tn_h*100, 'white')

    # Sensitivity line at TP-FN boundary (horizontal dashes)
    ax.plot([x_cursor, x_cursor + ns_w], [tp_h, tp_h],
            color='white', lw=0.5, linestyle='--', alpha=0.6, zorder=3)
    ax.plot([x_cursor + ns_w, x_cursor + col_w], [fp_h, fp_h],
            color='white', lw=0.5, linestyle='--', alpha=0.6, zorder=3)

    # AUC label at top
    ax.text(x_cursor + col_w / 2, 1.015,
            f'AUC {rec["AUC"]:.3f}',
            ha='center', va='bottom', fontsize=6.0,
            color='#333333', zorder=5)

    # Panel letter
    ax.text(x_cursor + col_w / 2, 1.042,
            f'{panel}',
            ha='center', va='bottom', fontsize=7.0,
            color='#111111', fontweight='bold', zorder=5)

    x_mids.append(x_cursor + col_w / 2)
    x_ns_mids.append(cx_ns)
    x_s_mids.append(cx_s)
    col_widths.append(col_w)
    x_cursor += col_w + GAP

# ── Axes ──────────────────────────────────────────────────────────
drug_labels = [r['drug'] for r in records]
ax.set_xticks(x_mids)
ax.set_xticklabels(drug_labels, fontsize=7.0, ha='center')
ax.tick_params(axis='x', length=0, pad=4)

ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=6.5)
ax.set_ylabel('Row proportion (within actual class)', fontsize=7, labelpad=5)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_edgecolor('#aaaaaa')

# ── Y-axis annotations (right side) ──────────────────────────────
ax.text(1.002, 0.25, 'Actual NS\n(TP / FN)',
        va='center', ha='left', fontsize=6.5,
        color='#2171b5', transform=ax.transAxes)
ax.text(1.002, 0.75, 'Actual S\n(FP / TN)',
        va='center', ha='left', fontsize=6.5,
        color='#a63603', transform=ax.transAxes)

# ── Legend ────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc=C_TP, label='TP – Actual NS, Predicted NS (Sensitivity)'),
    mpatches.Patch(fc=C_FN, label='FN – Actual NS, Predicted S  (1 − Sensitivity)'),
    mpatches.Patch(fc=C_TN, label='TN – Actual S,  Predicted S  (Specificity)'),
    mpatches.Patch(fc=C_FP, label='FP – Actual S,  Predicted NS (1 − Specificity)'),
]
fig.legend(handles=handles,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.04),
           ncol=4, fontsize=6.5,
           frameon=False, handlelength=1.2,
           columnspacing=1.0)

# ── Width note ────────────────────────────────────────────────────
ax.set_title(
    'Mosaic Confusion Matrix – Training Set (5-Fold CV Bootstrap)  '
    '|  Column width proportional to bootstrap sample size  |  NS = Non-susceptible  |  S = Susceptible',
    fontsize=8, fontweight='bold', pad=22)

plt.savefig('cm_mosaic.pdf', format='pdf',
            bbox_inches='tight', facecolor='white')
print('Saved: cm_mosaic.pdf')
