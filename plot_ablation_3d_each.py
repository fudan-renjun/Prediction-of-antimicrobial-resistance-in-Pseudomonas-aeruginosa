"""
8 individual 3D ribbon ablation plots – one per antibiotic.
Each page shows all retained models for that drug as separate ribbons.
Optimal feature count marked with a star (scatter marker='*').
Drug name + model list annotated in upper-right.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D                 # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import os

os.chdir(r'e:\RS\铜绿')

plt.rcParams.update({
    'font.family':  'Arial',
    'font.size':    7.5,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})

# ── Colour / marker per model ─────────────────────────────────────
MODEL_COLOR  = {'LGBM':'#2166AC', 'GB':'#D6604D', 'XGB':'#4DAC26', 'RF':'#8073AC'}
MODEL_MARKER = {'LGBM':'o',       'GB':'s',        'XGB':'^',       'RF':'D'}

# ── Drug list ─────────────────────────────────────────────────────
drugs_cn = ['环丙沙星','左氧氟沙星','氨曲南','庆大霉素',
            '妥布霉素','头孢吡肟','头孢他啶','哌拉西林']
drugs_en = ['Ciprofloxacin','Levofloxacin','Aztreonam','Gentamicin',
            'Tobramycin','Cefepime','Ceftazidime','Piperacillin']
PANEL_LABELS = list('ABCDEFGH')

FULL_FEAT  = 92
X_FULL     = 46      # visual x position for All-features point
Z_FLOOR    = 0.46
Y_SPACING  = 2.0     # depth gap between model ribbons

# ── Save all 8 pages to one PDF ───────────────────────────────────
with PdfPages('ablation_3d_each.pdf') as pdf:

    for drug_cn, drug_en, panel in zip(drugs_cn, drugs_en, PANEL_LABELS):
        xl = pd.ExcelFile(f'建模结果/{drug_cn}/results.xlsx')
        abl_sheets = sorted([s for s in xl.sheet_names
                             if s.startswith('Ablation_')])
        models_in_drug = [s.replace('Ablation_', '') for s in abl_sheets]

        # ── Figure ────────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 6.5))
        ax  = fig.add_subplot(111, projection='3d')

        all_z = []

        for m_idx, model in enumerate(models_in_drug):
            y_pos  = m_idx * Y_SPACING
            color  = MODEL_COLOR.get(model, '#555555')

            abl    = xl.parse(f'Ablation_{model}')
            delong = xl.parse(f'AblaDeLong_{model}')

            x_data   = abl['N_Features'].values.astype(float)
            z_data   = abl['AUC'].values
            p_vals   = delong['P_value'].values
            full_auc = float(delong['Full_AUC'].iloc[0])

            # Optimal n = first feature count where DeLong P >= 0.05
            opt_mask = delong['P_value'] >= 0.05
            opt_n    = float(delong.loc[opt_mask, 'N_Features'].iloc[0]) \
                       if opt_mask.any() else 40.0
            opt_z    = float(z_data[int(opt_n) - 1])

            all_z.extend(z_data.tolist())
            all_z.append(full_auc)

            fill_ret  = mcolors.to_rgba(color, alpha=0.28)
            fill_sig  = mcolors.to_rgba(color, alpha=0.09)

            # ── Ribbon (Poly3DCollection) ──────────────────────────
            verts_sig, verts_ret = [], []
            for i in range(len(x_data) - 1):
                quad = [
                    [x_data[i],   y_pos, Z_FLOOR],
                    [x_data[i],   y_pos, z_data[i]],
                    [x_data[i+1], y_pos, z_data[i+1]],
                    [x_data[i+1], y_pos, Z_FLOOR],
                ]
                (verts_sig if p_vals[i] < 0.05 else verts_ret).append(quad)

            if verts_sig:
                ax.add_collection3d(Poly3DCollection(
                    verts_sig,
                    facecolors=[fill_sig] * len(verts_sig),
                    edgecolors='none'))
            if verts_ret:
                ax.add_collection3d(Poly3DCollection(
                    verts_ret,
                    facecolors=[fill_ret] * len(verts_ret),
                    edgecolors='none'))

            # ── Top-edge AUC line ─────────────────────────────────
            sig_i = np.where(p_vals < 0.05)[0]
            ret_i = np.where(p_vals >= 0.05)[0]
            if len(sig_i):
                ax.plot(x_data[sig_i], [y_pos]*len(sig_i), z_data[sig_i],
                        color=color, lw=0.9, alpha=0.35)
            if len(ret_i):
                ax.plot(x_data[ret_i], [y_pos]*len(ret_i), z_data[ret_i],
                        color=color, lw=2.2, alpha=0.95)

            # ── Full-feature point (plain circle) ─────────────────
            ax.scatter([X_FULL], [y_pos], [full_auc],
                       s=45, marker='o', color=color,
                       edgecolors='white', linewidths=0.8,
                       depthshade=False, zorder=8)
            # Dotted connector from end of ablation to full-feat
            ax.plot([40, X_FULL], [y_pos, y_pos], [z_data[-1], full_auc],
                    color=color, lw=0.8, linestyle=':', alpha=0.55)

            # ── Star at optimal feature count (scatter marker='*') ─
            ax.scatter([opt_n], [y_pos], [opt_z],
                       s=180, marker='*', color=color,
                       edgecolors='white', linewidths=0.5,
                       depthshade=False, zorder=10)

            # Vertical dashed line from floor to star
            ax.plot([opt_n, opt_n], [y_pos, y_pos],
                    [Z_FLOOR, opt_z],
                    color=color, lw=1.0, linestyle='--', alpha=0.65)

            # ── AUC value label next to star ──────────────────────
            ax.text(opt_n + 0.8, y_pos, opt_z + 0.005,
                    f'{opt_z:.3f}',
                    fontsize=5.8, color=color, va='bottom', ha='left')

        # ── Axes setup ────────────────────────────────────────────
        n_models = len(models_in_drug)
        z_min = max(Z_FLOOR, min(all_z) - 0.04)
        z_max = min(1.01,    max(all_z) + 0.04)

        ax.set_xlim(0.5, X_FULL + 1)
        ax.set_ylim(-0.8, (n_models - 1) * Y_SPACING + 0.8)
        ax.set_zlim(z_min, z_max)

        # X ticks
        ax.set_xticks([10, 20, 30, 40, X_FULL])
        ax.set_xticklabels(['10', '20', '30', '40', f'All\n({FULL_FEAT})'],
                           fontsize=6.5)

        # Y ticks: model names at ribbon positions
        ax.set_yticks([i * Y_SPACING for i in range(n_models)])
        ax.set_yticklabels(models_in_drug, fontsize=7.0)

        # Z ticks
        z_ticks = np.arange(np.ceil(z_min * 20) / 20,
                            z_max + 0.01, 0.05)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([f'{v:.2f}' for v in z_ticks], fontsize=6.5)

        ax.set_xlabel('Number of features', fontsize=7.5, labelpad=7)
        ax.set_zlabel('5-fold CV AUC',      fontsize=7.5, labelpad=6)
        ax.set_ylabel('')                  # model names on ticks

        # Viewing angle
        ax.view_init(elev=22, azim=-55)

        # Pane style
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#d8d8d8')
        ax.grid(True, linewidth=0.4, color='#eeeeee')

        # ── Upper-right annotation: drug + models ─────────────────
        legend_lines = [drug_en]
        for model in models_in_drug:
            legend_lines.append(f'  {model}')

        # Use ax.text2D for fixed screen-space position
        ax.text2D(0.97, 0.96,
                  drug_en,
                  transform=ax.transAxes,
                  fontsize=10, fontweight='bold',
                  color='#1a1a1a', va='top', ha='right')

        for k, model in enumerate(models_in_drug):
            c = MODEL_COLOR.get(model, '#555')
            ax.text2D(0.97, 0.88 - k * 0.08,
                      f'- {model}',
                      transform=ax.transAxes,
                      fontsize=8, color=c,
                      va='top', ha='right', fontweight='semibold')

        # ── Legend (bottom-left) ──────────────────────────────────
        handles = [
            mpatches.Patch(facecolor='#999', alpha=0.55,
                           label='Retained region (P >= 0.05)'),
            mpatches.Patch(facecolor='#999', alpha=0.15,
                           label='Excluded region (P < 0.05)'),
            plt.Line2D([0],[0], linestyle='--', color='gray',
                       lw=1.0, label='Optimal feature (vertical)'),
            plt.Line2D([0],[0], marker='*', color='gray',
                       lw=0, markersize=9,
                       label='Optimal feature point'),
            plt.Line2D([0],[0], marker='o', color='gray',
                       lw=0, markersize=6,
                       label=f'All-feature AUC ({FULL_FEAT})'),
        ]
        ax.legend(handles=handles,
                  loc='lower left',
                  bbox_to_anchor=(-0.08, -0.04),
                  fontsize=6.0, frameon=False,
                  handlelength=1.4, ncol=1)

        # Panel letter
        ax.text2D(0.01, 0.97, panel,
                  transform=ax.transAxes,
                  fontsize=12, fontweight='bold',
                  color='#111111', va='top', ha='left')

        fig.subplots_adjust(left=0.0, right=0.92, top=0.95, bottom=0.05)
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Done: {drug_en}  ({len(models_in_drug)} models)')

print('Saved: ablation_3d_each.pdf  (8 pages)')
