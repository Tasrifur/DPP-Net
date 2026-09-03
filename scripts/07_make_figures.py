#!/usr/bin/env python3
"""Step 7: regenerate every figure in the paper from the recorded numbers.

All accuracies and attention weights are hard-coded below, exactly as measured
by scripts 03-06, and are cross-checked against the manuscript before anything
is drawn. Nothing here recomputes a result from a model.

The one exception is Figure 6, whose top row is GEI silhouettes: those images
are read from the structured GEI dict. Pass --gei to enable it.

Usage:
    python scripts/07_make_figures.py                     # all number-only figures
    python scripts/07_make_figures.py --gei data/casiab_gei_structured_view.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ap = argparse.ArgumentParser()
_ap.add_argument("--gei", default=None,
                 help="Structured CASIA-B GEI dict, needed only for Figure 6")
_ap.add_argument("--outdir", default="figures")
ARGS = _ap.parse_args()
os.makedirs(ARGS.outdir, exist_ok=True)
os.chdir(ARGS.outdir)

# ==============================================================================
# CELL 1 — Setup + ALL RESULT NUMBERS + verification
# ==============================================================================
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['savefig.dpi'] = 300
SAVE = dict(dpi=300, bbox_inches='tight')

C3 = {'NM': '#1565C0', 'BG': '#E65100', 'CL': '#B71C1C'}
PART_NAMES = ["Head", "Shoulders", "UpperTorso", "LowerTorso",
              "Pelvis", "Thighs", "Calves", "Feet"]
UNIFORM = 1.0 / 8

# ==============================================================================
# RESULT NUMBERS — single source of truth. Nothing below recomputes these.
# Same-view: Test_2.  Cross-view: Test_3.  Transfer: CASIA-E / OU-MVLP.
# ==============================================================================

# --- Table 3: CASIA-B same-view, 20-way 5-shot, 300 episodes/cell -------------
CASIAB_MEAN = {'NM': 99.21, 'BG': 98.59, 'CL': 92.42}
CASIAB_AVG  = 96.74

# --- Table 4: ablation, fixed uniform vs learned dynamic ----------------------
ABL = {'NM':  (99.13, 99.18),
       'BG':  (81.61, 98.56),
       'CL':  (35.66, 92.04),
       'Avg': (72.13, 96.59)}

# --- Table 10a: k-shot sensitivity, view 090, 20-way, 600 episodes ------------
SHOTS   = [1, 3, 5, 10]
SHOT_NM = [98.99, 99.79, 99.93, 100.00]; SHOT_NM_CI = [0.11, 0.06, 0.03, 0.01]
SHOT_BG = [98.04, 99.26, 99.61,  99.65]; SHOT_BG_CI = [0.18, 0.12, 0.08, 0.08]
SHOT_CL = [96.71, 97.71, 97.68,  98.03]; SHOT_CL_CI = [0.20, 0.16, 0.17, 0.15]

# --- Table 10b: N-way sensitivity, view 090, 5-shot, 600 episodes -------------
NWAY    = [5, 10, 15, 20]
NWAY_NM = [99.95, 99.93, 99.94, 99.95]; NWAY_NM_CI = [0.05, 0.05, 0.04, 0.03]
NWAY_BG = [99.85, 99.65, 99.66, 99.54]; NWAY_BG_CI = [0.11, 0.12, 0.09, 0.09]
NWAY_CL = [99.61, 98.73, 98.41, 97.69]; NWAY_CL_CI = [0.18, 0.21, 0.17, 0.16]

# --- CASIA-B reference attention (mean over all views and conditions) ---------
CASIAB_ATTN = np.array([0.0937, 0.0014, 0.0027, 0.0015, 0.0134, 0.2243, 0.3840, 0.2790])

# --- Per-condition attention, SUBJECT 001 at view 090 (Test_2 XAI cell) -------
# These pair with the subject-001 / view-090 silhouettes loaded in Cell 4.
XAI_SUBJECT = '001'
XAI_VIEW    = '090'
XAI_W = {
    'NM': np.array([0.0989, 0.0022, 0.0038, 0.0029, 0.0282, 0.3762, 0.2835, 0.2044]),
    'BG': np.array([0.1162, 0.0003, 0.0005, 0.0001, 0.0003, 0.0142, 0.5780, 0.2904]),
    'CL': np.array([0.0346, 0.0001, 0.0003, 0.0001, 0.0006, 0.2025, 0.4905, 0.2713]),
}

# --- Per-subject attention shift vs NM, 124 subjects at 090 ------------------
DELTA_BG = np.array([+0.05818, +0.00013, +0.00130, -0.00154, -0.01555, -0.30281, +0.15704, +0.10325])
DELTA_CL = np.array([-0.00467, -0.00107, -0.00132, -0.00252, -0.02438, -0.26763, +0.23765, +0.06394])

# --- Table 8: CASIA-E zero-shot per angle, 20-way 2-shot ---------------------
CE_ANGLES = ['000','015','030','045','060','075','090','105','120','135','150','165','180','270']
CE_NOVEL  = {'105', '120', '270'}
CE_NM = [99.19, 97.78, 97.33, 69.09, 99.39, 98.99, 99.29, 99.77, 99.77, 74.27, 99.07, 99.43, 99.40, 99.96]
CE_BG = [87.65, 76.60, 65.94, 50.63, 78.25, 68.88, 72.18, 77.98, 80.33, 53.39, 65.27, 74.06, 80.14, 81.03]
CE_CL = [84.73, 69.09, 57.20, 45.91, 76.05, 66.56, 70.30, 72.39, 76.47, 48.79, 62.25, 71.42, 75.36, 78.37]
CE_MEAN = {'NM': 95.19, 'BG': 72.31, 'CL': 68.21}
CE_AVG  = 78.57

# --- Table 8: OU-MVLP zero-shot per angle, 20-way 1-shot ---------------------
OU_ANGLES = ['000','015','030','045','060','075','090','180','195','210','225','240','255','270']
OU_SV     = [65.62, 90.34, 94.27, 95.87, 94.33, 94.63, 93.36, 75.72, 90.40, 92.83, 93.68, 87.86, 91.97, 90.64]
OU_SV_CI  = [ 0.93,  0.58,  0.45,  0.39,  0.44,  0.44,  0.46,  0.78,  0.56,  0.48,  0.48,  0.58,  0.54,  0.62]
OU_MEAN   = 89.39

# --- Table 11: OU-MVLP N-way scaling ----------------------------------------
OU_NWAY_N  = [5, 10, 20, 50, 100]
OU_NWAY_ZS = [97.24, 95.50, 93.21, 89.39, 86.81]
OU_NWAY_CI = [ 0.66,  0.58,  0.48,  0.40,  0.28]
OU_NWAY_ID = [99.56, 99.36, 98.88, 98.06, 96.99]

# --- Attention cosine to the CASIA-B reference ------------------------------
COS_ZS = {'CASIA-E': 0.9571, 'OU-MVLP': 0.9668}
COS_OU_IN_DOMAIN = 0.9434
COS_CE_PER_COND  = {'NM': 0.9322, 'BG': 0.9642, 'CL': 0.9750}
COS_OU_MIN       = 0.8969   # at 270 deg

# --- Table 9: CASIA-E zero-shot vs in-domain --------------------------------
ZS = {'NM': 95.19, 'BG': 72.31, 'CL': 68.21, 'Avg': 78.57}
ID = {'NM': 98.46, 'BG': 90.16, 'CL': 82.00, 'Avg': 90.21}

# ==============================================================================
# VERIFICATION — must agree with the manuscript before anything is drawn
# ==============================================================================
def _close(a, b, tol=0.02):
    return abs(a - b) <= tol

checks = [
    ("Table 3 mean",        _close(np.mean(list(CASIAB_MEAN.values())), CASIAB_AVG)),
    ("Table 4 fixed avg",   _close(np.mean([ABL[c][0] for c in 'NM BG CL'.split()]), ABL['Avg'][0])),
    ("Table 4 dynamic avg", _close(np.mean([ABL[c][1] for c in 'NM BG CL'.split()]), ABL['Avg'][1])),
    ("CASIA-E NM mean",     _close(np.mean(CE_NM), CE_MEAN['NM'], 0.05)),
    ("CASIA-E BG mean",     _close(np.mean(CE_BG), CE_MEAN['BG'], 0.05)),
    ("CASIA-E CL mean",     _close(np.mean(CE_CL), CE_MEAN['CL'], 0.05)),
    ("CASIA-E overall",     _close(np.mean(list(CE_MEAN.values())), CE_AVG, 0.05)),
    ("OU-MVLP mean",        _close(np.mean(OU_SV), OU_MEAN, 0.05)),
    ("attn sums to 1",      _close(CASIAB_ATTN.sum(), 1.0, 1e-3)),
    ("XAI NM sums to 1",    _close(XAI_W['NM'].sum(), 1.0, 2e-3)),
    ("XAI BG sums to 1",    _close(XAI_W['BG'].sum(), 1.0, 2e-3)),
    ("XAI CL sums to 1",    _close(XAI_W['CL'].sum(), 1.0, 2e-3)),
    ("Fig8 ZS == Table 8",  ZS['Avg'] == CE_AVG),
    ("Fig8 ID == Table 9",  _close(np.mean([ID[c] for c in 'NM BG CL'.split()]), ID['Avg'], 0.05)),
]
for name, ok in checks:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
assert all(ok for _, ok in checks), "Number verification failed — fix Cell 1 before plotting."

print("\nAll numbers verified.")
print(f"  CASIA-B same-view avg : {CASIAB_AVG:.2f}%")
print(f"  Ablation avg gain     : {ABL['Avg'][1] - ABL['Avg'][0]:+.2f} points")
print(f"  Largest ablation gain : {ABL['CL'][1] - ABL['CL'][0]:+.2f} points (CL)")
print(f"  CASIA-E zero-shot avg : {CE_AVG:.2f}%")
print(f"  OU-MVLP zero-shot avg : {OU_MEAN:.2f}%")
print(f"  Lower-limb attention  : {CASIAB_ATTN[5:].sum():.4f}")
print(f"  Above-uniform parts   : {[PART_NAMES[i] for i in range(8) if CASIAB_ATTN[i] > UNIFORM]}")

# ==============================================================================
# CELL 2 — part_weights_bar.png   →   THE INSET FOR FIGURE 2 / fig:episode
# ==============================================================================
# The old version used an illustrative head-dominant vector in which the calves
# had weight 0.0005. In the measured reference the calves are the LARGEST part
# (0.384) and the head falls BELOW uniform. Both facts change the picture.

weights = CASIAB_ATTN

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(8)
bars = ax.bar(x, weights, color=['#D84315' if w > UNIFORM else '#90A4AE' for w in weights],
              edgecolor='black', linewidth=0.9, alpha=0.92)

for bar, w in zip(bars, weights):
    if w > UNIFORM:
        bar.set_edgecolor('#B8860B'); bar.set_linewidth(2.2)
    ax.text(bar.get_x() + bar.get_width()/2, w + 0.008, f'{w:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='#B71C1C' if w > UNIFORM else '#555')

ax.axhline(y=UNIFORM, color='gray', linestyle='--', linewidth=1.4,
           label=f'Uniform (1/8 = {UNIFORM:.3f})')
ax.set_xticks(x); ax.set_xticklabels(PART_NAMES, rotation=35, ha='right', fontsize=10)
ax.set_ylabel('Attention Weight', fontsize=12)
ax.set_title('Dynamic Part Attention Weights $w$', fontsize=13, fontweight='bold')
ax.set_ylim(0, weights.max() * 1.20)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle=':', alpha=0.5); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("part_weights_bar.png", **SAVE)
plt.savefig("part_weights_bar.pdf", bbox_inches='tight')
plt.close()

print("Saved part_weights_bar.png / .pdf")
print(f"  above uniform : {[PART_NAMES[i] for i in range(8) if weights[i] > UNIFORM]}")
print(f"  head          : {weights[0]:.4f}  (BELOW uniform - remove its highlight)")
print(f"  calves        : {weights[6]:.4f}  (largest single part)")

# ==============================================================================
# CELL 3 — figure4_casiab_panels.png   (PDF Figure 5)
# ==============================================================================
# (a) same-view CCR by condition   (b) k-shot   (c) N-way   (d) ablation
#
# Panel (d) ylim starts at 0. The previous code used (75, 105), which would
# clip the CL fixed-weight bar (35.66%) clean off the axis.

fig = plt.figure(figsize=(20, 5))
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.30)
ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i]) for i in range(4)]

# (a) --------------------------------------------------------------------------
conds = ['NM', 'BG', 'CL']
vals  = [CASIAB_MEAN[c] for c in conds]
ax1.bar(conds, vals, color=[C3[c] for c in conds], edgecolor='black', alpha=0.88)
for i, v in enumerate(vals):
    ax1.text(i, v + 1.2, f'{v:.2f}', ha='center', fontweight='bold')
ax1.axhline(5, color='red', ls='--', lw=1.2, label='Random')
ax1.set_ylabel('CCR (%)'); ax1.set_ylim(0, 108)
ax1.set_title('(a) Same-View CCR by Condition', fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(axis='y', ls=':', alpha=0.4)

# (b) --------------------------------------------------------------------------
ax2.errorbar(SHOTS, SHOT_NM, yerr=SHOT_NM_CI, fmt='o-', color=C3['NM'], label='NM', lw=2, capsize=4)
ax2.errorbar(SHOTS, SHOT_BG, yerr=SHOT_BG_CI, fmt='s-', color=C3['BG'], label='BG', lw=2, capsize=4)
ax2.errorbar(SHOTS, SHOT_CL, yerr=SHOT_CL_CI, fmt='^-', color=C3['CL'], label='CL', lw=2, capsize=4)
ax2.set_xlabel('k (shots)'); ax2.set_ylabel('Accuracy (%)'); ax2.set_xticks(SHOTS)
ax2.set_ylim(95.5, 100.6)
ax2.set_title('(b) k-Shot Sensitivity (20-way, 090$^\\circ$)', fontweight='bold')
ax2.legend(fontsize=9, loc='lower right'); ax2.grid(ls=':', alpha=0.4)

# (c) --------------------------------------------------------------------------
ax3.errorbar(NWAY, NWAY_NM, yerr=NWAY_NM_CI, fmt='o-', color=C3['NM'], label='NM', lw=2, capsize=4)
ax3.errorbar(NWAY, NWAY_BG, yerr=NWAY_BG_CI, fmt='s-', color=C3['BG'], label='BG', lw=2, capsize=4)
ax3.errorbar(NWAY, NWAY_CL, yerr=NWAY_CL_CI, fmt='^-', color=C3['CL'], label='CL', lw=2, capsize=4)
ax3.set_xlabel('N (ways)'); ax3.set_ylabel('Accuracy (%)'); ax3.set_xticks(NWAY)
ax3.set_ylim(97.0, 100.4)
ax3.set_title('(c) N-Way Sensitivity (5-shot, 090$^\\circ$)', fontweight='bold')
ax3.legend(fontsize=9, loc='lower left'); ax3.grid(ls=':', alpha=0.4)

# (d) --------------------------------------------------------------------------
abl_conds = ['NM', 'BG', 'CL', 'Avg']
fixed_v = [ABL[c][0] for c in abl_conds]
dyn_v   = [ABL[c][1] for c in abl_conds]
xa = np.arange(len(abl_conds))
ax4.bar(xa - 0.2, fixed_v, 0.38, label='Fixed uniform ($w_i = 1/8$)',
        color='#90A4AE', edgecolor='black', alpha=0.88)
ax4.bar(xa + 0.2, dyn_v, 0.38, label='Residual dynamic (ours)',
        color='#FF5722', edgecolor='black', alpha=0.92)
for i, (f, d) in enumerate(zip(fixed_v, dyn_v)):
    ax4.text(i - 0.2, f + 1.5, f'{f:.2f}', ha='center', fontsize=8)
    ax4.text(i + 0.2, d + 1.5, f'{d:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax4.text(i, max(f, d) + 6.5, f'{d - f:+.2f}', ha='center', fontsize=10,
             color='darkgreen', fontweight='bold')
ax4.axhline(5, color='red', ls='--', lw=1.0)
ax4.set_xticks(xa); ax4.set_xticklabels(abl_conds)
ax4.set_ylabel('CCR (%)'); ax4.set_ylim(0, 118)
ax4.set_title('(d) Ablation: Fixed vs Dynamic Weighting', fontweight='bold')
ax4.legend(fontsize=8, loc='lower left'); ax4.grid(axis='y', ls=':', alpha=0.4)

plt.savefig("figure4_casiab_panels.png", **SAVE)
plt.close()
print("Saved figure4_casiab_panels.png")
print(f"  panel (d) CL fixed bar = {ABL['CL'][0]:.2f}% -- visible, ylim starts at 0")

# ==============================================================================
# FIGURE 6 (figure5_xai.png) — two-row layout, as published
# ==============================================================================
# The GEI dict is read for SILHOUETTE IMAGES ONLY. No model, no checkpoint, no
# inference. The attention weights come from XAI_W above.
#
# The silhouettes must be the ones the recorded weights were measured on:
# subject 001, view 090, first sequence of each condition. Both are asserted.

def make_figure5_xai(gei_path):
    import torch

    print(f"Loading GEI images from {gei_path} ...")
    gei_struct = torch.load(gei_path)
    print(f"Loaded ({len(gei_struct)} subjects)")

    assert XAI_SUBJECT in gei_struct, (
        f"Subject {XAI_SUBJECT} not in the GEI dict. The weights in XAI_W were "
        f"measured on this subject; do not substitute another one.")
    for c in ["NM", "BG", "CL"]:
        assert XAI_VIEW in gei_struct[XAI_SUBJECT].get(c, {}), \
            f"Subject {XAI_SUBJECT} has no {c} at view {XAI_VIEW}"

    silhouettes = {c: gei_struct[XAI_SUBJECT][c][XAI_VIEW][0].squeeze().cpu().numpy()
                   for c in ["NM", "BG", "CL"]}
    print(f"Silhouettes: subject {XAI_SUBJECT}, view {XAI_VIEW}, "
          f"first sequence per condition, shape {silhouettes['NM'].shape}")

    cond_titles = {"NM": "Normal (NM)", "BG": "Carrying Bag (BG)",
                   "CL": "Wearing Coat (CL)"}
    cond_keys = ["NM", "BG", "CL"]
    wmax = max(XAI_W[c].max() for c in cond_keys)
    ylim_top = wmax * 1.22      # BG calves reach 0.578; a fixed 0.6 ceiling clips it
    print(f"max weight = {wmax:.4f} (BG calves) -> ylim {ylim_top:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, cond in enumerate(cond_keys):
        img, w = silhouettes[cond], XAI_W[cond]

        ax = axes[0, i]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Probe: {cond_titles[cond]}\nView {XAI_VIEW}$^\\circ$",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
        h = img.shape[0]
        for p in range(1, 8):
            ax.axhline(y=p * (h / 8), color="yellow", linestyle=":", lw=1.2, alpha=0.7)
        for p in range(8):
            ax.text(1, (p + 0.5) * (h / 8), PART_NAMES[p], color="yellow",
                    fontsize=6.5, va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4))

        ax = axes[1, i]
        x = np.arange(8)
        bars = ax.bar(x, w, color=C3[cond], alpha=0.85, edgecolor="black", lw=0.8)
        ax.axhline(y=UNIFORM, color="gray", linestyle="--", lw=1.3,
                   label=f"Uniform = {UNIFORM:.3f}")
        for bar, v in zip(bars, w):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="darkred" if v > UNIFORM else "dimgray")
            if v > UNIFORM:
                bar.set_edgecolor("goldenrod")
                bar.set_linewidth(2.0)
        ax.set_ylabel("Attention Weight", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(PART_NAMES, rotation=40, ha="right", fontsize=9)
        ax.set_ylim(0, ylim_top)
        ax.set_title(f"Part Weights — {cond_titles[cond]}", fontsize=11,
                     fontweight="bold")
        ax.yaxis.grid(True, linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(fontsize=9)

    plt.suptitle(f"XAI: Residual-Conditioned Part Attention — "
                 f"Subject {XAI_SUBJECT} (CASIA-B)\n"
                 f"Gold border = above uniform | weights adapt per condition",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("figure5_xai.png", **SAVE)
    plt.close()
    print("Saved figure5_xai.png (two-row layout)")
    for c in cond_keys:
        top = [PART_NAMES[i] for i in np.argsort(-XAI_W[c])[:3]]
        print(f"  {c}: top-3 parts = {top}")


if ARGS.gei:
    make_figure5_xai(ARGS.gei)
else:
    print("Figure 6 skipped: pass --gei <structured GEI dict> to render it.")

# ==============================================================================
# CELL 5 — figure6_zeroshot_panels.png   (PDF Figure 7)
# ==============================================================================
# (a) CASIA-E per angle  (b) OU-MVLP per angle  (c) OU-MVLP N-way  (d) cosine

fig = plt.figure(figsize=(22, 5))
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.28)
axA, axB, axC, axD = [fig.add_subplot(gs[i]) for i in range(4)]

# (a) CASIA-E zero-shot per angle ----------------------------------------------
x = np.arange(len(CE_ANGLES)); w = 0.27
axA.bar(x - w, CE_NM, w, label='NM', color=C3['NM'], alpha=0.88)
axA.bar(x,     CE_BG, w, label='BG', color=C3['BG'], alpha=0.88)
axA.bar(x + w, CE_CL, w, label='CL', color=C3['CL'], alpha=0.88)
for xi, ang in enumerate(CE_ANGLES):
    if ang in CE_NOVEL:
        axA.text(xi, 2, '*', ha='center', fontsize=13, color='gold', fontweight='bold')
axA.axhline(5, color='red', ls='--', lw=1.0)
axA.set_xticks(x); axA.set_xticklabels([str(int(a)) for a in CE_ANGLES], rotation=45, fontsize=7)
axA.set_ylabel('CCR (%)'); axA.set_ylim(0, 108)
axA.set_title(f'(a) CASIA-E Zero-Shot per Angle\n'
              f'avg {CE_AVG:.2f}%; * = novel; dips at 45$^\\circ$/135$^\\circ$',
              fontweight='bold', fontsize=9)
axA.legend(fontsize=8, ncol=3, loc='lower center'); axA.grid(axis='y', ls=':', alpha=0.4)

# (b) OU-MVLP zero-shot per angle ----------------------------------------------
x2 = np.arange(len(OU_ANGLES))
bars = axB.bar(x2, OU_SV, yerr=OU_SV_CI, capsize=3, color='#00897B',
               edgecolor='black', alpha=0.85, error_kw={'elinewidth': 0.8})
for idx in (OU_ANGLES.index('000'), OU_ANGLES.index('180')):
    bars[idx].set_color('#C62828'); bars[idx].set_alpha(0.90)
axB.axhline(OU_MEAN, color='#1565C0', ls='-', lw=1.4, label=f'Mean {OU_MEAN:.2f}%')
axB.axhline(5, color='red', ls='--', lw=1.0, label='Random')
axB.set_xticks(x2); axB.set_xticklabels([str(int(a)) for a in OU_ANGLES], rotation=45, fontsize=7)
axB.set_ylabel('CCR (%)'); axB.set_ylim(0, 108)
axB.set_title('(b) OU-MVLP Zero-Shot per Angle\n'
              'weakest at the 0$^\\circ$/180$^\\circ$ extremes (red)',
              fontweight='bold', fontsize=9)
axB.legend(fontsize=8, loc='lower center'); axB.grid(axis='y', ls=':', alpha=0.4)

# (c) OU-MVLP N-way scaling ------------------------------------------------------
axC.errorbar(OU_NWAY_N, OU_NWAY_ZS, yerr=OU_NWAY_CI, fmt='o-', color='#1565C0',
             lw=2.2, ms=8, capsize=4, label='Zero-shot')
axC.plot(OU_NWAY_N, OU_NWAY_ID, 's--', color='#2E7D32', lw=2.0, ms=7, label='In-domain')
axC.plot(OU_NWAY_N, [100 / n for n in OU_NWAY_N], ':', color='red', lw=1.3, label='Random')
for n, v in zip(OU_NWAY_N, OU_NWAY_ZS):
    axC.text(n, v - 5.5, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')
axC.set_xscale('log'); axC.set_xticks(OU_NWAY_N)
axC.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axC.set_xlabel('N (ways)'); axC.set_ylabel('CCR (%)'); axC.set_ylim(0, 105)
axC.set_title('(c) OU-MVLP $N$-Way Scaling\nview 090$^\\circ$, 1-shot',
              fontweight='bold', fontsize=9)
axC.legend(fontsize=8, loc='center left'); axC.grid(ls=':', alpha=0.4)

# (d) attention cosine ------------------------------------------------------------
names = list(COS_ZS); vals = [COS_ZS[n] for n in names]
barsD = axD.bar(names, vals, color=['#1B5E20' if v > 0.90 else '#F57F17' for v in vals],
                edgecolor='black', lw=0.8, alpha=0.90, width=0.55)
axD.axhline(0.90, color='green', ls='--', lw=1.5, label='> 0.90')
for b, v in zip(barsD, vals):
    axD.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
axD.set_ylim(0.5, 1.05); axD.set_ylabel('Cosine Similarity')
axD.set_title('(d) Attention Consistency\nvs CASIA-B reference',
              fontweight='bold', fontsize=9)
axD.legend(fontsize=8); axD.grid(axis='y', ls=':', alpha=0.4)

plt.savefig("figure6_zeroshot_panels.png", **SAVE)
plt.close()
print("Saved figure6_zeroshot_panels.png")
print(f"  CASIA-E novel-angle mean : "
      f"{np.mean([np.mean([CE_NM[i], CE_BG[i], CE_CL[i]]) for i, a in enumerate(CE_ANGLES) if a in CE_NOVEL]):.2f}%")
print(f"  CASIA-E shared-angle mean: "
      f"{np.mean([np.mean([CE_NM[i], CE_BG[i], CE_CL[i]]) for i, a in enumerate(CE_ANGLES) if a not in CE_NOVEL]):.2f}%")

# ==============================================================================
# CELL 6 — figure8_zeroshot_vs_indomain.png   (PDF Figure 8)
# ==============================================================================

conds = ['NM', 'BG', 'CL', 'Avg']
zs_v = [ZS[c] for c in conds]
id_v = [ID[c] for c in conds]
x = np.arange(len(conds))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(x - 0.2, zs_v, 0.38, label='Zero-shot (2-shot)', color='#90CAF9', edgecolor='black')
ax.bar(x + 0.2, id_v, 0.38, label='In-domain (5-shot)', color='#1565C0', edgecolor='black')
for i, (z, d) in enumerate(zip(zs_v, id_v)):
    ax.text(i - 0.2, z + 0.8, f'{z:.1f}', ha='center', fontsize=9)
    ax.text(i + 0.2, d + 0.8, f'{d:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i, max(z, d) + 4.5, f'+{d - z:.1f}', ha='center', fontsize=10,
            color='darkgreen', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=11)
ax.set_ylabel('CCR (%)', fontsize=12); ax.set_ylim(0, 112)
ax.set_title('CASIA-E: Zero-Shot vs In-Domain Training\n'
             f'Average {ZS["Avg"]:.2f}% $\\rightarrow$ {ID["Avg"]:.2f}% '
             f'(+{ID["Avg"] - ZS["Avg"]:.2f})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11); ax.yaxis.grid(True, ls=':', alpha=0.5); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figure8_zeroshot_vs_indomain.png", **SAVE)
plt.close()
print("Saved figure8_zeroshot_vs_indomain.png")
print("  Note: in-domain is 5-shot vs 2-shot zero-shot; Section 4.7 states this caveat.")

# ==============================================================================
# CELL 7 (OPTIONAL) — figure7_attention_consistency.png
# ==============================================================================
# Not cited in the current manuscript: panel (d) of Figure 6 already carries
# this. Generate only if you decide to add it back.

COSINES = {
    'CASIA-E\n(zero-shot)': COS_ZS['CASIA-E'],
    'OU-MVLP\n(zero-shot)': COS_ZS['OU-MVLP'],
    'OU-MVLP\n(in-domain)': COS_OU_IN_DOMAIN,
}

fig, ax = plt.subplots(figsize=(8, 5.5))
names = list(COSINES); vals = [COSINES[n] for n in names]
bars = ax.bar(names, vals, color=['#1B5E20' if v > 0.90 else '#F57F17' for v in vals],
              edgecolor='black', lw=0.9, alpha=0.92)
ax.axhline(0.90, color='green', ls='--', lw=1.6, label='> 0.90')
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f'{v:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0.5, 1.04)
ax.set_ylabel('Cosine Similarity to CASIA-B Attention', fontsize=12)
ax.set_title('Attention Consistency Across Datasets', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.yaxis.grid(True, ls=':', alpha=0.5); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figure7_attention_consistency.png", **SAVE)
plt.close()
print("Saved figure7_attention_consistency.png (optional, not cited)")

# ==============================================================================
# FIGURE 4 (figure3_gei_grid.png) — OPTIONAL
# ==============================================================================
# GEI images only; unchanged by the view-major rebuild, so the existing file
# remains valid. Regenerate only if you want a consistent dpi. Needs --gei.

def make_figure3_gei_grid(gei_path):
    import torch

    GRID_ANGLES = ["000", "054", "090", "126", "180"]
    GRID_CONDS = ["NM", "BG", "CL"]

    gei_struct = torch.load(gei_path)
    subj_pick = next(s for s in sorted(gei_struct)
                     if all(a in gei_struct[s].get(c, {})
                            for c in GRID_CONDS for a in GRID_ANGLES))
    print(f"Using subject {subj_pick}")

    fig, axes = plt.subplots(len(GRID_CONDS), len(GRID_ANGLES),
                             figsize=(len(GRID_ANGLES) * 2, len(GRID_CONDS) * 2))
    for r, cond in enumerate(GRID_CONDS):
        for c, ang in enumerate(GRID_ANGLES):
            ax = axes[r, c]
            ax.imshow(gei_struct[subj_pick][cond][ang][0].squeeze().cpu().numpy(),
                      cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f"{int(ang)}$^\\circ$", fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(cond, fontsize=12, fontweight="bold", color=C3[cond])
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(f"Sample Gait Energy Images — Subject {subj_pick} (CASIA-B)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figure3_gei_grid.png", **SAVE)
    plt.close()
    print("Saved figure3_gei_grid.png")


if ARGS.gei:
    make_figure3_gei_grid(ARGS.gei)
else:
    print("Figure 4 skipped: pass --gei to regenerate it (the existing file is still valid).")


# ==============================================================================
# CELL 9 — Collect outputs
# ==============================================================================
produced = [
    ("part_weights_bar.png",              "inset for PDF Fig. 3 (fig:episode)"),
    ("figure4_casiab_panels.png",         "PDF Fig. 5  (fig:casiab)"),
    ("figure5_xai.png",                   "PDF Fig. 6  (fig:xai)"),
    ("figure6_zeroshot_panels.png",       "PDF Fig. 7  (fig:zeroshot)"),
    ("figure8_zeroshot_vs_indomain.png",  "PDF Fig. 8  (fig:recovery)"),
    ("figure3_gei_grid.png",              "PDF Fig. 4  (fig:gei)  -- optional"),
    ("figure7_attention_consistency.png", "optional, not cited"),
]
unchanged = [
    ("gei.png",                                "PDF Fig. 1 -- GEI images, unchanged"),
    ("figure1_architecture_diagram.png",       "PDF Fig. 2 -- schematic, unchanged"),
    ("figure2_episode_attention_schematic.png",
     "PDF Fig. 3 -- schematic; swap in the new part_weights_bar inset"),
]

print(f"{'file':<42} {'size':<10} role")
print("-" * 95)
for fn, role in produced:
    if os.path.exists(fn):
        print(f"{fn:<42} {os.path.getsize(fn)/1024:>6.0f} KB  {role}")
    else:
        print(f"{fn:<42} {'--':<10} {role}")

print("\nNot regenerated (keep your existing files):")
for fn, role in unchanged:
    print(f"  {fn:<42} {role}")

try:
    import zipfile
    made = [fn for fn, _ in produced if os.path.exists(fn)]
    if os.path.exists("part_weights_bar.pdf"):
        made.append("part_weights_bar.pdf")
    with zipfile.ZipFile('dppnet_figures.zip', 'w') as z:
        for fn in made:
            z.write(fn)
    print(f"\nBundled {len(made)} files -> dppnet_figures.zip")
except Exception as e:
    print(f"\nzip step skipped: {e}")

print('\nAll figures written to', os.getcwd())
