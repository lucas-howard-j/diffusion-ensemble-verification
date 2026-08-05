"""
Ensemble comparison figure — lead 20 (1 TU).

Three methods evaluated at matched spread (σ_ic = 0.158):
  DDPM (one-shot generative)
  Perfect-model dynamical (IC perturbation + RK4)
  Deterministic emulator   (IC perturbation + MSE ConvNet)

Two panels:
  (a) CRPS and VS skill side-by-side for each method
  (b) VS/CRPS ratio for each method

Style matches figure4.py.

Data:
    lorenz96/sweep/sweep_errors.npz                          — DDPM bootstrap CIs
    lorenz96/compare_ensembles/dyn_results{_sig0p01}.npz     — dynamical bootstrap CIs
    lorenz96/compare_ensembles/det_results{_sig0p01}.npz     — det emulator bootstrap CIs

Run:
    PYTHONPATH=. python scripts/figure_ensemble_comparison.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Style — match figure4 ─────────────────────────────────────────────────────
FS_TICK  = 6
FS_LABEL = 7
FS_TITLE = 7
FS_LEG   = 6.5

CRPS_COLOR = "#4C72B0"
VS_COLOR   = "#DD8452"
DIFF_COLOR = "#555555"
BOX_EDGE   = "#b8bcc8"
ERR_CAP    = 2.5
ERR_LW     = 0.9
ERR_COLOR  = "k"

OUT_DIR = Path("figures/main")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

# DDPM: bootstrap distributions from sweep at lead=20, n_train=249k (index [0, -1])
_sw         = np.load("lorenz96/sweep/sweep_errors.npz")
_crps_dist  = _sw["crps_skill_dist"][0, -1]   # (n_boot,)
_vs_dist    = _sw["vs_skill_dist"][0, -1]
_ratio_dist = _vs_dist / _crps_dist

def _ci95(dist):
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))

ddpm_crps  = float(np.median(_crps_dist))
ddpm_vs    = float(np.median(_vs_dist))
ddpm_ratio = float(np.median(_ratio_dist))
ddpm_crps_lo,  ddpm_crps_hi  = _ci95(_crps_dist)
ddpm_vs_lo,    ddpm_vs_hi    = _ci95(_vs_dist)
ddpm_ratio_lo, ddpm_ratio_hi = _ci95(_ratio_dist)

# Dynamical and det emulator: both σ_ic conditions
dyn_lin  = np.load("lorenz96/compare_ensembles/dyn_results_sig0p01.npz")
dyn_mat  = np.load("lorenz96/compare_ensembles/dyn_results.npz")
det_lin  = np.load("lorenz96/compare_ensembles/det_results_sig0p01.npz")
det_mat  = np.load("lorenz96/compare_ensembles/det_results.npz")

# Five conditions; methods grouped by background shading in the plot
labels = [
    "DDPM",
    "Dyn.\n$\\sigma_{ic}$=0.01",
    "Dyn.\n$\\sigma_{ic}$=0.16",
    "Det.\n$\\sigma_{ic}$=0.01",
    "Det.\n$\\sigma_{ic}$=0.16",
]
crps = np.array([
    ddpm_crps,
    float(dyn_lin["crps_skill"]), float(dyn_mat["crps_skill"]),
    float(det_lin["crps_skill"]), float(det_mat["crps_skill"]),
])
vs = np.array([
    ddpm_vs,
    float(dyn_lin["vs_skill"]), float(dyn_mat["vs_skill"]),
    float(det_lin["vs_skill"]), float(det_mat["vs_skill"]),
])
ratio = np.array([
    ddpm_ratio,
    float(dyn_lin["vs_skill"]) / float(dyn_lin["crps_skill"]),
    float(dyn_mat["vs_skill"]) / float(dyn_mat["crps_skill"]),
    float(det_lin["vs_skill"]) / float(det_lin["crps_skill"]),
    float(det_mat["vs_skill"]) / float(det_mat["crps_skill"]),
])

# CIs: (2, 5) arrays where row0=lo_err, row1=hi_err (distances from point)
def _asym(vals, los, his):
    return np.array([vals - los, his - vals])

crps_err = _asym(crps,
    np.array([ddpm_crps_lo,  float(dyn_lin["crps_lo"]), float(dyn_mat["crps_lo"]),
              float(det_lin["crps_lo"]), float(det_mat["crps_lo"])]),
    np.array([ddpm_crps_hi,  float(dyn_lin["crps_hi"]), float(dyn_mat["crps_hi"]),
              float(det_lin["crps_hi"]), float(det_mat["crps_hi"])]),
)
vs_err = _asym(vs,
    np.array([ddpm_vs_lo,  float(dyn_lin["vs_lo"]), float(dyn_mat["vs_lo"]),
              float(det_lin["vs_lo"]), float(det_mat["vs_lo"])]),
    np.array([ddpm_vs_hi,  float(dyn_lin["vs_hi"]), float(dyn_mat["vs_hi"]),
              float(det_lin["vs_hi"]), float(det_mat["vs_hi"])]),
)
ratio_err = _asym(ratio,
    np.array([ddpm_ratio_lo,  float(dyn_lin["ratio_lo"]), float(dyn_mat["ratio_lo"]),
              float(det_lin["ratio_lo"]), float(det_mat["ratio_lo"])]),
    np.array([ddpm_ratio_hi,  float(dyn_lin["ratio_hi"]), float(dyn_mat["ratio_hi"]),
              float(det_lin["ratio_hi"]), float(det_mat["ratio_hi"])]),
)

# ── x positions: gap between method groups ────────────────────────────────────
# Groups: [DDPM] [Dyn-lin  Dyn-mat] [Det-lin  Det-mat]
# Within-group x-gap = 0.65; bar pair width = 2*(w/2+off) must stay < 0.65
GROUP_GAP = 0.65   # extra space between method groups
x = np.array([0.0, 1.0, 1.65, 2.65, 3.30])
w   = 0.22   # individual bar width
off = 0.13   # offset from group centre (gap = 2*(off - w/2) = 0.02 between bars)
rw  = 0.32   # bar width for ratio panel

XLIM = (-0.45, x[-1] + 0.45)
METHOD_BG = "#f0f2f6"   # light shading behind each method group

def _shade_groups(ax):
    groups = [(0, 0, 0), (1, 2, 1), (3, 4, 2)]
    for first, last, shade in groups:
        if shade % 2 == 1:
            x0 = (x[first] + (x[first-1] if first > 0 else x[first] - GROUP_GAP)) / 2
            x1 = (x[last]  + (x[last+1]  if last < len(x)-1 else x[last] + GROUP_GAP)) / 2
            ax.axvspan(x0, x1, color=METHOD_BG, zorder=0, linewidth=0)

def _group_labels(ax, ydata_top, pad=0.02):
    """Method-group labels just inside the top of the axes, in data coords."""
    for lbl, xi in [("DDPM", x[0]), ("Dynamical", (x[1]+x[2])/2),
                    ("Det. emulator", (x[3]+x[4])/2)]:
        ax.text(xi, ydata_top - pad, lbl, ha="center", va="top",
                fontsize=FS_TICK, fontstyle="italic", color="0.4", zorder=5)

# ── Panel (a): CRPS and VS skill ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
ax = axes[0]

_shade_groups(ax)
bars_crps = ax.bar(x - off, crps, width=w, color=CRPS_COLOR, edgecolor="k",
                   linewidth=0.5, label="CRPS skill", zorder=3)
bars_vs   = ax.bar(x + off, vs,   width=w, color=VS_COLOR,   edgecolor="k",
                   linewidth=0.5, label="VS skill",   zorder=3)

ax.errorbar(x - off, crps, yerr=crps_err, fmt="none", color=ERR_COLOR,
            capsize=ERR_CAP, linewidth=ERR_LW, zorder=4)
ax.errorbar(x + off, vs,   yerr=vs_err,   fmt="none", color=ERR_COLOR,
            capsize=ERR_CAP, linewidth=ERR_LW, zorder=4)

ax.axhline(0, color="k", linewidth=0.6, linestyle=":", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FS_TICK)
ax.set_ylabel("Skill score", fontsize=FS_LABEL)
ax.set_title("(a)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_xlim(*XLIM)
ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
ax.legend(fontsize=FS_LEG, framealpha=0.95, handlelength=1.4,
          handletextpad=0.4, borderpad=0.5, loc="center right")
_group_labels(ax, ax.get_ylim()[1])

# ── Panel (b): VS / CRPS ratio ────────────────────────────────────────────────
ax = axes[1]

YLIM_B = (0.0, 1.22)
_shade_groups(ax)
bars_ratio = ax.bar(x, ratio, width=rw, color=DIFF_COLOR, edgecolor="k",
                    linewidth=0.5, zorder=3)

ax.errorbar(x, ratio, yerr=ratio_err, fmt="none", color=ERR_COLOR,
            capsize=ERR_CAP, linewidth=ERR_LW, zorder=4)

# Value labels above each bar's upper CI bound
ratio_hi = ratio + ratio_err[1]
for xi, v, hi in zip(x, ratio, ratio_hi):
    ax.text(xi, hi + 0.018, f"{v:.2f}", ha="center", va="bottom",
            fontsize=FS_TICK - 0.5, color="k")

ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", zorder=2)
# Annotate the reference line directly — no legend needed
ax.text(XLIM[1] - 0.04, 1.0 + 0.012, "VS/CRPS = 1",
        ha="right", va="bottom", fontsize=FS_TICK, color="k")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FS_TICK)
ax.set_ylabel("VS skill / CRPS skill", fontsize=FS_LABEL)
ax.set_title("(b)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM_B)
ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
_group_labels(ax, YLIM_B[1])

# ── Shared annotation ─────────────────────────────────────────────────────────
fig.suptitle(
    "Lead = 1.0 TU (20 steps)  |  $n_{\\mathrm{ens}}$ = 500  |  "
    "$\\sigma_{ic}$=0.01: linear regime  |  $\\sigma_{ic}$=0.16: spread-matched to DDPM",
    fontsize=FS_TITLE, y=1.01,
)

fig.tight_layout(w_pad=1.2)

out_pdf = OUT_DIR / "figure_ensemble_comparison.pdf"
out_png = OUT_DIR / "figure_ensemble_comparison.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")

# ── Console summary ───────────────────────────────────────────────────────────
print()
print(f"{'Method':<24}  {'CRPS':>8}  {'VS':>8}  {'VS/CRPS':>8}")
print("-" * 54)
for m, c, v, r in zip(labels, crps, vs, ratio):
    print(f"{m.replace(chr(10),' '):<28}  {c:>+8.4f}  {v:>+8.4f}  {r:>8.3f}")
