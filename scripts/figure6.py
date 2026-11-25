"""
Figure 6 — Lorenz-96: does the training objective close the CRPS–VS gap?  Lead 20 (1 TU).

Three generative emulators trained on the same data, differing in how the
distribution is learned:
  DDPM            (denoising diffusion, noise-prediction MSE)
  CRPS-trained    (noise-conditioned ConvNet, fair CRPS per node)
  ES-trained      (same ConvNet, fair energy score over the 40-node state)
and, as matched-spread references (σ_IC = 15.8 % of the climatological SD,
marked on the Figure 5 sweep), the MSE-trained deterministic emulator and the
perfect-model dynamical ensemble.

Two panels:
  (a) CRPS and VS skill for each model
  (b) VS/CRPS ratio for each model

Data:
    lorenz96/ensemble_results/ddpm_results.npz     — DDPM (lead 20, n_train 249k) bootstrap CIs
    lorenz96/ensemble_results/crps_results.npz     — CRPS-trained emulator bootstrap CIs
    lorenz96/ensemble_results/es_results.npz       — ES-trained emulator bootstrap CIs
    lorenz96/ensemble_results/det_results.npz      — deterministic emulator, matched spread
    lorenz96/ensemble_results/dyn_results.npz      — dynamical, matched spread

Run:
    PYTHONPATH=. python scripts/figure6.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Style — shared with the other main figures ─────────────────────────────────────────────────
FS_TICK  = 8
FS_LABEL = 9
FS_TITLE = 9
FS_LEG   = 7

# Blue = DDPM and green = dynamical across figures 2–6; CRPS/VS are dark/orange
CRPS_COLOR = "#444444"     # near-black (lines elsewhere are #252525); lighter so CIs show
VS_COLOR   = "#DD8452"
DDPM_COLOR = "#2166ac"
EMU_COLOR  = "#9e9e9e"     # CRPS- and ES-trained generators
DYN_COLOR  = "#1b7837"
DET_COLOR  = "#762a83"     # deterministic emulator, as in figure 4
BOX_EDGE   = "#b8bcc8"
ERR_CAP    = 2.5
ERR_LW     = 0.9
ERR_COLOR  = "k"

OUT_DIR = Path("figures/main")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

# All five models share one protocol: the same 2,000 test ICs, 500 members,
# the same 500-member climatology and paired bootstrap (seed 42; see
# lorenz96/ensemble_{ddpm,crps,deterministic,dynamical}.py).
# Generative emulators, then the matched-spread references (deterministic
# emulator and dynamical ensemble, both at σ_IC = 15.8 %):
# (tick label, results npz) in plotting order
_RESULTS = [
    ("DDPM",                          "ddpm_results.npz"),
    ("CRPS-\ntrained",                "crps_results.npz"),
    ("ES-\ntrained",                  "es_results.npz"),
    ("Det.\nemulator",             "det_results.npz"),
    ("Dynamical",                     "dyn_results.npz"),
]
_res = [np.load(f"lorenz96/ensemble_results/{f}") for _, f in _RESULTS]

labels = [lbl for lbl, _ in _RESULTS]
crps  = np.array([float(r["crps_skill"]) for r in _res])
vs    = np.array([float(r["vs_skill"])   for r in _res])
ratio = vs / crps

# CIs: (2, n) arrays where row0=lo_err, row1=hi_err (distances from point)
def _asym(vals, los, his):
    return np.array([vals - los, his - vals])

def _bounds(key):
    return (np.array([float(r[f"{key}_lo"]) for r in _res]),
            np.array([float(r[f"{key}_hi"]) for r in _res]))

crps_err  = _asym(crps,  *_bounds("crps"))
vs_err    = _asym(vs,    *_bounds("vs"))
ratio_err = _asym(ratio, *_bounds("ratio"))

# ── x positions: gap between method groups ────────────────────────────────────
# Groups: [DDPM  CRPS  ES] (generative emulators)  [Det.  Dynamical] (matched-spread references, shaded)
x = np.array([0.0, 0.9, 1.8, 3.05, 3.95])
w   = 0.22   # individual bar width
off = 0.13   # offset from group centre (gap = 2*(off - w/2) = 0.02 between bars)
rw  = 0.32   # bar width for ratio panel

XLIM = (-0.45, x[-1] + 0.6)
METHOD_BG = "#f0f2f6"   # light shading behind each method group

def _shade_groups(ax):
    """Shade the matched-spread reference group."""
    ax.axvspan(x[3] - 0.42, x[4] + 0.42, color=METHOD_BG, zorder=0, linewidth=0)

def _reference_label(ax):
    """Label the shaded column, just inside the top of the axes."""
    ax.text((x[3] + x[4]) / 2, 0.985, "matched-spread\nreferences",
            transform=ax.get_xaxis_transform(), linespacing=1.0,
            ha="center", va="top", fontsize=FS_TICK, fontstyle="italic",
            color="0.4", zorder=5)

# ── Panel (a): CRPS and VS skill ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
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
ax.set_ylabel("Skill Score", fontsize=FS_LABEL)
ax.set_title("(a) Marginal and joint skill", fontsize=FS_TITLE, fontweight="bold")
ax.set_ylim(0, (crps + crps_err[1]).max() * 1.14)
_reference_label(ax)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_xlim(*XLIM)
ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
ax.legend(fontsize=FS_LEG, framealpha=0.95, handlelength=1.4,
          handletextpad=0.4, borderpad=0.5, loc="upper left")

# ── Panel (b): VS / CRPS ratio ────────────────────────────────────────────────
ax = axes[1]

YLIM_B = (0.0, 1.22)
_shade_groups(ax)
bars_ratio = ax.bar(x, ratio, width=rw, color=[DDPM_COLOR, EMU_COLOR, EMU_COLOR, DET_COLOR, DYN_COLOR],
                    edgecolor="k", linewidth=0.5, zorder=3)

ax.errorbar(x, ratio, yerr=ratio_err, fmt="none", color=ERR_COLOR,
            capsize=ERR_CAP, linewidth=ERR_LW, zorder=4)

# Value labels above each bar's upper CI bound
ratio_hi = ratio + ratio_err[1]
for xi, v, hi in zip(x, ratio, ratio_hi):
    ax.text(xi, hi + 0.018, f"{v:.2f}", ha="center", va="bottom",
            fontsize=FS_TICK, fontweight="bold", color="k")

ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", zorder=2)
# Annotate the reference line directly — no legend needed
ax.text(XLIM[0] + 0.06, 1.0 + 0.012, "VS / CRPS = 1",
        ha="left", va="bottom", fontsize=FS_TICK, color="k")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FS_TICK)
ax.set_ylabel("VS / CRPS skill ratio", fontsize=FS_LABEL)
ax.set_title("(b) Joint relative to marginal", fontsize=FS_TITLE, fontweight="bold")
_reference_label(ax)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM_B)
ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)

fig.tight_layout(w_pad=1.2)

out_pdf = OUT_DIR / "figure6.pdf"
out_png = OUT_DIR / "figure6.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")

# ── Console summary ───────────────────────────────────────────────────────────
print()
print(f"{'Method':<24}  {'CRPS':>8}  {'VS':>8}  {'VS/CRPS':>8}")
print("-" * 54)
for m, c, v, r in zip(labels, crps, vs, ratio):
    print(f"{m.replace(chr(10),' '):<28}  {c:>+8.4f}  {v:>+8.4f}  {r:>8.3f}")
