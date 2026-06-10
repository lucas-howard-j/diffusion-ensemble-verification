"""
Figure 1: Permutation protocol for separating marginal from joint forecast skill.

The key comparison is the relative drop in scores:
  Δ = Score(original) − Score(shuffled)
applied to both DDPM and climatology draws separately.
ES: Δ_DDPM ≈ Δ_clim  → marginal information dominates ES
VS: Δ_DDPM > Δ_clim  → VS detects DDPM joint structure above climatology

Run from project root:
    PYTHONPATH=. python scripts/plots/figure1_protocol.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

OUT_DIR = Path("figures/main")
OUT_DIR.mkdir(exist_ok=True, parents=True)

CELL5  = ["#2166ac", "#74add1", "#f7f7f7", "#f4a582", "#d6604d"]
C_DDPM = "#1a4d6e"   # dark blue — DDPM track
C_CLIM = "#1a7a1a"   # vivid green — climatology track
C_GRAY = "#888888"

# ── Grid data (schematic) ──────────────────────────────────────────────────────
# DDPM original: columns 0-1 warm, 3-4 cold — structured spatial covariance
DDPM_ORIG = np.array([
    [4, 3, 2, 0, 1],
    [3, 4, 1, 0, 2],
    [4, 3, 2, 1, 0],
])
# Climatology: no column structure (independent marginal draws)
CLIM_ORIG = np.array([
    [2, 4, 1, 3, 0],
    [1, 2, 4, 0, 3],
    [3, 0, 2, 4, 1],
])
rng = np.random.default_rng(42)
DDPM_SHUF = np.array([rng.permutation(row) for row in DDPM_ORIG])
CLIM_SHUF = np.array([rng.permutation(row) for row in CLIM_ORIG])

NR, NC = DDPM_ORIG.shape   # 3, 5
CW, CH = 0.026, 0.10       # cell size in axes-fraction

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.0, 3.8))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def draw_grid(ax, data, left, bottom, outline_color, outline_lw=0.9):
    nr, nc = data.shape
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle(
                (left + j * CW, bottom + (nr - 1 - i) * CH), CW, CH,
                transform=ax.transAxes, clip_on=False,
                facecolor=CELL5[data[i, j]], edgecolor="white", linewidth=0.5,
            ))
    ax.add_patch(Rectangle(
        (left, bottom), nc * CW, nr * CH,
        transform=ax.transAxes, clip_on=False,
        facecolor="none", edgecolor=outline_color, linewidth=outline_lw,
    ))


def farrow(ax, xy, xytext, color, lw=1.0, ms=8):
    ax.annotate("", xy=xy, xytext=xytext,
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=ms))


# ── Layout ─────────────────────────────────────────────────────────────────────
D_CY = 0.735   # DDPM track y-center
C_CY = 0.265   # Clim track y-center
DIV_Y = 0.500  # divider between ES and VS results

ORIG_L = 0.09
D_ORIG_B = D_CY - NR * CH / 2   # 0.585
C_ORIG_B = C_CY - NR * CH / 2   # 0.115
ORIG_R = ORIG_L + NC * CW        # 0.220

SHUF_L = 0.345
D_SHUF_B = D_ORIG_B
C_SHUF_B = C_ORIG_B
SHUF_R = SHUF_L + NC * CW        # 0.475

CMP_L = 0.560   # comparison section left edge
SH_MX = (ORIG_R + SHUF_L) / 2   # midpoint of arrow section

# ── Track labels ──────────────────────────────────────────────────────────────
ax.text(0.01, D_CY, "DDPM",
        ha="left", va="center", fontsize=6.5, color=C_DDPM,
        fontweight="bold", transform=ax.transAxes)
ax.text(0.01, C_CY, "Clim.",
        ha="left", va="center", fontsize=6.5, color=C_CLIM,
        fontweight="bold", transform=ax.transAxes)

# ── Original grids ────────────────────────────────────────────────────────────
draw_grid(ax, DDPM_ORIG, ORIG_L, D_ORIG_B, C_DDPM, outline_lw=1.4)
draw_grid(ax, CLIM_ORIG, ORIG_L, C_ORIG_B, C_CLIM, outline_lw=1.4)
ax.text(ORIG_L + NC * CW / 2, D_ORIG_B + NR * CH + 0.03, "Original",
        ha="center", va="bottom", fontsize=5.5, color="#555555",
        transform=ax.transAxes)

# ── Shuffle arrows — one per track with individual labels ─────────────────────
farrow(ax, (SHUF_L, D_CY), (ORIG_R + 0.005, D_CY), C_DDPM, lw=1.1, ms=8)
farrow(ax, (SHUF_L, C_CY), (ORIG_R + 0.005, C_CY), C_CLIM, lw=1.1, ms=8)

shuf_kw = dict(ha="center", va="center", fontsize=5.5,
               style="italic", transform=ax.transAxes)
ax.text(SH_MX, D_CY + 0.09, "shuffle\n(per location)",
        color=C_DDPM,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor=C_DDPM, linewidth=0.5, alpha=0.92),
        **shuf_kw)
ax.text(SH_MX, C_CY - 0.09, "shuffle\n(per location)",
        color=C_CLIM,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor=C_CLIM, linewidth=0.5, alpha=0.92),
        **shuf_kw)

# ── Shuffled grids ────────────────────────────────────────────────────────────
draw_grid(ax, DDPM_SHUF, SHUF_L, D_SHUF_B, C_DDPM, outline_lw=1.4)
draw_grid(ax, CLIM_SHUF, SHUF_L, C_SHUF_B, C_CLIM, outline_lw=1.4)
ax.text(SHUF_L + NC * CW / 2, D_SHUF_B + NR * CH + 0.03, "Shuffled",
        ha="center", va="bottom", fontsize=5.5, color="#555555",
        transform=ax.transAxes)

# ── Bracket connecting both shuffled grids → comparison section ───────────────
BK_X = SHUF_R + 0.025
# Vertical spine
ax.plot([BK_X, BK_X], [C_CY, D_CY], color=C_GRAY, lw=1.2,
        transform=ax.transAxes, clip_on=False)
# Horizontal ticks at each track
for y in [D_CY, C_CY]:
    ax.plot([BK_X - 0.008, BK_X], [y, y], color=C_GRAY, lw=1.2,
            transform=ax.transAxes, clip_on=False)
# Horizontal line to comparison + arrowhead
ax.plot([BK_X, CMP_L - 0.01], [DIV_Y, DIV_Y], color=C_GRAY, lw=1.2,
        transform=ax.transAxes, clip_on=False)
farrow(ax, (CMP_L - 0.005, DIV_Y), (CMP_L - 0.015, DIV_Y), C_GRAY, lw=1.2, ms=9)

# ── Comparison section ────────────────────────────────────────────────────────
ax.text(CMP_L, 0.50,
        r"$\Delta = \mathrm{Score(original)} - \mathrm{Score(shuffled)}$",
        ha="left", va="center", fontsize=7.0, color="#333333",
        transform=ax.transAxes)


# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_DIR / "figure1.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "figure1.png", dpi=150, bbox_inches="tight")
print("Saved figures/figure1.pdf and figures/figure1.png")


if __name__ == "__main__":
    pass
