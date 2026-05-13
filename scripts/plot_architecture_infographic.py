#!/usr/bin/env python
"""
Infographic-style architecture figure for gauge-space DDPM.
Output: figures/architecture_infographic.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
C_BLUE_DARK  = "#2563A8"
C_BLUE_MED   = "#4A90C4"
C_BLUE_LIGHT = "#9DC3E6"
C_TEAL       = "#1F7A8C"
C_TEAL_LIGHT = "#5BB8C8"
C_SLATE      = "#5B6FAF"
C_STEEL      = "#3A6DB5"
C_SKY        = "#5BAAD0"
C_GRAY       = "#8C8C8C"
C_GRAY_LIGHT = "#EFEFEF"
C_GREEN      = "#4A9B52"
C_ORANGE     = "#D07030"
C_RED        = "#B83232"
C_TEXT       = "#1A1A1A"
C_TEXT_LIGHT = "#4A4A4A"

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, text, fc, fontsize=8, tc="white",
         bold=False, radius=0.015, lc="white", lw=0.7, linespacing=1.35):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=lc, facecolor=fc,
        transform=ax.transAxes, clip_on=True, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=tc,
            fontweight="bold" if bold else "normal",
            transform=ax.transAxes, zorder=4,
            linespacing=linespacing, clip_on=False)

def arr(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.1, ms=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, mutation_scale=ms), zorder=5)

def panel_label(ax, letter):
    ax.text(-0.01, 1.03, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=C_TEXT, va="bottom")


# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("white")
fig.text(0.5, 0.975,
         "MLPDenoiser — Gauge-Space DDPM Architecture",
         ha="center", va="top", fontsize=13.5, fontweight="bold", color=C_TEXT)

gs = GridSpec(2, 2, figure=fig,
              left=0.04, right=0.97, top=0.94, bottom=0.04,
              hspace=0.14, wspace=0.10)
ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

for ax in [ax_A, ax_B, ax_C, ax_D]:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # light border
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)
    rect = FancyBboxPatch((0, 0), 1, 1,
                          boxstyle="round,pad=0,rounding_size=0.01",
                          linewidth=0.8, edgecolor="#CCCCCC", facecolor="none",
                          transform=ax.transAxes, clip_on=False, zorder=0)
    ax.add_patch(rect)

for ax, lbl in zip([ax_A, ax_B, ax_C, ax_D], "ABCD"):
    panel_label(ax, lbl)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Architecture
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_A
ax.set_title("Model Architecture  (MLPDenoiser)", fontsize=9,
             fontweight="bold", color=C_TEXT, pad=5)

# ── input sub-boxes ────────────────────────────────────────────────────────
bw = 0.275; bh = 0.08; gap = 0.012
xs3 = [0.03, 0.03 + bw + gap, 0.03 + 2 * (bw + gap)]
y_in = 0.855

for x0, txt, fc in zip(xs3,
    ["Noisy target  x_t\n[dim = 8]",
     "Time embedding\n[dim = 128]  sinusoidal",
     "Condition vector  c\n[dim = 116]"],
    [C_BLUE_DARK, C_SLATE, C_TEAL]):
    rbox(ax, x0, y_in, bw, bh, txt, fc, fontsize=7.5)

# concat arrow
arr(ax, 0.50, y_in, 0.50, y_in - 0.025)
ax.text(0.52, y_in - 0.013, "concat  →  [252]",
        fontsize=7, color=C_TEXT_LIGHT, transform=ax.transAxes, va="center")

# ── hidden layers ──────────────────────────────────────────────────────────
layer_defs = [
    ("[512]   Linear → BetaSiLU(β = 0.775) → LayerNorm → Dropout(0.1)", C_BLUE_MED, 0.735),
    ("[512]   Linear → BetaSiLU(β = 0.894) → LayerNorm → Dropout(0.1)", C_BLUE_MED, 0.600),
    ("[256]   Linear → BetaSiLU(β = 0.926) → LayerNorm → Dropout(0.1)", C_BLUE_MED, 0.465),
]
for txt, fc, y0 in layer_defs:
    rbox(ax, 0.04, y0, 0.92, 0.08, txt, fc, fontsize=7.5)

# arrows between layers
for y_from, y_to in [(0.735, 0.680), (0.600, 0.545), (0.465, 0.410)]:
    arr(ax, 0.50, y_from, 0.50, y_to)
arr(ax, 0.50, 0.830, 0.50, 0.815)  # input → layer 1

# ── output box ────────────────────────────────────────────────────────────
rbox(ax, 0.04, 0.330, 0.92, 0.08,
     "Linear  →  Predicted noise  ε̂     [dim = 8]",
     C_TEAL, fontsize=8.5, bold=True)

# footer callouts
ax.text(0.50, 0.295, "~528 k parameters   |   EMA decay = 0.9999",
        ha="center", fontsize=7.5, color=C_TEXT_LIGHT, transform=ax.transAxes)
ax.text(0.50, 0.250,
        r"BetaSiLU:  $x \cdot \sigma(\beta x)$,   scalar $\beta$ per layer, learnable",
        ha="center", fontsize=7.5, color=C_TEXT_LIGHT,
        style="italic", transform=ax.transAxes)
ax.text(0.50, 0.210,
        r"$\beta$ values increase layer-by-layer (0.775 → 0.894 → 0.926)",
        ha="center", fontsize=7, color=C_GRAY, transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Diffusion Process
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_B
ax.set_title("Diffusion Process  (DDPM,  T = 1000 steps)", fontsize=9,
             fontweight="bold", color=C_TEXT, pad=5)

# forward arrow (top)
ax.annotate("", xy=(0.91, 0.82), xytext=(0.09, 0.82),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                            lw=1.2, mutation_scale=12))
ax.text(0.50, 0.870,
        "Forward  q(x_t | x_{t−1}) :  adds Gaussian noise",
        ha="center", fontsize=7.5, color=C_TEXT_LIGHT, transform=ax.transAxes)
ax.text(0.50, 0.835,
        "β schedule: linear,  β_start = 10⁻⁴ → β_end = 0.02",
        ha="center", fontsize=7, color=C_GRAY, style="italic",
        transform=ax.transAxes)

# signal snapshots
n_states = 7
xcs = np.linspace(0.09, 0.91, n_states)
y_mid = 0.565

for i, xc in enumerate(xcs):
    frac = i / (n_states - 1)
    amp  = 0.10 * (1 - frac * 0.65)
    alpha = 0.95 - frac * 0.35
    tt = np.linspace(xc - 0.055, xc + 0.055, 120)
    np.random.seed(i * 13 + 7)
    signal = amp * np.sin(2 * np.pi * (tt - xc) / 0.11 * 2)
    noise  = frac * 0.09 * np.random.randn(120)
    ax.plot(tt, y_mid + signal + noise,
            color=C_BLUE_MED, lw=1.3, alpha=alpha,
            transform=ax.transAxes, clip_on=True)
    # divider
    if i < n_states - 1:
        xmid = (xc + xcs[i + 1]) / 2
        ax.plot([xmid, xmid], [0.44, 0.70],
                color="#DDDDDD", lw=0.8, transform=ax.transAxes)

# labels under each snapshot
ax.text(xcs[0], 0.42, "x₀\nclean", ha="center", fontsize=7.5,
        color=C_BLUE_DARK, transform=ax.transAxes)
ax.text(xcs[-1], 0.42, "x_T\nnoise", ha="center", fontsize=7.5,
        color=C_GRAY, transform=ax.transAxes)
for i in range(1, n_states - 1):
    t_label = int(i / (n_states - 1) * 1000)
    ax.text(xcs[i], 0.42, f"t={t_label}", ha="center", fontsize=6.5,
            color=C_GRAY, transform=ax.transAxes)

# reverse arrow (bottom)
ax.annotate("", xy=(0.09, 0.30), xytext=(0.91, 0.30),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C_TEAL,
                            lw=1.4, mutation_scale=12))
ax.text(0.50, 0.265,
        "Reverse  p_θ(x_{t−1} | x_t, c) :  MLP denoiser removes noise",
        ha="center", fontsize=7.5, color=C_TEAL, transform=ax.transAxes)
ax.text(0.50, 0.228,
        "conditioned on climate state  c",
        ha="center", fontsize=7, color=C_TEAL_LIGHT, style="italic",
        transform=ax.transAxes)

# training objective
rbox(ax, 0.14, 0.07, 0.72, 0.075,
     "Training objective :   MSE( ε̂,  ε )",
     C_BLUE_DARK, fontsize=8.5, bold=True)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Conditioning Vector
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_C
ax.set_title("Conditioning Vector  c   [dim = 116]", fontsize=9,
             fontweight="bold", color=C_TEXT, pad=5)

segs = [
    ("SSH PCs\n15 modes × 3 lags\n(t−1, t−2, t−4)\n[45]", 45, C_STEEL),
    ("SST PCs\n15 modes × 3 lags\n(t, t−1, t−2)\n[45]",   45, C_SKY),
    ("Wind PCs\n10 modes\n× 1 lag\n[10]",                  10, C_SLATE),
    ("NAO\nlags\n0,1,2\n[3]",                               3,  C_GRAY),
    ("ENSO\nlags 0,4\n8,13 wk\n[4]",                       4,  "#8AABB8"),
    ("Gauge\nSSH at t\n(8 sta.)\n[8]",                     8,  C_TEAL),
    ("ld\n[1]",                                             1,  C_GRAY_LIGHT),
]
total = sum(s[1] for s in segs)
bar_y = 0.580; bar_h = 0.22
x_cur = 0.03

for label, count, fc in segs:
    w = (count / total) * 0.94
    tc = "white" if fc != C_GRAY_LIGHT else C_TEXT
    rbox(ax, x_cur, bar_y, w - 0.004, bar_h, label, fc,
         fontsize=6.5, tc=tc)
    x_cur += w

ax.text(0.50, bar_y + bar_h + 0.04,
        "←  116 dimensions  →",
        ha="center", fontsize=7.5, color=C_TEXT_LIGHT,
        transform=ax.transAxes)

# target box
rbox(ax, 0.10, 0.370, 0.80, 0.077,
     "Target :  SSH anomaly at 8 NOAA tide gauges  (z-scored)   [dim = 8]",
     C_TEAL, fontsize=8)

# station list
stations = ["Eastport ME", "Boston MA", "Atlantic City NJ", "Battery NY",
            "Sewells Point VA", "Charleston SC", "Mayport FL", "Key West FL"]
ax.text(0.50, 0.330, "  ·  ".join(stations[:4]),
        ha="center", fontsize=7.5, color=C_TEXT_LIGHT, transform=ax.transAxes)
ax.text(0.50, 0.290, "  ·  ".join(stations[4:]),
        ha="center", fontsize=7.5, color=C_TEXT_LIGHT, transform=ax.transAxes)
ax.text(0.50, 0.255, "(N → S,  indices 0–7)",
        ha="center", fontsize=7, color=C_GRAY, style="italic",
        transform=ax.transAxes)

# arrow from conditioning bar to architecture note
ax.annotate("feeds into\ndenoiser", xy=(0.50, 0.82), xytext=(0.50, 0.80),
            xycoords="axes fraction", textcoords="axes fraction",
            ha="center", fontsize=6.5, color=C_GRAY,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_GRAY, lw=0.6))


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — Training Config & Data Split
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_D
ax.set_title("Training Configuration & Data Splits", fontsize=9,
             fontweight="bold", color=C_TEXT, pad=5)

# ── timeline ───────────────────────────────────────────────────────────────
TMIN, TMAX = 1993, 2026
def tx(yr): return 0.04 + (yr - TMIN) / (TMAX - TMIN) * 0.92

bar_y2 = 0.760; bar_h2 = 0.115
splits_tl = [
    (1993, 2019, C_GREEN,  "Train\n1993–2018\n26 yr  ·  ~5350 pairs"),
    (2019, 2024, C_ORANGE, "Validation\n2019–2023\n5 yr  ·  ~245 init times"),
    (2024, 2026, C_RED,    "Test\n2024–2025\nheld out"),
]
for y1, y2, fc, lbl in splits_tl:
    x1 = tx(y1); x2 = tx(y2)
    rbox(ax, x1, bar_y2, x2 - x1 - 0.006, bar_h2, lbl, fc,
         fontsize=6.5, tc="white")

for yr in [1993, 2000, 2010, 2019, 2024, 2026]:
    xp = tx(yr)
    ax.plot([xp, xp], [bar_y2 - 0.025, bar_y2],
            color=C_GRAY, lw=0.8, transform=ax.transAxes)
    ax.text(xp, bar_y2 - 0.04, str(yr),
            ha="center", fontsize=6.5, color=C_TEXT_LIGHT,
            transform=ax.transAxes)

ax.text(0.50, 0.690,
        "* EOFs / indices fitted 1993–2020  (pre-split; fully clean refit is future work)",
        ha="center", fontsize=6.5, color=C_GRAY, style="italic",
        transform=ax.transAxes)

# ── hyperparameter table ───────────────────────────────────────────────────
rows = [
    ("Batch size",        "64"),
    ("Learning rate",     "3 × 10⁻⁵"),
    ("LR schedule",       "ReduceLROnPlateau  (factor 0.5, pat. 20)"),
    ("Min LR",            "1 × 10⁻⁷"),
    ("Weight decay",      "1 × 10⁻³"),
    ("Grad clip",         "1.0"),
    ("Early-stop pat.",   "150 epochs"),
    ("Hidden dims",       "(512, 512, 256)"),
    ("Best val loss",     "0.1588   (epoch 538)"),
]

ROW_H = 0.054
y_hdr = 0.620
rbox(ax, 0.04, y_hdr, 0.42, ROW_H * 0.88, "Hyperparameter",
     C_BLUE_DARK, fontsize=7.5, bold=True)
rbox(ax, 0.48, y_hdr, 0.50, ROW_H * 0.88, "Value",
     C_BLUE_DARK, fontsize=7.5, bold=True)

for i, (k, v) in enumerate(rows):
    yy = y_hdr - (i + 1) * ROW_H
    fc = C_GRAY_LIGHT if i % 2 == 0 else "white"
    rbox(ax, 0.04, yy, 0.42, ROW_H * 0.90, k,
         fc, fontsize=7, tc=C_TEXT, lc="#DDDDDD", lw=0.5)
    rbox(ax, 0.48, yy, 0.50, ROW_H * 0.90, v,
         fc, fontsize=7, tc=C_TEXT, lc="#DDDDDD", lw=0.5)


# ── save ──────────────────────────────────────────────────────────────────────
Path("figures").mkdir(exist_ok=True)
out = "figures/architecture_infographic.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
