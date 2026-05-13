"""
Figure 2: System schematics for both study systems.

Two-row parallel pipeline layout:
  (a) SSH: Inputs → EOFs/Indices → DDPM → Evaluation
  (b) L96: L96 system → DDPM → Ensemble → Scores

Run:
    PYTHONPATH=. python scripts/plots/figure2_schematics.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C_DATA  = "#4C72B0"
C_PROC  = "#2e8b57"
C_MODEL = "#C44E52"
C_EVAL  = "#8172B2"
C_ARROW = "#555555"
ALPHA   = 0.13


def fbox(ax, x, y, w, h, color, title, lines=(), fontsize=8.5):
    """Rounded box with bold title and optional body lines."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, alpha=ALPHA,
        edgecolor=color, linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(rect)
    n = len(lines)
    # Title sits at top of box
    title_y = y + h - 0.28 if n else y + h / 2
    ax.text(x + w / 2, title_y, title,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=color, zorder=3)
    # Body lines evenly spaced below title
    if lines:
        spacing = min((h - 0.55) / max(n, 1), 0.32)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, title_y - 0.38 - i * spacing, line,
                    ha="center", va="top", fontsize=6.8,
                    color="0.3", zorder=3)


def arrow(ax, x0, x1, y, color=C_ARROW):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=11),
                zorder=4)


def l96_ring(ax, cx, cy, R=0.55, N=40):
    """Small L96 ring schematic."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xs = cx + R * np.cos(angles)
    ys = cy + R * np.sin(angles)
    ax.scatter(xs, ys, s=8, color=C_MODEL, zorder=5, linewidths=0.3,
               edgecolors="white")
    # A few coupling arrows near the top node
    for i in [0, 1, 39]:
        for di in [1, -1]:
            j = (i + di) % N
            dx, dy = xs[j] - xs[i], ys[j] - ys[i]
            ax.annotate("",
                xy=(xs[i] + 0.6 * dx, ys[i] + 0.6 * dy),
                xytext=(xs[i] + 0.2 * dx, ys[i] + 0.2 * dy),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa",
                                lw=0.7, mutation_scale=5),
                zorder=4)
    ax.text(cx, cy, f"$N\\!=\\!40$\n$F\\!=\\!8$",
            ha="center", va="center", fontsize=6.5, color="0.4", zorder=6)


def draw_row(ax, label, boxes, y_center, BW=2.8, BH=1.55, gap=0.45):
    """Draw one pipeline row of boxes with arrows."""
    ax.set_xlim(0, len(boxes) * (BW + gap) + 0.3)
    total_w = len(boxes) * BW + (len(boxes) - 1) * gap
    x0 = (ax.get_xlim()[1] - total_w) / 2

    positions = []
    for i, (color, title, lines) in enumerate(boxes):
        x = x0 + i * (BW + gap)
        by = y_center - BH / 2
        fbox(ax, x, by, BW, BH, color, title, lines)
        positions.append((x, by, BW, BH))

    for i in range(len(positions) - 1):
        x_left  = positions[i][0] + BW
        x_right = positions[i + 1][0]
        arrow(ax, x_left, x_right, y_center)

    # Panel label
    ax.text(0.05, y_center + BH / 2 + 0.15, f"({label})",
            fontsize=9, fontweight="bold", color="0.4",
            transform=ax.transData, va="bottom")

    return positions


def main() -> None:
    fig = plt.figure(figsize=(13, 5.2))
    gs  = GridSpec(2, 1, figure=fig, hspace=0.55)

    # ── Panel (a): SSH ────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.axis("off")

    ssh_boxes = [
        (C_DATA,  "Inputs",
         ("GLORYS SSH, SST  (1993–2024)",
          "ERA5 u10, v10, MSLP",
          "GLORYS Niño3.4 SST")),
        (C_PROC,  "EOFs  &  Indices",
         ("SSH / SST EOFs  (15 modes, lags 0–4 wk)",
          "Wind EOFs  (10 modes, lag 0)",
          "NAO  (lags 0–2 wk)   ENSO  (lags 0–13 wk)")),
        (C_MODEL, "DDPM  ·  MLP",
         ("Target: SSH anomaly at 8 gauges",
          "1000 diffusion steps",
          "Train 1993–2018  ·  Val 2019–2021")),
        (C_EVAL,  "Evaluation",
         ("Test: 2022–2024  (132 inits)",
          "CRPS skill  (per station)",
          "Energy score  ·  Variogram score")),
    ]
    draw_row(ax_a, "a", ssh_boxes, y_center=1.0)

    # ── Panel (b): L96 ────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.axis("off")

    l96_boxes = [
        (C_MODEL, "Lorenz 96",
         ("$N=40$,  $F=8$",
          "RK4,  $\\Delta t = 0.05$ TU",
          "Lyapunov time $\\approx$ 0.5 TU")),
        (C_PROC,  "Training data",
         ("$(\\mathbf{x}_t,\\, \\mathbf{x}_{t+T})$ pairs",
          "Leads $T \\in \\{20, 40, 60\\}$ steps",
          "$n_{\\mathrm{train}} \\in \\{1\\mathrm{k},\\ldots,249\\mathrm{k}\\}$")),
        (C_MODEL, "DDPM  ·  ConvNet",
         ("Circular Conv1d, 4 ResBlocks",
          "1000 diff. steps, DDIM sampling",
          "Condition: $\\mathbf{x}_0$  +  lead $T$")),
        (C_EVAL,  "Scores",
         ("$n_{\\mathrm{eval}} = 500$,  $M = 20$ members",
          "CRPS skill  vs. climatological draw",
          "Variogram score skill  ($p = 0.5$)")),
    ]
    pos = draw_row(ax_b, "b", l96_boxes, y_center=1.0)

    # Small L96 ring inset inside first box of row (b)
    bx, by, bw, bh = pos[0]
    l96_ring(ax_b, cx=bx + bw / 2, cy=by + 0.38, R=0.30)

    fig.savefig(OUT_DIR / "figure2.png", dpi=150, bbox_inches="tight")
    print("Saved figures/figure2.png")


if __name__ == "__main__":
    main()
