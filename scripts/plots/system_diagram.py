"""System diagram: data I/O, preprocessing, model architecture, and evaluation."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

# ── Colour palette ────────────────────────────────────────────────────────────
C_DATA   = "#4C72B0"   # blue  — raw data
C_PROC   = "#55A868"   # green — preprocessing
C_MODEL  = "#C44E52"   # red   — model
C_EVAL   = "#8172B2"   # purple — evaluation
C_ARROW  = "#444444"
ALPHA    = 0.18

def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, alpha=ALPHA,
        edgecolor=color, linewidth=1.5,
    )
    ax.add_patch(rect)
    cy = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=color)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                ha="center", va="center", fontsize=7.5,
                color="0.3", style="italic")

def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.4, mutation_scale=12))

def hline(ax, x0, x1, y, color=C_ARROW):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=12))

# ── Column x positions ────────────────────────────────────────────────────────
X_RAW   = 0.2
X_PROC  = 4.0
X_EOF   = 7.6
X_MODEL = 11.2
X_EVAL  = 15.0

BW = 3.2   # box width
BH = 1.2   # box height

# ── Section headers ───────────────────────────────────────────────────────────
for x, lbl, col in [
    (X_RAW,   "RAW DATA",      C_DATA),
    (X_PROC,  "PREPROCESSING", C_PROC),
    (X_EOF,   "FEATURES",      C_PROC),
    (X_MODEL, "MODEL",         C_MODEL),
    (X_EVAL,  "EVALUATION",    C_EVAL),
]:
    ax.text(x + BW / 2, 9.6, lbl, ha="center", va="center",
            fontsize=9, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=col,
                      alpha=0.12, edgecolor=col, linewidth=1))

# ── Raw data boxes ────────────────────────────────────────────────────────────
box(ax, X_RAW, 7.9, BW, BH, C_DATA, "GLORYS 1/12°",
    "SSH, SST  |  1993–2025\n5–50°N, 98–50°W")
box(ax, X_RAW, 5.9, BW, BH, C_DATA, "ERA5 0.25°",
    "u10/v10 wind  |  1993–2025\nMSLP (20–80°N, 90°W–40°E)")
box(ax, X_RAW, 3.9, BW, BH, C_DATA, "Niño3.4 GLORYS",
    "SST  |  1993–2025\n5°S–5°N, 170–120°W")
box(ax, X_RAW, 1.9, BW, BH, C_DATA, "NOAA Tide Gauges",
    "SSH  |  8 East Coast stations\n(validation only)")

# ── Preprocessing boxes ───────────────────────────────────────────────────────
box(ax, X_PROC, 7.4, BW, BH, C_PROC, "Anomaly + Detrend",
    "Seasonal clim (train)\nDaily → weekly  |  linear trend")
box(ax, X_PROC, 5.4, BW, BH, C_PROC, "Anomaly + Detrend",
    "Seasonal clim (train)\nDaily → weekly  |  linear trend")
box(ax, X_PROC, 3.4, BW, BH, C_PROC, "Niño3.4 Index",
    "Area-mean SST anomaly\nWeekly mean")
box(ax, X_PROC, 1.4, BW, BH, C_PROC, "Gauge Preprocessing",
    "Hourly → weekly\nSeasonal removal")

# ── EOF / index boxes ─────────────────────────────────────────────────────────
box(ax, X_EOF, 7.4, BW, BH, C_PROC, "SSH / SST EOFs",
    "100 modes each\n~78% variance explained")
box(ax, X_EOF, 5.4, BW, BH, C_PROC, "Wind EOFs",
    "Joint u10/v10  |  20 modes\nRandomized SVD")
box(ax, X_EOF, 3.9, BW, BH * 0.9, C_PROC, "NAO Index",
    "EOF PC1 of MSLP\nTruncated SVD (k=1)")
box(ax, X_EOF, 2.6, BW, BH * 0.9, C_PROC, "ENSO Index",
    "Niño3.4  |  lags 0,4,8,13 wk")

# ── Model box ─────────────────────────────────────────────────────────────────
box(ax, X_MODEL, 4.8, BW, 3.6, C_MODEL, "DDPM  ·  MLP Denoiser",
    "")
# Internal text
model_text = (
    "Target:   SSH PCs (15 modes)\n"
    "Condition:  SST PCs (lags 0,1,2)\n"
    "             SSH PCs (lags 1,2,4)\n"
    "             Wind PCs  |  NAO  |  ENSO\n"
    "             Lead time (2,4,8,12 wk)\n\n"
    "T = 1000 steps  |  MLP denoiser\n"
    "EMA  |  train 1993–2020  |  val 2021–2023"
)
ax.text(X_MODEL + BW / 2, 6.4, model_text,
        ha="center", va="center", fontsize=7.5,
        color="0.2", family="monospace",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

# ── Evaluation box ────────────────────────────────────────────────────────────
box(ax, X_EVAL, 5.8, BW, 2.4, C_EVAL, "Evaluation  (test 2024–2025)",
    "")
eval_text = (
    "SSH reconstruction from PCs\n\n"
    "Metrics:\n"
    "  • Spatial RMSE / Anomaly Corr\n"
    "  • Tide gauge correlation\n"
    "  • Spread–skill (ensemble)\n"
    "  • vs. persistence & climatology"
)
ax.text(X_EVAL + BW / 2, 6.8, eval_text,
        ha="center", va="center", fontsize=7.5,
        color="0.2",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

# ── Arrows: raw → preprocessing ───────────────────────────────────────────────
for ya, yb in [(8.5, 8.0), (6.5, 6.0), (4.5, 3.85), (2.5, 1.9)]:
    hline(ax, X_RAW + BW, X_PROC, (ya + yb) / 2)

# ── Arrows: preprocessing → EOF ───────────────────────────────────────────────
hline(ax, X_PROC + BW, X_EOF, 8.0)   # SSH/SST anom → SSH/SST EOFs
hline(ax, X_PROC + BW, X_EOF, 6.0)   # wind → wind EOFs
hline(ax, X_PROC + BW, X_EOF, 3.85)  # niño → ENSO index
# NAO: ERA5 MSLP preprocessed → NAO EOF
ax.annotate("", xy=(X_EOF, 4.35), xytext=(X_PROC + BW, 5.9),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2,
                            connectionstyle="arc3,rad=0.3", mutation_scale=10))

# ── Arrows: features → model ──────────────────────────────────────────────────
for y in [8.0, 6.0, 4.35, 3.05]:
    ax.annotate("", xy=(X_MODEL, 6.6), xytext=(X_EOF + BW, y),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.0,
                                connectionstyle="arc3,rad=0.15",
                                mutation_scale=10))

# ── Arrow: model → eval ───────────────────────────────────────────────────────
hline(ax, X_MODEL + BW, X_EVAL, 6.6)

# ── Arrow: tide gauges → eval ─────────────────────────────────────────────────
ax.annotate("", xy=(X_EVAL + BW / 2, 5.8), xytext=(X_RAW + BW / 2, 2.5),
            arrowprops=dict(arrowstyle="-|>", color=C_DATA, lw=1.2,
                            connectionstyle="arc3,rad=-0.3",
                            mutation_scale=10, linestyle="dashed"))

# ── Train/val/test legend ─────────────────────────────────────────────────────
for i, (lbl, col) in enumerate([
    ("Train 1993–2020", C_MODEL),
    ("Val  2021–2023",  C_PROC),
    ("Test 2024–2025",  C_EVAL),
]):
    ax.add_patch(mpatches.Rectangle((0.2 + i * 3.2, 0.4), 0.3, 0.3,
                                     facecolor=col, alpha=0.6, edgecolor="none"))
    ax.text(0.6 + i * 3.2, 0.55, lbl, va="center", fontsize=8, color="0.3")

ax.set_title("SSH S2S Prediction System  —  GLORYS EOFs + DDPM",
             fontsize=13, fontweight="bold", pad=6)

plt.tight_layout()
plt.savefig("figures/system_diagram.png", dpi=150, bbox_inches="tight")
print("Saved figures/system_diagram.png")
