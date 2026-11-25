"""
Supplementary figure: SSH DDPM training curves and architecture comparison.

Panel (a): Train / val loss vs epoch for the final model (train_build2025.log, the
           2025 data build; checkpoint ddpm_best_build2025.pt used in Figs 1 and 2).
Panel (b): Best validation loss for one-at-a-time variants of the final
           configuration, 3 training seeds each (scripts/run_ablation.sh; logs in
           logs/ablation/).  Dots are seeds, lines their range; the shaded band is the
           final configuration's own seed range from the same sweep.

Output:
    figures/supplement/figure_training.pdf
    figures/supplement/figure_training.png

Run:
    python scripts/figure_training.py
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_DIR = Path("logs")
OUT_DIR = Path("figures/supplement")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Final model log (the run that produced the checkpoint used in the paper)
FINAL_LOG = LOG_DIR / "train_build2025.log"

# Ablation variants for panel (b): (log-name prefix in logs/ablation/, label).
# Each changes one setting from the final configuration (src/config.py).
ABL_DIR = LOG_DIR / "ablation"
ABL_SEEDS = [0, 1, 2]
VARIANTS = [
    ("final",      "Final config"),
    ("dropout05",  "Dropout 0.05 (vs 0.10)"),
    ("dropout15",  "Dropout 0.15"),
    ("plain_silu", "Plain SiLU (vs β-SiLU)"),
    ("nao_t_only", "NAO at t only (vs t, t−1, t−2)"),
    ("ssh_t_only", "SSH PCs at t only (vs 4 lags)"),
    ("larger_net", "Wider net 1024–1024–512, lr 10⁻⁴"),
    ("cosine",     "Cosine β schedule (vs linear)"),
]
# The noise-prediction loss depends on the β schedule (it averages over diffusion
# steps whose difficulty the schedule sets), so cosine-schedule losses are not on
# the same scale as the rest: they are listed at the panel edge, not plotted.
OFF_SCALE = {"cosine"}

# Style
FS_TICK  = 6
FS_LABEL = 7
FS_TITLE = 7
FS_LEG   = 6.5

TRAIN_COLOR = "#4C72B0"
VAL_COLOR   = "#DD8452"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> dict:
    """
    Parse a training log and return:
      epochs, train_loss, val_loss  — lists of floats (one per logged epoch)
      best_val                      — float
      best_epoch                    — int
      stop_epoch                    — int (epoch at early-stop, or last logged)
    """
    epochs, train_loss, val_loss = [], [], []
    best_val   = float("inf")
    best_epoch = 0
    stop_epoch = 0

    epoch_re = re.compile(
        r"Epoch\s+(\d+)\s+train=([0-9.]+)\s+val=([0-9.]+)"
    )
    best_re  = re.compile(r"Best val loss:\s*([0-9.]+)")
    stop_re  = re.compile(r"Early stopping at epoch\s+(\d+)")

    for line in path.read_text().splitlines():
        m = epoch_re.search(line)
        if m:
            ep, tr, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
            epochs.append(ep)
            train_loss.append(tr)
            val_loss.append(vl)
            stop_epoch = ep
            continue
        m = best_re.search(line)
        if m:
            best_val = float(m.group(1))
            continue
        m = stop_re.search(line)
        if m:
            stop_epoch = int(m.group(1))

    # If best_epoch not explicitly in log, find it from the val curve.
    if epochs:
        best_epoch = epochs[int(np.argmin(val_loss))]

    return dict(
        epochs     = np.array(epochs),
        train_loss = np.array(train_loss),
        val_loss   = np.array(val_loss),
        best_val   = best_val,
        best_epoch = best_epoch,
        stop_epoch = stop_epoch,
    )


# ── Figure ────────────────────────────────────────────────────────────────────

def plot() -> None:
    final = parse_log(FINAL_LOG)

    # Collect variant results for panel (b): best val loss per seed
    var_labels, var_vals, var_off = [], [], []
    for prefix, label in VARIANTS:
        vals = [parse_log(ABL_DIR / f"{prefix}_s{sd}.log")["best_val"]
                for sd in ABL_SEEDS if (ABL_DIR / f"{prefix}_s{sd}.log").exists()]
        if vals:
            var_labels.append(label)
            var_vals.append(np.array(vals))
            var_off.append(prefix in OFF_SCALE)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 3.0),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    # ── Panel (a): training curve ─────────────────────────────────────────────
    ep = final["epochs"]
    ax_a.plot(ep, final["train_loss"], "-", color=TRAIN_COLOR,
              linewidth=1.1, label="Train loss")
    ax_a.plot(ep, final["val_loss"],   "-", color=VAL_COLOR,
              linewidth=1.1, label="Val loss")

    best_val = final["best_val"]
    ax_a.axhline(best_val, color=VAL_COLOR, linewidth=0.6,
                 linestyle="--", alpha=0.7)
    ax_a.text(ep[-1] * 0.97, best_val * 0.97,      # just below the line, clear of the curves
              f"best = {best_val:.4f}",
              fontsize=FS_TICK - 0.5, color=VAL_COLOR, alpha=0.85,
              ha="right", va="top")

    ax_a.set_xlabel("Epoch", fontsize=FS_LABEL)
    ax_a.set_ylabel("Loss", fontsize=FS_LABEL)
    ax_a.set_title("(a)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
    ax_a.tick_params(axis="both", labelsize=FS_TICK)
    ax_a.legend(fontsize=FS_LEG, frameon=False)
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)

    # ── Panel (b): ablation, seed ranges ────────────────────────────────────────
    y_pos = np.arange(len(var_labels))
    ref = var_vals[0]
    ax_b.axvspan(ref.min(), ref.max(), color=VAL_COLOR, alpha=0.15, lw=0)
    on_scale = [v for v, off in zip(var_vals, var_off) if not off]
    lo = min(v.min() for v in on_scale)
    hi = max(v.max() for v in on_scale)
    pad = 0.12 * (hi - lo)
    ax_b.set_xlim(lo - pad, hi + 2.5 * pad)
    for i, (vals, off) in enumerate(zip(var_vals, var_off)):
        if off:
            ax_b.text(hi + 2.4 * pad, i, f"{vals.mean():.3f} →\n(different loss scale)",
                      fontsize=FS_TICK - 1, ha="right", va="center", color="#555555")
            continue
        color = VAL_COLOR if i == 0 else "#555555"
        ax_b.plot([vals.min(), vals.max()], [i, i], "-", color=color, lw=1.2, alpha=0.8)
        ax_b.plot(vals, np.full(len(vals), i), "o", color=color, ms=3)
        ax_b.text(vals.max(), i - 0.28, f"{vals.mean():.3f}", fontsize=FS_TICK - 1,
                  ha="right", va="bottom", color=color)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(var_labels, fontsize=FS_TICK)
    ax_b.set_ylim(len(var_labels) - 0.5, -0.6)
    ax_b.set_xlabel("Best val. loss (3 seeds; label = mean)", fontsize=FS_LABEL)
    ax_b.set_title("(b)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
    ax_b.tick_params(axis="both", labelsize=FS_TICK)
    ax_b.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_b.grid(True, axis="x", alpha=0.25, lw=0.5)

    fig.tight_layout(w_pad=2.0)

    out_pdf = OUT_DIR / "figure_training.pdf"
    out_png = OUT_DIR / "figure_training.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    plot()
