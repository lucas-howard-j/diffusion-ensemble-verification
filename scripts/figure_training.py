"""
Supplementary figure: SSH DDPM training curves and architecture comparison.

Panel (a): Train / val loss vs epoch for the final model (train_gauge_init2.log).
Panel (b): Best validation loss across architecture / hyperparameter variants.

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
FINAL_LOG = LOG_DIR / "train_gauge_init2.log"

# Architecture / config variants to compare in panel (b).
# Each entry: (log file, display label, include_in_plot)
VARIANTS = [
    ("train_gauge_init2.log",  "Final model\n(dropout=0.10, linear β)"),
    ("train_dropout05.log",    "Dropout=0.05"),
    ("train_dropout15.log",    "Dropout=0.15"),
    ("train_betasilu.log",     "β-SiLU activation"),
    ("train_nao_lags.log",     "NAO lag predictors"),
    ("train_ssh_lag0.log",     "No SSH lag"),
    ("train_large.log",        "Larger network\n(lr=1×10⁻⁴)"),
    ("train_cosine.log",       "Cosine β schedule"),
]

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

    # Collect variant results for panel (b)
    var_labels, var_vals = [], []
    for fname, label in VARIANTS:
        p = LOG_DIR / fname
        if not p.exists():
            continue
        d = parse_log(p)
        var_labels.append(label)
        var_vals.append(d["best_val"])

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
    ax_a.text(ep[-1] * 0.97, best_val * 1.015,
              f"best = {best_val:.4f}",
              fontsize=FS_TICK - 0.5, color=VAL_COLOR, alpha=0.85,
              ha="right", va="bottom")

    ax_a.set_xlabel("Epoch", fontsize=FS_LABEL)
    ax_a.set_ylabel("Loss", fontsize=FS_LABEL)
    ax_a.set_title("(a)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
    ax_a.tick_params(axis="both", labelsize=FS_TICK)
    ax_a.legend(fontsize=FS_LEG, frameon=False)
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)

    # ── Panel (b): variant comparison ─────────────────────────────────────────
    y_pos  = np.arange(len(var_labels))
    colors = [VAL_COLOR if i == 0 else "#888888" for i in range(len(var_labels))]

    ax_b.barh(y_pos, var_vals, height=0.55, color=colors, alpha=0.80)
    for i, v in enumerate(var_vals):
        ax_b.text(v + 0.001, y_pos[i], f"{v:.4f}",
                  fontsize=FS_TICK - 0.5, va="center", ha="left")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(var_labels, fontsize=FS_TICK)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Best validation loss", fontsize=FS_LABEL)
    ax_b.set_title("(b)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
    ax_b.tick_params(axis="both", labelsize=FS_TICK)
    ax_b.set_xlim(left=0.13)
    ax_b.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # vertical line at final model's best val
    ax_b.axvline(var_vals[0], color=VAL_COLOR, linewidth=0.7,
                 linestyle="--", alpha=0.65)

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
