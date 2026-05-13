"""Loss curve for the locked gauge-init model (train_gauge_init2.log)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG  = ROOT / "logs" / "train_new_split.log"
OUT  = ROOT / "figures" / "training_loss_gauge_init.pdf"

pat = re.compile(
    r"Epoch\s+(\d+)\s+train=([\d.]+)\s+val=([\d.]+)\s+lr=([\de.+-]+)"
)

epochs, train_loss, val_loss, lrs = [], [], [], []
with open(LOG) as f:
    for line in f:
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))
            lrs.append(float(m.group(4)))

epochs     = np.array(epochs)
train_loss = np.array(train_loss)
val_loss   = np.array(val_loss)
lrs        = np.array(lrs)

best_idx  = np.argmin(val_loss)
best_ep   = epochs[best_idx]
best_val  = val_loss[best_idx]

# LR drop epochs (where lr changes from previous logged value)
lr_change_epochs = [epochs[i] for i in range(1, len(lrs)) if lrs[i] != lrs[i-1]]

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.plot(epochs, train_loss, color="#3B82F6", lw=1.5, label="Train")
ax.plot(epochs, val_loss,   color="#EF4444", lw=1.5, label="Val")

# best val marker
ax.axvline(best_ep, color="#EF4444", lw=0.8, ls="--", alpha=0.7)
ax.scatter([best_ep], [best_val], color="#EF4444", zorder=5, s=40)
ax.annotate(
    f"best val {best_val:.4f}\n(epoch {best_ep})",
    xy=(best_ep, best_val),
    xytext=(best_ep + 15, best_val + 0.012),
    fontsize=7,
    color="#EF4444",
    arrowprops=dict(arrowstyle="-", color="#EF4444", lw=0.7),
)

# LR drop markers
for ep in lr_change_epochs:
    ax.axvline(ep, color="gray", lw=0.6, ls=":", alpha=0.6)

ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("DDPM loss", fontsize=9)
ax.set_title("Training loss — gauge-init locked model (VAL 2019–2023)", fontsize=9)
ax.legend(fontsize=8, framealpha=0.7)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.tick_params(labelsize=8)
ax.set_xlim(0, epochs[-1] + 5)

# secondary note: LR drops
if lr_change_epochs:
    note = "LR drops: " + ", ".join(str(e) for e in lr_change_epochs)
    ax.text(0.98, 0.96, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color="gray")

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT)
print(f"Saved → {OUT}")
