"""Quick sanity-check plot of the weekly ENSO (Niño 3.4) index."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.indices.enso import load_enso

enso = load_enso()

fig, ax = plt.subplots(figsize=(14, 4))

# Fill positive (El Niño) red, negative (La Niña) blue
ax.fill_between(enso.index, enso, where=enso >= 0, color="tomato",  alpha=0.7, label="El Niño (+)")
ax.fill_between(enso.index, enso, where=enso <  0, color="steelblue", alpha=0.7, label="La Niña (−)")
ax.plot(enso.index, enso, color="k", lw=0.4)

# Reference lines
ax.axhline(0,    color="k",    lw=0.8)
ax.axhline( 1.5, color="tomato",   lw=0.8, ls="--", alpha=0.6)
ax.axhline(-1.5, color="steelblue", lw=0.8, ls="--", alpha=0.6)

# Train/val/test shading
ymin, ymax = ax.get_ylim()
ax.axvspan("1993", "2021", alpha=0.04, color="gray", label="Train (1993–2020)")
ax.axvspan("2021", "2024", alpha=0.08, color="green", label="Val (2021–2023)")
ax.axvspan("2024", "2025", alpha=0.08, color="purple", label="Test (2024+)")

ax.set_ylabel("Niño 3.4 anomaly (standardized)")
ax.set_title("Weekly ENSO index (Niño 3.4 SST anomaly, GLORYS12)")
ax.legend(loc="upper left", fontsize=8, ncol=5)
ax.set_xlim(enso.index[0], enso.index[-1])

plt.tight_layout()
out = Path("figures/enso_timeseries.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved → {out}")

# Quick stats
print(f"\nRecord: {enso.index[0].date()} → {enso.index[-1].date()}  ({len(enso)} weeks)")
print(f"Mean={enso.mean():.3f}  Std={enso.std():.3f}  Min={enso.min():.2f}  Max={enso.max():.2f}")
strong = enso[enso.abs() > 1.5]
print(f"Weeks with |index| > 1.5: {len(strong)}  ({100*len(strong)/len(enso):.1f}%)")
