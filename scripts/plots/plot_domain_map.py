"""
Fig 1a: Map of 8 NOAA tide gauges on US East Coast with input domains.
Run:
    PYTHONPATH=. python scripts/plots/plot_domain_map.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from src.config import TIDE_GAUGES, DOMAIN, ERA5_DOMAIN

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

GAUGES = TIDE_GAUGES
SHORT = [g["name"].split("_")[0] for g in GAUGES]
LATS  = [g["lat"] for g in GAUGES]
LONS  = [g["lon"] for g in GAUGES]

# Label offsets (dx, dy) in degrees — staggered to avoid NE cluster overlap
OFFSETS = {
    "Eastport":    (-9.0,  1.5),   # left of marker to stay in map
    "Boston":      ( 1.5,  1.8),   # pushed well above Battery
    "Battery":     ( 1.5,  0.3),
    "Atlantic":    ( 1.5, -1.2),   # pushed below Battery
    "Sewells":     ( 1.5, -1.5),   # pushed below Atlantic City label
    "Charleston":  ( 1.5,  0.0),
    "Mayport":     ( 1.5, -0.5),
    "Key":         ( 1.5, -0.5),
}
LABELS = {
    "Eastport":    "Eastport, ME",
    "Boston":      "Boston, MA",
    "Atlantic":    "Atlantic City, NJ",
    "Battery":     "The Battery, NY",
    "Sewells":     "Sewells Point, VA",
    "Charleston":  "Charleston, SC",
    "Mayport":     "Mayport, FL",
    "Key":         "Key West, FL",
}

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(5, 7), subplot_kw={"projection": proj})
ax.set_extent([-90, -60, 22, 48], crs=proj)

ax.add_feature(cfeature.LAND,      facecolor="0.88", zorder=1)
ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":", zorder=2)
ax.add_feature(cfeature.STATES,    linewidth=0.3, zorder=2)

# ── Input domain rectangles ───────────────────────────────────────────────────
def _domain_rect(ax, d, color, lw, ls, label, zorder=3):
    x0, x1 = d["lon_min"], d["lon_max"]
    y0, y1 = d["lat_min"], d["lat_max"]
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=lw, linestyle=ls, edgecolor=color,
        facecolor="none", transform=proj, zorder=zorder, label=label,
    )
    ax.add_patch(rect)

_domain_rect(ax, DOMAIN,      color="#1a6faf", lw=1.4, ls="-",
             label="SSH / SST domain (GLORYS)")
_domain_rect(ax, ERA5_DOMAIN, color="#c45c00", lw=1.4, ls="--",
             label="Wind domain (ERA5  u, v)")

# ── Tide gauge markers ────────────────────────────────────────────────────────
order  = np.argsort(LATS)
colors = plt.cm.plasma(np.linspace(0.15, 0.75, len(GAUGES)))

for rank, idx in enumerate(order):
    lat, lon = LATS[idx], LONS[idx]
    short    = SHORT[idx]
    dx, dy   = OFFSETS.get(short, (1.2, 0.3))

    ax.plot(lon, lat, "o", color=colors[rank], ms=7, mec="k", mew=0.6,
            transform=proj, zorder=5)
    ax.annotate(
        LABELS.get(short, short),
        xy=(lon, lat), xytext=(lon + dx, lat + dy),
        fontsize=7, transform=proj, zorder=6,
        arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6),
        va="center",
    )

# ── Legend for domain boxes ───────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor="none", edgecolor="#1a6faf", lw=1.4,
                   label="SSH / SST  (GLORYS)"),
    mpatches.Patch(facecolor="none", edgecolor="#c45c00", lw=1.4,
                   linestyle="--", label="Wind  u, v  (ERA5)"),
]
ax.legend(handles=legend_handles, fontsize=7, loc="lower left",
          framealpha=0.85, edgecolor="0.6")

gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", linestyle="--",
                  xlocs=range(-90, -55, 10), ylocs=range(25, 50, 5),
                  x_inline=False, y_inline=False)
gl.top_labels   = False
gl.right_labels = False

ax.set_title("NOAA tide gauge network & input domains", fontsize=10, pad=6)

fig.tight_layout()
out = OUT_DIR / "domain_map.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved {out}")
