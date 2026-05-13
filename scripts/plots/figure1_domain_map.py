"""
Figure 1: US East Coast tide gauge network with pairwise SSH correlation overlay.

Pearson correlations are computed from GLORYS training-period (1993-2018) SSH
anomaly interpolated to gauge locations. Edges between station pairs are colored
and weighted by r.

Run:
    PYTHONPATH=. python scripts/plots/figure1_domain_map.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

from src.config import TIDE_GAUGES, TRAIN_YEARS, SSH_ANOM_WEEKLY
from src.model.dataset import _interp_to_gauges

OUT_DIR = Path("figures/main")
OUT_DIR.mkdir(exist_ok=True, parents=True)

LATS   = [g["lat"] for g in TIDE_GAUGES]
LONS   = [g["lon"] for g in TIDE_GAUGES]
SHORT  = [g["name"].split("_")[0] for g in TIDE_GAUGES]

LABELS = {
    "Eastport":   "Eastport, ME",
    "Boston":     "Boston, MA",
    "Atlantic":   "Atlantic City, NJ",
    "Battery":    "The Battery, NY",
    "Sewells":    "Sewells Point, VA",
    "Charleston": "Charleston, SC",
    "Mayport":    "Mayport, FL",
    "Key":        "Key West, FL",
}

OFFSETS = {
    "Eastport":   (-9.0,  1.5),
    "Boston":     ( 1.5,  1.8),
    "Battery":    ( 1.5,  0.3),
    "Atlantic":   ( 1.5, -1.2),
    "Sewells":    ( 1.5, -1.5),
    "Charleston": ( 1.5,  0.0),
    "Mayport":    ( 1.5, -0.5),
    "Key":        ( 1.5, -0.5),
}


def compute_training_correlations() -> np.ndarray:
    """Pearson r matrix from GLORYS training-period SSH anomaly at gauge locations."""
    print("Loading SSH anomaly and interpolating to gauges...")
    da = xr.open_dataarray(str(SSH_ANOM_WEEKLY))
    time = pd.to_datetime(da.time.values)
    train_mask = time.year.isin(list(TRAIN_YEARS))
    gauge_ssh = _interp_to_gauges(da)   # (n_time, n_gauges)
    da.close()
    return np.corrcoef(gauge_ssh[train_mask].T)   # (n_gauges, n_gauges)


def main() -> None:
    corr = compute_training_correlations()
    n = len(TIDE_GAUGES)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(5, 7), subplot_kw={"projection": proj})
    ax.set_extent([-90, -60, 22, 48], crs=proj)

    ax.add_feature(cfeature.LAND,      facecolor="0.88", zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":", zorder=2)
    ax.add_feature(cfeature.STATES,    linewidth=0.3, zorder=2)

    # ── Correlation network edges ─────────────────────────────────────────────
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            ax.plot(
                [LONS[i], LONS[j]], [LATS[i], LATS[j]],
                color=cmap(norm(r)),
                linewidth=0.3 + 0.7 * max(r, 0),
                alpha=0.25 + 0.75 * max(r, 0),
                transform=proj, zorder=3, solid_capstyle="round",
            )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03,
                        fraction=0.04, shrink=0.65, aspect=25)
    cbar.set_label("Pearson $r$  (GLORYS 1993–2018)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Gauge markers ─────────────────────────────────────────────────────────
    for lat, lon, short in zip(LATS, LONS, SHORT):
        ax.plot(lon, lat, "o", color="k", ms=6, mec="k", mew=0.8,
                transform=proj, zorder=5)
        dx, dy = OFFSETS.get(short, (1.2, 0.3))
        ax.annotate(
            LABELS.get(short, short),
            xy=(lon, lat), xytext=(lon + dx, lat + dy),
            fontsize=7, transform=proj, zorder=6,
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6),
            va="center",
        )

    # ── Gridlines ─────────────────────────────────────────────────────────────
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7",
                      linestyle="--", xlocs=range(-90, -55, 10),
                      ylocs=range(25, 50, 5), x_inline=False, y_inline=False)
    gl.top_labels   = False
    gl.right_labels = False

    fig.tight_layout()
    out = OUT_DIR / "figure2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
