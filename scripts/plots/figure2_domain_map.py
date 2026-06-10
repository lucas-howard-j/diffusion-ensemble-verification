"""
Figure 2: (a) US East Coast tide gauge network with pairwise SSH correlation overlay.
          (b) Per-pair VS failure vs observed Pearson r, points coloured by inter-station
              distance (km).

VS failure = 1 − VS_DDPM / VS_clim (ratio-of-means over inits × leads, test set).
Requires: results/vs_per_pair_test.npz  (run scripts/vs_per_pair.py --cached first)

Run:
    PYTHONPATH=. python scripts/plots/figure2_domain_map.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

from src.config import TIDE_GAUGES, TRAIN_YEARS, SSH_ANOM_WEEKLY
from src.model.dataset import _interp_to_gauges

OUT_DIR  = Path("figures/main")
VS_CACHE = Path("results/vs_per_pair_test.npz")
OUT_DIR.mkdir(exist_ok=True, parents=True)

LATS  = [g["lat"] for g in TIDE_GAUGES]
LONS  = [g["lon"] for g in TIDE_GAUGES]
SHORT = [g["name"].split("_")[0] for g in TIDE_GAUGES]

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def compute_training_correlations() -> np.ndarray:
    """Pearson r matrix (n_gauges × n_gauges) from GLORYS training-period SSH."""
    print("Loading SSH anomaly and interpolating to gauges...")
    da = xr.open_dataarray(str(SSH_ANOM_WEEKLY))
    time = pd.to_datetime(da.time.values)
    train_mask = time.year.isin(list(TRAIN_YEARS))
    gauge_ssh = _interp_to_gauges(da)
    da.close()
    return np.corrcoef(gauge_ssh[train_mask].T)   # (n_gauges, n_gauges)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    corr = compute_training_correlations()
    n    = len(TIDE_GAUGES)

    # ── Load per-pair VS data ─────────────────────────────────────────────────
    vd          = np.load(VS_CACHE, allow_pickle=True)
    vs_pairs    = vd["vs_pairs"]      # (n_inits, n_leads, n_pairs, 4)
    pair_labels = vd["pair_labels"]   # (n_pairs,)

    # VS failure = 1 − mean(DDPM) / mean(clim), ratio-of-means over inits×leads
    vs_ddpm = vs_pairs[:, :, :, 0]   # (n_inits, n_leads, n_pairs)
    vs_clim = vs_pairs[:, :, :, 2]
    vs_fail = 1.0 - vs_ddpm.mean(axis=(0, 1)) / vs_clim.mean(axis=(0, 1))  # (n_pairs,)

    # Per-pair observed r and distance
    pair_r    = np.zeros(len(pair_labels))
    pair_dist = np.zeros(len(pair_labels))
    for k, label in enumerate(pair_labels):
        a_name, b_name = label.split("–")
        i = SHORT.index(a_name)
        j = SHORT.index(b_name)
        pair_r[k]    = corr[i, j]
        pair_dist[k] = haversine_km(LATS[i], LONS[i], LATS[j], LONS[j])

    # ── Figure layout ─────────────────────────────────────────────────────────
    FS_TICK  = 6
    FS_LABEL = 7
    FS_TITLE = 7

    proj = ccrs.PlateCarree()
    fig  = plt.figure(figsize=(7.0, 5.2))
    gs   = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0],
                             wspace=0.08)

    ax_map = fig.add_subplot(gs[0], projection=proj)

    # Broken y-axis: top panel (main data) + bottom panel (outlier)
    gs_r   = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1], height_ratios=[5, 1], hspace=0.06,
    )
    ax_top = fig.add_subplot(gs_r[0])
    ax_bot = fig.add_subplot(gs_r[1], sharex=ax_top)

    # ── Panel (a): map ────────────────────────────────────────────────────────
    ax_map.set_extent([-82, -64, 22, 48], crs=proj)
    ax_map.add_feature(cfeature.LAND,      facecolor="0.88", zorder=1)
    ax_map.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax_map.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":", zorder=2)
    ax_map.add_feature(cfeature.STATES,    linewidth=0.3, zorder=2)

    cmap_net = plt.cm.YlOrRd
    norm_net = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            ax_map.plot(
                [LONS[i], LONS[j]], [LATS[i], LATS[j]],
                color=cmap_net(norm_net(r)),
                linewidth=0.3 + 0.7 * max(r, 0),
                alpha=0.25 + 0.75 * max(r, 0),
                transform=proj, zorder=3, solid_capstyle="round",
            )

    sm_net = cm.ScalarMappable(cmap=cmap_net, norm=norm_net)
    sm_net.set_array([])
    cbar = fig.colorbar(sm_net, ax=ax_map, orientation="horizontal",
                        pad=0.03, fraction=0.04, shrink=0.75, aspect=22)
    cbar.set_label("Pearson $r$  (GLORYS 1993–2018)", fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    for lat, lon, short in zip(LATS, LONS, SHORT):
        ax_map.plot(lon, lat, "o", color="k", ms=5, mec="k", mew=0.8,
                    transform=proj, zorder=5)
        dx, dy = OFFSETS.get(short, (1.2, 0.3))
        ax_map.annotate(
            LABELS.get(short, short),
            xy=(lon, lat), xytext=(lon + dx, lat + dy),
            fontsize=6, transform=proj, zorder=6,
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5),
            va="center",
        )

    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color="0.7",
                          linestyle="--", xlocs=range(-80, -60, 5),
                          ylocs=range(25, 50, 5), x_inline=False, y_inline=False)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": FS_TICK - 1}
    gl.ylabel_style = {"size": FS_TICK - 1}

    ax_map.set_title("(a)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)

    # ── Panel (b): per-pair VS failure vs r, broken y-axis ───────────────────
    cmap_dist = plt.cm.viridis
    dist_max  = pair_dist.max()
    norm_dist = mcolors.Normalize(vmin=0, vmax=dist_max)

    sckw = dict(c=pair_dist, cmap=cmap_dist, norm=norm_dist,
                s=22, zorder=3, linewidths=0.3, edgecolors="0.3")

    # Find outlier bounds
    atl_idx  = list(pair_labels).index("Atlantic–Battery")
    atl_val  = float(vs_fail[atl_idx])
    bot_pad  = 0.04
    bot_ylim = (atl_val - bot_pad, atl_val + bot_pad)

    ax_top.scatter(pair_r, vs_fail, **sckw)
    sc = ax_bot.scatter(pair_r, vs_fail, **sckw)

    ax_top.set_ylim(-0.30, 0.06)
    ax_bot.set_ylim(*bot_ylim)

    ax_top.axhline(0, color="k", lw=0.8, ls="--")
    ax_top.grid(True, alpha=0.2, lw=0.5)
    ax_bot.grid(True, alpha=0.2, lw=0.5)

    # ── Broken-axis spines ────────────────────────────────────────────────────
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bot.tick_params(labelsize=FS_TICK)
    ax_top.tick_params(axis="y", labelsize=FS_TICK)

    # Diagonal break marks on left spine only (right side has colorbar).
    # d_y must scale inversely with panel height to keep the same physical slope.
    # height_ratios = [5, 1], so bot panel is 5× shorter → d_y_bot = 5 × d_y_top.
    d_x     = 0.03
    d_y_top = 0.04
    d_y_bot = d_y_top * 5   # compensate for height ratio
    kw  = dict(color="k", clip_on=False, lw=0.8, zorder=5)
    ax_top.plot((-d_x, +d_x), (-d_y_top, +d_y_top),
                transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d_x, +d_x), (1 - d_y_bot, 1 + d_y_bot),
                transform=ax_bot.transAxes, **kw)

    # Labels
    ax_bot.set_xlabel("Observed Pearson $r$", fontsize=FS_LABEL)
    ax_top.set_ylabel(r"VS failure  ($1 - \mathrm{VS_{DDPM}}/\mathrm{VS_{clim}}$)",
                      fontsize=FS_LABEL)
    ax_top.set_title("(b)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)

    cbar2 = fig.colorbar(sc, ax=[ax_top, ax_bot], orientation="vertical",
                         pad=0.03, fraction=0.05, aspect=28)
    cbar2.set_label("Distance (km)", fontsize=FS_TICK)
    cbar2.ax.tick_params(labelsize=FS_TICK - 1)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUT_DIR / "figure2.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure2.pdf", bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'figure2.png'} and figure2.pdf")


if __name__ == "__main__":
    main()
