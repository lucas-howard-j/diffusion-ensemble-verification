"""
Figure 2: (a) Arc diagram — US East Coast tide gauge network, stations ordered N→S,
              curved arcs coloured by pairwise SSH Pearson r (GLORYS training period).
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
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

    fig  = plt.figure(figsize=(7.0, 5.2))
    gs   = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0],
                             wspace=0.30)

    ax_arc = fig.add_subplot(gs[0])

    # Broken y-axis: top panel (main data) + bottom panel (outlier)
    gs_r   = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1], height_ratios=[5, 1], hspace=0.06,
    )
    ax_top = fig.add_subplot(gs_r[0])
    ax_bot = fig.add_subplot(gs_r[1], sharex=ax_top)

    # ── Panel (a): arc diagram ────────────────────────────────────────────────
    # Stations ordered north → south (left → right along x-axis)
    order = list(np.argsort(LATS)[::-1])   # indices into TIDE_GAUGES, N→S
    xs    = np.arange(n, dtype=float)

    cmap_net = plt.cm.YlOrRd
    norm_net = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Collect all pairs, sort by r ascending so high-r arcs render on top
    arc_pairs = []
    for ii in range(n):
        for jj in range(ii + 1, n):
            r = float(corr[order[ii], order[jj]])
            arc_pairs.append((ii, jj, r))
    arc_pairs.sort(key=lambda t: t[2])

    for ii, jj, r in arc_pairs:
        x1, x2 = xs[ii], xs[jj]
        h = (x2 - x1) * 0.35
        verts = [(x1, 0.0), (x1, h), (x2, h), (x2, 0.0)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
        path  = MPath(verts, codes)
        color = cmap_net(norm_net(r))
        lw    = 0.3 + 1.2 * max(r, 0.0)
        alpha = 0.25 + 0.75 * max(r, 0.0)
        ax_arc.add_patch(mpatches.PathPatch(
            path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha,
        ))

    # Baseline and station dots
    ax_arc.axhline(0, color="0.7", lw=0.6, zorder=0)
    for ii in range(n):
        ax_arc.plot(xs[ii], 0, "o", color="k", ms=4, zorder=5)

    # Station labels below baseline, rotated
    for ii, orig_i in enumerate(order):
        short = SHORT[orig_i]
        label = LABELS.get(short, short)
        ax_arc.text(xs[ii], -0.22, label, fontsize=5.5,
                    ha="right", va="top", rotation=40,
                    transform=ax_arc.transData, clip_on=False)

    max_h = (n - 1) * 0.35
    ax_arc.set_xlim(-0.5, n - 0.5)
    ax_arc.set_ylim(-0.9, max_h + 0.2)
    ax_arc.axis("off")
    ax_arc.add_patch(mpatches.FancyBboxPatch(
        (-0.26, -0.02), 1.26, 1.02, boxstyle="square,pad=0",
        transform=ax_arc.transAxes, facecolor="none",
        edgecolor="0.4", linewidth=0.8, zorder=10, clip_on=False,
    ))

    # N→S directionality arrow
    ax_arc.annotate("", xy=(n - 0.3, 0), xytext=(-0.3, 0),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=0.8),
                    annotation_clip=False)
    ax_arc.text(-0.35, 0.08, "N", fontsize=6.5, ha="center", va="bottom", color="0.4")
    ax_arc.text(n - 0.25, 0.08, "S", fontsize=6.5, ha="center", va="bottom", color="0.4")

    ax_arc.set_title("(a)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)

    # Colorbar in upper-left whitespace above the arcs
    bbox_arc = ax_arc.get_position()
    cb_h = 0.014
    cb_w = bbox_arc.width * 0.42
    cb_l = bbox_arc.x0 + bbox_arc.width * 0.02
    cb_b = bbox_arc.y0 + bbox_arc.height * 0.74
    cax_cb = fig.add_axes([cb_l, cb_b, cb_w, cb_h])
    sm_net = cm.ScalarMappable(cmap=cmap_net, norm=norm_net)
    sm_net.set_array([])
    cbar = fig.colorbar(sm_net, cax=cax_cb, orientation="horizontal")
    cbar.set_label("Pearson $r$  (GLORYS 1993–2018)", fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    # Inset map: station locations, upper-right
    iw = bbox_arc.width * 0.30
    ih = bbox_arc.height * 0.44
    il = bbox_arc.x1 - iw - bbox_arc.width * 0.01
    ib = bbox_arc.y1 - ih + bbox_arc.height * 0.04
    ax_inset = fig.add_axes([il, ib, iw, ih], projection=ccrs.PlateCarree())
    ax_inset.set_extent([-82, -64, 22, 48], crs=ccrs.PlateCarree())
    ax_inset.add_feature(cfeature.LAND,      facecolor="0.88", zorder=1)
    ax_inset.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax_inset.add_feature(cfeature.STATES,    linewidth=0.25, zorder=2)
    for lat, lon in zip(LATS, LONS):
        ax_inset.plot(lon, lat, "o", color="k", ms=3,
                      transform=ccrs.PlateCarree(), zorder=5)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.5)

    # ── Panel (b): per-pair VS failure vs r, broken y-axis ───────────────────
    sckw = dict(color="#2166ac", s=22, zorder=3, linewidths=0.3, edgecolors="0.3")

    # Find outlier bounds
    atl_idx  = list(pair_labels).index("Atlantic–Battery")
    atl_val  = float(vs_fail[atl_idx])
    bot_pad  = 0.04
    bot_ylim = (atl_val - bot_pad, atl_val + bot_pad)

    ax_top.scatter(pair_r, vs_fail, **sckw)
    ax_bot.scatter(pair_r, vs_fail, **sckw)

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
    ax_top.set_ylabel("VS skill deficit", fontsize=FS_LABEL)
    ax_top.set_title("(b)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUT_DIR / "figure2.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure2.pdf", bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'figure2.png'} and figure2.pdf")


if __name__ == "__main__":
    main()
