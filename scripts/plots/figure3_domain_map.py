"""
Figure 3: (a) Arc diagram — US East Coast tide gauge network, stations ordered N→S,
              curved arcs coloured by pairwise SSH Pearson r (GLORYS training period).
          (b) Per-pair VS skill vs observed Pearson r.

Per-pair VS skill = 1 − VS_DDPM / VS_clim (ratio of means over inits × leads, test set).
Requires: results/vs_per_pair_test.npz  (run python -m scripts.vs_per_pair
          --ckpt checkpoints/ddpm_best_build2025.pt --split test first)

Run:
    PYTHONPATH=. python scripts/plots/figure3_domain_map.py
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

from src.config import TIDE_GAUGES, TRAIN_YEARS, SSH_ANOM_WEEKLY, PROC
from src.eof.compute_eofs import load_eofs
from src.model.dataset import _interp_to_gauges

GAUGE_CACHE = PROC / "gauge_ssh_interp.npy"

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
    """Pearson r matrix (n_gauges × n_gauges) from GLORYS training-period SSH.

    Uses the gauge-interpolated cache (data/processed/gauge_ssh_interp.npy, on the
    EOF time axis) when present; it is identical to interpolating the full field.
    """
    if GAUGE_CACHE.exists():
        gauge_ssh = np.load(str(GAUGE_CACHE))
        time = pd.to_datetime(load_eofs("ssh")["time"])
        if len(time) != len(gauge_ssh):
            raise ValueError("gauge_ssh_interp.npy does not match the EOF time axis")
        train_mask = time.year.isin(list(TRAIN_YEARS))
        return np.corrcoef(gauge_ssh[train_mask].T)
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
    FS_TICK  = 8
    FS_LABEL = 9
    FS_TITLE = 9

    fig  = plt.figure(figsize=(7.0, 5.5))
    gs   = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.85, 1.0],
                             wspace=0.55)

    ax_arc = fig.add_subplot(gs[0])

    ax_vs = fig.add_subplot(gs[1])

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

    # Baseline and station dots — hollow circles with number inside
    ax_arc.axhline(0, color="0.7", lw=0.6, zorder=0)
    for ii in range(n):
        ax_arc.plot(xs[ii], 0, "o", color="k", ms=11, zorder=5,
                    markerfacecolor="white", markeredgewidth=0.8)
        ax_arc.text(xs[ii], 0, str(ii + 1), fontsize=6.5,
                    ha="center", va="center", color="k", zorder=6,
                    fontweight="bold")

    # Station labels below baseline, rotated — pushed down to clear the dot
    for ii, orig_i in enumerate(order):
        short = SHORT[orig_i]
        label = LABELS.get(short, short)
        ax_arc.text(xs[ii], -0.30, label, fontsize=7,
                    ha="right", va="top", rotation=40,
                    transform=ax_arc.transData, clip_on=False)

    max_h = (n - 1) * 0.35
    ax_arc.set_xlim(-0.5, n - 0.5)
    ax_arc.set_ylim(-0.85, max_h * 1.02)
    ax_arc.axis("off")
    ax_arc.add_patch(mpatches.FancyBboxPatch(
        (-0.26, -0.02), 1.42, 1.02, boxstyle="square,pad=0",
        transform=ax_arc.transAxes, facecolor="none",
        edgecolor="0.4", linewidth=0.8, zorder=10, clip_on=False,
    ))

    # N→S directionality arrow
    ax_arc.annotate("", xy=(n - 0.3, 0), xytext=(-0.3, 0),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=0.8),
                    annotation_clip=False)
    ax_arc.text(-0.35, 0.08, "N", fontsize=8, ha="center", va="bottom", color="0.4")
    ax_arc.text(n - 0.25, 0.08, "S", fontsize=8, ha="center", va="bottom", color="0.4")

    ax_arc.set_title("(a)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)

    # Colorbar in upper-left whitespace above the arcs
    bbox_arc = ax_arc.get_position()
    cb_h = 0.014
    cb_w = bbox_arc.width * 0.38
    cb_l = bbox_arc.x0 - bbox_arc.width * 0.02
    cb_b = bbox_arc.y0 + bbox_arc.height * 0.88
    cax_cb = fig.add_axes([cb_l, cb_b, cb_w, cb_h])
    sm_net = cm.ScalarMappable(cmap=cmap_net, norm=norm_net)
    sm_net.set_array([])
    cbar = fig.colorbar(sm_net, cax=cax_cb, orientation="horizontal")
    cbar.set_label("Pearson $r$\n(GLORYS 1993–2018)", fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    # Inset map: station locations, upper-right
    iw = bbox_arc.width * 0.58
    ih = bbox_arc.height * 0.64
    il = bbox_arc.x1 - iw + bbox_arc.width * 0.14
    ib = bbox_arc.y1 - ih + bbox_arc.height * 0.10
    ax_inset = fig.add_axes([il, ib, iw, ih], projection=ccrs.PlateCarree())
    ax_inset.set_extent([-90, -62, 22, 48], crs=ccrs.PlateCarree())
    ax_inset.add_feature(cfeature.LAND,      facecolor="0.88", zorder=1)
    ax_inset.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5", zorder=0)
    ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax_inset.add_feature(cfeature.STATES,    linewidth=0.25, zorder=2)
    # Number stations in N→S order (matching arc plot) — hollow dots, number inside
    map_order = list(np.argsort(LATS)[::-1])
    for rank, orig_i in enumerate(map_order):
        lat, lon = LATS[orig_i], LONS[orig_i]
        ax_inset.plot(lon, lat, "o", color="k", ms=8.5,
                      markerfacecolor="white", markeredgewidth=0.6,
                      transform=ccrs.PlateCarree(), zorder=5)
        ax_inset.text(lon, lat, str(rank + 1), fontsize=5.5,
                      ha="center", va="center", color="k", fontweight="bold",
                      transform=ccrs.PlateCarree(), zorder=6)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.5)

    # Align the top of the colorbar legend (label text) with the top of the inset
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    top_label = cbar.ax.xaxis.label.get_window_extent(rend).y1
    top_inset = ax_inset.get_window_extent(rend).y1
    pos = cax_cb.get_position()
    shift = (top_inset - top_label) / fig.bbox.height
    cax_cb.set_position([pos.x0, pos.y0 + shift, pos.width, pos.height])

    # ── Panel (b): per-pair VS skill vs r ─────────────────────────────────────
    ax_vs.scatter(pair_r, vs_fail, color="#2166ac", s=22, zorder=3,
                  linewidths=0.3, edgecolors="0.3")
    lo = min(float(np.min(vs_fail)), 0.0)
    hi = max(float(np.max(vs_fail)), 0.0)
    pad = 0.08 * (hi - lo)
    ax_vs.set_ylim(lo - pad, hi + pad)
    ax_vs.axhline(0, color="k", lw=0.8, ls="--")
    ax_vs.grid(True, alpha=0.2, lw=0.5)
    ax_vs.tick_params(labelsize=FS_TICK)
    ax_vs.set_xlabel("Observed Pearson $r$\n(GLORYS 1993–2018)", fontsize=FS_LABEL)
    ax_vs.set_ylabel("Per-pair VS skill\n$1-\\mathrm{VS}_\\mathrm{DDPM}\\,/\\,\\mathrm{VS}_\\mathrm{clim}$",
                     fontsize=FS_LABEL)
    for k in (int(np.argmax(vs_fail)), int(np.argmin(vs_fail))):
        a_n, b_n = str(pair_labels[k]).split("–")
        na, nb = sorted((order.index(SHORT.index(a_n)) + 1,
                         order.index(SHORT.index(b_n)) + 1))
        ax_vs.annotate(f"{pair_labels[k]} ({na}–{nb})", (pair_r[k], vs_fail[k]),
                       xytext=(-6, 0), textcoords="offset points",
                       ha="right", va="center", fontsize=FS_TICK)
    ax_vs.set_title("(b)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUT_DIR / "figure3.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_DIR / "figure3.pdf", bbox_inches="tight", pad_inches=0.08)
    print(f"Saved {OUT_DIR / 'figure3.png'} and figure3.pdf")


if __name__ == "__main__":
    main()
