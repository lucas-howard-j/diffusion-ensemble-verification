"""
Supplementary figure: per-gauge VS failure scatter.

8 panels (2 rows × 4 cols), one per tide gauge.  Each panel shows the 7 pairs
involving that gauge: x = observed Pearson r, y = VS failure (1 − VS_DDPM/VS_clim,
ratio-of-means over inits × leads, test set), colour = inter-station distance (km).
Shared axes across all panels for direct comparison.

Requires: results/vs_per_pair_test.npz

Run:
    PYTHONPATH=. python scripts/plots/vs_by_gauge.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
from scipy import stats

from src.config import TIDE_GAUGES, TRAIN_YEARS, SSH_ANOM_WEEKLY
from src.model.dataset import _interp_to_gauges

VS_CACHE = Path("results/vs_per_pair_test.npz")
OUT_DIR  = Path("figures/supplement")
OUT_DIR.mkdir(exist_ok=True, parents=True)

LATS  = [g["lat"] for g in TIDE_GAUGES]
LONS  = [g["lon"] for g in TIDE_GAUGES]
SHORT = [g["name"].split("_")[0] for g in TIDE_GAUGES]

PANEL_LABELS = {
    "Eastport":   "Eastport, ME",
    "Boston":     "Boston, MA",
    "Atlantic":   "Atlantic City, NJ",
    "Battery":    "The Battery, NY",
    "Sewells":    "Sewells Point, VA",
    "Charleston": "Charleston, SC",
    "Mayport":    "Mayport, FL",
    "Key":        "Key West, FL",
}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def compute_training_correlations() -> np.ndarray:
    print("Loading SSH anomaly...")
    da = xr.open_dataarray(str(SSH_ANOM_WEEKLY))
    time = pd.to_datetime(da.time.values)
    train_mask = time.year.isin(list(TRAIN_YEARS))
    gauge_ssh = _interp_to_gauges(da)
    da.close()
    return np.corrcoef(gauge_ssh[train_mask].T)


def main() -> None:
    corr = compute_training_correlations()

    vd          = np.load(VS_CACHE, allow_pickle=True)
    vs_pairs    = vd["vs_pairs"]      # (n_inits, n_leads, n_pairs, 4)
    pair_labels = list(vd["pair_labels"])

    vs_ddpm = vs_pairs[:, :, :, 0]
    vs_clim = vs_pairs[:, :, :, 2]
    vs_fail = 1.0 - vs_ddpm.mean(axis=(0, 1)) / vs_clim.mean(axis=(0, 1))

    # Per-pair r and distance
    pair_r    = np.zeros(len(pair_labels))
    pair_dist = np.zeros(len(pair_labels))
    for k, label in enumerate(pair_labels):
        a, b = label.split("–")
        i, j = SHORT.index(a), SHORT.index(b)
        pair_r[k]    = corr[i, j]
        pair_dist[k] = haversine_km(LATS[i], LONS[i], LATS[j], LONS[j])

    # For each gauge: find all pair indices that include it
    gauge_pairs: dict[str, list[int]] = {s: [] for s in SHORT}
    for k, label in enumerate(pair_labels):
        a, b = label.split("–")
        gauge_pairs[a].append(k)
        gauge_pairs[b].append(k)

    # Shared colour scale
    dist_max  = pair_dist.max()
    cmap_dist = plt.cm.viridis
    norm_dist = mcolors.Normalize(vmin=0, vmax=dist_max)

    # Shared axis limits
    y_all = vs_fail
    ymin  = y_all.min() * 1.15
    ymax  = max(y_all.max() * 1.15, 0.06)

    FS_TICK  = 6
    FS_LABEL = 7
    FS_TITLE = 6.5

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.0),
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.18)

    panel_letters = list("abcdefgh")

    print(f"\n{'Gauge':<18}  {'slope':>8}  {'intercept':>10}  {'r²':>6}  {'p':>8}  sig")
    print("-" * 62)

    for idx, (short, ax) in enumerate(zip(SHORT, axes.flat)):
        pidx = gauge_pairs[short]
        r_g  = pair_r[pidx]
        f_g  = vs_fail[pidx]
        d_g  = pair_dist[pidx]

        sc = ax.scatter(r_g, f_g, c=d_g, cmap=cmap_dist, norm=norm_dist,
                        s=20, zorder=3, linewidths=0.3, edgecolors="0.3")

        # OLS fit
        slope, intercept, r_val, p_val, se = stats.linregress(r_g, f_g)
        sig = p_val < 0.05
        x_fit = np.array([-0.05, 1.05])
        ax.plot(x_fit, slope * x_fit + intercept,
                color="#b2182b" if sig else "0.55",
                lw=1.1, ls="-", zorder=4)

        print(f"{PANEL_LABELS[short]:<18}  {slope:>+8.4f}  {intercept:>+10.4f}"
              f"  {r_val**2:>6.3f}  {p_val:>8.4f}  {'*' if sig else ''}")

        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.05, 1.05)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.2, lw=0.4)

        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        ax.text(0.97, 0.04, p_str,
                transform=ax.transAxes, fontsize=5.5,
                color="#b2182b" if sig else "0.4",
                ha="right", va="bottom")

        label = f"({panel_letters[idx]}) {PANEL_LABELS[short]}"
        ax.set_title(label, fontsize=FS_TITLE, fontweight="bold", pad=3)

    # Shared axis labels
    for ax in axes[1]:
        ax.set_xlabel("Observed Pearson $r$", fontsize=FS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"VS failure", fontsize=FS_LABEL)

    # Single colorbar to the right
    sm = plt.cm.ScalarMappable(cmap=cmap_dist, norm=norm_dist)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.02, aspect=30)
    cbar.set_label("Distance (km)", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)

    out_pdf = OUT_DIR / "vs_by_gauge.pdf"
    out_png = OUT_DIR / "vs_by_gauge.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
