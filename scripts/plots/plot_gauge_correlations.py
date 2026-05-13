"""
Compute and plot pairwise tide gauge correlations.

Usage:
    python scripts/plot_gauge_correlations.py [--out figures/gauge_correlations.png]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gauge_loader import load_gauges, gauge_locations


def shorten(name: str) -> str:
    """'Eastport_ME' -> 'Eastport\nME'"""
    parts = name.split("_")
    return "\n".join([" ".join(parts[:-1]), parts[-1]])


def fisher_ci(r: float, n_obs: int, z_crit: float = 1.96):
    """95% CI for Pearson r via Fisher z-transform."""
    if n_obs < 4:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n_obs - 3)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


def mean_ci(df, corr, stations):
    """
    Per-station mean CI: average Fisher CIs across all pairwise partners,
    excluding self-pairs. n_obs per pair = number of joint non-NaN months.
    """
    lo_all, hi_all = [], []
    for stn in stations:
        los, his = [], []
        for other in stations:
            if other == stn:
                continue
            valid = df[[stn, other]].dropna()
            n = len(valid)
            r = corr.loc[stn, other]
            lo, hi = fisher_ci(r, n)
            if np.isfinite(lo):
                los.append(lo)
                his.append(hi)
        lo_all.append(np.mean(los) if los else np.nan)
        hi_all.append(np.mean(his) if his else np.nan)
    return np.array(lo_all), np.array(hi_all)


def detrend(df):
    """Remove per-station linear trend fitted on non-NaN values."""
    out = df.copy()
    t = np.arange(len(df), dtype=float)
    for col in df.columns:
        mask = df[col].notna()
        if mask.sum() < 3:
            continue
        slope, intercept = np.polyfit(t[mask], df[col].values[mask], 1)
        out[col] = df[col] - (slope * t + intercept)
    return out


def main(out_path: Path) -> None:
    # load_gauges already removes mean and monthly seasonal cycle
    df = load_gauges(remove_seasonal=True)

    # Drop stations with too many missing values
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))

    # Remove remaining linear trend (sea level rise etc.)
    df = detrend(df)

    corr = df.corr(method="pearson", min_periods=12)
    labels = [shorten(c) for c in corr.columns]
    n = len(corr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # ── Left: correlation matrix ──────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise Pearson r (detrended monthly SSH anomalies)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="r")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="k" if abs(v) < 0.7 else "w")

    # ── Right: mean correlation vs. latitude (coast-wide coherence) ───────────
    ax = axes[1]
    locs = gauge_locations()
    mean_r = corr.mean()  # mean r with all other stations (incl. self=1)
    # Exclude self-correlation
    mean_r = (corr.sum() - 1) / (n - 1)

    lats = locs.loc[mean_r.index, "lat"]
    station_labels = [shorten(s) for s in mean_r.index]

    ci_lo, ci_hi = mean_ci(df, corr, list(mean_r.index))
    xerr = np.array([mean_r.values - ci_lo, ci_hi - mean_r.values])

    ax.barh(range(n), mean_r.values, xerr=xerr, color="steelblue",
            edgecolor="k", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=4, elinewidth=1.2))
    ax.set_yticks(range(n))
    ax.set_yticklabels(station_labels, fontsize=8)
    ax.set_xlabel("Mean Pearson r with other stations")
    ax.set_title("Coast-wide coherence by station\n(error bars: 95% CI, Fisher z)")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(0, 1)

    # Annotate with lat
    for i, (r, lat) in enumerate(zip(mean_r.values, lats.values)):
        ax.text(r + 0.01, i, f"{lat:.1f}°N", va="center", fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print summary table
    print("\nMean inter-station correlation:")
    for name, r in mean_r.sort_values(ascending=False).items():
        print(f"  {name:<25} r = {r:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("figures/gauge_correlations.png"))
    args = parser.parse_args()
    main(args.out)
