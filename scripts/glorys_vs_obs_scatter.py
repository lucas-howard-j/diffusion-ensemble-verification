"""
Scatter plot of GLORYS SSH anomalies vs NOAA tide gauge SSH anomalies at
gauge locations.

GLORYS weekly anomalies (already deseasonalized + detrended over training
period in src/preprocessing/anomalies.py) are interpolated to gauge locations
via the same nearest-ocean-cell method used in the diffusion dataset, then
aggregated to monthly means.

Observed gauge series are loaded with seasonal cycle removed; here we
additionally remove a linear trend (slope fit on TRAIN_YEARS only, applied
to all months) for parity with the GLORYS pipeline.

Run:
    PYTHONPATH=. python scripts/glorys_vs_obs_scatter.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from src.config import SSH_ANOM_WEEKLY, TIDE_GAUGES, TRAIN_YEARS
from src.data.gauge_loader import load_gauges
from src.model.dataset import _interp_to_gauges
from src.evaluation.metrics import pearson_r

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def detrend_train(series: pd.Series) -> pd.Series:
    """Linear detrend; slope fit on TRAIN_YEARS only, applied to all months."""
    s = series.dropna()
    train = s[s.index.year.isin(list(TRAIN_YEARS))]
    if len(train) < 12:
        return series
    t_train = (train.index - train.index[0]).days.values.astype(float)
    slope, intercept = np.polyfit(t_train, train.values, 1)
    t_all = (series.index - train.index[0]).days.values.astype(float)
    return series - (intercept + slope * t_all)


def main() -> None:
    print(f"Loading GLORYS SSH anomalies: {SSH_ANOM_WEEKLY}")
    da = xr.open_dataarray(str(SSH_ANOM_WEEKLY))

    print("Interpolating to gauge locations ...")
    arr = _interp_to_gauges(da)  # (n_time, n_gauges)
    times = pd.DatetimeIndex(da.time.values)
    gauge_names = [g["name"] for g in TIDE_GAUGES]
    glorys_weekly = pd.DataFrame(arr, index=times, columns=gauge_names)

    print("Aggregating GLORYS weekly → monthly ...")
    glorys_monthly = glorys_weekly.resample("MS").mean()

    print("Loading observed gauges (deseasonalized) and detrending ...")
    obs = load_gauges()  # already mean-removed + seasonal-removed
    obs.index = obs.index.to_period("M").to_timestamp()  # snap to month start
    obs = obs.apply(detrend_train, axis=0)

    # Align on common monthly index and stations actually present in obs
    stations = [s for s in gauge_names if s in obs.columns]
    common_idx = glorys_monthly.index.intersection(obs.index)
    G = glorys_monthly.loc[common_idx, stations]
    O = obs.loc[common_idx, stations]
    print(f"Common months: {len(common_idx)} ({common_idx.min().date()} → {common_idx.max().date()})")

    # ── Per-station scatter panels ──────────────────────────────────────────
    n = len(stations)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             sharex=False, sharey=False)
    axes_flat = axes.flatten()

    rs = {}
    for ax, s in zip(axes_flat, stations):
        x = G[s].values
        y = O[s].values
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        r = pearson_r(x, y) if len(x) > 2 else np.nan
        rs[s] = r

        ax.scatter(x, y, s=10, alpha=0.5, color="steelblue", edgecolor="none")
        lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.05 if len(x) else 1.0
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7, alpha=0.6)
        if len(x) > 2:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([-lim, lim])
            ax.plot(xs, intercept + slope * xs, color="tomato", linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.4)
        ax.axvline(0, color="grey", linewidth=0.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{s}  (r={r:+.2f}, n={len(x)})", fontsize=9)
        ax.set_xlabel("GLORYS SSHA (m)", fontsize=8)
        ax.set_ylabel("Obs SSHA (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("GLORYS vs NOAA gauge SSH anomalies (monthly, detrended + deseasonalized)",
                 y=1.00, fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "glorys_vs_obs_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")

    print("\nPearson r (GLORYS vs obs, monthly):")
    for s in stations:
        print(f"  {s:<22}  r = {rs[s]:+.3f}")


if __name__ == "__main__":
    main()
