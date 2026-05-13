"""
Temporal autocorrelation of detrended gauge SSH by station.

Compares training (1993-2018) vs test (2022-2024) periods to show:
  - How quickly autocorrelation decays (relevant to init-time conditioning utility)
  - Whether the test period has different persistence structure

Run:
    PYTHONPATH=. python scripts/gauge_autocorrelation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import TIDE_GAUGES, TRAIN_YEARS, TEST_YEARS, LEAD_WEEKS
from src.model.dataset import SSHDiffusionDataset

MAX_LAG = 16   # weeks
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased sample ACF at lags 0 … max_lag."""
    x = x - x.mean()
    n = len(x)
    var = float(np.dot(x, x))
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        out[lag] = float(np.dot(x[:n - lag], x[lag:])) / var
    return out


def main() -> None:
    ds = SSHDiffusionDataset("train")   # loads full time axis
    short = [g["name"].split("_")[0] for g in TIDE_GAUGES]
    n_st = len(short)
    lags = np.arange(MAX_LAG + 1)

    train_mask = ds.time.year.isin(TRAIN_YEARS)
    test_mask  = ds.time.year.isin(TEST_YEARS)

    # Physical detrended anomaly (same units as rank histogram truth)
    train_phys = ds.gauge_ssh[train_mask] * ds.gauge_std + ds.gauge_mu
    test_phys  = ds.gauge_ssh[test_mask]  * ds.gauge_std + ds.gauge_mu

    print(f"Train n={train_mask.sum()}  Test n={test_mask.sum()} (weekly steps)")

    # ── Compute ACF ──────────────────────────────────────────────────────────
    acf_train = np.zeros((n_st, MAX_LAG + 1))
    acf_test  = np.zeros((n_st, MAX_LAG + 1))
    for k in range(n_st):
        acf_train[k] = acf(train_phys[:, k], MAX_LAG)
        acf_test[k]  = acf(test_phys[:, k],  MAX_LAG)

    # ── Print ACF at lead times ───────────────────────────────────────────────
    lead_lags = sorted(LEAD_WEEKS)
    print("\nACF at model lead times:")
    print(f"{'Station':<14}", end="")
    for l in lead_lags:
        print(f"  train@{l}wk  test@{l}wk", end="")
    print()
    for k, s in enumerate(short):
        print(f"{s:<14}", end="")
        for l in lead_lags:
            print(f"  {acf_train[k, l]:>+9.3f}  {acf_test[k, l]:>+8.3f}", end="")
        print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                             constrained_layout=True, sharey=False)
    axes_flat = axes.ravel()

    for k, (ax, s) in enumerate(zip(axes_flat, short)):
        ax.plot(lags, acf_train[k], color="steelblue", lw=1.8,
                marker="o", ms=3.5, label="Train (1993–2018)")
        ax.plot(lags, acf_test[k],  color="tomato",    lw=1.8,
                marker="s", ms=3.5, label="Test (2022–2024)")

        # Mark model lead times
        for l in LEAD_WEEKS:
            ax.axvline(l, color="0.6", lw=0.7, ls="--")

        ax.axhline(0, color="k", lw=0.6, ls="-")

        # 95% significance band (approx ±1.96/√n)
        n_tr = train_mask.sum()
        n_te = test_mask.sum()
        ax.axhspan(-1.96 / n_tr**0.5, 1.96 / n_tr**0.5,
                   color="steelblue", alpha=0.10, lw=0)
        ax.axhspan(-1.96 / n_te**0.5, 1.96 / n_te**0.5,
                   color="tomato",    alpha=0.10, lw=0)

        ax.set_title(s, fontsize=9)
        ax.set_xlim(0, MAX_LAG)
        ax.set_ylim(-0.5, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.set_xlabel("Lag (weeks)", fontsize=8)
        ax.set_ylabel("ACF", fontsize=8)
        ax.tick_params(labelsize=7)

    axes_flat[0].legend(fontsize=7, loc="upper right")

    # Annotate lead-time markers once on the first panel
    for l in LEAD_WEEKS:
        axes_flat[0].annotate(f"{l}w", xy=(l, 1.02), xycoords=("data", "axes fraction"),
                              ha="center", va="bottom", fontsize=6, color="0.45")

    fig.suptitle(
        "Temporal autocorrelation of detrended gauge SSH\n"
        "Dashed verticals = model lead times  ·  shaded bands = 95% sig. threshold",
        fontsize=10,
    )
    out = OUT_DIR / "gauge_autocorrelation"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}.pdf/.png")


if __name__ == "__main__":
    main()
