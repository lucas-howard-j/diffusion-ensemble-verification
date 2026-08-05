"""
Scatterplot of ΔVS vs ΔCRPS for a single sigma level.

Usage:
    python lorenz96/plot_scatter.py --sigma 0.005
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR     = Path("lorenz96/weight_sensitivity/figures")
RESULTS_DIR = Path("lorenz96/weight_sensitivity/results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.005)
    args = parser.parse_args()

    tag   = f"{args.sigma:.4f}".replace(".", "p")
    files = sorted(RESULTS_DIR.glob(f"sigma{tag}_n*.npz"))
    if not files:
        raise FileNotFoundError(f"No results found for sigma={args.sigma} in {RESULTS_DIR}")
    d = np.load(files[-1])

    crps = d["crps_draws"]
    vs   = d["vs_draws"]
    r    = float(np.corrcoef(crps, vs)[0, 1])

    lim = max(np.abs(crps).max(), np.abs(vs).max()) * 1.1
    lim = (-lim, lim)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Reference lines
    ax.axhline(0, color="0.7", lw=0.8, zorder=1)
    ax.axvline(0, color="0.7", lw=0.8, zorder=1)
    ax.plot(lim, lim, color="0.6", lw=0.8, ls="--", zorder=1, label="1:1")

    # Draws
    ax.scatter(crps, vs, s=14, color="#4C72B0", alpha=0.45, linewidths=0, zorder=2)

    # Mean crosshair
    ax.axhline(vs.mean(),   color="#DD8452", lw=1.2, ls=":", zorder=3)
    ax.axvline(crps.mean(), color="#DD8452", lw=1.2, ls=":", zorder=3)
    ax.scatter([crps.mean()], [vs.mean()], s=60, color="#DD8452",
               zorder=4, label=f"mean  ({crps.mean():+.4f}, {vs.mean():+.4f})")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ΔCRPS skill  (marginal)", fontsize=9)
    ax.set_ylabel("ΔVS skill  (joint / dependence)", fontsize=9)
    ax.set_title(f"σ_rel = {args.sigma}  |  N = {len(crps)} draws  |  r = {r:.3f}",
                 fontsize=9, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, lw=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"scatter_vs_crps_sigma{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"r(CRPS, VS) = {r:.3f}")


if __name__ == "__main__":
    main()
