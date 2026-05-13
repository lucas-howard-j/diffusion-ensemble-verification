"""
Block-length sensitivity of VS Δ−Δ_clim significance.

Loads the cached test-split scores from results/figure3_scores_test.npz and
re-runs the pooled block-bootstrap for VS Δ−Δ_clim across a range of block
sizes.  Answers the reviewer concern: does the keystone CI lower bound stay
positive when block size is varied from conservative to aggressive?

VS Δ−Δ_clim = (VS_skill_ddpm − VS_skill_ddpm_shuf) − (VS_skill_clim − VS_skill_clim_shuf)
            = excess joint-structure benefit of DDPM relative to climatological draw
            = the core claim that DDPM captures multivariate dependence beyond a
              marginal-only baseline.

Outputs:
  figures/supplement/block_sensitivity_vs.pdf   — CI bounds vs block size (supplement figure)
  figures/supplement/block_sensitivity_vs.png
  Printed table: block | mean | lo | hi | sig?

Run:
    PYTHONPATH=. python scripts/block_sensitivity_vs.py [--split val|test]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import block_bootstrap_ci

N_BOOT    = 2000          # more resamples for stable CI tails
CI_ALPHA  = 0.05
BLOCK_SIZES = [2, 3, 4, 5, 6, 8, 10, 12, 16]   # weeks
OUT_DIR   = Path("figures/supplement")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def _vs_delta_excess(vs_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-init VS Δ−Δ_clim pooled across leads.

    vs_arr: (N, n_leads, 4)  columns: [orig, shuf, clim, clim_shuf]
    Returns:
        delta_excess  (N, n_leads) — per-init, per-lead excess joint contribution
        pooled        (N,)         — lead-mean of delta_excess (the stat we bootstrap)
    """
    ok = np.all(np.isfinite(vs_arr), axis=(1, 2))
    a   = vs_arr[ok]
    ref = a[:, :, 2:3]                              # VS_clim reference
    per_init = 1.0 - a / (ref + 1e-12)              # skill (N, n_leads, 4)
    delta_ddpm   = per_init[:, :, 0] - per_init[:, :, 1]   # (N, n_leads)
    delta_clim   = per_init[:, :, 2] - per_init[:, :, 3]
    delta_excess = delta_ddpm - delta_clim           # (N, n_leads)
    pooled = delta_excess.mean(axis=1)               # (N,) — lead-mean
    return delta_excess, pooled


def _per_lead_vs_skill(vs_arr: np.ndarray) -> np.ndarray:
    """DDPM VS skill per init per lead: (N, n_leads)."""
    ok = np.all(np.isfinite(vs_arr), axis=(1, 2))
    a   = vs_arr[ok]
    ref = a[:, :, 2]   # VS_clim, shape (N, n_leads)
    return 1.0 - a[:, :, 0] / (ref + 1e-12)


def _block_bootstrap_mean_ci(
    x: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Moving-block bootstrap 95 % CI on the mean of 1-D array x."""
    N = len(x)
    nb = int(np.ceil(N / block_size))
    ms = max(0, N - block_size)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, ms + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:N]
        boot[b] = x[idx].mean()
    lo = float(np.percentile(boot, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(boot, 100 * (1 - CI_ALPHA / 2)))
    return lo, hi


def main(split: str = "test") -> None:
    cache = Path("results") / f"figure3_scores_{split}.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found — run figure3.py first to populate the cache."
        )

    scores = np.load(cache, allow_pickle=True)
    vs_arr = scores["vs"]          # (N, n_leads, 4)
    leads  = list(scores["leads"])
    n_leads = len(leads)

    delta_excess, pooled = _vs_delta_excess(vs_arr)
    per_lead_skill = _per_lead_vs_skill(vs_arr)   # (N, n_leads)
    N = len(pooled)
    print(f"Split: {split}  |  valid inits: {N}  |  leads: {leads}")
    print(f"\nVS Δ−Δ_clim point estimate (lead-mean): {pooled.mean():+.4f}")

    rng = np.random.default_rng(42)

    # ── Sweep: pooled VS Δ−Δ_clim ────────────────────────────────────────────
    print(f"\n{'Block':>6}  {'Mean':>8}  {'lo':>8}  {'hi':>8}  {'sig?':>5}")
    results = {}
    for bs in BLOCK_SIZES:
        lo, hi = _block_bootstrap_mean_ci(pooled, bs, rng)
        sig = "yes *" if lo > 0 else ("neg *" if hi < 0 else "no")
        print(f"  {bs:>2} wk   {pooled.mean():>+.4f}  {lo:>+.4f}  {hi:>+.4f}  {sig}")
        results[bs] = (pooled.mean(), lo, hi)

    # ── Per-lead VS skill CI sweep ────────────────────────────────────────────
    print(f"\n{'Block':>6}  " + "  ".join(f"{l:>2}wk [lo, hi]" for l in leads))
    per_lead_results: dict[int, list[tuple[float, float, float]]] = {}
    for bs in BLOCK_SIZES:
        row = []
        for li in range(n_leads):
            skill_li = per_lead_skill[:, li]
            lo, hi = _block_bootstrap_mean_ci(skill_li, bs, rng)
            row.append((skill_li.mean(), lo, hi))
        per_lead_results[bs] = row
        vals = "  ".join(f"{m:>+.3f}[{lo:>+.3f},{hi:>+.3f}]"
                         for m, lo, hi in row)
        print(f"  {bs:>2} wk  {vals}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel left: pooled VS Δ−Δ_clim CI vs block size
    ax = axes[0]
    means_p = [results[bs][0] for bs in BLOCK_SIZES]
    los_p   = [results[bs][1] for bs in BLOCK_SIZES]
    his_p   = [results[bs][2] for bs in BLOCK_SIZES]
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.plot(BLOCK_SIZES, means_p, "o-", color="#2166ac", lw=1.6, ms=5, label="Mean")
    ax.fill_between(BLOCK_SIZES, los_p, his_p, color="#2166ac", alpha=0.25,
                    label="95% CI")
    ax.plot(BLOCK_SIZES, los_p, "s--", color="#2166ac", lw=1.0, ms=4, alpha=0.7)
    ax.plot(BLOCK_SIZES, his_p, "^--", color="#2166ac", lw=1.0, ms=4, alpha=0.7)
    ax.set_xlabel("Block size (weeks)")
    ax.set_ylabel(r"VS $\Delta - \Delta_\mathrm{clim}$  (lead-mean)")
    ax.set_title(
        f"(a) Pooled VS $\\Delta - \\Delta_{{\\mathrm{{clim}}}}$ vs block size\n"
        f"{split} set  ($N$={N} inits)"
    )
    ax.set_xticks(BLOCK_SIZES)
    ax.legend(fontsize=8)

    # Panel right: per-lead VS DDPM skill lower bound vs block size
    ax2 = axes[1]
    lead_colors = plt.cm.Blues(np.linspace(0.45, 0.9, n_leads))
    for li, (l, color) in enumerate(zip(leads, lead_colors)):
        lo_vals = [per_lead_results[bs][li][1] for bs in BLOCK_SIZES]
        hi_vals = [per_lead_results[bs][li][2] for bs in BLOCK_SIZES]
        mn_vals = [per_lead_results[bs][li][0] for bs in BLOCK_SIZES]
        ax2.plot(BLOCK_SIZES, lo_vals, "s--", color=color, lw=1.2, ms=4,
                 label=f"{l}wk (lo)")
        ax2.plot(BLOCK_SIZES, mn_vals, "o-",  color=color, lw=1.6, ms=5)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Block size (weeks)")
    ax2.set_ylabel("VS skill lower CI bound  (1 − VS$_\\mathrm{DDPM}$ / VS$_\\mathrm{clim}$)")
    ax2.set_title(
        "(b) Per-lead VS skill: lower 95% CI bound\n"
        "dashed = lower bound, solid = mean"
    )
    ax2.set_xticks(BLOCK_SIZES)
    ax2.legend(fontsize=7, title="Lead", loc="upper right")

    fig.suptitle(
        f"Block-length sensitivity of VS significance  —  {split} set\n"
        "Figure 3 block size = 3 wk",
        fontsize=9,
    )
    fig.tight_layout()
    out_stem = OUT_DIR / "block_sensitivity_vs"
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_stem}.pdf / .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    main(args.split)
