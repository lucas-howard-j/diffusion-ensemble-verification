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
    vs_arr     = scores["vs"]           # (N, n_leads, 4)
    crps_ddpm  = scores["crps_ddpm"]    # (N, n_leads, n_stations)
    crps_ref   = scores["crps_ref"]
    leads  = list(scores["leads"])
    n_leads = len(leads)

    delta_excess, pooled = _vs_delta_excess(vs_arr)
    per_lead_skill = _per_lead_vs_skill(vs_arr)   # (N, n_leads)
    N = len(pooled)
    print(f"Split: {split}  |  valid inits: {N}  |  leads: {leads}")
    print(f"\nVS Δ−Δ_clim point estimate (lead-mean): {pooled.mean():+.4f}")

    rng = np.random.default_rng(42)

    # ── CRPS skill — ratio-of-means bootstrap (matches figure3.py) ──────────
    # Point estimate: 1 - mean_N(crps_ddpm) / mean_N(crps_ref), per lead, station-mean
    ok_crps = np.all(np.isfinite(crps_ddpm) & np.isfinite(crps_ref), axis=(1, 2))
    cd_all = crps_ddpm[ok_crps]   # (N, n_leads, n_st)
    cr_all = crps_ref[ok_crps]
    N_crps = cd_all.shape[0]

    def _crps_skill_rom(cd_sub, cr_sub):
        """Ratio-of-means skill: (n_leads,) station-mean."""
        return (1.0 - cd_sub.mean(0) / (cr_sub.mean(0) + 1e-12)).mean(-1)  # (n_leads,)

    crps_skill_pt = _crps_skill_rom(cd_all, cr_all)   # (n_leads,)
    crps_lead_mean_pt = crps_skill_pt.mean()
    print(f"CRPS skill point estimate (ratio-of-means, station-mean):")
    for li, l in enumerate(leads):
        print(f"  lead {l:>2}wk: {crps_skill_pt[li]:>+.4f}")
    print(f"  lead-mean:  {crps_lead_mean_pt:>+.4f}")

    def _crps_boot_ci(cd, cr, bs):
        N_b = len(cd)
        nb = int(np.ceil(N_b / bs))
        ms = max(0, N_b - bs)
        boot = np.empty((N_BOOT, len(leads)))
        for b in range(N_BOOT):
            starts = rng.integers(0, ms + 1, size=nb)
            idx = np.concatenate([np.arange(s, s + bs) for s in starts])[:N_b]
            boot[b] = _crps_skill_rom(cd[idx], cr[idx])
        lead_mean_boot = boot.mean(axis=1)
        lo = float(np.percentile(lead_mean_boot, 100 * CI_ALPHA / 2))
        hi = float(np.percentile(lead_mean_boot, 100 * (1 - CI_ALPHA / 2)))
        per_lead_lo = np.percentile(boot, 100 * CI_ALPHA / 2, axis=0)
        per_lead_hi = np.percentile(boot, 100 * (1 - CI_ALPHA / 2), axis=0)
        return lo, hi, per_lead_lo, per_lead_hi

    print(f"\n── CRPS skill block sensitivity (ratio-of-means, lead-mean) ──")
    print(f"{'Block':>6}  {'Mean':>8}  {'lo':>8}  {'hi':>8}  {'sig?':>5}")
    crps_results = {}
    crps_per_lead_results = {}
    for bs in BLOCK_SIZES:
        lo, hi, pl_lo, pl_hi = _crps_boot_ci(cd_all, cr_all, bs)
        sig = "yes *" if lo > 0 else ("neg *" if hi < 0 else "no")
        print(f"  {bs:>2} wk   {crps_lead_mean_pt:>+.4f}  {lo:>+.4f}  {hi:>+.4f}  {sig}")
        crps_results[bs] = (crps_lead_mean_pt, lo, hi)
        crps_per_lead_results[bs] = list(zip(crps_skill_pt, pl_lo, pl_hi))

    print(f"\n── CRPS skill per-lead block sensitivity ──")
    print(f"{'Block':>6}  " + "  ".join(f"{l:>2}wk [lo, hi]" for l in leads))
    for bs in BLOCK_SIZES:
        vals = "  ".join(f"{m:>+.3f}[{lo:>+.3f},{hi:>+.3f}]"
                         for m, lo, hi in crps_per_lead_results[bs])
        print(f"  {bs:>2} wk  {vals}")

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

    # ── ACF of bootstrapped series ───────────────────────────────────────────
    def _acf(x, max_lag=20):
        x = x - x.mean()
        c0 = np.dot(x, x) / len(x)
        return np.array([np.dot(x[:len(x)-k], x[k:]) / (len(x) * c0)
                         for k in range(max_lag + 1)])

    # CRPS per-init series for ACF (per-init ratio, lead-mean)
    crps_per_init = (1.0 - cd_all.mean(axis=2) / (cr_all.mean(axis=2) + 1e-12)).mean(axis=1)
    acf_vs   = _acf(pooled)
    acf_crps = _acf(crps_per_init)
    sig95    = 1.96 / np.sqrt(len(pooled))

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13, 7))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.12, wspace=0.35)
    ax  = fig.add_subplot(gs[:, 0])
    ax2_rows = [fig.add_subplot(gs[0, 1])] + \
               [fig.add_subplot(gs[i, 1], sharex=fig.axes[-1]) for i in range(1, 4)]
    ax3 = fig.add_subplot(gs[:, 2])

    # Panel left: pooled VS Δ−Δ_clim CI vs block size
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

    # Panel middle: per-lead CRPS skill — one row per lead
    lead_colors = plt.cm.Reds(np.linspace(0.4, 0.85, n_leads))
    for li, (l, color, ax2) in enumerate(zip(leads, lead_colors, ax2_rows)):
        mn_vals = np.array([crps_per_lead_results[bs][li][0] for bs in BLOCK_SIZES])
        lo_vals = np.array([crps_per_lead_results[bs][li][1] for bs in BLOCK_SIZES])
        hi_vals = np.array([crps_per_lead_results[bs][li][2] for bs in BLOCK_SIZES])
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.fill_between(BLOCK_SIZES, lo_vals, hi_vals, color=color, alpha=0.35)
        ax2.plot(BLOCK_SIZES, mn_vals, "o-", color=color, lw=1.6, ms=4)
        ax2.set_ylabel(f"{l}wk", fontsize=8)
        ax2.set_xticks(BLOCK_SIZES)
        ax2.tick_params(axis="x", labelbottom=(li == n_leads - 1), labelsize=7)
        ax2.tick_params(axis="y", labelsize=7)
        if li == 0:
            ax2.set_title("(b) Per-lead CRPS skill vs block size\nshading = 95% CI")
    ax2_rows[-1].set_xlabel("Block size (weeks)")

    # Panel 3: CRPS skill CI vs block size (lead-mean)
    means_c = [crps_results[bs][0] for bs in BLOCK_SIZES]
    los_c   = [crps_results[bs][1] for bs in BLOCK_SIZES]
    his_c   = [crps_results[bs][2] for bs in BLOCK_SIZES]
    ax3.axhline(0, color="k", lw=0.8, ls="--")
    ax3.plot(BLOCK_SIZES, means_c, "o-", color="#d6604d", lw=1.6, ms=5, label="Mean")
    ax3.fill_between(BLOCK_SIZES, los_c, his_c, color="#d6604d", alpha=0.25,
                     label="95% CI")
    ax3.plot(BLOCK_SIZES, los_c, "s--", color="#d6604d", lw=1.0, ms=4, alpha=0.7)
    ax3.plot(BLOCK_SIZES, his_c, "^--", color="#d6604d", lw=1.0, ms=4, alpha=0.7)
    ax3.set_xlabel("Block size (weeks)")
    ax3.set_ylabel("CRPS skill  (1 − CRPS$_\\mathrm{DDPM}$ / CRPS$_\\mathrm{clim}$)")
    ax3.set_title(
        f"(c) CRPS skill vs block size\n"
        f"{split} set  ($N$={N} inits, station-mean, lead-mean)"
    )
    ax3.set_xticks(BLOCK_SIZES)
    ax3.legend(fontsize=8)

    fig.suptitle(
        f"Block-length sensitivity  —  {split} set  |  Figure 3 block size = 3 wk",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_stem = OUT_DIR / "block_sensitivity_vs"
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_stem}.pdf / .png")

    # ── ACF figure ────────────────────────────────────────────────────────────
    fig_acf, (ax_acf_vs, ax_acf_crps) = plt.subplots(1, 2, figsize=(8, 3.5))
    lags = np.arange(len(acf_vs))
    for ax_acf, acf, label, color in [
        (ax_acf_vs,   acf_vs,   r"VS $\Delta{-}\Delta_\mathrm{clim}$", "#2166ac"),
        (ax_acf_crps, acf_crps, "CRPS skill",                          "#d6604d"),
    ]:
        ax_acf.bar(lags, acf, color=color, alpha=0.7, width=0.8)
        ax_acf.axhline(0,      color="k", lw=0.8)
        ax_acf.axhline( sig95, color="k", lw=0.8, ls="--", alpha=0.6,
                        label=r"95% threshold ($\pm 1.96/\sqrt{N}$)")
        ax_acf.axhline(-sig95, color="k", lw=0.8, ls="--", alpha=0.6)
        ax_acf.set_xlabel("Lag (weeks)")
        ax_acf.set_ylabel("ACF")
        ax_acf.set_title(f"ACF — {label} (lead-mean, $N$={len(pooled)})")
        ax_acf.set_xlim(-0.5, len(acf_vs) - 0.5)
        ax_acf.legend(fontsize=7)
    fig_acf.suptitle(f"Score residual autocorrelation — {split} set", fontsize=9)
    fig_acf.tight_layout()
    out_acf = OUT_DIR / "score_acf"
    fig_acf.savefig(out_acf.with_suffix(".pdf"), bbox_inches="tight")
    fig_acf.savefig(out_acf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_acf}.pdf / .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    main(args.split)
