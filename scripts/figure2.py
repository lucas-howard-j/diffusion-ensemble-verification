"""
Figure 3 — SSH central result (three panels).

(a) Marginal CRPS skill vs lead — station-mean, one bar per lead
(b) Energy score skill vs lead — 4 grouped bars: DDPM / DDPM-shuf / GLORYS-draw / GLORYS-shuf
(c) Variogram score skill vs lead — same 4-bar structure (p=0.5, equal weights)

Reference draws use per-(init, lead) seeds so results are fully reproducible and
independent of processing order.  Scores are cached to results/figure3_scores_{split}.npz;
use --rerun to force recomputation.

Run:
    PYTHONPATH=. python scripts/figure2.py --ckpt checkpoints/ddpm_best_gauge_init.pt [--split val] [--rerun]
    PYTHONPATH=. python scripts/figure2.py --split val          # uses cached scores
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches
from matplotlib.lines import Line2D

from src.config import LEAD_WEEKS, TIDE_GAUGES, TRAIN_YEARS
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint
from src.evaluation.metrics import block_bootstrap_ci
from src.evaluation.scores import crps, energy_score, variogram_score

N_MEMBERS  = 500
BLOCK_SIZE = 3      # weeks; matches ~2–3wk score decorrelation time
N_BOOT     = 1000
CI_ALPHA   = 0.05
OUT_DIR    = Path("figures/main")
CACHE_DIR  = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

_LEAD_COLORS     = ["#08519c", "#2171b5", "#6baed6", "#bdd7e7"]
_DDPM_COLOR      = "#2166ac"
_DDPM_SHUF_COLOR = "#92c5de"
_CLIM_COLOR      = "#555555"
_CLIM_SHUF_COLOR = "#aaaaaa"
_BAR_COLORS = [_DDPM_COLOR, _DDPM_SHUF_COLOR, _CLIM_SHUF_COLOR]
_BAR_LABELS = ["DDPM", "DDPM shuffled", "GLORYS draw shuffled"]
_BAR_COLS   = [0, 1, 3]   # column indices into the (N, n_leads, 4) score arrays


def _shuffle_columns(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = arr.copy()
    for k in range(arr.shape[1]):
        out[:, k] = arr[rng.permutation(arr.shape[0]), k]
    return out


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(ckpt_path: Path | None, split: str) -> dict:
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    gauge_mu   = ckpt["gauge_mu"]
    gauge_std  = ckpt["gauge_std"]
    print(f"Device: {device}  |  target_dim: {target_dim}  |  split: {split}")

    ds            = SSHDiffusionDataset(split)
    station_names = [g["name"] for g in TIDE_GAUGES][:target_dim]
    short         = [s.split("_")[0] for s in station_names]
    n_st          = target_dim

    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx
    leads    = sorted(LEAD_WEEKS)
    n_leads  = len(leads)
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Init times: {len(valid_ts)}")

    train_mask  = ds.time.year.isin(TRAIN_YEARS)
    glorys_phys = ds.gauge_ssh[train_mask] * gauge_std + gauge_mu
    glorys_time = ds.time[train_mask]
    glorys_pool = {m: glorys_phys[glorys_time.month == m] for m in range(1, 13)}

    crps_ddpm_rec, crps_ref_rec, es_rec, vs_rec = [], [], [], []

    # Shuffles use a single global rng (averaged over many inits; order doesn't matter).
    # Reference draws use per-(t_val, lead) seeds so results are independent of
    # processing order and match any other script using the same scheme.
    shuf_rng = np.random.default_rng(42)

    def _ref_rng(t_val: int, lead: int) -> np.random.Generator:
        """Deterministic, order-independent seed for GLORYS reference draws."""
        return np.random.default_rng(hash((t_val, lead)) % (2 ** 31))

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(valid_ts):
            if i % 20 == 0:
                print(f"  {i}/{len(valid_ts)}")

            conds, glorys_targets, target_months = [], [], []
            for lead in leads:
                cond, target = ds[t_to_lead_idx[t_val][lead]]
                conds.extend([cond] * N_MEMBERS)
                glorys_targets.append(target.numpy() * gauge_std + gauge_mu)
                target_months.append(ds.time[t_val + lead].month)

            cond_batch = torch.stack(conds).to(device)
            x = torch.randn(len(cond_batch), target_dim, device=device)
            for step in reversed(range(schedule.T)):
                x = schedule.p_sample(model, x, step, cond_batch)
            preds = x.cpu().numpy()

            row_crps_d = np.zeros((n_leads, n_st))
            row_crps_r = np.zeros((n_leads, n_st))
            row_es     = np.full((n_leads, 4), np.nan)
            row_vs     = np.full((n_leads, 4), np.nan)

            for j, lead in enumerate(leads):
                true    = glorys_targets[j]
                members = preds[j * N_MEMBERS:(j + 1) * N_MEMBERS] * gauge_std + gauge_mu
                month   = target_months[j]
                pool    = glorys_pool[month]
                rng_ref = _ref_rng(t_val, lead)

                ref_members   = pool[rng_ref.choice(len(pool), N_MEMBERS, replace=True)]
                row_crps_d[j] = crps(members,     true)
                row_crps_r[j] = crps(ref_members, true)

                if len(pool) >= 2:
                    shuf_ddpm = _shuffle_columns(members,     shuf_rng)
                    clim      = pool[rng_ref.choice(len(pool), N_MEMBERS, replace=True)]
                    shuf_clim = _shuffle_columns(clim, shuf_rng)
                    row_es[j] = [
                        energy_score(members,   true),
                        energy_score(shuf_ddpm, true),
                        energy_score(clim,      true),
                        energy_score(shuf_clim, true),
                    ]
                    row_vs[j] = [
                        variogram_score(members,   true),
                        variogram_score(shuf_ddpm, true),
                        variogram_score(clim,      true),
                        variogram_score(shuf_clim, true),
                    ]

            crps_ddpm_rec.append(row_crps_d)
            crps_ref_rec.append(row_crps_r)
            es_rec.append(row_es)
            vs_rec.append(row_vs)

    return dict(
        crps_ddpm = np.stack(crps_ddpm_rec),   # (N, n_leads, n_st)
        crps_ref  = np.stack(crps_ref_rec),
        es        = np.stack(es_rec),           # (N, n_leads, 4)
        vs        = np.stack(vs_rec),
        leads     = np.array(leads),
        short     = np.array(short),
    )


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def _ci(skill_ts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Block bootstrap 95 % CI on column means.  skill_ts: (N, n_cols)"""
    return block_bootstrap_ci(skill_ts, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng)


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(scores: dict) -> None:
    crps_ddpm_arr = scores["crps_ddpm"]
    crps_ref_arr  = scores["crps_ref"]
    es_arr        = scores["es"]
    vs_arr        = scores["vs"]
    leads         = list(scores["leads"])
    short         = list(scores["short"])
    n_leads       = len(leads)
    n_st          = crps_ddpm_arr.shape[2]

    es_ok = np.all(np.isfinite(es_arr), axis=(1, 2))
    vs_ok = np.all(np.isfinite(vs_arr), axis=(1, 2))
    print(f"Valid inits — CRPS: {len(crps_ddpm_arr)}  ES: {es_ok.sum()}  VS: {vs_ok.sum()}")

    rng_boot = np.random.default_rng(42)

    # Panel (a): station-mean CRPS skill per lead.
    # Use ratio-of-time-means (matches rmse_vs_lead.py) — robust to outlier reference draws.
    # mean-of-ratios is biased negative when any single ref draw is near zero.
    crps_skill = (
        1 - crps_ddpm_arr.mean(0) / crps_ref_arr.mean(0)
    ).mean(-1)   # (n_leads,): ratio-of-means per station, then station-mean

    # Bootstrap: resample raw scores and recompute ratio-of-means, same as rmse_vs_lead.py
    N = crps_ddpm_arr.shape[0]
    n_blocks  = int(np.ceil(N / BLOCK_SIZE))
    max_start = max(0, N - BLOCK_SIZE)
    boot_crps = np.empty((N_BOOT, n_leads))
    for b in range(N_BOOT):
        starts = rng_boot.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:N]
        skill  = 1 - crps_ddpm_arr[idx].mean(0) / crps_ref_arr[idx].mean(0)  # (n_leads, n_st)
        boot_crps[b] = skill.mean(-1)   # station-mean
    crps_lo = np.percentile(boot_crps, 100 * CI_ALPHA / 2,       axis=0)
    crps_hi = np.percentile(boot_crps, 100 * (1 - CI_ALPHA / 2), axis=0)

    # Panels (b) and (c): skill scores only
    def _skill_and_ci(arr, ok_mask):
        a        = arr[ok_mask]
        ref      = a[:, :, 2:3]
        per_init = 1 - a / (ref + 1e-12)           # (N', n_leads, 4)
        mean     = per_init.mean(0)
        lo       = np.zeros_like(mean)
        hi       = np.zeros_like(mean)
        for col in [0, 1, 3]:
            l, h = _ci(per_init[:, :, col], rng_boot)
            lo[:, col] = l
            hi[:, col] = h
        # Return per-init delta time series for pooled bootstrap in panel (d)
        delta_ddpm   = per_init[:, :, 0] - per_init[:, :, 1]   # (N', n_leads)
        delta_clim   = per_init[:, :, 2] - per_init[:, :, 3]
        delta_excess = delta_ddpm - delta_clim
        return mean, lo, hi, delta_ddpm, delta_clim, delta_excess

    es_mean, es_lo, es_hi, es_dd, es_dc, es_de = _skill_and_ci(es_arr, es_ok)
    vs_mean, vs_lo, vs_hi, vs_dd, vs_dc, vs_de = _skill_and_ci(vs_arr, vs_ok)

    # Panel (d): pool each delta across leads (lead-mean per init), then block-bootstrap
    def _pooled_ci(ts: np.ndarray) -> tuple[float, float]:
        """ts: (N, n_leads) → average across leads, block-bootstrap CI on the mean."""
        lead_mean = ts.mean(axis=1)   # (N,)
        N_p = len(lead_mean)
        nb  = int(np.ceil(N_p / BLOCK_SIZE))
        ms  = max(0, N_p - BLOCK_SIZE)
        boot = np.array([
            lead_mean[np.concatenate([np.arange(s, s + BLOCK_SIZE)
                       for s in rng_boot.integers(0, ms + 1, size=nb)])[:N_p]].mean()
            for _ in range(N_BOOT)
        ])
        return float(np.percentile(boot, 100 * CI_ALPHA / 2)), \
               float(np.percentile(boot, 100 * (1 - CI_ALPHA / 2)))

    d_series = [es_dd, es_dc, es_de, vs_dd, vs_dc, vs_de]
    d_vals   = np.array([ts.mean() for ts in d_series])
    d_lo_p   = np.zeros(6)
    d_hi_p   = np.zeros(6)
    for i, ts in enumerate(d_series):
        d_lo_p[i], d_hi_p[i] = _pooled_ci(ts)

    # Summary tables
    print("\nCRPS skill (1 − CRPS/CRPS_ref)  [station-mean per lead]:")
    for j, l in enumerate(leads):
        print(f"  {l:>2}wk  {crps_skill[j]:>+.4f}  [{crps_lo[j]:>+.4f}, {crps_hi[j]:>+.4f}]")
    print("\nEnergy score skill  (1 − ES/ES_clim):")
    print(f"{'Lead':>6}  {'DDPM':>8}  {'DDPM-shuf':>10}  {'GLORYS-shuf':>12}")
    for j, l in enumerate(leads):
        print(f"  {l:>2}wk  {es_mean[j,0]:>+.4f}  {es_mean[j,1]:>+.4f}  {es_mean[j,3]:>+.4f}")
    print("\nVariogram score skill  (1 − VS/VS_clim, p=0.5):")
    print(f"{'Lead':>6}  {'DDPM':>8}  {'DDPM-shuf':>10}  {'GLORYS-shuf':>12}")
    for j, l in enumerate(leads):
        print(f"  {l:>2}wk  {vs_mean[j,0]:>+.4f}  {vs_mean[j,1]:>+.4f}  {vs_mean[j,3]:>+.4f}")
    print("\nPooled deltas (lead-mean ± 95% CI):")
    d_names = ["ES Δ_DDPM", "ES Δ_GLORYS", "ES Δ−Δ_clim",
               "VS Δ_DDPM", "VS Δ_GLORYS", "VS Δ−Δ_clim"]
    for name, val, lo, hi in zip(d_names, d_vals, d_lo_p, d_hi_p):
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {name:<16}  {val:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}] {sig}")

    # Build figure: 2×2 layout
    fig = plt.figure(figsize=(7.0, 5.2))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    FS_TICK  = 6    # tick labels
    FS_LABEL = 7    # axis labels
    FS_TITLE = 7    # panel titles
    FS_LEG   = 5.5  # legends

    ld = np.array(leads)   # actual week values for x-axis

    def _style_lead_ax(ax, ylabel, title, leg_loc="lower left",
                       zero_line=True, leg_outside=False):
        if zero_line:
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(ld)
        ax.set_xticklabels([f"{l}wk" for l in leads], fontsize=FS_TICK)
        ax.set_xlabel("Lead time", fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontweight="bold", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        if leg_outside:
            ax.legend(fontsize=FS_LEG, loc="upper center",
                      bbox_to_anchor=(0.5, -0.22), ncol=3,
                      borderaxespad=0, frameon=True)
        else:
            ax.legend(fontsize=FS_LEG, loc=leg_loc)
        ax.grid(True, alpha=0.2, lw=0.5)

    # Panel (a): single line + shaded 95 % CI
    ax_a.plot(ld, crps_skill, "o-", color=_DDPM_COLOR, lw=1.2, ms=3)
    ax_a.fill_between(ld, crps_lo, crps_hi, color=_DDPM_COLOR, alpha=0.18)
    _style_lead_ax(ax_a, "CRPS skill", "(a) Marginal CRPS skill", leg_loc="upper right")

    # Panels (b) and (c): one line per series with shaded CI
    _LINE_STYLES = ["-", "--", ":"]
    _MARKERS     = ["o", "s", "^"]

    def _plot_score_lines(ax, mean, lo, hi, title, ylabel,
                          leg_loc="lower left", zero_line=True, leg_outside=False):
        for col_idx, label, color, ls, mk in zip(
                _BAR_COLS, _BAR_LABELS, _BAR_COLORS, _LINE_STYLES, _MARKERS):
            vals = mean[:, col_idx]
            ax.plot(ld, vals, ls + mk, color=color, lw=1.2, ms=3, label=label)
            ax.fill_between(ld, lo[:, col_idx], hi[:, col_idx], color=color, alpha=0.15)
        _style_lead_ax(ax, ylabel, title, leg_loc,
                       zero_line=zero_line, leg_outside=leg_outside)

    _plot_score_lines(ax_b, es_mean, es_lo, es_hi,
                      "(b) Energy score skill", "ES skill")
    _plot_score_lines(ax_c, vs_mean, vs_lo, vs_hi,
                      "(c) Variogram score skill", r"VS skill  ($p$=0.5)",
                      zero_line=True, leg_outside=True)
    ax_c.set_ylim(top=0.1)

    err_kw = dict(elinewidth=0.6, capsize=2, ecolor="0.25")

    # Panel (d): pooled deltas — ES on left axis, VS on right axis (scales differ ~50×)
    _D_LABELS = [r"$\Delta$DDPM", r"$\Delta$GLORYS", r"$\Delta$DDPM$-\Delta$GLORYS"]
    _D_COLORS = [_DDPM_COLOR, _CLIM_COLOR, "#b2182b"]
    w_d   = 0.22
    x_es  = np.array([-1, 0, 1]) * w_d          # ES bars centred at x=0
    x_vs  = np.array([-1, 0, 1]) * w_d + 1.6    # VS bars centred at x=1.6

    ax_d2 = ax_d.twinx()   # right axis for VS

    for di, (label, color) in enumerate(zip(_D_LABELS, _D_COLORS)):
        # ES bar — left axis
        val    = d_vals[di]
        err_lo = np.maximum(0, val - d_lo_p[di])
        err_hi = np.maximum(0, d_hi_p[di] - val)
        ax_d.bar(x_es[di], val, w_d, color=color, hatch="///", alpha=0.8,
                 label=label, yerr=[[err_lo], [err_hi]], error_kw=err_kw)
        if d_lo_p[di] > 0 or d_hi_p[di] < 0:
            ax_d.text(x_es[di], d_hi_p[di] + 0.002, "*",
                      ha="center", va="bottom", fontsize=8, color="k")

        # VS bar — right axis
        vi     = di + 3
        val    = d_vals[vi]
        err_lo = np.maximum(0, val - d_lo_p[vi])
        err_hi = np.maximum(0, d_hi_p[vi] - val)
        ax_d2.bar(x_vs[di], val, w_d, color=color, hatch="///", alpha=0.8,
                  yerr=[[err_lo], [err_hi]], error_kw=err_kw)
        if d_lo_p[vi] > 0 or d_hi_p[vi] < 0:
            ax_d2.text(x_vs[di], d_hi_p[vi] + 0.01, "*",
                       ha="center", va="bottom", fontsize=8, color="k")

    # Align zeros: keep the data-driven upper limits, push lower limits down so
    # that y=0 sits at the same fractional height on both axes.
    es_hi = max(d_hi_p[0], d_hi_p[1], d_hi_p[2]) * 1.25
    vs_hi = max(d_hi_p[3], d_hi_p[4], d_hi_p[5]) * 1.25
    # natural negative extents (may be 0 if all bars positive)
    es_lo_nat = min(d_lo_p[0], d_lo_p[1], d_lo_p[2], 0) * 1.25
    vs_lo_nat = min(d_lo_p[3], d_lo_p[4], d_lo_p[5], 0) * 1.25
    zero_frac = max(-es_lo_nat / (es_hi - es_lo_nat),
                    -vs_lo_nat / (vs_hi - vs_lo_nat))
    # given zero_frac and hi, lo = -zero_frac / (1 - zero_frac) * hi
    ax_d.set_ylim(-zero_frac / (1 - zero_frac) * es_hi, es_hi)
    ax_d2.set_ylim(-zero_frac / (1 - zero_frac) * vs_hi, vs_hi)

    ax_d.axhline(0, color="k", lw=0.8, ls="--")
    ax_d.set_xticks([0, 1.6])
    ax_d.set_xticklabels(["ES", "VS"], fontsize=FS_TICK + 1, fontweight="bold")
    ax_d.set_xlim(-0.55, 2.15)
    ax_d.set_ylabel(r"ES  $\Delta$ skill  (pooled)", fontsize=FS_LABEL, color=_DDPM_COLOR)
    ax_d.tick_params(axis="y", labelsize=FS_TICK, labelcolor=_DDPM_COLOR)
    ax_d2.set_ylabel(r"VS  $\Delta$ skill  (pooled)", fontsize=FS_LABEL, color="#555555")
    ax_d2.tick_params(axis="y", labelsize=FS_TICK, labelcolor="#555555")
    ax_d.set_title("(d) Joint structure\ncontribution", fontweight="bold", fontsize=FS_TITLE)
    ax_d.grid(True, axis="y", alpha=0.25, lw=0.6)

    # Legend: bar labels + star explanation (no double-asterisk)
    legend_handles = [
        matplotlib.patches.Patch(facecolor=c, hatch="///", alpha=0.8, label=l)
        for c, l in zip(_D_COLORS, _D_LABELS)
    ] + [Line2D([], [], marker="*", color="k", linestyle="none", markersize=6,
                label="p < 0.05  (block bootstrap)")]
    ax_d.legend(handles=legend_handles, fontsize=FS_LEG, loc="upper center",
                bbox_to_anchor=(0.5, -0.22), ncol=2, borderaxespad=0, frameon=True)

    fig.savefig(OUT_DIR / "figure2.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "figure2.png", dpi=300, bbox_inches="tight")
    print("\nSaved figures/figure2.pdf and figures/figure2.png")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, default=None,
                        help="Checkpoint path (only needed when running inference)")
    parser.add_argument("--split", type=str,  default="val",
                        help="Dataset split: val or test")
    parser.add_argument("--rerun", action="store_true",
                        help="Force rerun inference even if cache exists")
    args = parser.parse_args()

    cache_path = CACHE_DIR / f"figure3_scores_{args.split}.npz"

    if cache_path.exists() and not args.rerun:
        print(f"Loading cached scores from {cache_path}")
        scores = dict(np.load(cache_path, allow_pickle=True))
    else:
        print("Running inference ...")
        scores = run_inference(args.ckpt, args.split)
        np.savez(cache_path, **scores)
        print(f"Saved scores to {cache_path}")

    make_figure(scores)


if __name__ == "__main__":
    main()
