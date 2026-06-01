"""
Figure 3 (inset variant) — identical to figure3.py except panel (d) uses a
single y-axis scaled to VS with an inset zooming into the ES bars.

Motivation: after extending the test set through 2025, ES Δ−Δ_clim became
marginally significant (+0.011), breaking the original twin-axis story. The
inset design shows the true scale contrast (ES bars near zero vs VS bars full
height) while still making the ES CIs legible.

Outputs: figures/main/figure3_inset.pdf, figures/main/figure3_inset.png

Run (from cache):
    PYTHONPATH=. python scripts/figure3_inset.py --split test
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

N_MEMBERS  = 25
BLOCK_SIZE = 3
N_BOOT     = 1000
CI_ALPHA   = 0.05
OUT_DIR    = Path("figures/main")
CACHE_DIR  = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

_DDPM_COLOR      = "#2166ac"
_DDPM_SHUF_COLOR = "#92c5de"
_CLIM_COLOR      = "#555555"
_BAR_COLORS = [_DDPM_COLOR, _DDPM_SHUF_COLOR, _CLIM_COLOR]
_BAR_LABELS = ["DDPM", "DDPM shuffled", "GLORYS draw shuffled"]
_BAR_COLS   = [0, 1, 3]


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
    shuf_rng = np.random.default_rng(42)

    def _ref_rng(t_val: int, lead: int) -> np.random.Generator:
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
        crps_ddpm = np.stack(crps_ddpm_rec),
        crps_ref  = np.stack(crps_ref_rec),
        es        = np.stack(es_rec),
        vs        = np.stack(vs_rec),
        leads     = np.array(leads),
        short     = np.array(short),
    )


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def _ci(skill_ts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return block_bootstrap_ci(skill_ts, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng)


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(scores: dict) -> None:
    crps_ddpm_arr = scores["crps_ddpm"]
    crps_ref_arr  = scores["crps_ref"]
    es_arr        = scores["es"]
    vs_arr        = scores["vs"]
    leads         = list(scores["leads"])
    n_leads       = len(leads)

    es_ok = np.all(np.isfinite(es_arr), axis=(1, 2))
    vs_ok = np.all(np.isfinite(vs_arr), axis=(1, 2))
    print(f"Valid inits — CRPS: {len(crps_ddpm_arr)}  ES: {es_ok.sum()}  VS: {vs_ok.sum()}")

    rng_boot = np.random.default_rng(42)

    # Panel (a): ratio-of-time-means CRPS skill
    crps_skill = (
        1 - crps_ddpm_arr.mean(0) / crps_ref_arr.mean(0)
    ).mean(-1)

    N = crps_ddpm_arr.shape[0]
    n_blocks  = int(np.ceil(N / BLOCK_SIZE))
    max_start = max(0, N - BLOCK_SIZE)
    boot_crps = np.empty((N_BOOT, n_leads))
    for b in range(N_BOOT):
        starts = rng_boot.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:N]
        skill  = 1 - crps_ddpm_arr[idx].mean(0) / crps_ref_arr[idx].mean(0)
        boot_crps[b] = skill.mean(-1)
    crps_lo = np.percentile(boot_crps, 100 * CI_ALPHA / 2,       axis=0)
    crps_hi = np.percentile(boot_crps, 100 * (1 - CI_ALPHA / 2), axis=0)

    # Panels (b) and (c)
    def _skill_and_ci(arr, ok_mask):
        a        = arr[ok_mask]
        ref      = a[:, :, 2:3]
        per_init = 1 - a / (ref + 1e-12)
        mean     = per_init.mean(0)
        lo       = np.zeros_like(mean)
        hi       = np.zeros_like(mean)
        for col in [0, 1, 3]:
            l, h = _ci(per_init[:, :, col], rng_boot)
            lo[:, col] = l
            hi[:, col] = h
        delta_ddpm   = per_init[:, :, 0] - per_init[:, :, 1]
        delta_clim   = per_init[:, :, 2] - per_init[:, :, 3]
        delta_excess = delta_ddpm - delta_clim
        return mean, lo, hi, delta_ddpm, delta_clim, delta_excess

    es_mean, es_lo, es_hi, es_dd, es_dc, es_de = _skill_and_ci(es_arr, es_ok)
    vs_mean, vs_lo, vs_hi, vs_dd, vs_dc, vs_de = _skill_and_ci(vs_arr, vs_ok)

    # Panel (d): pooled deltas
    def _pooled_ci(ts: np.ndarray) -> tuple[float, float]:
        lead_mean = ts.mean(axis=1)
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

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.0, 5.2))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    FS_TICK  = 6
    FS_LABEL = 7
    FS_TITLE = 7
    FS_LEG   = 5.5

    ld = np.array(leads)

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
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if leg_outside:
                ax.legend(fontsize=FS_LEG, loc="upper center",
                          bbox_to_anchor=(0.5, -0.22), ncol=3,
                          borderaxespad=0, frameon=True)
            else:
                ax.legend(fontsize=FS_LEG, loc=leg_loc)
        ax.grid(True, alpha=0.2, lw=0.5)

    # Panel (a)
    ax_a.plot(ld, crps_skill, "o-", color=_DDPM_COLOR, lw=1.2, ms=3)
    ax_a.fill_between(ld, crps_lo, crps_hi, color=_DDPM_COLOR, alpha=0.18)
    _style_lead_ax(ax_a, "CRPS skill", "(a) Marginal CRPS skill", leg_loc="upper right")

    # Panels (b) and (c)
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
                      zero_line=False, leg_outside=True)

    # ── Panel (d): single y-axis (VS scale) + inset for ES ───────────────────
    _D_LABELS = [r"$\Delta$DDPM", r"$\Delta$GLORYS", r"$\Delta$DDPM$-\Delta$GLORYS"]
    _D_COLORS = [_DDPM_COLOR, _CLIM_COLOR, "#b2182b"]
    w_d  = 0.22
    x_es = np.array([-1, 0, 1]) * w_d
    x_vs = np.array([-1, 0, 1]) * w_d + 1.6

    err_kw_vs = dict(elinewidth=0.6, capsize=2, ecolor="0.25")
    err_kw_es = dict(elinewidth=0.6, capsize=2, capthick=0.1, ecolor="0.25")

    vs_ymax = max(d_hi_p[3], d_hi_p[4], d_hi_p[5]) * 1.25
    es_ymax = max(d_hi_p[0], d_hi_p[1], d_hi_p[2]) * 1.25

    # Draw all bars on main axis
    for di, (label, color) in enumerate(zip(_D_LABELS, _D_COLORS)):
        # ES bar — tiny on VS scale, detailed in inset
        val    = d_vals[di]
        err_lo = np.maximum(0, val - d_lo_p[di])
        err_hi = np.maximum(0, d_hi_p[di] - val)
        ax_d.bar(x_es[di], val, w_d, color=color, hatch="///", alpha=0.8,
                 label=label, yerr=[[err_lo], [err_hi]], error_kw=err_kw_es)

        # VS bar — full scale
        vi     = di + 3
        val_v  = d_vals[vi]
        err_lo = np.maximum(0, val_v - d_lo_p[vi])
        err_hi = np.maximum(0, d_hi_p[vi] - val_v)
        ax_d.bar(x_vs[di], val_v, w_d, color=color, hatch="///", alpha=0.8,
                 yerr=[[err_lo], [err_hi]], error_kw=err_kw_vs)
        if d_lo_p[vi] > 0 or d_hi_p[vi] < 0:
            ax_d.text(x_vs[di], d_hi_p[vi] + vs_ymax * 0.03, "*",
                      ha="center", va="bottom", fontsize=8, color="k")

    ax_d.set_ylim(0, vs_ymax)
    ax_d.axhline(0, color="k", lw=0.8, ls="--")
    ax_d.set_xticks([0, 1.6])
    ax_d.set_xticklabels(["ES", "VS"], fontsize=FS_TICK + 1, fontweight="bold")
    ax_d.set_xlim(-0.55, 2.15)
    ax_d.set_ylabel(r"$\Delta$ skill  (pooled)", fontsize=FS_LABEL)
    ax_d.tick_params(axis="y", labelsize=FS_TICK)
    ax_d.set_title("(d) Joint structure\ncontribution", fontweight="bold", fontsize=FS_TITLE)
    ax_d.grid(True, axis="y", alpha=0.25, lw=0.6)

    # Inset: zoomed view of ES bars — centred over the ES bar group so
    # indicate_inset_zoom connectors stay roughly parallel (no V-cross).
    ax_ins = ax_d.inset_axes([0.04, 0.50, 0.37, 0.46])

    err_kw_ins = dict(elinewidth=0.6, capsize=2, ecolor="0.25")
    for di, (label, color) in enumerate(zip(_D_LABELS, _D_COLORS)):
        val    = d_vals[di]
        err_lo = np.maximum(0, val - d_lo_p[di])
        err_hi = np.maximum(0, d_hi_p[di] - val)
        ax_ins.bar(x_es[di], val, w_d, color=color, hatch="///", alpha=0.8,
                   yerr=[[err_lo], [err_hi]], error_kw=err_kw_ins)
        if d_lo_p[di] > 0 or d_hi_p[di] < 0:
            ax_ins.text(x_es[di], d_hi_p[di] + es_ymax * 0.04, "*",
                        ha="center", va="bottom", fontsize=6, color="k")

    ax_ins.set_xlim(-0.45, 0.45)
    ax_ins.set_ylim(0, es_ymax)
    ax_ins.set_xticks([])
    ax_ins.yaxis.tick_right()
    ax_ins.tick_params(labelsize=6)
    ax_ins.axhline(0, color="k", lw=0.5, ls="--")
    ax_ins.grid(True, axis="y", alpha=0.2, lw=0.4)
    # Label inside the box to avoid y-axis label clipping
    ax_ins.text(0.97, 0.97, "ES Δ skill", transform=ax_ins.transAxes,
                fontsize=6, ha="right", va="top", color="k")

    ax_d.indicate_inset_zoom(ax_ins, edgecolor="0.4")

    # Legend
    legend_handles = [
        matplotlib.patches.Patch(facecolor=c, hatch="///", alpha=0.8, label=l)
        for c, l in zip(_D_COLORS, _D_LABELS)
    ] + [Line2D([], [], marker="*", color="k", linestyle="none", markersize=6,
                label="p < 0.05  (block bootstrap)")]
    ax_d.legend(handles=legend_handles, fontsize=FS_LEG, loc="upper center",
                bbox_to_anchor=(0.5, -0.22), ncol=2, borderaxespad=0, frameon=True)

    fig.savefig(OUT_DIR / "figure3_inset.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "figure3_inset.png", dpi=300, bbox_inches="tight")
    print("\nSaved figures/main/figure3_inset.pdf and figures/main/figure3_inset.png")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, default=None)
    parser.add_argument("--split", type=str,  default="val")
    parser.add_argument("--rerun", action="store_true")
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
