"""
Pearson correlation between DDPM ensemble-mean SSH predictions and NOAA
tide gauge observations, stratified by lead time.

Model output is SSH anomaly at gauge locations directly (no EOF reconstruction).
Gauge data is monthly; weekly model predictions are averaged to monthly
before correlating.

Run:
    PYTHONPATH=. python scripts/gauge_correlation.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def savefig(fig, path, **kwargs):
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", **kwargs)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight", **kwargs)

from src.config import LEAD_WEEKS, TIDE_GAUGES
from src.data.gauge_loader import load_gauges
from src.evaluation.metrics import pearson_r, pairwise_corr_matrix, pattern_correlation
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS = 25
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def main(ckpt_path: Path | None = None, split: str = "val") -> None:
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    gauge_mu  = ckpt["gauge_mu"]   # (n_gauges,)
    gauge_std = ckpt["gauge_std"]  # (n_gauges,)
    print(f"Device: {device}")

    ds = SSHDiffusionDataset(split)
    print(f"{split.capitalize()} pairs: {len(ds)}")

    # Ordered gauge names matching model output indices
    all_gauge_names = [g["name"] for g in TIDE_GAUGES]

    # Index all val pairs by t
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx

    leads = sorted(LEAD_WEEKS)
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Val time steps with all leads: {len(valid_ts)}")

    # Load observed gauge anomalies (monthly)
    gauge_obs = load_gauges()
    station_names = [n for n in all_gauge_names if n in gauge_obs.columns]
    station_indices = [all_gauge_names.index(n) for n in station_names]

    # pred_accum[lead][station] = {ym: [values]}
    pred_accum: dict[int, dict[str, dict[str, list[float]]]] = {
        l: {s: defaultdict(list) for s in station_names} for l in leads
    }

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(valid_ts):
            if i % 20 == 0:
                print(f"  t {i}/{len(valid_ts)}")

            conds, target_times = [], []
            for lead in leads:
                cond, _ = ds[t_to_lead_idx[t_val][lead]]
                conds.extend([cond] * N_MEMBERS)
                target_times.append(ds.time[t_val + lead])

            cond_batch = torch.stack(conds).to(device)
            x = torch.randn(len(cond_batch), target_dim, device=device)
            for step in reversed(range(schedule.T)):
                x = schedule.p_sample(model, x, step, cond_batch)

            gauge_z = x.cpu().numpy()  # (n_leads*N_MEMBERS, n_gauges)

            for j, lead in enumerate(leads):
                block_z = gauge_z[j * N_MEMBERS : (j + 1) * N_MEMBERS]   # (25, n_gauges)
                block_phys = block_z * gauge_std + gauge_mu               # (25, n_gauges)
                ens_mean = block_phys.mean(axis=0)                        # (n_gauges,)

                ym = target_times[j].to_period("M").to_timestamp()
                for k, s in zip(station_indices, station_names):
                    val = ens_mean[k]
                    if np.isfinite(val):
                        pred_accum[lead][s][str(ym)].append(float(val))

    # ── Build monthly pred DataFrames ────────────────────────────────────────
    pred_dfs: dict[int, pd.DataFrame] = {}
    corr_table: dict[int, dict[str, float]] = {l: {} for l in leads}

    for l in leads:
        rows = {}
        for s in station_names:
            monthly = pred_accum[l][s]
            if monthly:
                rows[s] = pd.Series(
                    {pd.Timestamp(ym): np.mean(v) for ym, v in monthly.items()}
                )
        pred_dfs[l] = pd.DataFrame(rows)

    # ── Marginal skill ───────────────────────────────────────────────────────
    print("\nCorrelation (Pearson r) — ensemble mean vs NOAA gauge")
    print(f"{'Station':<22}", end="")
    for l in leads:
        print(f"  {l:>2}wk", end="")
    print()
    print("-" * (22 + 6 * len(leads)))

    for s in station_names:
        print(f"{s:<22}", end="")
        for l in leads:
            df = pred_dfs[l]
            if s not in df.columns:
                corr_table[l][s] = np.nan
                print(f"   nan", end="")
                continue
            obs_vals  = gauge_obs[s].reindex(df.index).values
            pred_vals = df[s].values
            r = pearson_r(obs_vals, pred_vals)
            corr_table[l][s] = r
            print(f"  {r:+.3f}", end="")
        print()

    # ── Coherence: pairwise station correlation matrices ─────────────────────
    val_dates = sorted(set(ts for l in leads for ts in pred_dfs[l].index))
    obs_val = gauge_obs.reindex(pd.DatetimeIndex(val_dates))[station_names].dropna(how="all")
    obs_corr = pairwise_corr_matrix(obs_val)

    print("\nPattern correlation (predicted vs observed pairwise station corr)")
    print(f"{'Lead':>6}  {'Pattern r':>10}")
    pattern_rs = {}
    for l in leads:
        pred_corr = pairwise_corr_matrix(pred_dfs[l][station_names].dropna(how="all"))
        pr = pattern_correlation(obs_corr, pred_corr)
        pattern_rs[l] = pr
        print(f"  {l:>2}wk    {pr:+.3f}")

    # ── Figure 1: marginal skill bar chart ───────────────────────────────────
    station_labels = [s.replace("_", "\n") for s in station_names]
    fig1, axes = plt.subplots(1, len(leads), figsize=(3 * len(leads), 4.5), sharey=True)
    for ax, lead in zip(axes, leads):
        rs = [corr_table[lead].get(s, np.nan) for s in station_names]
        colors = ["steelblue" if (r >= 0 or np.isnan(r)) else "tomato" for r in rs]
        ax.barh(station_labels, rs, color=colors)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(f"{lead}wk")
        ax.set_xlabel("Pearson r")
        if ax == axes[0]:
            ax.set_ylabel("Station")
    fig1.suptitle(f"DDPM vs NOAA gauge — {split} set (ens={N_MEMBERS})", y=1.01)
    fig1.tight_layout()
    savefig(fig1, OUT_DIR / f"gauge_correlation_{split}.pdf")
    print(f"\nSaved figures/gauge_correlation_{split}.pdf")

    # ── Figure 2: coherence matrices ─────────────────────────────────────────
    ncols = 1 + len(leads)
    fig2, axes2 = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5))
    short = [s.split("_")[0] for s in station_names]

    def _plot_mat(ax, mat, title):
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_yticks(range(len(mat.index)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(title, fontsize=9)
        for (ii, jj), v in np.ndenumerate(mat.values):
            if np.isfinite(v):
                ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="k")
        return im

    im = _plot_mat(axes2[0], obs_corr, "Observed")
    fig2.colorbar(im, ax=axes2[0], shrink=0.7, label="r")
    for ax, l in zip(axes2[1:], leads):
        pred_corr = pairwise_corr_matrix(pred_dfs[l][station_names].dropna(how="all"))
        im = _plot_mat(ax, pred_corr, f"Predicted {l}wk\n(pattern r={pattern_rs[l]:+.2f})")
        fig2.colorbar(im, ax=ax, shrink=0.7)
    fig2.suptitle(f"Pairwise station correlation — observed vs DDPM ensemble mean ({split} set)", y=1.01)
    fig2.tight_layout()
    savefig(fig2, OUT_DIR / f"coherence_matrices_{split}.pdf")
    print(f"Saved figures/coherence_matrices_{split}.pdf")

    # ── Residual correlation matrices ─────────────────────────────────────────
    print("\nResidual correlation matrices (pred - obs)")
    fig3, axes3 = plt.subplots(1, len(leads), figsize=(4 * len(leads), 4.5))
    for ax, l in zip(axes3, leads):
        resid = {}
        for s in station_names:
            if s not in pred_dfs[l].columns:
                continue
            shared = pred_dfs[l].index.intersection(gauge_obs.index)
            r = pred_dfs[l].loc[shared, s] - gauge_obs.loc[shared, s]
            mask = np.isfinite(r)
            resid[s] = r[mask]
        resid_df = pd.DataFrame(resid).dropna(how="all")
        resid_corr = pairwise_corr_matrix(resid_df)
        im = _plot_mat(ax, resid_corr, f"{l}wk residuals")
        fig3.colorbar(im, ax=ax, shrink=0.7)
        vals = resid_corr.values.copy()
        np.fill_diagonal(vals, np.nan)
        print(f"  {l:>2}wk  mean off-diagonal r = {np.nanmean(vals):+.3f}")
    fig3.suptitle(f"Residual (pred − obs) pairwise correlation — {split} set", y=1.01)
    fig3.tight_layout()
    savefig(fig3, OUT_DIR / f"residual_correlation_{split}.pdf")
    print(f"Saved figures/residual_correlation_{split}.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, default=None,
                        help="Path to checkpoint (default: ddpm_best_large.pt)")
    parser.add_argument("--split", type=str,  default="val",
                        help="Dataset split to evaluate: val or test")
    args = parser.parse_args()
    main(args.ckpt, args.split)
