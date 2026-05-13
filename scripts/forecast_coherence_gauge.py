"""
Forecast spatial coherence vs lead time — gauge-space model variant.

DDPM output is SSH anomaly at gauge locations directly (no EOF reconstruction).
Random EOF baseline is still computed via EOF reconstruction so the paper
comparison remains valid: DDPM (gauge-space) vs what the EOF basis alone imposes.

Three lines per panel:
  DDPM          — ensemble mean gauge SSH (steelblue)
  Random EOF    — 25 random N(0,1) PCs → EOF reconstruction → gauge (orange)
  Observed      — actual gauge correlations during val period (dashed black)

95% CI from block bootstrap over val time steps.

Run:
    PYTHONPATH=. python scripts/forecast_coherence_gauge.py
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

from src.config import LEAD_WEEKS, TIDE_GAUGES, N_SSH_MODES
from src.data.gauge_loader import load_gauges
from src.eof.reconstruct import reconstruct_field
from src.evaluation.gauge_comparison import build_gauge_series
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS = 25
N_BOOT    = 500
SEED      = 42
OUT_DIR   = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def _mean_off_diag(mat: np.ndarray, i: int) -> float:
    row = mat[i].copy()
    row[i] = np.nan
    return float(np.nanmean(row))


def _bootstrap_mean_coherence(data: np.ndarray, n_boot: int,
                               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_t, n_st = data.shape
    boot = np.zeros((n_boot, n_st))
    for b in range(n_boot):
        idx  = rng.integers(0, n_t, size=n_t)
        corr = np.corrcoef(data[idx].T)
        boot[b] = [_mean_off_diag(corr, i) for i in range(n_st)]
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def main() -> None:
    rng = np.random.default_rng(SEED)

    model, schedule, ckpt, device = load_checkpoint()
    target_dim  = int(ckpt["target_dim"])
    gauge_mu    = ckpt["gauge_mu"]    # (n_gauges,)
    gauge_std   = ckpt["gauge_std"]   # (n_gauges,)
    ssh_mu      = ckpt["ssh_mu"]      # kept for random EOF baseline
    ssh_std     = ckpt["ssh_std"]
    print(f"Device: {device}")

    ds = SSHDiffusionDataset("val")
    all_gauge_names = [g["name"] for g in TIDE_GAUGES]

    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx

    leads = sorted(LEAD_WEEKS)
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Val time steps: {len(valid_ts)}")

    gauge_obs = load_gauges()
    station_names = [n for n in all_gauge_names if n in gauge_obs.columns]
    station_indices = [all_gauge_names.index(n) for n in station_names]
    n_st = len(station_names)

    def _rand_eof_gauge(tgt_time: pd.Timestamp) -> np.ndarray:
        """N(0,1) PCs → EOF reconstruction → ensemble mean gauge row (n_st,)."""
        rand_pcs_z = rng.standard_normal((N_MEMBERS, N_SSH_MODES)).astype(np.float32)
        rand_phys  = rand_pcs_z * ssh_std[:N_SSH_MODES] + ssh_mu[:N_SSH_MODES]
        ens_mean   = rand_phys.mean(axis=0, keepdims=True)
        ssh_field  = reconstruct_field(ens_mean, "ssh")
        gdf        = build_gauge_series(ssh_field, pd.DatetimeIndex([tgt_time]))
        return np.array([gdf[s].iloc[0] if s in gdf.columns else np.nan
                         for s in station_names], dtype=float)

    # Collect (n_t, n_st) ensemble-mean gauge values per lead
    ddpm_gauge: dict[int, list[np.ndarray]] = {l: [] for l in leads}
    rand_gauge: dict[int, list[np.ndarray]] = {l: [] for l in leads}

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
            gauge_z = x.cpu().numpy()   # (n_leads*N_MEMBERS, n_gauges)

            for j, lead in enumerate(leads):
                block_z   = gauge_z[j * N_MEMBERS : (j + 1) * N_MEMBERS]
                block_phys = block_z * gauge_std + gauge_mu          # (25, n_gauges)
                ens_mean   = block_phys[:, station_indices].mean(axis=0)
                ddpm_gauge[lead].append(ens_mean)
                rand_gauge[lead].append(_rand_eof_gauge(target_times[j]))

    # ── Observed coherence ────────────────────────────────────────────────────
    all_dates = sorted(set(
        ds.time[t + l].to_period("M").to_timestamp()
        for t in valid_ts for l in leads
    ))
    obs_val  = gauge_obs.reindex(pd.DatetimeIndex(all_dates))[station_names]
    obs_corr = obs_val.corr().values
    obs_coh  = np.array([_mean_off_diag(obs_corr, i) for i in range(n_st)])

    # ── Per-lead coherence + bootstrap CI ────────────────────────────────────
    n_leads = len(leads)
    store = {
        "ddpm": (ddpm_gauge, np.zeros((n_leads, n_st)),
                 np.zeros((n_leads, n_st)), np.zeros((n_leads, n_st))),
        "rand": (rand_gauge, np.zeros((n_leads, n_st)),
                 np.zeros((n_leads, n_st)), np.zeros((n_leads, n_st))),
    }
    for key, (src, mean_arr, lo_arr, hi_arr) in store.items():
        for li, lead in enumerate(leads):
            data  = np.stack(src[lead])
            valid = ~np.isnan(data).all(axis=0)
            d     = data[:, valid]
            corr  = np.corrcoef(d.T)
            lo, hi = _bootstrap_mean_coherence(d, N_BOOT, rng)
            vi = 0
            for si in range(n_st):
                if valid[si]:
                    mean_arr[li, si] = _mean_off_diag(corr, vi)
                    lo_arr[li, si]   = lo[vi]
                    hi_arr[li, si]   = hi[vi]
                    vi += 1
                else:
                    mean_arr[li, si] = lo_arr[li, si] = hi_arr[li, si] = np.nan

    # ── Plot ─────────────────────────────────────────────────────────────────
    short = [s.replace("_", " ") for s in station_names]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True, sharex=True)
    axes = axes.ravel()

    cfg = {
        "ddpm": ("steelblue",  "DDPM ens. mean (gauge-space)"),
        "rand": ("darkorange", "Random EOF baseline"),
    }

    for si, (ax, label) in enumerate(zip(axes, short)):
        for key, (color, lbl) in cfg.items():
            _, mean_arr, lo_arr, hi_arr = store[key]
            ax.plot(leads, mean_arr[:, si], "o-", color=color, lw=1.8, label=lbl)
            ax.fill_between(leads, lo_arr[:, si], hi_arr[:, si],
                            color=color, alpha=0.2)
        ax.axhline(obs_coh[si], color="k", lw=1.3, ls="--", label="Observed")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(leads)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_ylim(-0.3, 1.05)
        if si % 4 == 0:
            ax.set_ylabel("Mean coherence (r)", fontsize=8)
        if si >= 4:
            ax.set_xlabel("Lead (weeks)", fontsize=8)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle(
        "Forecast spatial coherence vs lead time  —  val set  "
        f"(ens={N_MEMBERS}, 95% CI shaded, n_boot={N_BOOT})",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "forecast_coherence_gauge_vs_lead.pdf"
    savefig(fig, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
