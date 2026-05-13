"""
Spatial coherence of DDPM forecasts vs random EOF baseline as a function of
lead time.

For each lead, collects ensemble-mean predicted gauge SSH across all val time
steps to form a (n_t, n_stations) matrix, then computes pairwise Pearson
correlations between stations. "Coherence" for station i = mean r with all
other stations.

Three lines per panel:
  DDPM          — ensemble mean predictions (steelblue)
  Random EOF    — 25 random N(0,1) PCs → ensemble mean (orange); isolates
                  the coherence imposed by the EOF basis alone
  Observed      — actual gauge correlations during val period (dashed black)

95% CI from block bootstrap over val time steps.

Run:
    conda run -n ssh_ml python -m scripts.forecast_coherence
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
    """
    Bootstrap 95% CI on mean off-diagonal correlation per station.
    data : (n_t, n_stations)
    Returns lo, hi each shape (n_stations,).
    """
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
    target_dim = int(ckpt["target_dim"])
    ssh_mu  = ckpt["ssh_mu"]
    ssh_std = ckpt["ssh_std"]
    print(f"Device: {device}")

    ds = SSHDiffusionDataset("val")
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx

    leads = sorted(LEAD_WEEKS)
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Val time steps: {len(valid_ts)}")

    gauge_obs = load_gauges()
    station_names = [g["name"] for g in TIDE_GAUGES if g["name"] in gauge_obs.columns]
    n_st = len(station_names)

    # Collect (n_t, n_st) ensemble-mean gauge values per lead
    ddpm_gauge: dict[int, list[np.ndarray]] = {l: [] for l in leads}
    rand_gauge: dict[int, list[np.ndarray]] = {l: [] for l in leads}

    def _gauge_from_pcs(pcs_block_z: np.ndarray, tgt_time: pd.Timestamp) -> np.ndarray:
        """Ensemble mean PCs → denorm → reconstruct → gauge row (n_st,)."""
        ens_mean_z   = pcs_block_z.mean(axis=0, keepdims=True)
        ens_phys     = ens_mean_z * ssh_std[:target_dim] + ssh_mu[:target_dim]
        ssh_field    = reconstruct_field(ens_phys, "ssh")
        gdf          = build_gauge_series(ssh_field, pd.DatetimeIndex([tgt_time]))
        return np.array([gdf[s].iloc[0] if s in gdf.columns else np.nan
                         for s in station_names], dtype=float)

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
            pcs_z = x.cpu().numpy()

            # Random baseline: N(0,1) PCs, same shape
            rand_z = rng.standard_normal(pcs_z.shape).astype(np.float32)

            for j, lead in enumerate(leads):
                sl = slice(j * N_MEMBERS, (j + 1) * N_MEMBERS)
                ddpm_gauge[lead].append(_gauge_from_pcs(pcs_z[sl],   target_times[j]))
                rand_gauge[lead].append(_gauge_from_pcs(rand_z[sl],  target_times[j]))

    # ── Observed coherence (val period, lead-independent) ────────────────────
    all_dates = sorted(set(
        ds.time[t + l].to_period("M").to_timestamp()
        for t in valid_ts for l in leads
    ))
    obs_val  = gauge_obs.reindex(pd.DatetimeIndex(all_dates))[station_names]
    obs_corr = obs_val.corr().values
    obs_coh  = np.array([_mean_off_diag(obs_corr, i) for i in range(n_st)])

    # ── Per-lead pairwise correlation + bootstrap CI ──────────────────────────
    n_leads = len(leads)
    store = {
        "ddpm": (ddpm_gauge, np.zeros((n_leads, n_st)),
                 np.zeros((n_leads, n_st)), np.zeros((n_leads, n_st))),
        "rand": (rand_gauge, np.zeros((n_leads, n_st)),
                 np.zeros((n_leads, n_st)), np.zeros((n_leads, n_st))),
    }

    for key, (src, mean_arr, lo_arr, hi_arr) in store.items():
        for li, lead in enumerate(leads):
            data = np.stack(src[lead])           # (n_t, n_st)
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
        "ddpm": ("steelblue",  "DDPM ens. mean"),
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
    out = OUT_DIR / "forecast_coherence_vs_lead.pdf"
    savefig(fig, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
