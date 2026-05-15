"""
Rank (Talagrand) histograms for the gauge-space DDPM ensemble.

Two figure layouts:
  1. rank_hist_by_lead.pdf   — one panel per lead, ranks aggregated over all stations
  2. rank_hist_by_station.pdf — one panel per station, ranks aggregated over all leads

Truth: GLORYS synthetic gauge values at t+lead (same as energy/variogram score).
Rank of truth among M ensemble members ∈ {1, …, M+1}.  Flat histogram → well-calibrated.

--bias_correct: subtract per-lead, per-station mean ensemble error before ranking,
  isolating spread calibration from marginal bias.

Run:
    PYTHONPATH=. python scripts/rank_histograms.py --ckpt checkpoints/ddpm_best_gauge_init.pt
    PYTHONPATH=. python scripts/rank_histograms.py --ckpt checkpoints/ddpm_best_gauge_init.pt --bias_correct
    PYTHONPATH=. python scripts/rank_histograms.py --ckpt checkpoints/ddpm_best_gauge_init.pt --split test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import LEAD_WEEKS, TIDE_GAUGES, TRAIN_YEARS
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS  = 500
N_PLOT_BINS = 10   # 10 equal percentile bins (0–10%, 10–20%, …, 90–100%)
OUT_DIR = Path("figures/supplement")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def savefig(fig: plt.Figure, stem: Path | str, **kwargs) -> None:
    p = Path(stem)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", **kwargs)
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight", **kwargs)


def _rank(members: np.ndarray, truth: float) -> int:
    """Rank of scalar truth among 1-D member array.  Result in {1, …, M+1}."""
    return int(np.sum(members < truth)) + 1


def coarsen_counts(counts: np.ndarray, n_plot_bins: int) -> np.ndarray:
    """Merge fine rank bins into coarser display bins by summing."""
    n_fine = len(counts)
    edges = np.round(np.linspace(0, n_fine, n_plot_bins + 1)).astype(int)
    return np.array([counts[edges[i]:edges[i + 1]].sum() for i in range(n_plot_bins)])


def plot_rank_hist(
    ax: plt.Axes,
    counts: np.ndarray,
    n_members: int,
    title: str,
    color: str = "steelblue",
) -> None:
    """Draw one rank histogram panel using percentile bins."""
    coarse  = coarsen_counts(counts, N_PLOT_BINS)
    freq    = coarse / coarse.sum()
    uniform = 1.0 / N_PLOT_BINS

    bin_edges = np.linspace(0, 100, N_PLOT_BINS + 1)
    ax.bar(bin_edges[:-1], freq, width=10.0, align="edge",
           color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axhline(uniform, color="k", lw=1.0, ls="--", label="uniform")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(freq.max() * 1.25, uniform * 1.5))
    ax.set_xlabel("Percentile", fontsize=8)
    ax.set_ylabel("Relative frequency", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    n = int(counts.sum())
    ax.annotate(f"n={n}", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=7)


def main(ckpt_path: Path | None = None, split: str = "val",
         bias_correct: bool = False) -> None:
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    gauge_mu   = ckpt["gauge_mu"]
    gauge_std  = ckpt["gauge_std"]
    print(f"Device: {device}  |  split: {split}  |  bias_correct: {bias_correct}")

    ds = SSHDiffusionDataset(split)
    all_gauge_names = [g["name"] for g in TIDE_GAUGES]
    n_gauges = len(all_gauge_names)

    # Build (t → lead → dataset_index) lookup
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx

    leads = sorted(LEAD_WEEKS)
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Init times: {len(valid_ts)}")

    n_st   = n_gauges
    n_fine = N_MEMBERS + 1

    # ── Pass 1: collect raw ensemble + obs in physical space ─────────────────
    # all_ens[j] : list of (M, n_gauges) arrays
    # all_obs[j] : list of (n_gauges,) arrays
    all_ens: dict[int, list] = {j: [] for j in range(len(leads))}
    all_obs: dict[int, list] = {j: [] for j in range(len(leads))}

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(valid_ts):
            if i % 20 == 0:
                print(f"  init {i}/{len(valid_ts)}")

            conds = []
            for lead in leads:
                cond, _ = ds[t_to_lead_idx[t_val][lead]]
                conds.extend([cond] * N_MEMBERS)

            cond_batch = torch.stack(conds).to(device)
            x = torch.randn(len(cond_batch), target_dim, device=device)
            for step in reversed(range(schedule.T)):
                x = schedule.p_sample(model, x, step, cond_batch)

            gauge_z = x.cpu().numpy()   # (n_leads * N_MEMBERS, n_gauges)

            for j, lead in enumerate(leads):
                block_z    = gauge_z[j * N_MEMBERS : (j + 1) * N_MEMBERS]
                block_phys = block_z * gauge_std + gauge_mu              # (M, n_gauges)
                obs_phys   = (ds.gauge_ssh[t_val + lead] * ds.gauge_std
                              + ds.gauge_mu)                              # (n_gauges,)
                if not np.all(np.isfinite(obs_phys)):
                    continue
                all_ens[j].append(block_phys)
                all_obs[j].append(obs_phys)

    # ── Bias correction (per-lead, per-station) ───────────────────────────────
    bias: dict[int, np.ndarray] = {}
    for j, lead in enumerate(leads):
        ens_stack = np.stack(all_ens[j])        # (n_cases, M, n_gauges)
        obs_stack = np.stack(all_obs[j])        # (n_cases, n_gauges)
        ens_mean  = ens_stack.mean(axis=1)      # (n_cases, n_gauges)
        bias[j]   = (ens_mean - obs_stack).mean(axis=0)   # (n_gauges,)
        print(f"  Lead {lead} wk bias (physical): {bias[j].round(3)}")

    # ── Pass 2: compute ranks with optional bias correction ───────────────────
    # counts shape: (n_leads, n_st, n_fine)
    counts = np.zeros((len(leads), n_st, n_fine), dtype=np.int64)

    for j in range(len(leads)):
        for ens_phys, obs_phys in zip(all_ens[j], all_obs[j]):
            if bias_correct:
                ens_phys = ens_phys - bias[j]
            for k in range(n_st):
                r = _rank(ens_phys[:, k], float(obs_phys[k]))
                counts[j, k, r - 1] += 1

    tag = f"{split}_bc" if bias_correct else split
    bc_note = " (bias-corrected)" if bias_correct else ""

    # ── Figure 1: one panel per lead, averaged over stations ─────────────────
    lead_labels = [f"{l} wk" for l in leads]
    fig1, axes1 = plt.subplots(1, len(leads), figsize=(3.2 * len(leads), 3.5),
                                constrained_layout=True)
    if len(leads) == 1:
        axes1 = [axes1]

    for j, (ax, lead_lbl) in enumerate(zip(axes1, lead_labels)):
        c = counts[j].sum(axis=0)
        plot_rank_hist(ax, c, N_MEMBERS, title=f"Lead: {lead_lbl}")

    fig1.suptitle(
        f"Rank histograms — averaged over all stations  ({split} set{bc_note})",
        fontsize=11,
    )
    out1 = OUT_DIR / f"rank_hist_by_lead_{tag}"
    savefig(fig1, out1)
    print(f"Saved {out1}.pdf/.png")

    # ── Figure 2: one panel per station, averaged over leads ─────────────────
    ncols = 4
    nrows = int(np.ceil(n_st / ncols))
    fig2, axes2 = plt.subplots(nrows, ncols,
                                figsize=(3.5 * ncols, 3.2 * nrows),
                                constrained_layout=True)
    axes2_flat = np.array(axes2).ravel()

    short_names = [name.split("_")[0] for name in all_gauge_names]

    for k, (ax, sname) in enumerate(zip(axes2_flat, short_names)):
        c = counts[:, k, :].sum(axis=0)
        plot_rank_hist(ax, c, N_MEMBERS, title=sname)

    for ax in axes2_flat[n_st:]:
        ax.set_visible(False)

    fig2.suptitle(
        f"Rank histograms — averaged over all lead times  ({split} set{bc_note})",
        fontsize=11,
    )
    out2 = OUT_DIR / f"rank_hist_by_station_{tag}"
    savefig(fig2, out2)
    print(f"Saved {out2}.pdf/.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Checkpoint path (default: ddpm_best_large.pt)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--bias_correct", action="store_true",
                        help="Subtract per-lead per-station mean ensemble error before ranking")
    args = parser.parse_args()
    main(args.ckpt, args.split, args.bias_correct)
