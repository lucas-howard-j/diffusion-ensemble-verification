"""
Multivariate rank histograms for the gauge-space DDPM ensemble.

Uses the average-distance pre-rank function (energy-score family):
  s(z) = (1/M) * sum_{j != z} ||z - x_j||  in normalised 8-D gauge space.
  (symmetric, leave-one-out over the M+1 augmented set)

Rank of truth among M+1 pre-ranked values -> {1,...,M+1}.
  Flat  -> jointly calibrated
  U-shaped -> joint underdispersion (ensemble too tightly clustered)
  Hump     -> joint overdispersion

--bias_correct: subtract per-lead, per-station mean ensemble error (computed on
  the same split) before ranking, isolating joint dispersion from marginal bias.

Run:
    PYTHONPATH=. python scripts/rank_histograms_mv.py [--split test] [--bias_correct]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import LEAD_WEEKS, TIDE_GAUGES
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS   = 500
N_PLOT_BINS = 10   # 10 equal percentile bins (0–10%, 10–20%, …, 90–100%)
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def coarsen_counts(counts: np.ndarray, n_plot_bins: int) -> np.ndarray:
    """Merge fine rank bins into coarser display bins by summing."""
    n_fine = len(counts)
    edges = np.round(np.linspace(0, n_fine, n_plot_bins + 1)).astype(int)
    return np.array([counts[edges[i]:edges[i + 1]].sum() for i in range(n_plot_bins)])


# ── Multivariate pre-rank ──────────────────────────────────────────────────────

def mv_rank(obs_z: np.ndarray, ens_z: np.ndarray) -> int:
    """
    Multivariate rank of obs_z using average-distance pre-rank.

    obs_z : (d,)   observation in normalised space
    ens_z : (M, d) ensemble members in normalised space

    Returns rank in {1, ..., M+1}.  Under exchangeability -> Uniform{1,...,M+1}.
    """
    M = ens_z.shape[0]
    aug  = np.concatenate([obs_z[None], ens_z], axis=0)   # (M+1, d)
    diff = aug[:, None, :] - aug[None, :, :]               # (M+1, M+1, d)
    D    = np.sqrt((diff ** 2).sum(-1))                    # (M+1, M+1)
    np.fill_diagonal(D, 0.0)
    s    = D.sum(1) / M                                    # avg dist to M others
    return int(np.sum(s[1:] < s[0])) + 1


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_mv_hist(ax: plt.Axes, counts: np.ndarray, n_members: int,
                 title: str, color: str = "steelblue") -> None:
    coarse  = coarsen_counts(counts, N_PLOT_BINS)
    freq    = coarse / coarse.sum()
    uniform = 1.0 / N_PLOT_BINS

    bin_edges = np.linspace(0, 100, N_PLOT_BINS + 1)
    ax.bar(bin_edges[:-1], freq, width=10.0, align="edge",
           color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axhline(uniform, color="k", lw=1.0, ls="--", label="uniform")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(freq.max() * 1.3, uniform * 1.6))
    ax.set_xlabel("Percentile", fontsize=8)
    ax.set_ylabel("Relative frequency", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.annotate(f"n={int(counts.sum())}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=7)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(ckpt_path: Path | None = None, split: str = "val",
         bias_correct: bool = False) -> None:
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    print(f"Device: {device}  |  split: {split}  |  target_dim: {target_dim}"
          f"  |  bias_correct: {bias_correct}")

    ds     = SSHDiffusionDataset(split)
    leads  = sorted(LEAD_WEEKS)
    n_bins = N_MEMBERS + 1

    # Build (t -> lead -> dataset_index) lookup
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx

    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Init times: {len(valid_ts)}")

    # ── Pass 1: generate samples, store raw data ───────────────────────────────
    # all_ens[j] : list of (M, d) arrays, one per valid init
    # all_obs[j] : list of (d,)  arrays, one per valid init
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

            ens_z_all = x.cpu().numpy()   # (n_leads * N_MEMBERS, d)

            for j, lead in enumerate(leads):
                ens_z = ens_z_all[j * N_MEMBERS : (j + 1) * N_MEMBERS]
                obs_z = ds.gauge_ssh[t_val + lead]

                if not np.all(np.isfinite(obs_z)):
                    continue

                all_ens[j].append(ens_z)
                all_obs[j].append(obs_z)

    # ── Bias correction (per-lead, per-station) ────────────────────────────────
    bias: dict[int, np.ndarray] = {}
    for j, lead in enumerate(leads):
        ens_stack = np.stack(all_ens[j])   # (n_cases, M, d)
        obs_stack = np.stack(all_obs[j])   # (n_cases, d)
        ens_mean  = ens_stack.mean(axis=1) # (n_cases, d)
        bias[j]   = (ens_mean - obs_stack).mean(axis=0)   # (d,)  mean error per station
        print(f"  Lead {lead} wk bias (normalised): {bias[j].round(3)}")

    # ── Pass 2: rank with optional bias correction ─────────────────────────────
    counts = np.zeros((len(leads), n_bins), dtype=np.int64)

    for j in range(len(leads)):
        for ens_z, obs_z in zip(all_ens[j], all_obs[j]):
            if bias_correct:
                ens_z = ens_z - bias[j]   # shift ensemble so mean error = 0
            r = mv_rank(obs_z, ens_z)
            counts[j, r - 1] += 1

    # ── Figures ───────────────────────────────────────────────────────────────
    lead_labels = [f"Lead {l} wk" for l in leads]
    tag = f"{split}_bc" if bias_correct else split

    fig, axes = plt.subplots(1, len(leads), figsize=(3.2 * len(leads), 3.5),
                             constrained_layout=True)
    if len(leads) == 1:
        axes = [axes]

    for j, (ax, lbl) in enumerate(zip(axes, lead_labels)):
        plot_mv_hist(ax, counts[j], N_MEMBERS, title=lbl)

    bc_note = " (bias-corrected)" if bias_correct else ""
    fig.suptitle(
        f"Multivariate rank histograms — all 8 stations jointly  ({split} set{bc_note})",
        fontsize=10,
    )

    stem = OUT_DIR / f"rank_hist_mv_{tag}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {stem}.pdf/.png")

    pooled = counts.sum(axis=0)
    fig2, ax2 = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
    plot_mv_hist(ax2, pooled, N_MEMBERS,
                 title=f"Multivariate rank — pooled over all leads  ({split} set{bc_note})")
    stem2 = OUT_DIR / f"rank_hist_mv_pooled_{tag}"
    fig2.savefig(stem2.with_suffix(".pdf"), bbox_inches="tight")
    fig2.savefig(stem2.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {stem2}.pdf/.png")

    n_total       = int(pooled.sum())
    uniform       = n_total / n_bins
    expected_tail = 3 * uniform
    low_tail      = int(pooled[:3].sum())
    high_tail     = int(pooled[-3:].sum())
    print(f"\nPooled summary (n={n_total}{', bias-corrected' if bias_correct else ''}):")
    print(f"  Low tail  (ranks 1–3):    {low_tail}  (expected {expected_tail:.0f})")
    print(f"  High tail (ranks {n_bins-2}–{n_bins}): {high_tail}  (expected {expected_tail:.0f})")
    if high_tail > low_tail * 1.3 and low_tail < expected_tail:
        print("  Shape: U-skewed high -> high-tail excess")
    elif low_tail > high_tail * 1.3:
        print("  Shape: U-skewed low -> overdispersion")
    elif high_tail + low_tail > expected_tail * 2 * 1.3:
        print("  Shape: U-shaped -> underdispersion")
    else:
        print("  Shape: approximately flat -> near joint calibration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         type=Path, default=Path("checkpoints/ddpm_best_gauge_init.pt"))
    parser.add_argument("--split",        type=str,  default="val", choices=["val", "test"])
    parser.add_argument("--bias_correct", action="store_true",
                        help="Subtract per-lead per-station mean ensemble error before ranking")
    args = parser.parse_args()
    main(args.ckpt, args.split, args.bias_correct)
