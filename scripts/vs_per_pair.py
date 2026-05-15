"""
Per-pair variogram score analysis — test set.

For each station pair (i,j), computes:
  VS_ddpm, VS_shuf, VS_clim, VS_clim_shuf  averaged across inits and leads
  → per-pair skill     = 1 − VS_ddpm / VS_clim
  → per-pair Δ_ddpm    = VS_skill_ddpm − VS_skill_ddpm_shuf
  → per-pair Δ_clim    = VS_skill_clim − VS_skill_clim_shuf
  → per-pair Δ−Δ_clim  = excess joint structure vs climatology

Also prints block-bootstrap CI on the POOLED Δ−Δ_clim restricted to each pair,
so we can see which pairs are individually significant.

Run:
    PYTHONPATH=. python -m scripts.vs_per_pair [--split test]
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

from src.config import LEAD_WEEKS, TIDE_GAUGES, TRAIN_YEARS
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS  = 500
N_BOOT     = 2000
BLOCK_SIZES = [3, 5, 8]
OUT_DIR    = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def _vs_pair(members: np.ndarray, obs: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Per-pair VS contributions.  members: (M, d)  obs: (d,) → (n_pairs,)"""
    d = obs.shape[0]
    out = np.full(d * (d - 1) // 2, np.nan)
    k = 0
    for i in range(d):
        for j in range(i + 1, d):
            if np.isfinite(obs[i]) and np.isfinite(obs[j]):
                obs_diff  = abs(float(obs[i]) - float(obs[j])) ** p
                ens_diffs = np.abs(members[:, i] - members[:, j]) ** p
                out[k] = (obs_diff - ens_diffs.mean()) ** 2
            k += 1
    return out


def _shuffle(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = arr.copy()
    for k in range(arr.shape[1]):
        out[:, k] = arr[rng.permutation(arr.shape[0]), k]
    return out


def _ref_rng(t: int, lead: int) -> np.random.Generator:
    return np.random.default_rng(hash((t, lead)) % (2 ** 31))


def _bb_ci(x: np.ndarray, bs: int, rng: np.random.Generator) -> tuple[float, float]:
    N  = len(x)
    nb = int(np.ceil(N / bs))
    ms = max(0, N - bs)
    boot = np.array([
        x[np.concatenate([np.arange(s, s + bs)
                           for s in rng.integers(0, ms + 1, size=nb)])[:N]].mean()
        for _ in range(N_BOOT)
    ])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main(ckpt_path: Path | None, split: str) -> None:
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    gauge_mu   = ckpt["gauge_mu"]
    gauge_std  = ckpt["gauge_std"]
    print(f"Device: {device}  split: {split}")

    ds = SSHDiffusionDataset(split)
    leads = sorted(LEAD_WEEKS)
    n_leads = len(leads)

    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    N = len(valid_ts)
    print(f"Valid inits: {N}  leads: {leads}")

    station_names = [g["name"] for g in TIDE_GAUGES][:target_dim]
    short = [s.split("_")[0] for s in station_names]
    n_st  = target_dim
    n_pairs = n_st * (n_st - 1) // 2
    pair_labels = [f"{short[i]}–{short[j]}"
                   for i in range(n_st) for j in range(i + 1, n_st)]

    train_mask  = ds.time.year.isin(TRAIN_YEARS)
    glorys_phys = ds.gauge_ssh[train_mask] * gauge_std + gauge_mu
    glorys_time = ds.time[train_mask]
    glorys_pool = {m: glorys_phys[glorys_time.month == m] for m in range(1, 13)}

    # vs_pairs_rec: (N, n_leads, n_pairs, 4) — [orig, shuf, clim, clim_shuf]
    vs_pairs_rec = np.full((N, n_leads, n_pairs, 4), np.nan)
    shuf_rng = np.random.default_rng(42)

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(valid_ts):
            if i % 20 == 0:
                print(f"  {i}/{N}")

            conds = []
            for lead in leads:
                cond, _ = ds[t_to_lead_idx[t_val][lead]]
                conds.extend([cond] * N_MEMBERS)

            cond_batch = torch.stack(conds).to(device)
            x = torch.randn(len(cond_batch), target_dim, device=device)
            for step in reversed(range(schedule.T)):
                x = schedule.p_sample(model, x, step, cond_batch)
            preds = x.cpu().numpy()

            for j, lead in enumerate(leads):
                members = preds[j * N_MEMBERS:(j + 1) * N_MEMBERS] * gauge_std + gauge_mu
                true    = ds.gauge_ssh[t_val + lead] * gauge_std + gauge_mu
                month   = int(ds.time[t_val + lead].month)
                pool    = glorys_pool[month]
                if len(pool) < 2:
                    continue
                rng_ref   = _ref_rng(t_val, lead)
                clim      = pool[rng_ref.choice(len(pool), N_MEMBERS, replace=True)]
                shuf_ddpm = _shuffle(members, shuf_rng)
                shuf_clim = _shuffle(clim,    shuf_rng)

                vs_pairs_rec[i, j, :, 0] = _vs_pair(members,   true)
                vs_pairs_rec[i, j, :, 1] = _vs_pair(shuf_ddpm, true)
                vs_pairs_rec[i, j, :, 2] = _vs_pair(clim,      true)
                vs_pairs_rec[i, j, :, 3] = _vs_pair(shuf_clim, true)

    # Save cache
    cache = Path("results") / f"vs_per_pair_{split}.npz"
    np.savez(cache, vs_pairs=vs_pairs_rec, leads=leads,
             pair_labels=pair_labels, short=short)
    print(f"Saved {cache}")

    _analyse(vs_pairs_rec, leads, pair_labels, short, split)


def _analyse(vs_pairs_rec, leads, pair_labels, short, split):
    n_pairs = len(pair_labels)
    n_leads = len(leads)

    # Per-pair, per-lead: nanmean across inits
    ok = np.all(np.isfinite(vs_pairs_rec), axis=(1, 2, 3))  # (N,)
    a  = vs_pairs_rec[ok]   # (N_ok, n_leads, n_pairs, 4)
    N_ok = a.shape[0]

    # Raw VS differences: positive = DDPM captures more joint structure than shuffled,
    # relative to climatology's advantage.  Avoids dividing by near-zero per-pair VS_clim.
    # delta_raw = (VS_shuf − VS_ddpm) − (VS_clim_shuf − VS_clim)
    delta_ddpm_raw   = a[:, :, :, 1] - a[:, :, :, 0]   # VS_shuf − VS_ddpm  (N, n_leads, n_pairs)
    delta_clim_raw   = a[:, :, :, 3] - a[:, :, :, 2]   # VS_clim_shuf − VS_clim
    delta_excess     = delta_ddpm_raw - delta_clim_raw  # (N, n_leads, n_pairs)
    pooled_lead      = delta_excess.mean(axis=1)        # (N, n_pairs)

    # Per-pair mean raw VS and Δ−Δ_clim (averaged over leads and inits)
    mean_vs_ddpm = a[:, :, :, 0].mean(axis=(0, 1))     # (n_pairs,)
    mean_vs_clim = a[:, :, :, 2].mean(axis=(0, 1))
    mean_skill   = 1.0 - mean_vs_ddpm / (mean_vs_clim + 1e-12)
    mean_de      = pooled_lead.mean(axis=0)             # (n_pairs,)

    print(f"\nPer-pair raw VS Δ−Δ_clim  (raw units; {split} set, N={N_ok} inits)")
    print(f"{'Pair':<22}  {'VS skill':>9}  {'Δ−Δ_raw':>12}  "
          + "  ".join(f"CI b={bs}" for bs in BLOCK_SIZES))

    rng = np.random.default_rng(42)
    ci_results = {}
    for p, label in enumerate(pair_labels):
        x  = pooled_lead[:, p]
        cis = []
        for bs in BLOCK_SIZES:
            lo, hi = _bb_ci(x, bs, rng)
            cis.append((lo, hi))
        ci_results[label] = cis
        ci_str = "  ".join(f"[{lo:+.4f},{hi:+.4f}]" for lo, hi in cis)
        sig3 = "*" if cis[0][0] > 0 else " "
        print(f"  {label:<20}  {mean_skill[p]:>+.4f}   {mean_de[p]:>+.6f}  {sig3}  {ci_str}")

    # Sort by raw Δ−Δ_clim
    order = np.argsort(mean_de)[::-1]
    print(f"\nTop 5 pairs by raw Δ−Δ_clim:")
    for p in order[:5]:
        print(f"  {pair_labels[p]:<22}  Δ−Δ_raw={mean_de[p]:>+.6f}  "
              f"VS skill={mean_skill[p]:>+.4f}")
    print(f"\nBottom 5 pairs by raw Δ−Δ_clim:")
    for p in order[-5:]:
        print(f"  {pair_labels[p]:<22}  Δ−Δ_raw={mean_de[p]:>+.6f}  "
              f"VS skill={mean_skill[p]:>+.4f}")

    # ── Figure: per-pair Δ−Δ_clim ranked ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Δ−Δ_clim ranked bar with block=3 CI
    ax = axes[0]
    ord2 = np.argsort(mean_de)[::-1]
    xs  = np.arange(n_pairs)
    colors = ["#2166ac" if mean_de[p] > 0 else "#d62728" for p in ord2]
    ax.bar(xs, mean_de[ord2], color=colors, alpha=0.8)
    lo3 = np.array([ci_results[pair_labels[p]][0][0] for p in ord2])
    hi3 = np.array([ci_results[pair_labels[p]][0][1] for p in ord2])
    ax.errorbar(xs, mean_de[ord2],
                yerr=[mean_de[ord2] - lo3, hi3 - mean_de[ord2]],
                fmt="none", color="k", capsize=3, lw=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([pair_labels[p] for p in ord2],
                       rotation=55, ha="right", fontsize=6)
    ax.set_ylabel(r"Per-pair VS $\Delta - \Delta_\mathrm{clim}$ (raw units, lead-mean)")
    ax.set_title(f"(a) Per-pair Δ−Δ_clim — {split} set\nerror bars: block=3 bootstrap CI")

    # Right: VS skill vs observed correlation scatter
    # Compute observed pairwise correlation from GLORYS training data
    ds_tmp = SSHDiffusionDataset(split)
    train_mask = ds_tmp.time.year.isin(TRAIN_YEARS)
    glorys_tr  = ds_tmp.gauge_ssh[train_mask] * ds_tmp.gauge_std + ds_tmp.gauge_mu
    obs_corr   = np.corrcoef(glorys_tr[:, :len(short)].T)
    triu_i, triu_j = np.triu_indices(len(short), k=1)
    obs_pair_corr = obs_corr[triu_i, triu_j]

    lead_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_leads))
    ax2 = axes[1]
    for li, (lead, color) in enumerate(zip(leads, lead_colors)):
        vs_d = a[:, li, :, 0].mean(axis=0)   # mean VS_ddpm per pair at this lead
        vs_c = a[:, li, :, 2].mean(axis=0)   # mean VS_clim
        ps   = 1.0 - vs_d / (vs_c + 1e-12)  # pooled-clim-normalised skill
        ax2.scatter(obs_pair_corr, ps, color=color, s=20, alpha=0.8, label=f"{lead}wk")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Observed pairwise Pearson correlation (GLORYS training)")
    ax2.set_ylabel("Per-pair VS skill  (1 − VS$_\\mathrm{DDPM}$ / VS$_\\mathrm{clim}$)")
    ax2.set_title("(b) VS skill vs observed correlation\n(each dot = one station pair)")
    ax2.legend(fontsize=8, title="Lead")

    fig.suptitle(f"Per-pair variogram score analysis — {split} set", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / f"vs_per_pair_{split}"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}.pdf / .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--cached", action="store_true",
                        help="Load from results/vs_per_pair_{split}.npz instead of re-running")
    args = parser.parse_args()

    if args.cached:
        cache = Path("results") / f"vs_per_pair_{args.split}.npz"
        d = np.load(cache, allow_pickle=True)
        _analyse(d["vs_pairs"], list(d["leads"]),
                 list(d["pair_labels"]), list(d["short"]), args.split)
    else:
        main(args.ckpt, args.split)
