"""
Figure 3: SSH central results — CRPS skill, Energy Score skill, Variogram Score skill.

Panels:
  (a) CRPS skill per station × lead (horizontal grouped bar chart)
  (b) ES skill vs lead: DDPM / DDPM-shuffled / GLORYS-shuffled  (line + CI)
  (c) VS skill vs lead: same three lines                         (line + CI)

Results are cached to figures/figure1_cache.npz so only the first run is slow.
Delete the cache file to force a recompute.

Run:
    PYTHONPATH=. python scripts/plots/figure1_results.py \
        --ckpt checkpoints/ddpm_best_gauge_init.pt --split test
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
from src.evaluation.metrics import block_bootstrap_ci
from src.evaluation.scores import crps, energy_score, variogram_score

N_MEMBERS  = 500
BLOCK_SIZE = 8
N_BOOT     = 1000
CI_ALPHA   = 0.05
OUT_DIR    = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

LEADS = sorted(LEAD_WEEKS)
GAUGE_NAMES = [g["name"] for g in TIDE_GAUGES]
SHORT = [g["name"].split("_")[0] for g in TIDE_GAUGES]

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(ckpt_path: Path, split: str) -> dict:
    rng = np.random.default_rng(42)

    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    gauge_mu   = ckpt["gauge_mu"]
    gauge_std  = ckpt["gauge_std"]
    print(f"Device: {device}  |  split: {split}  |  target_dim: {target_dim}")

    ds = SSHDiffusionDataset(split)

    # GLORYS seasonal pool
    train_mask     = ds.time.year.isin(TRAIN_YEARS)
    glorys_phys    = ds.gauge_ssh[train_mask] * ds.gauge_std + ds.gauge_mu
    glorys_time    = ds.time[train_mask]
    glorys_pool    = {m: glorys_phys[glorys_time.month == m] for m in range(1, 13)}

    # Valid init times
    t_to_lead: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead[t][lead] = idx
    valid_ts = sorted(t for t, ld in t_to_lead.items() if all(l in ld for l in LEADS))
    print(f"Init times: {len(valid_ts)}")

    n_st = target_dim
    # Per-init records: (n_inits, n_leads) for ES/VS; (n_inits, n_leads, n_st) for CRPS
    crps_ddpm_rec = np.zeros((len(valid_ts), len(LEADS), n_st))
    crps_ref_rec  = np.zeros((len(valid_ts), len(LEADS), n_st))
    es_rec        = np.zeros((len(valid_ts), len(LEADS), 3))   # DDPM, DDPM-shuf, GLORYS-shuf
    vs_rec        = np.zeros((len(valid_ts), len(LEADS), 3))

    model.eval()
    with torch.no_grad():
        for ii, t_val in enumerate(valid_ts):
            if ii % 20 == 0:
                print(f"  {ii}/{len(valid_ts)}")

            conds, targets = [], []
            for lead in LEADS:
                cond, tgt = ds[t_to_lead[t_val][lead]]
                conds.extend([cond] * N_MEMBERS)
                targets.append(tgt.numpy() * gauge_std + gauge_mu)

            cond_batch = torch.stack(conds).to(device)
            x = torch.randn(len(cond_batch), target_dim, device=device)
            for step in reversed(range(schedule.T)):
                x = schedule.p_sample(model, x, step, cond_batch)
            preds = x.cpu().numpy()

            for ji, lead in enumerate(LEADS):
                true        = targets[ji]                                           # (n_st,)
                members     = preds[ji * N_MEMBERS:(ji + 1) * N_MEMBERS] * gauge_std + gauge_mu

                month       = ds.time[t_val + lead].month
                pool        = glorys_pool[month]
                ref_members = pool[rng.choice(len(pool), N_MEMBERS, replace=True)]

                # CRPS
                crps_ddpm_rec[ii, ji] = crps(members, true)
                crps_ref_rec[ii, ji]  = crps(ref_members, true)

                # Shuffled members (destroy joint structure)
                shuf = members.copy()
                for k in range(n_st):
                    shuf[:, k] = members[rng.permutation(N_MEMBERS), k]
                ref_shuf = ref_members.copy()
                for k in range(n_st):
                    ref_shuf[:, k] = ref_members[rng.permutation(N_MEMBERS), k]

                # ES
                es_rec[ii, ji, 0] = energy_score(members,     true)
                es_rec[ii, ji, 1] = energy_score(shuf,        true)
                es_rec[ii, ji, 2] = energy_score(ref_shuf,    true)

                # VS
                vs_rec[ii, ji, 0] = variogram_score(members,  true)
                vs_rec[ii, ji, 1] = variogram_score(shuf,     true)
                vs_rec[ii, ji, 2] = variogram_score(ref_shuf, true)

    # ES/VS clim reference (unshuffled GLORYS draw) — separate pass for skill denom
    es_clim = np.zeros((len(valid_ts), len(LEADS)))
    vs_clim = np.zeros((len(valid_ts), len(LEADS)))
    rng2    = np.random.default_rng(42)
    with torch.no_grad():
        for ii, t_val in enumerate(valid_ts):
            for ji, lead in enumerate(LEADS):
                month       = ds.time[t_val + lead].month
                pool        = glorys_pool[month]
                ref_members = pool[rng2.choice(len(pool), N_MEMBERS, replace=True)]
                es_clim[ii, ji] = energy_score(ref_members, targets[ji])
                vs_clim[ii, ji] = variogram_score(ref_members, targets[ji])

    return dict(
        crps_ddpm=crps_ddpm_rec, crps_ref=crps_ref_rec,
        es=es_rec, es_clim=es_clim,
        vs=vs_rec, vs_clim=vs_clim,
        n_st=n_st,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

LEAD_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]   # 2/4/8/12wk
LINE_COLORS  = {"DDPM": "#1f77b4", "DDPM shuf": "#1f77b4", "GLORYS shuf": "0.5"}
LINE_STYLES  = {"DDPM": "-", "DDPM shuf": "--", "GLORYS shuf": "--"}
LINE_MARKERS = {"DDPM": "o", "DDPM shuf": "s", "GLORYS shuf": "^"}


def plot(data: dict, out_path: Path) -> None:
    rng = np.random.default_rng(42)

    crps_skill = 1 - data["crps_ddpm"].mean(0) / data["crps_ref"].mean(0)  # (n_leads, n_st)
    n_st       = data["n_st"]
    short      = SHORT[:n_st]

    # ES/VS skill per init: (n_inits, n_leads)
    es_clim = data["es_clim"]
    vs_clim = data["vs_clim"]
    # 3 lines: DDPM, DDPM-shuf, GLORYS-shuf
    es_skill = {
        "DDPM":       1 - data["es"][:, :, 0] / (es_clim + 1e-12),
        "DDPM shuf":  1 - data["es"][:, :, 1] / (es_clim + 1e-12),
        "GLORYS shuf":1 - data["es"][:, :, 2] / (es_clim + 1e-12),
    }
    vs_skill = {
        "DDPM":       1 - data["vs"][:, :, 0] / (vs_clim + 1e-12),
        "DDPM shuf":  1 - data["vs"][:, :, 1] / (vs_clim + 1e-12),
        "GLORYS shuf":1 - data["vs"][:, :, 2] / (vs_clim + 1e-12),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             gridspec_kw={"width_ratios": [1.6, 1, 1]})

    # ── Panel (a): CRPS skill grouped horizontal bar chart ────────────────────
    ax = axes[0]
    n_leads = len(LEADS)
    bar_h   = 0.18
    gap     = 0.06
    group_h = n_leads * bar_h + gap

    y_centers = np.arange(n_st) * group_h
    offsets   = np.linspace(-(n_leads - 1) / 2, (n_leads - 1) / 2, n_leads) * bar_h

    for li, (lead, color) in enumerate(zip(LEADS, LEAD_COLORS)):
        vals = crps_skill[li]   # (n_st,)
        ys   = y_centers + offsets[li]
        ax.barh(ys, vals, height=bar_h * 0.85, color=color,
                label=f"{lead} wk", zorder=3)

    ax.axvline(0, color="k", lw=0.8, zorder=4)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("CRPS skill  (1 − CRPS / CRPS$_{ref}$)", fontsize=8.5)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.legend(fontsize=7.5, title="Lead", title_fontsize=7.5,
              loc="lower right", framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.set_title("(a)  Marginal CRPS skill", fontsize=9, fontweight="bold", pad=5)

    # ── Panels (b) and (c): ES and VS skill ───────────────────────────────────
    for ax, skill_dict, score_name, panel_label in [
        (axes[1], es_skill, "Energy score", "(b)"),
        (axes[2], vs_skill, "Variogram score", "(c)"),
    ]:
        for label, vals in skill_dict.items():
            mean_skill = vals.mean(axis=0)   # (n_leads,)
            lo, hi = block_bootstrap_ci(vals, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng)
            ax.plot(LEADS, mean_skill,
                    color=LINE_COLORS[label], ls=LINE_STYLES[label],
                    marker=LINE_MARKERS[label], ms=5, lw=1.6, label=label, zorder=3)
            ax.fill_between(LEADS, lo, hi,
                            color=LINE_COLORS[label], alpha=0.15, zorder=2)

        ax.axhline(0, color="k", lw=0.8, ls="--", zorder=4)
        ax.set_xticks(LEADS)
        ax.set_xticklabels([f"{l}w" for l in LEADS], fontsize=8)
        ax.set_xlabel("Lead time (weeks)", fontsize=8.5)
        ax.set_ylabel(f"{score_name} skill", fontsize=8.5)
        ax.legend(fontsize=7.5, framealpha=0.9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_title(f"{panel_label}  {score_name} skill", fontsize=9,
                     fontweight="bold", pad=5)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path.with_suffix('.pdf')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, default=Path("checkpoints/ddpm_best_gauge_init.pt"))
    parser.add_argument("--split", type=str,  default="test")
    parser.add_argument("--nocache", action="store_true",
                        help="Ignore existing cache and rerun inference")
    args = parser.parse_args()

    cache_path = OUT_DIR / f"figure1_cache_{args.split}.npz"

    if cache_path.exists() and not args.nocache:
        print(f"Loading cached results from {cache_path}")
        raw  = np.load(cache_path)
        data = {k: raw[k] for k in raw.files}
        data["n_st"] = int(data["n_st"])
    else:
        data = run_inference(args.ckpt, args.split)
        save_dict = {k: (np.array([v]) if isinstance(v, int) else v)
                     for k, v in data.items()}
        np.savez(cache_path, **save_dict)
        print(f"Cached to {cache_path}")

    plot(data, OUT_DIR / "figure1")


if __name__ == "__main__":
    main()
