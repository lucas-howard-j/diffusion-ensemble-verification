"""
Figure 1 — SSH marginal vs joint skill of the DDPM (three panels).

(a) Marginal CRPS skill vs lead — station mean
(b) Energy score skill vs lead
(c) Variogram score skill vs lead (p=0.5, equal weights)

All skills are ratio of means, 1 − mean(score)/mean(clim score), against
same-calendar-month GLORYS training-period draws, with 95 % block-bootstrap CIs.
This module also computes the per-init scores (original and shuffled ensembles)
used by figure2.py and ddpm_seed_spread.py.

Reference draws use per-(init, lead) seeds so results are fully reproducible and
independent of processing order.  Scores are cached to results/figure3_scores_{split}.npz;
use --rerun to force recomputation.

Run:
    PYTHONPATH=. python scripts/figure1.py [--split test|val] [--ckpt CKPT] [--rerun]
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

from src.config import CHECKPOINTS, LEAD_WEEKS, TIDE_GAUGES, TRAIN_YEARS
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint
from src.evaluation.metrics import joint_skill_summary
from src.evaluation.scores import crps, energy_score, variogram_score

DEFAULT_CKPT = CHECKPOINTS / "ddpm_best_build2025.pt"   # primary 2025-build DDPM
N_MEMBERS  = 500
BLOCK_SIZE = 3      # weeks; matches ~2–3wk score decorrelation time
N_BOOT     = 1000
CI_ALPHA   = 0.05
OUT_DIR    = Path("figures/main")
CACHE_DIR  = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

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

    torch.manual_seed(0)   # reproducible DDPM sampling
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


# ── Summary (shared with figure2.py and ddpm_seed_spread.py) ──────────────────

def load_scores(split: str, ckpt: Path | None = None, rerun: bool = False) -> dict:
    """Cached per-init scores for `split`; runs inference with `ckpt` if missing or rerun.

    The primary checkpoint uses results/figure3_scores_{split}.npz (legacy name,
    also read by figure2.py); any other checkpoint gets its own cache file, so
    --ckpt never silently returns another model's scores or overwrites Figure 2's.
    """
    ckpt = Path(ckpt) if ckpt is not None else DEFAULT_CKPT
    if ckpt.name == DEFAULT_CKPT.name:
        cache_path = CACHE_DIR / f"figure3_scores_{split}.npz"   # legacy cache name
    else:
        cache_path = CACHE_DIR / f"figure3_scores_{split}_{ckpt.stem}.npz"
    if cache_path.exists() and not rerun:
        cached = dict(np.load(cache_path, allow_pickle=True))
        cached_ckpt = str(cached.get("ckpt", DEFAULT_CKPT.name))   # older caches: primary model
        if cached_ckpt != ckpt.name:
            raise ValueError(f"{cache_path} holds scores for {cached_ckpt}, not {ckpt.name}; "
                             "use --rerun to recompute.")
        print(f"Loading cached scores from {cache_path}")
        return cached
    print("Running inference ...")
    scores = run_inference(ckpt, split)
    scores["ckpt"] = np.array(ckpt.name)
    np.savez(cache_path, **scores)
    print(f"Saved scores to {cache_path}")
    return scores


def summarise(scores: dict) -> dict:
    """Ratio-of-means skills with 95 % block-bootstrap CIs; prints the summary tables.

    Returns leads, crps (skill, lo, hi per lead), and es / vs dicts from
    joint_skill_summary (skill per lead and column [model, model shuffled, clim,
    clim shuffled]; pooled shuffle gaps Δ_model, Δ_clim, Δ_model − Δ_clim).
    """
    crps_ddpm_arr = scores["crps_ddpm"]
    crps_ref_arr  = scores["crps_ref"]
    es_arr, vs_arr = scores["es"], scores["vs"]
    leads   = list(scores["leads"])
    n_leads = len(leads)

    es_ok = np.all(np.isfinite(es_arr), axis=(1, 2))
    vs_ok = np.all(np.isfinite(vs_arr), axis=(1, 2))
    print(f"Valid inits — CRPS: {len(crps_ddpm_arr)}  ES: {es_ok.sum()}  VS: {vs_ok.sum()}")

    rng_boot = np.random.default_rng(42)

    # CRPS: ratio of means per station, then station mean (matches rmse_vs_lead.py)
    crps_skill = (1 - crps_ddpm_arr.mean(0) / crps_ref_arr.mean(0)).mean(-1)
    N = crps_ddpm_arr.shape[0]
    n_blocks, max_start = int(np.ceil(N / BLOCK_SIZE)), max(0, N - BLOCK_SIZE)
    boot_crps = np.empty((N_BOOT, n_leads))
    for b in range(N_BOOT):
        starts = rng_boot.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:N]
        boot_crps[b] = (1 - crps_ddpm_arr[idx].mean(0) / crps_ref_arr[idx].mean(0)).mean(-1)
    crps_lo = np.percentile(boot_crps, 100 * CI_ALPHA / 2, axis=0)
    crps_hi = np.percentile(boot_crps, 100 * (1 - CI_ALPHA / 2), axis=0)

    # ES / VS: same ratio-of-means convention; shuffle gaps pooled over leads
    es = joint_skill_summary(es_arr[es_ok], BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot)
    vs = joint_skill_summary(vs_arr[vs_ok], BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot)

    print("\nCRPS skill (1 − CRPS/CRPS_ref)  [station-mean per lead]:")
    for j, l in enumerate(leads):
        print(f"  {l:>2}wk  {crps_skill[j]:>+.4f}  [{crps_lo[j]:>+.4f}, {crps_hi[j]:>+.4f}]")
    for name, sm in [("Energy score skill  (1 − ES/ES_clim)", es),
                     ("Variogram score skill  (1 − VS/VS_clim, p=0.5)", vs)]:
        print(f"\n{name}:")
        print(f"{'Lead':>6}  {'DDPM':>22}  {'DDPM-shuf':>10}  {'GLORYS-shuf':>12}")
        for j, l in enumerate(leads):
            k = sm["skill"][j]
            print(f"  {l:>2}wk  {k[0]:>+.4f} [{sm['skill_lo'][j, 0]:>+.4f},{sm['skill_hi'][j, 0]:>+.4f}]"
                  f"  {k[1]:>+.4f}  {k[3]:>+.4f}")
    print("\nPooled shuffle gaps (lead-mean ± 95% CI):")
    for score, sm in [("ES", es), ("VS", vs)]:
        for lbl, v, lo, hi in zip(["Δ_DDPM", "Δ_GLORYS", "Δ−Δ_clim"],
                                  sm["delta"], sm["delta_lo"], sm["delta_hi"]):
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {score} {lbl:<10}  {v:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}] {sig}")

    return dict(leads=leads, crps=(crps_skill, crps_lo, crps_hi), es=es, vs=vs)


# ── Figure style (shared with figure2.py) ─────────────────────────────────────

FS_TICK, FS_LABEL, FS_TITLE, FS_LEG = 6, 7, 7, 5.5


def style_lead_ax(ax, leads, ylabel, title):
    ld = np.array(leads)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(ld)
    ax.set_xticklabels([f"{l}wk" for l in leads], fontsize=FS_TICK)
    ax.set_xlabel("Lead time", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontweight="bold", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.2, lw=0.5)


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def make_figure(summ: dict) -> None:
    """Figure 1: marginal vs joint skill of the DDPM — CRPS, ES and VS skill vs lead."""
    leads = summ["leads"]
    ld = np.array(leads)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.16))
    panels = [
        (summ["crps"], "", "(a) CRPS"),
        ((summ["es"]["skill"][:, 0], summ["es"]["skill_lo"][:, 0], summ["es"]["skill_hi"][:, 0]),
         "", "(b) Energy Score"),
        ((summ["vs"]["skill"][:, 0], summ["vs"]["skill_lo"][:, 0], summ["vs"]["skill_hi"][:, 0]),
         "", "(c) Variogram Score"),
    ]
    for ax, ((val, lo, hi), ylabel, title) in zip(axes, panels):
        ax.plot(ld, val, "o-", color=_DDPM_COLOR, lw=1.2, ms=3, label="DDPM")
        ax.fill_between(ld, lo, hi, color=_DDPM_COLOR, alpha=0.18, label="95% CI")
        style_lead_ax(ax, leads, ylabel, title)
        # Figure-1-local font sizes (larger than the shared defaults)
        ax.set_xticklabels([str(l) for l in leads])
        ax.set_xlabel("Lead time (weeks)")
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_fontsize(9)
        ax.title.set_fontsize(9)
        for t in ax.get_xticklabels():
            t.set_fontsize(8)
        ax.legend(fontsize=7, loc="lower left")
    fig.supylabel("Skill Score", fontsize=9, x=0.04)
    # common y-range so the three skills are visually comparable
    lo_all = min(min(p[0][1]) for p in panels)
    hi_all = max(max(p[0][2]) for p in panels)
    pad = 0.1 * (hi_all - lo_all)
    for ax in axes:
        ax.set_ylim(lo_all - pad - 0.03, hi_all + pad)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.savefig(OUT_DIR / "figure1.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "figure1.png", dpi=300, bbox_inches="tight")
    print("\nSaved figures/main/figure1.pdf and figures/main/figure1.png")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, default=None,
                        help=f"Checkpoint for inference (default {DEFAULT_CKPT.name})")
    parser.add_argument("--split", type=str,  default="test", help="val or test")
    parser.add_argument("--rerun", action="store_true",
                        help="Force rerun inference even if cache exists")
    args = parser.parse_args()
    make_figure(summarise(load_scores(args.split, args.ckpt, args.rerun)))


if __name__ == "__main__":
    main()
