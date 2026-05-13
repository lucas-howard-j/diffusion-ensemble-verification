"""
CRPS (and RMSE) skill vs lead time — full val set, scored against GLORYS targets.

CRPS = E[|X-y|] - (1/2)E[|X-X'|]  (proper scoring rule, 1-D per station)
Reference baseline: GLORYS seasonal draw (same-month random draw from training period).
Persistence baseline: predict gauge SSH at t for all lead times.
Skill = 1 - score / score_ref  (positive = beats baseline)

Run:
    PYTHONPATH=. python scripts/rmse_vs_lead.py --ckpt checkpoints/ddpm_best_gauge_init.pt
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def savefig(fig, path, **kwargs):
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", **kwargs)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight", **kwargs)

from src.config import LEAD_WEEKS, TIDE_GAUGES, TRAIN_YEARS
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS  = 25
BLOCK_SIZE = 8     # weeks; accounts for ~2-month autocorrelation in SSH
N_BOOT     = 1000  # bootstrap resamples
CI_ALPHA   = 0.05  # 95 % CI
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def _skill_ci_crps(
    ddpm_rec: np.ndarray,
    ref_rec: np.ndarray,
    block_size: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
    station_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Block bootstrap CI for CRPS skill = 1 − mean(ddpm)/mean(ref).

    Resamples raw score arrays jointly so the CI is centered on exactly the
    same quantity that is plotted (ratio of time-averaged scores), avoiding a
    ratio-of-means vs. mean-of-ratios mismatch.

    ddpm_rec, ref_rec : (n_inits, n_leads, n_st)
    station_mean=True : returns lo, hi of shape (n_leads,)   — mean across stations
    station_mean=False: returns lo, hi of shape (n_leads, n_st) — per-station CI
    """
    N = ddpm_rec.shape[0]
    n_blocks = int(np.ceil(N / block_size))
    max_start = max(0, N - block_size)
    out_shape = (n_boot, ddpm_rec.shape[1]) if station_mean \
                else (n_boot, ddpm_rec.shape[1], ddpm_rec.shape[2])
    boot = np.empty(out_shape)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])[:N]
        skill  = 1 - ddpm_rec[idx].mean(0) / ref_rec[idx].mean(0)  # (n_leads, n_st)
        boot[b] = skill.mean(-1) if station_mean else skill
    lo = np.percentile(boot, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def _skill_ci_rmse(
    sq_rec: np.ndarray,
    clim_sq_rec: np.ndarray,
    block_size: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Block bootstrap CI for station-mean RMSE skill = 1 − rmse(sq)/rmse(clim_sq).

    sq_rec, clim_sq_rec : (n_inits, n_leads, n_st)  — squared errors per init
    Returns lo, hi      : (n_leads,)
    """
    N = sq_rec.shape[0]
    n_blocks = int(np.ceil(N / block_size))
    max_start = max(0, N - block_size)
    boot = np.empty((n_boot, sq_rec.shape[1]))
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])[:N]
        skill  = (
            1 - np.sqrt(sq_rec[idx].mean(0)) / np.sqrt(clim_sq_rec[idx].mean(0))
        ).mean(-1)
        boot[b] = skill
    lo = np.percentile(boot, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def _crps(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    CRPS per station.
    members : (M, n_st)
    obs     : (n_st,)
    returns : (n_st,)
    """
    M = members.shape[0]
    term1 = np.mean(np.abs(members - obs[None, :]), axis=0)
    diff  = np.abs(members[:, None, :] - members[None, :, :])
    off   = ~np.eye(M, dtype=bool)
    term2 = diff[off].reshape(M * (M - 1), -1).mean(axis=0) / 2
    return term1 - term2


def _crps_det(forecast: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS for a deterministic (point) forecast = MAE."""
    return np.abs(forecast - obs)


def main(ckpt_path: Path | None = None, split: str = "val") -> None:
    rng = np.random.default_rng(42)

    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    gauge_mu   = ckpt["gauge_mu"]
    gauge_std  = ckpt["gauge_std"]
    print(f"Device: {device}  |  target_dim: {target_dim}")

    ds = SSHDiffusionDataset(split)

    all_gauge_names = [g["name"] for g in TIDE_GAUGES]
    station_names   = all_gauge_names[:target_dim]
    short           = [s.split("_")[0] for s in station_names]
    n_st = target_dim

    # ── Index val pairs by t ──────────────────────────────────────────────────
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx

    leads = sorted(LEAD_WEEKS)
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Val time steps: {len(valid_ts)}")

    # ── GLORYS seasonal draw pool ─────────────────────────────────────────────
    train_mask  = ds.time.year.isin(TRAIN_YEARS)
    glorys_phys = ds.gauge_ssh[train_mask] * gauge_std + gauge_mu
    glorys_time = ds.time[train_mask]
    glorys_pool: dict[int, np.ndarray] = {
        m: glorys_phys[glorys_time.month == m] for m in range(1, 13)
    }

    # ── Per-init score records — shape will be (n_inits, n_leads, n_st) ──────
    # Storing individual init scores (not just sums) enables block bootstrap CIs.
    crps_ddpm_rec = []   # each entry: (n_leads, n_st)
    crps_ref_rec  = []
    crps_pers_rec = []
    rmse_ddpm_rec = []
    rmse_clim_rec = []
    rmse_pers_rec = []

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(valid_ts):
            if i % 20 == 0:
                print(f"  t {i}/{len(valid_ts)}")

            # Physical gauge SSH at init time t (persistence forecast)
            persist_phys = ds.gauge_ssh[t_val] * gauge_std + gauge_mu   # (n_st,)

            conds, glorys_targets = [], []
            for lead in leads:
                cond, target = ds[t_to_lead_idx[t_val][lead]]
                conds.extend([cond] * N_MEMBERS)
                glorys_targets.append(target.numpy() * gauge_std + gauge_mu)

            cond_batch = torch.stack(conds).to(device)
            x = torch.randn(len(cond_batch), target_dim, device=device)
            for step in reversed(range(schedule.T)):
                x = schedule.p_sample(model, x, step, cond_batch)
            preds = x.cpu().numpy()

            row_crps_ddpm = np.zeros((len(leads), n_st))
            row_crps_ref  = np.zeros((len(leads), n_st))
            row_crps_pers = np.zeros((len(leads), n_st))
            row_rmse_ddpm = np.zeros((len(leads), n_st))
            row_rmse_clim = np.zeros((len(leads), n_st))
            row_rmse_pers = np.zeros((len(leads), n_st))

            for j, lead in enumerate(leads):
                true       = glorys_targets[j]
                block_phys = preds[j*N_MEMBERS:(j+1)*N_MEMBERS] * gauge_std + gauge_mu

                month = ds.time[t_val + lead].month
                pool  = glorys_pool[month]
                ref_members = pool[rng.choice(len(pool), N_MEMBERS, replace=True)]

                row_crps_ddpm[j] = _crps(block_phys,  true)
                row_crps_ref[j]  = _crps(ref_members, true)
                row_crps_pers[j] = _crps_det(persist_phys, true)
                row_rmse_ddpm[j] = (block_phys.mean(axis=0) - true) ** 2
                row_rmse_clim[j] = true ** 2
                row_rmse_pers[j] = (persist_phys - true) ** 2

            crps_ddpm_rec.append(row_crps_ddpm)
            crps_ref_rec.append(row_crps_ref)
            crps_pers_rec.append(row_crps_pers)
            rmse_ddpm_rec.append(row_rmse_ddpm)
            rmse_clim_rec.append(row_rmse_clim)
            rmse_pers_rec.append(row_rmse_pers)

    # Stack to (n_inits, n_leads, n_st)
    crps_ddpm_rec = np.stack(crps_ddpm_rec)
    crps_ref_rec  = np.stack(crps_ref_rec)
    crps_pers_rec = np.stack(crps_pers_rec)
    rmse_ddpm_rec = np.stack(rmse_ddpm_rec)
    rmse_clim_rec = np.stack(rmse_clim_rec)
    rmse_pers_rec = np.stack(rmse_pers_rec)

    # Mean scores — identical to the old sum/n_inits values
    crps_ddpm  = crps_ddpm_rec.mean(axis=0)   # (n_leads, n_st)
    crps_ref   = crps_ref_rec.mean(axis=0)
    crps_pers  = crps_pers_rec.mean(axis=0)
    rmse_ddpm  = np.sqrt(rmse_ddpm_rec.mean(axis=0))
    rmse_clim  = np.sqrt(rmse_clim_rec.mean(axis=0))
    rmse_pers  = np.sqrt(rmse_pers_rec.mean(axis=0))

    crps_skill      = 1 - crps_ddpm / crps_ref
    crps_skill_pers = 1 - crps_pers / crps_ref
    rmse_skill      = 1 - rmse_ddpm / rmse_clim
    rmse_skill_pers = 1 - rmse_pers / rmse_clim


    # ── Block bootstrap CIs — computed once here, used in both figures ────────
    rng_boot = np.random.default_rng(42)
    print(f"\nComputing block bootstrap CIs (block={BLOCK_SIZE}wk, n_boot={N_BOOT}) ...")
    # Station-mean CIs for skill_summary.pdf — shape (n_leads,)
    crps_lo,    crps_hi    = _skill_ci_crps(crps_ddpm_rec, crps_ref_rec,  BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot, station_mean=True)
    pers_lo,    pers_hi    = _skill_ci_crps(crps_pers_rec, crps_ref_rec,  BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot, station_mean=True)
    rmse_lo,    rmse_hi    = _skill_ci_rmse(rmse_ddpm_rec, rmse_clim_rec, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot)
    rpers_lo,   rpers_hi   = _skill_ci_rmse(rmse_pers_rec, rmse_clim_rec, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot)
    # Per-station CIs for crps_skill_vs_lead.pdf — shape (n_leads, n_st)
    crps_lo_st, crps_hi_st = _skill_ci_crps(crps_ddpm_rec, crps_ref_rec,  BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot, station_mean=False)
    print("  Done.")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nCRPS skill (1 − CRPS/CRPS_ref):")
    print(f"{'Lead':>6}  " + "  ".join(f"{s:>8}" for s in short))
    for j, l in enumerate(leads):
        vals = "  ".join(f"{crps_skill[j, k]:>+.3f}" for k in range(n_st))
        print(f"  {l:>2}wk  DDPM  {vals}")
        vals = "  ".join(f"{crps_skill_pers[j, k]:>+.3f}" for k in range(n_st))
        print(f"       Pers  {vals}")

    # ── Figure 1: per-station CRPS skill vs lead ──────────────────────────────
    cmap = plt.cm.tab10
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    for k, name in enumerate(short):
        color = cmap(k / n_st)
        ax1.fill_between(leads, crps_lo_st[:, k], crps_hi_st[:, k],
                         color=color, alpha=0.12)
        ax1.plot(leads, crps_skill[:, k], "o-", color=color,
                 lw=1.5, ms=5, label=name)
    # Persistence: mean across stations as a single reference line
    ax1.plot(leads, crps_skill_pers.mean(axis=1), "k^--",
             lw=1.4, ms=6, label="Persistence (mean)")
    ax1.axhline(0, color="k", lw=0.7, ls=":")
    ax1.set_xticks(leads)
    ax1.set_xlabel("Lead time (weeks)")
    ax1.set_ylabel("CRPS skill  (1 − CRPS / CRPS$_{ref}$)")
    ax1.set_title(
        f"Per-station CRPS skill vs lead — test set\n"
        f"(ref = GLORYS seasonal draw)  —  95 % CI block={BLOCK_SIZE} wk"
    )
    ax1.legend(fontsize=7, ncol=3, loc="lower left")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    savefig(fig1, OUT_DIR / "crps_skill_vs_lead.pdf")
    print("\nSaved figures/crps_skill_vs_lead.pdf")

    # ── Figure 2: mean skill — CRPS and RMSE with bootstrap CIs ─────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    ax = axes2[0]
    ax.plot(leads, crps_skill.mean(axis=1), "o-", color="steelblue",
            lw=1.8, ms=6, label="DDPM")
    ax.fill_between(leads, crps_lo, crps_hi, color="steelblue", alpha=0.18,
                    label=f"95 % CI (block={BLOCK_SIZE}wk)")
    ax.plot(leads, crps_skill_pers.mean(axis=1), "^--", color="darkorange",
            lw=1.5, ms=6, label="Persistence")
    ax.fill_between(leads, pers_lo, pers_hi, color="darkorange", alpha=0.15)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xticks(leads)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("Mean CRPS skill")
    ax.set_title("CRPS skill (mean across stations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(leads, rmse_skill.mean(axis=1), "o-", color="firebrick",
            lw=1.8, ms=6, label="DDPM")
    ax.fill_between(leads, rmse_lo, rmse_hi, color="firebrick", alpha=0.18,
                    label=f"95 % CI (block={BLOCK_SIZE}wk)")
    ax.plot(leads, rmse_skill_pers.mean(axis=1), "^--", color="darkorange",
            lw=1.5, ms=6, label="Persistence")
    ax.fill_between(leads, rpers_lo, rpers_hi, color="darkorange", alpha=0.15)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xticks(leads)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("Mean RMSE skill")
    ax.set_title("RMSE skill (mean across stations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig2.suptitle(
        f"Forecast skill vs lead — val set, GLORYS targets\n"
        f"Shading: 95 % block-bootstrap CI (block={BLOCK_SIZE} wk, n={N_BOOT})",
        y=1.02,
    )
    fig2.tight_layout()
    savefig(fig2, OUT_DIR / "skill_summary.pdf")
    print("Saved figures/skill_summary.pdf")

    # ── Figure 3: per-station RMSE skill with persistence ────────────────────
    fig3, axes3 = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    fig3.suptitle("RMSE skill vs lead — per station  (ref = climatology)",
                  fontsize=11, fontweight="bold")
    for si, ax in enumerate(axes3.flat):
        ax.plot(leads, rmse_skill[:, si], "o-", color="firebrick",
                lw=1.5, ms=5, label="DDPM")
        ax.plot(leads, rmse_skill_pers[:, si], "^--", color="darkorange",
                lw=1.4, ms=5, label="Persistence")
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_title(short[si], fontsize=9)
        ax.set_xticks(leads)
        ax.set_xlabel("Lead (wk)", fontsize=8)
        ax.set_ylabel("RMSE skill", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if si == 0:
            ax.legend(fontsize=7)
    savefig(fig3, OUT_DIR / "rmse_skill_vs_lead.pdf")
    print("Saved figures/rmse_skill_vs_lead.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Path to checkpoint (default: ddpm_best_large.pt)")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to evaluate: val or test")
    args = parser.parse_args()
    main(args.ckpt, args.split)
