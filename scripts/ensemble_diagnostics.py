"""
Two diagnostics testing whether DDPM ensemble members capture the right
joint spatial structure:

1. Within-ensemble gauge covariance vs observed covariance.
2. Spread-error calibration per station and lead.

Model output is SSH anomaly at gauge locations directly (no EOF reconstruction).

Run:
    PYTHONPATH=. python scripts/ensemble_diagnostics.py --ckpt checkpoints/ddpm_best_gauge_init.pt
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import LEAD_WEEKS, TIDE_GAUGES, TRAIN_YEARS
from src.data.gauge_loader import load_gauges
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint
from src.evaluation.metrics import block_bootstrap_ci

N_MEMBERS  = 25
BLOCK_SIZE = 8     # weeks; moving-block bootstrap block length
N_BOOT     = 1000  # bootstrap resamples
CI_ALPHA   = 0.05  # 95 % CI
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def savefig(fig, path, **kwargs):
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", **kwargs)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight", **kwargs)


def _variogram_score(members: np.ndarray, obs: np.ndarray, p: float = 0.5) -> float:
    """Variogram score (Scheuerer & Hamill 2015) for one forecast.
    members : (M, d)   obs : (d,)
    Equal weights w_ij = 1/n_pairs across all i<j station pairs.
    Lower is better; skill = 1 - VS/VS_ref.
    """
    valid = np.isfinite(obs)
    if valid.sum() < 2:
        return np.nan
    m, y = members[:, valid], obs[valid]
    d = y.shape[0]
    vs = 0.0
    n_pairs = 0
    for i in range(d):
        for j in range(i + 1, d):
            obs_diff  = abs(float(y[i]) - float(y[j])) ** p
            ens_diffs = np.abs(m[:, i] - m[:, j]) ** p
            vs += (obs_diff - ens_diffs.mean()) ** 2
            n_pairs += 1
    return float(vs / n_pairs)


def _variogram_score_per_pair(
    members: np.ndarray, obs: np.ndarray, p: float = 0.5
) -> np.ndarray:
    """Per-pair VS contributions for one forecast. Returns shape (n_pairs,) where
    n_pairs = d*(d-1)/2 over ALL d stations (original ordering). Pairs where either
    obs is NaN are returned as NaN. No normalization — caller averages across inits.
    members : (M, d)  obs : (d,)
    """
    d = obs.shape[0]
    n_pairs = d * (d - 1) // 2
    out = np.full(n_pairs, np.nan)
    k = 0
    for i in range(d):
        for j in range(i + 1, d):
            if np.isfinite(obs[i]) and np.isfinite(obs[j]):
                obs_diff  = abs(float(obs[i]) - float(obs[j])) ** p
                ens_diffs = np.abs(members[:, i] - members[:, j]) ** p
                out[k] = (obs_diff - ens_diffs.mean()) ** 2
            k += 1
    return out


def _energy_score(members: np.ndarray, obs: np.ndarray) -> float:
    """Energy score for one forecast: members (M, d), obs (d,).
    Stations with NaN obs are dropped before scoring.
    """
    valid = np.isfinite(obs)
    if valid.sum() < 2:
        return np.nan
    m, y = members[:, valid], obs[valid]
    M = m.shape[0]
    term1 = np.linalg.norm(m - y[None, :], axis=1).mean()
    # unbiased pairwise term via broadcasting; M=25 so (25,25,d) is fine
    diffs = m[:, None, :] - m[None, :, :]          # (M, M, d)
    pair_norms = np.linalg.norm(diffs, axis=2)      # (M, M)
    off = ~np.eye(M, dtype=bool)
    term2 = pair_norms[off].mean() / 2
    return float(term1 - term2)


def main(ckpt_path: Path | None = None, split: str = "val") -> None:
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])
    gauge_mu  = ckpt["gauge_mu"]
    gauge_std = ckpt["gauge_std"]
    print(f"Device: {device}")

    ds = SSHDiffusionDataset(split)
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

    # ── GLORYS synthetic-gauge pool (training years, same normalization as model) ─
    train_mask_ds = ds.time.year.isin(TRAIN_YEARS)
    glorys_train_phys = (
        ds.gauge_ssh[train_mask_ds] * ds.gauge_std + ds.gauge_mu
    )                                                           # (n_train, n_gauges)
    glorys_train_time = ds.time[train_mask_ds]
    glorys_pool: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        month_mask = glorys_train_time.month == month
        glorys_pool[month] = glorys_train_phys[month_mask][:, station_indices]

    within_cov_sum = {l: np.zeros((n_st, n_st)) for l in leads}
    within_cov_n   = {l: 0 for l in leads}
    spread_error: dict[int, dict[str, list[tuple[float, float]]]] = {
        l: {s: [] for s in station_names} for l in leads
    }
    # per-init records for per-pair scatter: list of (ens_cov_upper_tri, obs_vec)
    pair_cov_records: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        l: [] for l in leads
    }
    # per-init energy score: list of (es_orig, es_shuf, es_clim, es_clim_shuf)
    es_records: dict[int, list[tuple[float, float, float, float]]] = {l: [] for l in leads}
    # per-init variogram score (p=0.5): same 4-tuple layout
    vs_records: dict[int, list[tuple[float, float, float, float]]] = {l: [] for l in leads}
    # per-pair VS contributions: list of (ddpm_pairs, clim_pairs) each shape (n_pairs,)
    vs_pair_records: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {l: [] for l in leads}

    rng = np.random.default_rng()
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
                block_z = gauge_z[j * N_MEMBERS : (j + 1) * N_MEMBERS]   # (25, n_gauges)
                block_phys = block_z * gauge_std + gauge_mu               # (25, n_gauges)

                # Select only stations present in NOAA obs
                member_vals = block_phys[:, station_indices]   # (25, n_st)

                # Within-ensemble sample covariance
                centered = member_vals - member_vals.mean(axis=0, keepdims=True)
                cov = (centered.T @ centered) / (N_MEMBERS - 1)
                within_cov_sum[lead] += cov
                within_cov_n[lead]   += 1

                # Spread and error per station
                ens_mean = member_vals.mean(axis=0)
                ens_std  = member_vals.std(axis=0)
                ym = target_times[j].to_period("M").to_timestamp()

                # Store upper-triangle ens cov + obs vector for per-init scatter
                triu_i_tmp, triu_j_tmp = np.triu_indices(n_st, k=1)
                obs_row = np.array([gauge_obs[s].get(ym, np.nan) for s in station_names])
                pair_cov_records[lead].append(
                    (cov[triu_i_tmp, triu_j_tmp].copy(), obs_row)
                )

                # GLORYS target at t+lead — truth for energy score
                glorys_target = (
                    ds.gauge_ssh[t_val + lead] * ds.gauge_std + ds.gauge_mu
                )[station_indices]

                # Energy score vs GLORYS target: original, shuffled, GLORYS draw
                es_orig = _energy_score(member_vals, glorys_target)
                shuffled = member_vals.copy()
                for k in range(n_st):
                    shuffled[:, k] = member_vals[rng.permutation(N_MEMBERS), k]
                es_shuf = _energy_score(shuffled, glorys_target)

                target_month = ym.month
                pool = glorys_pool.get(target_month, np.empty((0, n_st)))
                if len(pool) >= 2:
                    glorys_members = pool[rng.choice(len(pool), N_MEMBERS, replace=True)]
                    es_clim = _energy_score(glorys_members, glorys_target)
                    glorys_shuf = glorys_members.copy()
                    for k in range(n_st):
                        glorys_shuf[:, k] = glorys_members[rng.permutation(N_MEMBERS), k]
                    es_clim_shuf = _energy_score(glorys_shuf, glorys_target)
                else:
                    es_clim = es_clim_shuf = np.nan

                if all(np.isfinite(v) for v in (es_orig, es_shuf, es_clim, es_clim_shuf)):
                    es_records[lead].append((es_orig, es_shuf, es_clim, es_clim_shuf))

                # Variogram score (p=0.5)
                vs_orig = _variogram_score(member_vals, glorys_target)
                vs_shuf = _variogram_score(shuffled,    glorys_target)
                if len(pool) >= 2:
                    vs_clim      = _variogram_score(glorys_members, glorys_target)
                    vs_clim_shuf = _variogram_score(glorys_shuf,    glorys_target)
                else:
                    vs_clim = vs_clim_shuf = np.nan
                if all(np.isfinite(v) for v in (vs_orig, vs_shuf, vs_clim, vs_clim_shuf)):
                    vs_records[lead].append((vs_orig, vs_shuf, vs_clim, vs_clim_shuf))

                # Per-pair contributions (always collect when pool is available)
                if len(pool) >= 2:
                    ddpm_pairs = _variogram_score_per_pair(member_vals, glorys_target)
                    clim_pairs = _variogram_score_per_pair(glorys_members, glorys_target)
                    vs_pair_records[lead].append((ddpm_pairs, clim_pairs))

                for k, s in enumerate(station_names):
                    obs_val = gauge_obs[s].get(ym, np.nan)
                    if np.isfinite(obs_val) and np.isfinite(ens_mean[k]):
                        spread_error[lead][s].append(
                            (float(ens_std[k]), abs(float(ens_mean[k]) - obs_val))
                        )

    # ── Observed covariance — GLORYS training period (1993–2018, ~1300 weekly) ─
    # Using GLORYS training data rather than NOAA test-period monthly obs (~36
    # samples) gives a stable, NaN-free estimate of the true spatial covariance.
    obs_cov = np.cov(glorys_train_phys[:, station_indices].T)   # (n_st, n_st)

    # ── Figure 1: within-ensemble / observed covariance ratio ────────────────
    # Ratio = 1 → perfectly calibrated; > 1 → over-amplified.
    # Diagonal = dispersivity² (ens_std/obs_std)²; off-diagonal inflated more
    # than diagonal → model has right correlation shape at wrong scale.
    from matplotlib.colors import TwoSlopeNorm

    short = [s.split("_")[0] for s in station_names]
    RATIO_CEIL = 4.0   # clip ceiling; values above this are extreme outliers

    # Build ratio matrices and print summary
    ratio_mats = {}
    off_mask = ~np.eye(n_st, dtype=bool)

    print("\nWithin-ensemble vs observed covariance")
    print(f"{'Lead':>6}  {'diag frac':>10}  {'off-diag ratio':>14}")
    for l in leads:
        wc = within_cov_sum[l] / within_cov_n[l]
        # Element-wise ratio; guard against near-zero observed entries
        safe_obs = np.where(np.abs(obs_cov) > 1e-9, obs_cov, np.nan)
        rm = np.clip(wc / safe_obs, 0, RATIO_CEIL)
        ratio_mats[l] = rm
        diag_frac = np.trace(wc) / (np.sum(wc) + 1e-12)
        off_ratio  = np.mean(np.abs(wc[off_mask])) / (np.mean(np.abs(obs_cov[off_mask])) + 1e-12)
        print(f"  {l:>2}wk    {diag_frac:>+.3f}        {off_ratio:>+.3f}")

    # Shared colour scale: diverging, centred at 1
    all_vals = np.concatenate([rm.ravel() for rm in ratio_mats.values()])
    vmax = min(RATIO_CEIL, float(np.nanpercentile(all_vals, 98)))
    vmax = max(vmax, 1.05)   # ensure vcenter=1 is not at the edge
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)

    fig1, axes1 = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, l in zip(axes1.flat, leads):
        wc = within_cov_sum[l] / within_cov_n[l]
        rm = ratio_mats[l]
        im = ax.imshow(rm, cmap="RdBu_r", norm=norm)

        # Cell annotations
        for i in range(n_st):
            for j in range(n_st):
                v = rm[i, j]
                fc = "white" if (v > vmax * 0.75 or v < 0.4) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color=fc)

        ax.set_xticks(range(n_st)); ax.set_yticks(range(n_st))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)

        # Use the same formulas as the printed summary so numbers match.
        # dispersivity = mean(ens_std_i / obs_std_i)  matches mean_marg_ratio
        # off_ratio    = mean(|wc_off|) / mean(|obs_off|)  matches off_diag_ratio
        ens_std_v  = np.sqrt(np.diag(wc).clip(0))
        obs_std_v  = np.sqrt(np.diag(obs_cov).clip(0))
        mean_disp  = float(np.mean(ens_std_v / (obs_std_v + 1e-12)))
        off_ratio  = float(np.mean(np.abs(wc[off_mask])) / (np.mean(np.abs(obs_cov[off_mask])) + 1e-12))
        ax.set_title(
            f"{l} wk  |  dispersivity {mean_disp:.2f}×   off-diag cov {off_ratio:.2f}×",
            fontsize=9,
        )
        fig1.colorbar(im, ax=ax, shrink=0.75, label="ens cov / obs cov")

    fig1.suptitle(
        "Within-ensemble covariance ratio to observed\n"
        "Ratio = 1 (white) → calibrated  ·  > 1 (red) → over-amplified  ·  "
        f"clipped at {RATIO_CEIL}×",
        fontsize=10,
    )
    savefig(fig1, OUT_DIR / "within_ensemble_cov.pdf")
    print(f"Saved figures/within_ensemble_cov.pdf")

    # ── Marginal vs off-diagonal dispersivity ────────────────────────────────
    # Per-station: ratio of ensemble std to observed std. Ratio ~1 = calibrated,
    # <1 = underdispersive marginal, >1 = overdispersive. Compared against the
    # off-diagonal covariance ratio so marginal and joint miscalibration are
    # disentangled.
    obs_std = np.sqrt(np.diag(obs_cov))                   # (n_st,)
    print("\nMarginal dispersivity — ens_std / obs_std per station/lead "
          "(1=calibrated, <1=underdispersive)")
    header = "  Lead  " + "  ".join(f"{s:>10}" for s in short)
    print(header)
    marg_ratio = np.zeros((len(leads), n_st))
    for li, l in enumerate(leads):
        wc = within_cov_sum[l] / within_cov_n[l]
        ens_std_l = np.sqrt(np.diag(wc))
        marg_ratio[li] = ens_std_l / (obs_std + 1e-12)
        row = "  ".join(f"{r:>10.3f}" for r in marg_ratio[li])
        print(f"  {l:>2}wk  {row}")

    off_mask = ~np.eye(n_st, dtype=bool)
    obs_off_mean = np.mean(np.abs(obs_cov[off_mask]))
    print("\nOff-diagonal vs marginal — same units, side-by-side per lead")
    print(f"{'Lead':>6}  {'mean_marg_ratio':>16}  {'off_diag_ratio':>15}")
    for li, l in enumerate(leads):
        wc = within_cov_sum[l] / within_cov_n[l]
        off_ratio = np.mean(np.abs(wc[off_mask])) / (obs_off_mean + 1e-12)
        print(f"  {l:>2}wk  {marg_ratio[li].mean():>16.3f}  {off_ratio:>15.3f}")

    fig_md, ax_md = plt.subplots(figsize=(7, 4))
    for k, s in enumerate(short):
        ax_md.plot(leads, marg_ratio[:, k], "o-", lw=1.4, ms=5, label=s)
    ax_md.axhline(1, color="k", lw=0.8, ls="--", label="calibrated")
    ax_md.set_xticks(leads)
    ax_md.set_xlabel("Lead time (weeks)")
    ax_md.set_ylabel("Ensemble std / observed std")
    ax_md.set_title("Marginal dispersivity per station — val set")
    ax_md.legend(fontsize=7, ncol=2, loc="best")
    fig_md.tight_layout()
    savefig(fig_md, OUT_DIR / "marginal_dispersivity.pdf")
    print("Saved figures/marginal_dispersivity.pdf")



    # ── Figure 4: energy score — original vs. shuffled vs. climatology ──────
    es_orig_mean, es_shuf_mean, es_clim_mean = [], [], []
    es_orig_se,   es_shuf_se,   es_clim_se   = [], [], []
    es_diff_mean, es_diff_se = [], []
    es_clim_shuf_mean, es_clim_shuf_se = [], []
    es_clim_diff_mean, es_clim_diff_se = [], []
    es_clim_shuf_mean, es_clim_shuf_se = [], []
    es_clim_diff_mean, es_clim_diff_se = [], []
    print("\nEnergy score: DDPM and GLORYS baseline vs. shuffled, scored on GLORYS truth")
    print(f"{'Lead':>6}  {'ES_orig':>9}  {'ES_shuf':>9}  {'ES_clim':>9}  {'ES_clim_sh':>10}  "
          f"{'Δ_ddpm':>8}  {'Δ_clim':>8}")
    for l in leads:
        arr = np.array(es_records[l])          # (N, 4)
        orig, shuf, clim, clim_sh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        diff_ddpm = orig - shuf
        diff_clim = clim - clim_sh
        n = len(orig)
        es_orig_mean.append(orig.mean());      es_orig_se.append(orig.std() / n**0.5)
        es_shuf_mean.append(shuf.mean());      es_shuf_se.append(shuf.std() / n**0.5)
        es_clim_mean.append(clim.mean());      es_clim_se.append(clim.std() / n**0.5)
        es_clim_shuf_mean.append(clim_sh.mean()); es_clim_shuf_se.append(clim_sh.std() / n**0.5)
        es_diff_mean.append(diff_ddpm.mean()); es_diff_se.append(diff_ddpm.std() / n**0.5)
        es_clim_diff_mean.append(diff_clim.mean()); es_clim_diff_se.append(diff_clim.std() / n**0.5)
        print(f"  {l:>2}wk   {orig.mean():>+.4f}   {shuf.mean():>+.4f}   "
              f"{clim.mean():>+.4f}   {clim_sh.mean():>+.4f}   "
              f"{diff_ddpm.mean():>+.4f}   {diff_clim.mean():>+.4f}")

    x = np.arange(len(leads))
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 4))

    w = 0.2
    ax4a.bar(x - 1.5*w, es_orig_mean,      w, yerr=es_orig_se,      capsize=3,
             color="steelblue",   label="DDPM")
    ax4a.bar(x - 0.5*w, es_shuf_mean,      w, yerr=es_shuf_se,      capsize=3,
             color="steelblue",   alpha=0.45, label="DDPM shuffled")
    ax4a.bar(x + 0.5*w, es_clim_mean,      w, yerr=es_clim_se,      capsize=3,
             color="0.45",        label="GLORYS draw")
    ax4a.bar(x + 1.5*w, es_clim_shuf_mean, w, yerr=es_clim_shuf_se, capsize=3,
             color="0.45",        alpha=0.45, label="GLORYS draw shuffled")
    ax4a.set_xticks(x); ax4a.set_xticklabels([f"{l}wk" for l in leads])
    ax4a.set_ylabel("Mean energy score (m)")
    ax4a.set_title("Energy score by lead\n(lower = better)")
    ax4a.legend(fontsize=7)

    w2 = 0.35
    ax4b.bar(x - w2/2, es_diff_mean,      w2, yerr=es_diff_se,      capsize=3,
             color="steelblue", label="DDPM: orig − shuffled")
    ax4b.bar(x + w2/2, es_clim_diff_mean, w2, yerr=es_clim_diff_se, capsize=3,
             color="0.45",      label="GLORYS: joint − shuffled")
    ax4b.axhline(0, color="k", lw=0.8, ls="--")
    ax4b.set_xticks(x); ax4b.set_xticklabels([f"{l}wk" for l in leads])
    ax4b.set_ylabel("ES_joint − ES_shuffled (m)")
    ax4b.set_title("Joint structure contribution\n(negative = spatial covariance helps)")
    ax4b.legend(fontsize=8)

    fig4.suptitle("Energy score: DDPM vs. GLORYS training draw — val set", y=1.01)
    fig4.tight_layout()
    savefig(fig4, OUT_DIR / "energy_score_permutation.pdf")
    print("Saved figures/energy_score_permutation.pdf")

    # ── Figure 5: energy skill score vs lead ─────────────────────────────────
    # Skill = 1 - ES / ES_clim, computed per init then averaged
    skill_lines = {
        "DDPM":             {"col": 0, "color": "steelblue",  "ls": "-",  "marker": "o"},
        "DDPM shuffled":    {"col": 1, "color": "steelblue",  "ls": "--", "marker": "s"},
        "GLORYS shuffled":  {"col": 3, "color": "0.45",       "ls": "--", "marker": "^"},
    }

    rng_boot = np.random.default_rng(42)
    fig5, ax5 = plt.subplots(figsize=(5.5, 4))
    for label, cfg in skill_lines.items():
        skill_all  = []   # (n_leads, N) — per-init skill at each lead
        skill_mean = []
        for l in leads:
            arr    = np.array(es_records[l])   # (N, 4)
            es_ref = arr[:, 2]                 # ES_clim per init
            skill  = 1 - arr[:, cfg["col"]] / es_ref
            skill_all.append(skill)
            skill_mean.append(skill.mean())
        # skill_all transposed: (N, n_leads) for block_bootstrap_ci (axis-0 = time)
        skill_ts = np.stack(skill_all, axis=1)          # (N, n_leads)
        lo, hi   = block_bootstrap_ci(skill_ts, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot)
        skill_mean = np.array(skill_mean)
        ax5.plot(leads, skill_mean, color=cfg["color"], ls=cfg["ls"],
                 marker=cfg["marker"], ms=6, lw=1.8, label=label)
        ax5.fill_between(leads, lo, hi, color=cfg["color"], alpha=0.18)

    ax5.axhline(0, color="k", lw=0.8, ls="--")
    ax5.set_xticks(leads)
    ax5.set_xlabel("Lead time (weeks)")
    ax5.set_ylabel("Energy skill score  (1 − ES / ES$_{clim}$)")
    ax5.set_title(
        f"Energy skill score vs lead — val set\n"
        f"Shading: 95 % block-bootstrap CI (block={BLOCK_SIZE} wk)"
    )
    ax5.legend(fontsize=9)
    fig5.tight_layout()
    savefig(fig5, OUT_DIR / "energy_skill_vs_lead.pdf")
    print("Saved figures/energy_skill_vs_lead.pdf")


    # ── Variogram score skill vs lead ─────────────────────────────────────────
    print("\nVariogram score skill (1 − VS / VS_clim):  p=0.5, equal weights")
    print(f"{'Lead':>6}  {'DDPM':>8}  {'DDPM_shuf':>10}  {'GLORYS_shuf':>12}")
    vs_skill_lines = {
        "DDPM":            {"col": 0, "color": "steelblue",  "ls": "-",  "marker": "o"},
        "DDPM shuffled":   {"col": 1, "color": "steelblue",  "ls": "--", "marker": "s"},
        "GLORYS shuffled": {"col": 3, "color": "0.45",       "ls": "--", "marker": "^"},
    }
    vs_skill_data: dict[str, list[float]] = {k: [] for k in vs_skill_lines}

    for l in leads:
        arr = np.array(vs_records[l])          # (N, 4): orig, shuf, clim, clim_shuf
        vs_ref = arr[:, 2]                     # VS_clim per init
        row_vals = {}
        for label, cfg in vs_skill_lines.items():
            skill = 1 - arr[:, cfg["col"]] / vs_ref
            vs_skill_data[label].append(skill.mean())
            row_vals[label] = skill.mean()
        print(f"  {l:>2}wk  {row_vals['DDPM']:>+.4f}  "
              f"{row_vals['DDPM shuffled']:>+.4f}      "
              f"{row_vals['GLORYS shuffled']:>+.4f}")

    fig6, ax6 = plt.subplots(figsize=(5.5, 4))
    for label, cfg in vs_skill_lines.items():
        sm = np.array(vs_skill_data[label])
        # Per-init skill time series: (N, n_leads)
        vs_ts = np.stack(
            [1 - np.array(vs_records[l])[:, cfg["col"]] / np.array(vs_records[l])[:, 2]
             for l in leads],
            axis=1,
        )
        lo, hi = block_bootstrap_ci(vs_ts, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot)
        ax6.plot(leads, sm, color=cfg["color"], ls=cfg["ls"],
                 marker=cfg["marker"], ms=6, lw=1.8, label=label)
        ax6.fill_between(leads, lo, hi, color=cfg["color"], alpha=0.18)
    ax6.axhline(0, color="k", lw=0.8, ls="--")
    ax6.set_xticks(leads)
    ax6.set_xlabel("Lead time (weeks)")
    ax6.set_ylabel("VS skill  (1 − VS / VS$_{clim}$,  p=0.5)")
    ax6.set_title(
        f"Variogram score skill vs lead — val set\n"
        f"(ref = GLORYS seasonal draw)  —  95 % CI block={BLOCK_SIZE} wk"
    )
    ax6.legend(fontsize=9)
    fig6.tight_layout()
    savefig(fig6, OUT_DIR / "variogram_skill_vs_lead.pdf")
    print("Saved figures/variogram_skill_vs_lead.pdf")


    # ── Fig 4b: per-pair VS skill vs observed pairwise correlation ───────────
    # Hypothesis: low-correlation pairs have worse VS skill (model over-amplifies
    # off-diagonal variance for uncorrelated stations).
    triu_ii, triu_jj = np.triu_indices(n_st, k=1)   # (n_pairs,)
    pair_labels = [f"{short[i]}–{short[j]}" for i, j in zip(triu_ii, triu_jj)]

    # Observed pairwise Pearson correlation from obs_cov
    obs_std_v = np.sqrt(np.diag(obs_cov))
    obs_corr_mat = obs_cov / np.outer(obs_std_v + 1e-12, obs_std_v + 1e-12)
    obs_pair_corr = obs_corr_mat[triu_ii, triu_jj]   # (n_pairs,)

    lead_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(leads)))

    fig_4b, ax_4b = plt.subplots(figsize=(6, 4.5))

    print("\nPer-pair VS skill (1 − VS_ddpm / VS_clim) vs observed pairwise correlation")
    print(f"{'Lead':>6}  " + "  ".join(f"{lb:>12}" for lb in pair_labels))

    for li, (l, color) in enumerate(zip(leads, lead_colors)):
        recs = vs_pair_records[l]
        if not recs:
            continue
        ddpm_stack = np.array([r[0] for r in recs])   # (N, n_pairs)
        clim_stack = np.array([r[1] for r in recs])   # (N, n_pairs)

        # Nanmean across inits for each pair
        mean_ddpm = np.nanmean(ddpm_stack, axis=0)    # (n_pairs,)
        mean_clim = np.nanmean(clim_stack, axis=0)    # (n_pairs,)
        pair_skill = 1.0 - mean_ddpm / (mean_clim + 1e-12)  # (n_pairs,)

        row = "  ".join(f"{v:>+12.4f}" for v in pair_skill)
        print(f"  {l:>2}wk  {row}")

        ax_4b.plot(obs_pair_corr, pair_skill, "o", color=color,
                   ms=6, alpha=0.85, label=f"{l}wk", zorder=3)

    ax_4b.axhline(0, color="k", lw=0.8, ls="--")
    ax_4b.set_xlabel("Observed pairwise Pearson correlation")
    ax_4b.set_ylabel("Per-pair VS skill  (1 − VS$_{ddpm}$ / VS$_{clim}$)")
    ax_4b.set_title("Per-pair variogram skill vs observed correlation\n(val set, p=0.5)")
    ax_4b.legend(fontsize=8, title="Lead", loc="lower right")
    fig_4b.tight_layout()
    savefig(fig_4b, OUT_DIR / "variogram_skill_per_pair.pdf")
    print("Saved figures/variogram_skill_per_pair.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Path to checkpoint (default: ddpm_best_large.pt)")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to evaluate: val or test")
    args = parser.parse_args()
    main(args.ckpt, args.split)
