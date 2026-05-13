"""
Dynamical ensemble baseline: perturbed-IC AR(1) forecast.

For each init time t and each member m:
  x_init_m  = x_obs_t + δ_0_m                   δ_0_m ~ N(0, f_ic² Σ_train)
  x_m(lead) = ρ(lead) ⊙ x_init_m + ε_proc_m     ε_proc ~ N(0, Q(lead))

where:
  Σ_train  = training-period covariance (8×8, physical units)
  ρ_j(lead) = per-station ACF at lag=lead weeks, estimated from training data
  Q(lead)  = spatially correlated process noise so ensemble variance = Σ_train*(1-ρ²)
               Q_{ij}(lead) = sqrt(σ_i²(1-ρ_i²) - f_ic²*ρ_i²*σ_i²)
                              * R_{ij}
                              * sqrt(σ_j²(1-ρ_j²) - f_ic²*ρ_j²*σ_j²)
             (diagonal term = σ_i²*(1-ρ_i²*(1+f_ic²)), reduces to
              σ_i²*(1-ρ_i²) when f_ic=0)

Ensemble mean is damped persistence; spread grows from zero to climatology.
Spatial correlations are drawn from the training-period covariance structure.

Produces the same CRPS / ES / VS skill metrics as DDPM (vs GLORYS seasonal draw).

Run:
    PYTHONPATH=. python scripts/dynamical_baseline.py --split test
    PYTHONPATH=. python scripts/dynamical_baseline.py --split test --sigma_ic 0.2
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import LEAD_WEEKS, TIDE_GAUGES, TRAIN_YEARS
from src.model.dataset import SSHDiffusionDataset
from src.evaluation.metrics import block_bootstrap_ci

N_MEMBERS  = 25
BLOCK_SIZE = 8
N_BOOT     = 1000
CI_ALPHA   = 0.05
OUT_DIR    = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def savefig(fig, path, **kwargs):
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", **kwargs)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight", **kwargs)


def _crps(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS per station.  members: (M, n_st)  obs: (n_st,)  → (n_st,)"""
    M = members.shape[0]
    term1 = np.mean(np.abs(members - obs[None, :]), axis=0)
    diff  = np.abs(members[:, None, :] - members[None, :, :])
    off   = ~np.eye(M, dtype=bool)
    term2 = diff[off].reshape(M * (M - 1), -1).mean(axis=0) / 2
    return term1 - term2


def _energy_score(members: np.ndarray, obs: np.ndarray) -> float:
    """Energy score.  members: (M, d)  obs: (d,)"""
    valid = np.isfinite(obs)
    if valid.sum() < 2:
        return np.nan
    m, y = members[:, valid], obs[valid]
    M = m.shape[0]
    term1 = np.linalg.norm(m - y[None, :], axis=1).mean()
    diffs = m[:, None, :] - m[None, :, :]
    pair_norms = np.linalg.norm(diffs, axis=2)
    off = ~np.eye(M, dtype=bool)
    term2 = pair_norms[off].mean() / 2
    return float(term1 - term2)


def _variogram_score(members: np.ndarray, obs: np.ndarray, p: float = 0.5) -> float:
    """Variogram score (Scheuerer & Hamill 2015).  members: (M, d)  obs: (d,)"""
    valid = np.isfinite(obs)
    if valid.sum() < 2:
        return np.nan
    m, y = members[:, valid], obs[valid]
    d = y.shape[0]
    vs, n_pairs = 0.0, 0
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


def _compute_acf_and_cov(
    gauge_phys: np.ndarray,   # (T, n_st) physical-unit SSH on training period
    leads: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      acf  : (n_leads, n_st)  lag-k autocorrelation per station
      Sigma: (n_st, n_st)     training-period sample covariance
    """
    n_st = gauge_phys.shape[1]
    n_leads = len(leads)
    acf = np.zeros((n_leads, n_st))
    for li, lag in enumerate(leads):
        x0 = gauge_phys[:-lag]
        xl = gauge_phys[lag:]
        for j in range(n_st):
            mask = np.isfinite(x0[:, j]) & np.isfinite(xl[:, j])
            if mask.sum() > 10:
                acf[li, j] = float(np.corrcoef(x0[mask, j], xl[mask, j])[0, 1])
    Sigma = np.cov(gauge_phys.T)    # (n_st, n_st)
    return acf, Sigma


def _make_noise_cholesky(
    Sigma: np.ndarray,        # (n_st, n_st)
    acf_lead: np.ndarray,     # (n_st,)
    sigma_std: np.ndarray,    # (n_st,) per-station std (diag of Sigma)^0.5
    f_ic: float,
) -> np.ndarray:
    """
    Cholesky factor of the process noise covariance Q(lead) such that the
    total ensemble variance equals Sigma*(1-rho^2) per station.

    Q_{ij} = d_i * R_{ij} * d_j
    where d_i = sqrt(max(0, sigma_i^2 * (1 - rho_i^2 * (1 + f_ic^2))))
    and R is the training correlation matrix.

    Returns the Cholesky factor L s.t. L @ L.T = Q (or a near-PSD fallback).
    """
    n = len(acf_lead)
    # Residual variance per station after IC + AR propagation
    resid_var = np.maximum(0.0, sigma_std**2 * (1.0 - acf_lead**2 * (1.0 + f_ic**2)))
    d = np.sqrt(resid_var)                              # (n,)
    # Training correlation matrix
    corr = Sigma / np.outer(sigma_std + 1e-12, sigma_std + 1e-12)
    np.fill_diagonal(corr, 1.0)
    Q = np.outer(d, d) * corr                           # (n, n)
    # Regularise for numerical PSD
    Q += np.eye(n) * 1e-10 * np.diag(Q).max()
    try:
        L = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        # Fall back to nearest PSD via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(Q)
        eigvals = np.maximum(eigvals, 0.0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    return L                                             # (n, n)


def main(split: str = "test", sigma_ic: float = 0.1) -> None:
    rng = np.random.default_rng(42)

    ds = SSHDiffusionDataset(split)
    gauge_mu  = ds.gauge_mu
    gauge_std = ds.gauge_std
    n_st = len(gauge_mu)

    all_gauge_names = [g["name"] for g in TIDE_GAUGES][:n_st]
    short = [s.split("_")[0] for s in all_gauge_names]

    # ── Training data in physical units ──────────────────────────────────────
    train_mask = ds.time.year.isin(TRAIN_YEARS)
    gauge_train_phys = ds.gauge_ssh[train_mask] * gauge_std + gauge_mu   # (T_train, n_st)
    glorys_time_train = ds.time[train_mask]

    leads = sorted(LEAD_WEEKS)
    print(f"Split: {split}  |  sigma_ic: {sigma_ic}")
    print(f"Training period: {glorys_time_train.year.min()}–{glorys_time_train.year.max()}"
          f"  ({train_mask.sum()} steps)")

    # ── ACF and covariance from training data ─────────────────────────────────
    acf, Sigma = _compute_acf_and_cov(gauge_train_phys, leads)
    sigma_std  = np.sqrt(np.diag(Sigma))    # per-station std

    print("\nPer-station ACF at each lead (from training data):")
    print("  Lead  " + "  ".join(f"{s:>8}" for s in short))
    for li, l in enumerate(leads):
        row = "  ".join(f"{acf[li, j]:>+.3f}" for j in range(n_st))
        print(f"  {l:>2}wk  {row}")

    # Precompute IC covariance Cholesky: Sigma_ic = (f_ic * sigma_std)^2 * R
    # IC perturbation drawn once per init, shared across leads.
    ic_d   = f_ic_val = sigma_ic * sigma_std   # per-station IC noise std (n_st,)
    corr_train = Sigma / np.outer(sigma_std + 1e-12, sigma_std + 1e-12)
    np.fill_diagonal(corr_train, 1.0)
    Sigma_ic = np.outer(ic_d, ic_d) * corr_train    # (n_st, n_st)
    Sigma_ic += np.eye(n_st) * 1e-10 * np.diag(Sigma_ic).max()
    L_ic = np.linalg.cholesky(Sigma_ic)             # (n_st, n_st)

    # Process noise Cholesky per lead
    L_proc = {}
    for li, l in enumerate(leads):
        L_proc[l] = _make_noise_cholesky(Sigma, acf[li], sigma_std, sigma_ic)

    # ── GLORYS seasonal pool (same as other scripts) ──────────────────────────
    glorys_pool: dict[int, np.ndarray] = {
        m: gauge_train_phys[glorys_time_train.month == m]
        for m in range(1, 13)
    }

    # ── Identify valid test init times ────────────────────────────────────────
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"\nInit times: {len(valid_ts)}")

    # ── Per-init score records ────────────────────────────────────────────────
    crps_dyn_rec  = []
    crps_ref_rec  = []
    es_records    = {l: [] for l in leads}   # (es_dyn, es_clim) per init
    vs_records    = {l: [] for l in leads}   # (vs_dyn, vs_clim) per init
    vs_pair_recs  = {l: [] for l in leads}   # (ddpm_pairs, clim_pairs) per init

    for i, t_val in enumerate(valid_ts):
        if i % 20 == 0:
            print(f"  init {i}/{len(valid_ts)}")

        x_obs = ds.gauge_ssh[t_val] * gauge_std + gauge_mu   # (n_st,) physical

        # IC perturbations — shared across leads for each member
        ic_pert = (L_ic @ rng.standard_normal((n_st, N_MEMBERS))).T   # (M, n_st)
        x_init  = x_obs[None, :] + ic_pert                            # (M, n_st)

        row_crps_dyn = np.zeros((len(leads), n_st))
        row_crps_ref = np.zeros((len(leads), n_st))

        for j, lead in enumerate(leads):
            # Ensemble forecast at this lead
            rho    = acf[j]                                          # (n_st,)
            L_p    = L_proc[lead]                                    # (n_st, n_st)
            proc_n = (L_p @ rng.standard_normal((n_st, N_MEMBERS))).T  # (M, n_st)
            members = rho[None, :] * x_init + proc_n                # (M, n_st)

            # GLORYS truth
            idx_t  = t_to_lead_idx[t_val][lead]
            glorys_target = ds.gauge_ssh[t_val + lead] * gauge_std + gauge_mu  # (n_st,)

            # Reference: GLORYS seasonal draw
            month = ds.time[t_val + lead].month
            pool  = glorys_pool[month]
            ref_m = pool[rng.choice(len(pool), N_MEMBERS, replace=True)]   # (M, n_st)

            row_crps_dyn[j] = _crps(members, glorys_target)
            row_crps_ref[j] = _crps(ref_m,   glorys_target)

            es_dyn  = _energy_score(members, glorys_target)
            es_clim = _energy_score(ref_m,   glorys_target)
            if np.isfinite(es_dyn) and np.isfinite(es_clim):
                es_records[lead].append((es_dyn, es_clim))

            vs_dyn  = _variogram_score(members, glorys_target)
            vs_clim = _variogram_score(ref_m,   glorys_target)
            if np.isfinite(vs_dyn) and np.isfinite(vs_clim):
                vs_records[lead].append((vs_dyn, vs_clim))

            pp_dyn  = _variogram_score_per_pair(members, glorys_target)
            pp_clim = _variogram_score_per_pair(ref_m,   glorys_target)
            vs_pair_recs[lead].append((pp_dyn, pp_clim))

        crps_dyn_rec.append(row_crps_dyn)
        crps_ref_rec.append(row_crps_ref)

    crps_dyn_rec = np.stack(crps_dyn_rec)   # (N_inits, n_leads, n_st)
    crps_ref_rec = np.stack(crps_ref_rec)

    crps_dyn   = crps_dyn_rec.mean(0)       # (n_leads, n_st)
    crps_ref   = crps_ref_rec.mean(0)
    crps_skill = 1 - crps_dyn / crps_ref

    # ── Print CRPS skill ──────────────────────────────────────────────────────
    print(f"\nCRPS skill (1 − CRPS_dyn / CRPS_ref)  [sigma_ic={sigma_ic}]")
    print("  Lead  " + "  ".join(f"{s:>8}" for s in short) + "   Mean")
    for j, l in enumerate(leads):
        vals = "  ".join(f"{crps_skill[j, k]:>+.3f}" for k in range(n_st))
        print(f"  {l:>2}wk  {vals}   {crps_skill[j].mean():>+.3f}")

    # ── Energy score skill ────────────────────────────────────────────────────
    print(f"\nEnergy score skill (1 − ES_dyn / ES_clim)")
    print(f"  {'Lead':>6}  {'ES_dyn':>9}  {'ES_clim':>9}  {'Skill':>8}")
    for l in leads:
        arr = np.array(es_records[l])
        es_d, es_c = arr[:, 0], arr[:, 1]
        skill = 1 - es_d.mean() / es_c.mean()
        print(f"  {l:>2}wk   {es_d.mean():>+.4f}   {es_c.mean():>+.4f}  {skill:>+.4f}")

    # ── Variogram score skill ─────────────────────────────────────────────────
    print(f"\nVariogram score skill (1 − VS_dyn / VS_clim,  p=0.5)")
    print(f"  {'Lead':>6}  {'VS_dyn':>9}  {'VS_clim':>9}  {'Skill':>8}")
    for l in leads:
        arr = np.array(vs_records[l])
        vs_d, vs_c = arr[:, 0], arr[:, 1]
        skill = 1 - vs_d.mean() / vs_c.mean()
        print(f"  {l:>2}wk   {vs_d.mean():>+.4f}   {vs_c.mean():>+.4f}  {skill:>+.4f}")

    # ── Bootstrap CIs for summary plot ───────────────────────────────────────
    rng_boot = np.random.default_rng(42)
    print(f"\nComputing block bootstrap CIs (block={BLOCK_SIZE}, n_boot={N_BOOT}) ...")
    crps_lo, crps_hi = _skill_ci_crps(
        crps_dyn_rec, crps_ref_rec, BLOCK_SIZE, N_BOOT, CI_ALPHA, rng_boot
    )
    print("  Done.")

    # ── Summary figure: CRPS skill vs lead ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.tab10
    for k, name in enumerate(short):
        color = cmap(k / n_st)
        ax.plot(leads, crps_skill[:, k], "o-", color=color, lw=1.5, ms=5, label=name)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.fill_between(leads, crps_lo, crps_hi, color="gray", alpha=0.18,
                    label=f"95 % CI station mean")
    ax.plot(leads, crps_skill.mean(1), "k^-", lw=2, ms=7, label="Mean")
    ax.set_xticks(leads)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("CRPS skill  (1 − CRPS / CRPS$_{ref}$)")
    ax.set_title(
        f"Dynamical baseline CRPS skill — {split} set\n"
        f"(AR(1) perturbed-IC,  σ_ic={sigma_ic:.2f}×σ_clim)"
    )
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"dynamical_crps_skill_{split}_sic{int(sigma_ic*100):02d}"
    savefig(fig, OUT_DIR / fname)
    print(f"\nSaved figures/{fname}.pdf")
    plt.close(fig)


def _skill_ci_crps(
    dyn_rec: np.ndarray,
    ref_rec: np.ndarray,
    block_size: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Block bootstrap CI on station-mean CRPS skill. Returns (lo, hi) shape (n_leads,)."""
    N = dyn_rec.shape[0]
    n_blocks = int(np.ceil(N / block_size))
    max_start = max(0, N - block_size)
    boot = np.empty((n_boot, dyn_rec.shape[1]))
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])[:N]
        skill  = 1 - dyn_rec[idx].mean(0) / ref_rec[idx].mean(0)   # (n_leads, n_st)
        boot[b] = skill.mean(-1)
    lo = np.percentile(boot, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    type=str,   default="test",
                        help="Dataset split: val or test")
    parser.add_argument("--sigma_ic", type=float, default=0.1,
                        help="IC perturbation as fraction of per-station training std")
    args = parser.parse_args()
    main(args.split, args.sigma_ic)
