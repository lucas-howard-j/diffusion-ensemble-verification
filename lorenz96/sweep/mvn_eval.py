"""
MVN (linear Gaussian) baseline for the L96 sweep.

For each (lead, n_train):
  1. Load mu/sigma from the corresponding DDPM checkpoint (same normalization)
  2. Fit OLS regression  target ~ W @ cond + b  on the same n_train pairs
  3. Estimate residual covariance Sigma from training residuals
  4. At test time draw N_ENS samples from N(W @ cond_test + b, Sigma)
  5. Score CRPS and VS skill vs climatological draw

If MVN VS skill is near zero at lead=40 (same as DDPM), the training data
contains no learnable joint signal at that lead — the DDPM is not failing
to find available signal, the signal is absent.

If MVN VS skill is positive at lead=40 while DDPM VS is near zero, the
signal exists but the DDPM architecture is not capturing it.

Saves mvn_results.npz in the same format as sweep_results.npz so that
plot_sweep.py can overlay the MVN line on the sweep figure.

Usage:
    PYTHONPATH=. python lorenz96/sweep/mvn_eval.py
"""

import os
import numpy as np
import torch

from lorenz96.config  import N, TEST_START, TRAIN_END
from lorenz96.dataset import load_trajectory

CKPT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_PATH = "lorenz96/data/l96_trajectory.npz"
OUT_PATH  = os.path.join(os.path.dirname(__file__), "mvn_results.npz")
N_EVAL    = 200
N_ENS     = 20   # match DDPM and dynamical ensemble size
N_BOOT    = 1000
SEED      = 42

LEAD_VALUES    = [20, 40, 60]
N_TRAIN_VALUES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 249_000]


def crps_per_sample(truth: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """truth:(n,D)  ens:(n,M,D) → (n,)"""
    M      = ens.shape[1]
    mae    = np.abs(ens - truth[:, None, :]).mean(axis=1)
    spread = np.abs(ens[:, :, None, :] - ens[:, None, :, :]).sum(axis=(1, 2)) / (2 * M * M)
    return (mae - spread).mean(axis=1)


def vs_per_sample(truth: np.ndarray, ens: np.ndarray, p: float = 0.5) -> np.ndarray:
    """truth:(n,D)  ens:(n,M,D) → (n,)"""
    D = truth.shape[1]
    i_idx, j_idx = np.triu_indices(D, k=1)
    obs_diff = np.abs(truth[:, i_idx] - truth[:, j_idx]) ** p
    ens_diff = np.abs(ens[:, :, i_idx] - ens[:, :, j_idx]) ** p
    return ((obs_diff - ens_diff.mean(axis=1)) ** 2).mean(axis=1)


def bootstrap_skill(model_crps, clim_crps, model_vs, clim_vs, n_boot, rng):
    """Paired bootstrap → (crps_dist, vs_dist) each shape (n_boot,)."""
    n = len(model_crps)
    crps_dist = np.empty(n_boot)
    vs_dist   = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        crps_dist[k] = 1.0 - model_crps[idx].mean() / clim_crps[idx].mean()
        vs_dist[k]   = 1.0 - model_vs[idx].mean()   / clim_vs[idx].mean()
    return crps_dist, vs_dist


def fit_mvn(cond_tr: np.ndarray, target_tr: np.ndarray):
    """OLS regression + residual covariance.

    Returns (W, b, Sigma) where the conditional mean is cond @ W.T + b.
    Sigma has a small diagonal ridge (1e-6) to ensure positive definiteness.
    """
    n = len(cond_tr)
    X = np.hstack([cond_tr, np.ones((n, 1))])          # (n, N+1)
    coeffs, _, _, _ = np.linalg.lstsq(X, target_tr, rcond=None)  # (N+1, N)
    W = coeffs[:-1].T                                   # (N, N)
    b = coeffs[-1]                                      # (N,)
    resid = target_tr - X @ coeffs                      # (n, N)
    Sigma = (resid.T @ resid) / n + 1e-6 * np.eye(N)
    return W, b, Sigma


def run():
    states, _ = load_trajectory(DATA_PATH)
    train_pool = states[:TRAIN_END]
    test_raw   = states[TEST_START:]

    n_leads = len(LEAD_VALUES)
    n_sizes = len(N_TRAIN_VALUES)
    crps_arr      = np.full((n_leads, n_sizes),          np.nan)
    vs_arr        = np.full((n_leads, n_sizes),          np.nan)
    crps_dist_arr = np.full((n_leads, n_sizes, N_BOOT),  np.nan)
    vs_dist_arr   = np.full((n_leads, n_sizes, N_BOOT),  np.nan)

    for i, lead in enumerate(LEAD_VALUES):
        for j, n_train in enumerate(N_TRAIN_VALUES):
            ckpt_path = os.path.join(CKPT_DIR, f"ddpm_lead{lead}_{n_train}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  skip  lead={lead}  n_train={n_train:,}  (no checkpoint)")
                continue

            print(f"lead={lead}  n_train={n_train:,} ...", flush=True)

            rng = np.random.default_rng(SEED + i * 1000 + j)

            ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mu    = ckpt["mu"]
            sigma = ckpt["sigma"]

            cond_tr   = (states[:n_train]              - mu) / sigma
            target_tr = (states[lead : n_train + lead] - mu) / sigma

            W, b, Sigma = fit_mvn(cond_tr, target_tr)

            try:
                L = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                Sigma += 1e-3 * np.eye(N)
                L = np.linalg.cholesky(Sigma)

            # Test pairs
            max_test = len(test_raw) - lead
            n_eval   = min(N_EVAL, max_test)
            idx = rng.choice(max_test, size=n_eval, replace=False)
            idx.sort()

            cond_norm   = (test_raw[idx]        - mu) / sigma
            target_norm = (test_raw[idx + lead] - mu) / sigma

            # MVN ensemble
            mu_pred = cond_norm @ W.T + b
            z       = rng.standard_normal((n_eval, N_ENS, N))
            mvn_ens = mu_pred[:, None, :] + (z @ L.T)

            # Climatological ensemble
            clim_idx = rng.integers(0, len(train_pool), size=(n_eval, N_ENS))
            clim_ens = (train_pool[clim_idx] - mu) / sigma

            # Per-sample scores for bootstrap
            crps_mvn  = crps_per_sample(target_norm, mvn_ens)
            crps_clim = crps_per_sample(target_norm, clim_ens)
            vs_mvn    = vs_per_sample(target_norm, mvn_ens)
            vs_clim   = vs_per_sample(target_norm, clim_ens)

            crps_arr[i, j] = 1.0 - crps_mvn.mean() / crps_clim.mean()
            vs_arr[i, j]   = 1.0 - vs_mvn.mean()   / vs_clim.mean()

            boot_rng = np.random.default_rng(SEED + i * 1000 + j + 500)
            crps_dist_arr[i, j], vs_dist_arr[i, j] = bootstrap_skill(
                crps_mvn, crps_clim, vs_mvn, vs_clim, N_BOOT, boot_rng)

            cm = np.median(crps_dist_arr[i, j])
            vm = np.median(vs_dist_arr[i, j])
            c2, c97 = np.percentile(crps_dist_arr[i, j], [2.5, 97.5])
            v2, v97 = np.percentile(vs_dist_arr[i, j],   [2.5, 97.5])
            print(f"  CRPS skill={cm:+.4f} [{c2:+.4f}, {c97:+.4f}]  "
                  f"VS skill={vm:+.4f} [{v2:+.4f}, {v97:+.4f}]", flush=True)

    np.savez(OUT_PATH,
             lead_values=LEAD_VALUES, n_train_values=N_TRAIN_VALUES,
             crps_skill=crps_arr, vs_skill=vs_arr,
             crps_skill_dist=crps_dist_arr, vs_skill_dist=vs_dist_arr)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    run()
