"""
Dynamical ensemble baseline for the L96 sweep.

For each lead, generates a perturbed-IC ensemble by integrating the true L96
equations forward from N_ENS members with perturbed initial conditions, with
additive Gaussian noise injected at each RK4 step to simulate model error:

    x_init^m  = x_t  + sigma_ic  * N(0, I_N)
    x_{t+1}^m = RK4(x_t^m) + sigma_model * N(0, I_N)

sigma_ic    controls initial-condition uncertainty.
sigma_model controls per-step model error growth.

Both are in physical units.  Defaults are chosen so ensemble spread is
comparable to short-lead DDPM spread.

Saves dyn_results.npz in the same layout as mvn_results.npz so figure4.py
can overlay a dynamical baseline line.

Usage:
    PYTHONPATH=. python lorenz96/sweep/dyn_eval.py
    PYTHONPATH=. python lorenz96/sweep/dyn_eval.py --sigma_ic 0.5 --sigma_model 0.02
"""

import argparse
import os

import numpy as np

from lorenz96.config  import N, F, DT, TEST_START, TRAIN_END
from lorenz96.dataset import load_trajectory

OUT_PATH  = os.path.join(os.path.dirname(__file__), "dyn_results.npz")
DATA_PATH = "lorenz96/data/l96_trajectory.npz"
N_EVAL    = 200    # match mvn_eval.py
N_ENS     = 20     # match DDPM sweep ensemble size
N_BOOT    = 1000
SEED      = 42

LEAD_VALUES = [20, 40, 60]

# IC noise as a fraction of the training-period per-node std (sigma_clim).
# Converted to physical units inside run() using the actual training sigma.
SIGMA_IC_FRAC_DEFAULT = 0.10          # 10 % of sigma_clim; also sets shared IC bias scale
SIGMA_MODEL_DEFAULT   = 0.00          # per-step residual noise (physical units)
SIGMA_BIAS_DEFAULT    = 0.00          # per-step shared bias std (physical units)


# ---------------------------------------------------------------------------
# L96 integrator (matches generate_data.py)
# ---------------------------------------------------------------------------

def _l96_deriv(x: np.ndarray) -> np.ndarray:
    """L96 tendency; x: (N,) or (M, N), cyclic."""
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + F


def rk4_step(x: np.ndarray, dt: float = DT) -> np.ndarray:
    """Single RK4 step; x: (M, N), returns (M, N)."""
    k1 = _l96_deriv(x)
    k2 = _l96_deriv(x + 0.5 * dt * k1)
    k3 = _l96_deriv(x + 0.5 * dt * k2)
    k4 = _l96_deriv(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_ensemble(
    x0: np.ndarray,        # (N,) initial state (physical)
    n_steps: int,
    n_ens: int,
    sigma_ic: float,       # IC perturbation std (physical)
    sigma_model: float,    # per-step residual noise std (physical), i.i.d. per member per node
    rng: np.random.Generator,
    sigma_bias: float = 0.0,  # per-step shared bias std: b~N(0,sigma_bias²·I) across members/steps
) -> np.ndarray:
    """
    Integrate N_ENS perturbed members forward for n_steps.
    Returns (n_ens, N) final states in physical units.

    IC = shared bias + per-member noise, both N(0, sigma_ic²·I):
      b     ~ N(0, sigma_ic²·I)  shared across all members
      eps_m ~ N(0, sigma_ic²·I)  i.i.d. per member
    Makes truth exchangeable with any member at t=0.
    """
    b   = sigma_ic * rng.standard_normal((1, N))
    ens = x0[None, :] + b + sigma_ic * rng.standard_normal((n_ens, N))
    noise_mean = sigma_bias * rng.standard_normal((1, N)) if sigma_bias > 0.0 else 0.0

    for _ in range(n_steps):
        ens = rk4_step(ens)
        noise = noise_mean
        if sigma_model > 0.0:
            noise = noise + sigma_model * rng.standard_normal((n_ens, N))
        if sigma_bias > 0.0 or sigma_model > 0.0:
            ens += noise

    return ens   # (M, N)


# ---------------------------------------------------------------------------
# Score functions (same as mvn_eval.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(sigma_ic_frac: float = SIGMA_IC_FRAC_DEFAULT,
        sigma_model: float = SIGMA_MODEL_DEFAULT,
        sigma_bias: float = SIGMA_BIAS_DEFAULT):
    states, _ = load_trajectory(DATA_PATH)

    # Normalization from training data (same as what checkpoints store)
    train_pool = states[:TRAIN_END]
    mu    = train_pool.mean(axis=0)   # (N,)
    sigma = train_pool.std(axis=0)    # (N,)
    sigma_clim = float(sigma.mean())

    # IC perturbation in physical units derived from fraction of sigma_clim
    sigma_ic = sigma_ic_frac * sigma_clim
    print(f"Training stats:  mu mean={mu.mean():.4f}  sigma_clim={sigma_clim:.4f}")
    print(f"sigma_ic = {sigma_ic_frac:.2f} × sigma_clim = {sigma_ic:.4f}  "
          f"sigma_model = {sigma_model}  sigma_bias = {sigma_bias}")

    test_raw = states[TEST_START:]

    n_leads = len(LEAD_VALUES)
    crps_arr      = np.full(n_leads,          np.nan)
    vs_arr        = np.full(n_leads,          np.nan)
    crps_dist_arr = np.full((n_leads, N_BOOT), np.nan)
    vs_dist_arr   = np.full((n_leads, N_BOOT), np.nan)

    for i, lead in enumerate(LEAD_VALUES):
        rng = np.random.default_rng(SEED + i * 1000)

        max_test = len(test_raw) - lead
        n_eval   = min(N_EVAL, max_test)
        idx = rng.choice(max_test, size=n_eval, replace=False)
        idx.sort()

        cond_phys   = test_raw[idx]           # (n_eval, N) physical
        target_phys = test_raw[idx + lead]    # (n_eval, N) physical

        target_norm = (target_phys - mu) / sigma   # (n_eval, N)

        # Climatological ensemble (same as mvn_eval.py)
        clim_idx = rng.integers(0, len(train_pool), size=(n_eval, N_ENS))
        clim_ens = (train_pool[clim_idx] - mu) / sigma   # (n_eval, N_ENS, N)

        # Dynamical ensemble
        dyn_ens_norm = np.empty((n_eval, N_ENS, N), dtype=np.float64)
        for t in range(n_eval):
            if t % 50 == 0:
                print(f"  lead={lead}  sample {t}/{n_eval} ...", flush=True)
            members_phys = integrate_ensemble(
                cond_phys[t], lead, N_ENS, sigma_ic, sigma_model,
                rng=np.random.default_rng(SEED + i * 1000 + t + 1),
                sigma_bias=sigma_bias,
            )
            dyn_ens_norm[t] = (members_phys - mu) / sigma

        crps_dyn  = crps_per_sample(target_norm, dyn_ens_norm)
        crps_clim = crps_per_sample(target_norm, clim_ens)
        vs_dyn    = vs_per_sample(target_norm, dyn_ens_norm)
        vs_clim   = vs_per_sample(target_norm, clim_ens)

        crps_arr[i] = 1.0 - crps_dyn.mean() / crps_clim.mean()
        vs_arr[i]   = 1.0 - vs_dyn.mean()   / vs_clim.mean()

        boot_rng = np.random.default_rng(SEED + i * 1000 + 500)
        crps_dist_arr[i], vs_dist_arr[i] = bootstrap_skill(
            crps_dyn, crps_clim, vs_dyn, vs_clim, N_BOOT, boot_rng
        )

        cm = np.median(crps_dist_arr[i])
        vm = np.median(vs_dist_arr[i])
        c2, c97 = np.percentile(crps_dist_arr[i], [2.5, 97.5])
        v2, v97 = np.percentile(vs_dist_arr[i],   [2.5, 97.5])
        print(f"lead={lead}  CRPS skill={cm:+.4f} [{c2:+.4f}, {c97:+.4f}]  "
              f"VS skill={vm:+.4f} [{v2:+.4f}, {v97:+.4f}]")

    np.savez(
        OUT_PATH,
        lead_values=LEAD_VALUES,
        crps_skill=crps_arr,
        vs_skill=vs_arr,
        crps_skill_dist=crps_dist_arr,
        vs_skill_dist=vs_dist_arr,
        sigma_ic_frac=sigma_ic_frac,
        sigma_ic=sigma_ic,
        sigma_model=sigma_model,
        sigma_bias=sigma_bias,
    )
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_ic_frac", type=float, default=SIGMA_IC_FRAC_DEFAULT,
                        help="IC perturbation as fraction of sigma_clim (default %(default)s)")
    parser.add_argument("--sigma_model",   type=float, default=SIGMA_MODEL_DEFAULT,
                        help="Per-step residual noise std, i.i.d. per member per node (default %(default)s)")
    parser.add_argument("--sigma_bias",    type=float, default=SIGMA_BIAS_DEFAULT,
                        help="Per-initialisation shared bias std b~N(0,σ²I) (default %(default)s)")
    parser.add_argument("--data",        type=str, default=DATA_PATH)
    parser.add_argument("--out",         type=str, default=OUT_PATH)
    args = parser.parse_args()
    OUT_PATH  = args.out
    DATA_PATH = args.data
    run(args.sigma_ic_frac, args.sigma_model, args.sigma_bias)
