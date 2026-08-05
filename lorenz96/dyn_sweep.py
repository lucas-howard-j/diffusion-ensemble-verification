"""
Dynamical σ_ic sweep — canonical rewrite.

Replaces lorenz96/sweep/dyn_eval.py.  Uses:
  - lorenz96.ensemble_dynamical.integrate_batch  (vectorised RK4)
  - lorenz96.eval.crps_ensemble, vs_per_ic       (canonical scoring)

Writes lorenz96/sweep/dyn_results_sic{tag}.npz, overwriting the old
n_eval=200 / n_ens=20 files with n_eval=2000 / n_ens=500 results that
are consistent with the ensemble comparison figure.

Output format (per σ_ic file) matches dyn_eval.py so figure_dyn_sweep.py
works unchanged:
    lead_values      (n_leads,)     int
    crps_skill       (n_leads,)     float  — point estimate
    vs_skill         (n_leads,)     float
    crps_skill_dist  (n_leads, n_boot) float  — full bootstrap distributions
    vs_skill_dist    (n_leads, n_boot) float
    sigma_ic_frac    scalar
    sigma_ic         scalar  — physical units (mean physical σ_clim × frac)

Run:
    PYTHONPATH=. python -m lorenz96.dyn_sweep
    PYTHONPATH=. python -m lorenz96.dyn_sweep --dry-run   # smoke test (n_eval=20, n_ens=10)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lorenz96.config import TRAIN_END, TEST_START
from lorenz96.dataset import load_trajectory
from lorenz96.ensemble_dynamical import integrate_batch
from lorenz96.eval import crps_ensemble, vs_per_ic

CKPT_PATH   = Path("lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt")
DATA_PATH   = Path("lorenz96/l96_trajectory.npz")
OUT_DIR     = Path("lorenz96/sweep")

LEAD_VALUES    = [20, 40, 60]
SIGMA_IC_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

N_EVAL   = 2_000
N_ENS    = 500
N_BOOT   = 1_000
CHUNK    = 200   # ICs per integrate_batch call
SEED     = 42


def _tag(frac: float) -> str:
    return f"sic{int(round(frac * 100)):02d}"


def _bootstrap(
    crps_ic_ens:  np.ndarray,   # (n,)
    crps_ic_clim: np.ndarray,   # (n,)
    vs_ic_ens:    np.ndarray,   # (n,)
    vs_ic_clim:   np.ndarray,   # (n,)
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired bootstrap → (crps_dist, vs_dist) each (n_boot,)."""
    n = len(crps_ic_ens)
    crps_b = np.empty(n_boot)
    vs_b   = np.empty(n_boot)
    for b in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        crps_b[b] = 1 - crps_ic_ens[idx].mean() / crps_ic_clim[idx].mean()
        vs_b[b]   = 1 - vs_ic_ens[idx].mean()   / vs_ic_clim[idx].mean()
    return crps_b, vs_b


def run(n_eval: int, n_ens: int, n_boot: int) -> None:
    rng = np.random.default_rng(SEED)

    # Load normalisation stats and dt from the lead-20 checkpoint.
    # mu/sigma are dataset-level (same for all leads); lead_steps read per call.
    ckpt  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    mu    = ckpt["mu"]      # (N,)
    sigma = ckpt["sigma"]   # (N,)

    states, data_cfg = load_trajectory(str(DATA_PATH))
    dt = data_cfg["dt"]

    # σ_clim: mean physical std across nodes (for reporting only — perturbations
    # are applied in normalised space where σ_clim = 1 per node by construction)
    sigma_clim = float(sigma.mean())

    test_raw  = states[TEST_START:]
    train_raw = states[:TRAIN_END]

    # Pre-select test ICs for the longest lead so all leads use the same ICs.
    max_lead  = max(LEAD_VALUES)
    max_avail = len(test_raw) - max_lead
    n_use     = min(n_eval, max_avail)
    ic_idx    = rng.choice(max_avail, size=n_use, replace=False)
    ic_idx.sort()

    cond_raw = test_raw[ic_idx]   # (n, N) — IC states (physical)

    # Climatological baseline: same random training draws for all runs so CI
    # comparisons are paired.  Regenerate per lead so clim lead = model lead.
    clim_idx = rng.integers(0, len(train_raw), size=(n_use, n_ens))

    for frac in SIGMA_IC_FRACS:
        sigma_ic_norm = frac   # in normalised space, σ_clim ≡ 1

        crps_skill_arr = np.empty(len(LEAD_VALUES))
        vs_skill_arr   = np.empty(len(LEAD_VALUES))
        crps_dist_arr  = np.empty((len(LEAD_VALUES), n_boot))
        vs_dist_arr    = np.empty((len(LEAD_VALUES), n_boot))

        print(f"\n── σ_ic_frac={frac:.2f}  (σ_ic_norm={sigma_ic_norm:.4f}) ──")

        for li, lead in enumerate(LEAD_VALUES):
            target_raw  = test_raw[ic_idx + lead]         # (n, N)
            target_norm = (target_raw - mu) / sigma        # (n, N)

            clim_ens_norm = (train_raw[clim_idx] - mu) / sigma   # (n, M, N)

            # Integrate ensemble members in chunks.
            ens_norm = np.empty((n_use, n_ens, len(mu)), dtype=np.float32)

            for i0 in range(0, n_use, CHUNK):
                i1 = min(i0 + CHUNK, n_use)
                ic_norm = (cond_raw[i0:i1] - mu) / sigma          # (nc, N)
                eps     = rng.standard_normal((i1 - i0, n_ens, len(mu))) * sigma_ic_norm
                ic_pert_norm = ic_norm[:, None, :] + eps           # (nc, M, N)
                ic_pert_raw  = ic_pert_norm * sigma + mu           # (nc, M, N) — broadcast
                batch_raw    = ic_pert_raw.reshape(-1, len(mu))    # (nc*M, N)
                final_raw    = integrate_batch(batch_raw, lead, dt)
                final_norm   = (final_raw - mu) / sigma
                ens_norm[i0:i1] = final_norm.reshape(i1 - i0, n_ens, len(mu))

            # Point estimates
            crps_ic_ens  = crps_ensemble(target_norm, ens_norm).mean(1)    # (n,)
            crps_ic_clim = crps_ensemble(target_norm, clim_ens_norm).mean(1)
            vs_ic_ens_   = vs_per_ic(target_norm, ens_norm)                 # (n,)
            vs_ic_clim_  = vs_per_ic(target_norm, clim_ens_norm)

            crps_pt = float(1 - crps_ic_ens.mean() / crps_ic_clim.mean())
            vs_pt   = float(1 - vs_ic_ens_.mean()  / vs_ic_clim_.mean())
            crps_skill_arr[li] = crps_pt
            vs_skill_arr[li]   = vs_pt

            print(f"  lead={lead:2d}  CRPS={crps_pt:+.4f}  VS={vs_pt:+.4f}", flush=True)

            # Bootstrap distributions (needed for violin plots)
            crps_b, vs_b = _bootstrap(
                crps_ic_ens, crps_ic_clim, vs_ic_ens_, vs_ic_clim_, n_boot, rng
            )
            crps_dist_arr[li] = crps_b
            vs_dist_arr[li]   = vs_b

        tag     = _tag(frac)
        outpath = OUT_DIR / f"dyn_results_{tag}.npz"
        np.savez(
            outpath,
            lead_values     = np.array(LEAD_VALUES),
            crps_skill      = crps_skill_arr,
            vs_skill        = vs_skill_arr,
            crps_skill_dist = crps_dist_arr,
            vs_skill_dist   = vs_dist_arr,
            sigma_ic_frac   = np.float64(frac),
            sigma_ic        = np.float64(frac * sigma_clim),   # physical, for reference
            sigma_model     = np.float64(0.0),
            sigma_bias      = np.float64(0.0),
        )
        print(f"  Saved → {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: n_eval=20, n_ens=10, n_boot=50")
    args = parser.parse_args()

    if args.dry_run:
        run(n_eval=20, n_ens=10, n_boot=50)
    else:
        run(n_eval=N_EVAL, n_ens=N_ENS, n_boot=N_BOOT)


if __name__ == "__main__":
    main()
