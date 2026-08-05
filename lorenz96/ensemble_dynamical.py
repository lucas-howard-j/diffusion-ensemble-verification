"""
Perfect-model ensemble forecast for L96.

For each test IC x₀, generates M ensemble members by perturbing x₀ in
normalised space by N(0, σ_ic²·I), denormalises, and integrates each for
lead_steps using the exact L96 RK4 integrator.

Integration is fully vectorised over ensemble members (no Python loop per
member), so 2000 × 500 members at lead 20 runs in ~5 s on a single CPU core.

Usage:
    python -m lorenz96.ensemble_dynamical \\
        [--ckpt CKPT] [--data PATH] [--sigma-ic FLOAT] \\
        [--n-eval N] [--n-ens M] [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lorenz96.config import F as L96_F, TRAIN_END, TEST_START
from lorenz96.dataset import load_trajectory
from lorenz96.eval import crps_ensemble, variogram_score, bootstrap_skill_ci


# ---------------------------------------------------------------------------
# Vectorised L96 integrator  (operates on batches of shape (B, N))
# ---------------------------------------------------------------------------

def _l96_deriv_batch(x: np.ndarray) -> np.ndarray:
    """L96 tendency for a batch of states.  x: (B, N) → (B, N)."""
    return (
        (np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * np.roll(x, 1, axis=1)
        - x + L96_F
    )


def _rk4_step_batch(x: np.ndarray, dt: float) -> np.ndarray:
    k1 = _l96_deriv_batch(x)
    k2 = _l96_deriv_batch(x + 0.5 * dt * k1)
    k3 = _l96_deriv_batch(x + 0.5 * dt * k2)
    k4 = _l96_deriv_batch(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_batch(x0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Integrate n_steps from a batch of ICs.  x0: (B, N) → x_T: (B, N)."""
    x = x0.copy()
    for _ in range(n_steps):
        x = _rk4_step_batch(x, dt)
    return x


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_dynamical_ensemble(
    ckpt_path:  str   = "lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt",
    data_path:  str   = "lorenz96/data/l96_trajectory.npz",
    sigma_ic:   float = 0.01,
    n_eval:     int   = 2_000,
    n_ens:      int   = 500,
    out_path:   str   = "lorenz96/ensemble_dynamical/results.npz",
    seed:       int   = 42,
    chunk_size: int   = 200,
) -> dict:
    """
    Returns dict with keys: crps_skill, vs_skill, spread, sigma_ic.
    Also writes an npz to out_path with the same keys plus crps_node.
    """
    rng = np.random.default_rng(seed)

    states, data_cfg = load_trajectory(data_path)
    dt   = data_cfg["dt"]

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mu    = ckpt["mu"]           # (N,) raw-space mean
    sigma = ckpt["sigma"]        # (N,) raw-space std
    lead  = ckpt["lead_steps"]
    N     = len(mu)

    # Test IC selection (same logic as eval.py)
    test_raw = states[TEST_START:]
    max_test = len(test_raw) - lead
    idx = rng.choice(max_test, size=min(n_eval, max_test), replace=False)
    idx.sort()

    cond_raw    = test_raw[idx]          # (n, N) raw
    target_raw  = test_raw[idx + lead]   # (n, N) raw
    target_norm = (target_raw - mu) / sigma

    # Climatological baseline
    train_raw = states[:TRAIN_END]
    clim_idx  = rng.integers(0, len(train_raw), size=(len(idx), n_ens))
    clim_ens_norm = (train_raw[clim_idx] - mu) / sigma   # (n, M, N)

    n = len(idx)
    ens_norm = np.empty((n, n_ens, N), dtype=np.float32)

    print(f"Integrating {n} × {n_ens} members  "
          f"(lead={lead} steps = {lead*dt:.2f} TU, σ_ic={sigma_ic:.4f}) ...", flush=True)

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        nc = i1 - i0

        if i0 % (chunk_size * 5) == 0:
            print(f"  IC {i0}/{n}", flush=True)

        ic_raw  = cond_raw[i0:i1]               # (nc, N) raw
        ic_norm = (ic_raw - mu) / sigma          # (nc, N) normalised

        # Draw perturbations: (nc, n_ens, N), then reshape to (nc*n_ens, N)
        eps = rng.standard_normal((nc, n_ens, N)) * sigma_ic
        ic_pert_norm = ic_norm[:, None, :] + eps   # (nc, n_ens, N)
        ic_pert_raw  = ic_pert_norm * sigma[None, None, :] + mu[None, None, :]

        batch_raw = ic_pert_raw.reshape(nc * n_ens, N)  # (nc*n_ens, N)
        final_raw = integrate_batch(batch_raw, lead, dt) # (nc*n_ens, N)
        final_norm = (final_raw - mu) / sigma            # (nc*n_ens, N)
        ens_norm[i0:i1] = final_norm.reshape(nc, n_ens, N).astype(np.float32)

    print(f"  Done.", flush=True)

    # Scores
    crps_ens  = crps_ensemble(target_norm, ens_norm).mean(0)   # (N,)
    crps_clim = crps_ensemble(target_norm, clim_ens_norm).mean(0)
    crps_node = 1 - crps_ens / crps_clim
    crps_skill = float(crps_node.mean())

    vs_ens  = variogram_score(target_norm, ens_norm)
    vs_clim = variogram_score(target_norm, clim_ens_norm)
    vs_skill = float(1 - vs_ens / vs_clim)

    spread = float(ens_norm.std(axis=1).mean())

    print(f"CRPS skill={crps_skill:+.4f}  VS skill={vs_skill:+.4f}  "
          f"spread={spread:.4f}  σ_ic={sigma_ic:.4f}")

    print("Bootstrapping CIs ...", flush=True)
    ci = bootstrap_skill_ci(target_norm, ens_norm, clim_ens_norm)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        crps_skill=crps_skill, vs_skill=vs_skill,
        crps_node=crps_node, spread=spread, sigma_ic=sigma_ic,
        **{k: np.float64(v) for k, v in ci.items()},
    )
    print(f"Saved → {out_path}")
    return {"crps_skill": crps_skill, "vs_skill": vs_skill,
            "spread": spread, "sigma_ic": sigma_ic}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt")
    parser.add_argument("--data",      default="lorenz96/data/l96_trajectory.npz")
    parser.add_argument("--sigma-ic",  type=float, default=0.01)
    parser.add_argument("--n-eval",    type=int, default=2_000)
    parser.add_argument("--n-ens",     type=int, default=500)
    parser.add_argument("--out",       default="lorenz96/ensemble_dynamical/results.npz")
    parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()

    run_dynamical_ensemble(
        ckpt_path=args.ckpt, data_path=args.data,
        sigma_ic=args.sigma_ic, n_eval=args.n_eval, n_ens=args.n_ens,
        out_path=args.out, chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
