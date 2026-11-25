"""
Deterministic emulator σ_ic sweep — canonical code.

For each σ_ic fraction, runs 2000 test ICs × 500 ensemble members through
the trained ConvDeterministicNet (lead=20 only) and bootstraps CRPS/VS skill.

Output: lorenz96/sweep/det_results_sic{tag}.npz per σ_ic value.
Format matches dyn_results_sic*.npz (single-lead case):
    crps_skill       scalar float  — point estimate
    vs_skill         scalar float
    crps_skill_dist  (n_boot,) float  — full bootstrap distributions
    vs_skill_dist    (n_boot,) float
    sigma_ic_frac    scalar
    sigma_ic         scalar  — physical (σ_clim × frac)

Run:
    PYTHONPATH=. python -m lorenz96.det_sweep
    PYTHONPATH=. python -m lorenz96.det_sweep --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lorenz96.config import TRAIN_END, TEST_START
from lorenz96.dataset import load_trajectory
from lorenz96.eval import crps_ensemble, vs_per_ic
from lorenz96.model import ConvDeterministicNet

CKPT_PATH   = Path("lorenz96/checkpoints/det_emulator.pt")
DATA_PATH   = Path("lorenz96/l96_trajectory.npz")
OUT_DIR     = Path("lorenz96/sweep")

SIGMA_IC_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
N_EVAL   = 2_000
N_ENS    = 500
N_BOOT   = 1_000
CHUNK    = 100
SEED     = 42
IC_WINDOW_LEAD = 60   # = max(dyn_sweep.LEAD_VALUES); shared IC selection


def _tag(frac: float) -> str:
    return f"sic{int(round(frac * 100)):02d}"


def _bootstrap(
    crps_ic_ens:  np.ndarray,
    crps_ic_clim: np.ndarray,
    vs_ic_ens:    np.ndarray,
    vs_ic_clim:   np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(crps_ic_ens)
    crps_b = np.empty(n_boot)
    vs_b   = np.empty(n_boot)
    for b in range(n_boot):
        idx       = rng.integers(0, n, size=n)
        crps_b[b] = 1 - crps_ic_ens[idx].mean() / crps_ic_clim[idx].mean()
        vs_b[b]   = 1 - vs_ic_ens[idx].mean()   / vs_ic_clim[idx].mean()
    return crps_b, vs_b


def run(n_eval: int, n_ens: int, n_boot: int) -> None:
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    rng = np.random.default_rng(SEED)

    ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    mu    = ckpt["mu"]
    sigma = ckpt["sigma"]
    lead  = int(ckpt["lead_steps"])
    cfg   = json.loads(ckpt["config"])
    N     = len(mu)

    sigma_clim = float(sigma.mean())

    model = ConvDeterministicNet(
        hidden_ch=cfg["hidden_ch"], n_res=cfg["n_res"], kernel_size=cfg["kernel"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    states, _ = load_trajectory(str(DATA_PATH))
    test_raw  = states[TEST_START:]
    train_raw = states[:TRAIN_END]

    # Select ICs exactly as dyn_sweep.py does (window = its longest lead, 60
    # steps; same seed), so the two σ_IC sweeps in Figure 5 share their test
    # ICs and climatological draws.
    max_avail = len(test_raw) - IC_WINDOW_LEAD
    n_use     = min(n_eval, max_avail)
    ic_idx    = rng.choice(max_avail, size=n_use, replace=False)
    ic_idx.sort()

    cond_raw    = test_raw[ic_idx]
    target_raw  = test_raw[ic_idx + lead]
    target_norm = (target_raw - mu) / sigma        # (n, N)
    cond_norm   = (cond_raw - mu) / sigma          # (n, N)

    clim_idx      = rng.integers(0, len(train_raw), size=(n_use, n_ens))
    clim_ens_norm = (train_raw[clim_idx] - mu) / sigma   # (n, M, N)

    crps_ic_clim = crps_ensemble(target_norm, clim_ens_norm).mean(1)   # (n,)
    vs_ic_clim   = vs_per_ic(target_norm, clim_ens_norm)               # (n,)

    print(f"Det. emulator sweep  lead={lead}  n_eval={n_use}  n_ens={n_ens}  device={device}")

    for frac in SIGMA_IC_FRACS:
        sigma_ic_norm = frac

        ens_norm = np.empty((n_use, n_ens, N), dtype=np.float32)

        with torch.no_grad():
            for i0 in range(0, n_use, CHUNK):
                i1  = min(i0 + CHUNK, n_use)
                nc  = i1 - i0
                eps = rng.standard_normal((nc, n_ens, N)) * sigma_ic_norm
                ic_pert = cond_norm[i0:i1, None, :] + eps          # (nc, M, N)
                batch   = torch.from_numpy(
                    ic_pert.reshape(nc * n_ens, N).astype(np.float32)
                ).to(device)
                pred = model(batch).cpu().numpy()
                ens_norm[i0:i1] = pred.reshape(nc, n_ens, N)

        crps_ic_ens = crps_ensemble(target_norm, ens_norm).mean(1)
        vs_ic_ens   = vs_per_ic(target_norm, ens_norm)

        crps_pt = float(1 - crps_ic_ens.mean() / crps_ic_clim.mean())
        vs_pt   = float(1 - vs_ic_ens.mean()   / vs_ic_clim.mean())

        crps_b, vs_b = _bootstrap(
            crps_ic_ens, crps_ic_clim, vs_ic_ens, vs_ic_clim, n_boot, rng
        )

        tag     = _tag(frac)
        outpath = OUT_DIR / f"det_results_{tag}.npz"
        np.savez(
            outpath,
            crps_skill      = np.float64(crps_pt),
            vs_skill        = np.float64(vs_pt),
            crps_skill_dist = crps_b,
            vs_skill_dist   = vs_b,
            sigma_ic_frac   = np.float64(frac),
            sigma_ic        = np.float64(frac * sigma_clim),
        )
        print(f"  σ_ic={frac:.2f}  CRPS={crps_pt:+.4f}  VS={vs_pt:+.4f}  → {outpath}")


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
