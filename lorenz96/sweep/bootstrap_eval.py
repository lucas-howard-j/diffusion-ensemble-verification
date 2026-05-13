"""
Bootstrap error estimates for CRPS skill and VS skill across all sweep checkpoints.

For each (lead, n_train):
  1. Load checkpoint, generate n_eval × n_ens ensemble (single pass)
  2. Compute per-sample CRPS and per-sample VS
  3. Bootstrap N_BOOT times, save full distribution

Saves sweep_errors.npz with arrays:
    crps_skill_dist, vs_skill_dist  — shape (n_leads, n_sizes, N_BOOT)
    lead_values, n_train_values

Usage:
    python lorenz96/sweep/bootstrap_eval.py [--device DEVICE]
"""

import argparse
import os
import numpy as np
import torch

from lorenz96.config  import N, TEST_START, TRAIN_END
from lorenz96.dataset import load_trajectory
from lorenz96.model   import DDPM, ConvScoreNet

CKPT_DIR     = "lorenz96/sweep/checkpoints"
DATA_PATH    = "lorenz96/data/l96_trajectory.npz"
OUT_PATH     = "lorenz96/sweep/sweep_errors.npz"
N_EVAL       = 200
N_ENS        = 20
SAMPLE_STEPS = 200
N_BOOT       = 1000
SEED         = 0

LEAD_VALUES    = [20, 40, 60]
N_TRAIN_VALUES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 249_000]


# ---------------------------------------------------------------------------
# Per-sample score functions
# ---------------------------------------------------------------------------

def crps_per_sample(truth: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """CRPS per sample, mean over D nodes.  truth:(n,D)  ens:(n,M,D) → (n,)"""
    M      = ens.shape[1]
    mae    = np.abs(ens - truth[:, None, :]).mean(axis=1)
    spread = np.abs(ens[:, :, None, :] - ens[:, None, :, :]).sum(axis=(1, 2)) / (2 * M * M)
    return (mae - spread).mean(axis=1)


def vs_per_sample(truth: np.ndarray, ens: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Variogram score per sample, mean over all D*(D-1)/2 pairs. → (n,)"""
    D = truth.shape[1]
    i_idx, j_idx = np.triu_indices(D, k=1)
    obs_diff = np.abs(truth[:, i_idx] - truth[:, j_idx]) ** p
    ens_diff = np.abs(ens[:, :, i_idx] - ens[:, :, j_idx]) ** p
    ens_mean = ens_diff.mean(axis=1)
    return ((obs_diff - ens_mean) ** 2).mean(axis=1)


def bootstrap_paired(ddpm_crps, clim_crps, ddpm_vs, clim_vs, n_boot=N_BOOT, rng=None):
    """Bootstrap CRPS skill, VS skill, and their difference using the same resamples.

    Using the same idx per replicate captures the correlation between the two metrics,
    giving a valid CI on the difference CRPS_skill - VS_skill.

    Returns (crps_dist, vs_dist, diff_dist) each shape (n_boot,).
    """
    n = len(ddpm_crps)
    crps_dist = np.empty(n_boot)
    vs_dist   = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        crps_dist[k] = 1.0 - ddpm_crps[idx].mean() / clim_crps[idx].mean()
        vs_dist[k]   = 1.0 - ddpm_vs[idx].mean()   / clim_vs[idx].mean()
    return crps_dist, vs_dist, crps_dist - vs_dist


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(SEED)

    states, data_cfg = load_trajectory(DATA_PATH)

    n_leads = len(LEAD_VALUES)
    n_sizes = len(N_TRAIN_VALUES)
    crps_dist = np.full((n_leads, n_sizes, N_BOOT), np.nan)
    vs_dist   = np.full((n_leads, n_sizes, N_BOOT), np.nan)
    diff_dist = np.full((n_leads, n_sizes, N_BOOT), np.nan)

    for i, lead in enumerate(LEAD_VALUES):
        for j, n_train in enumerate(N_TRAIN_VALUES):
            ckpt_path = os.path.join(CKPT_DIR, f"ddpm_lead{lead}_{n_train}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  skip  lead={lead} n_train={n_train:,}  (no checkpoint)")
                continue

            print(f"lead={lead}  n_train={n_train:,} ...", flush=True)

            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            mu    = ckpt["mu"]
            sigma = ckpt["sigma"]

            cfg = __import__("json").loads(ckpt["config"])
            net = ConvScoreNet(
                hidden_ch=cfg["hidden_ch"], n_res=cfg["n_res"],
                emb_dim=cfg["emb_dim"], kernel_size=cfg["kernel"],
            )
            model = DDPM(net, T=cfg["T_diff"]).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            # Test pairs
            test_raw = states[TEST_START:]
            max_test = len(test_raw) - lead
            idx = rng.choice(max_test, size=min(N_EVAL, max_test), replace=False)
            idx.sort()

            cond_norm   = (test_raw[idx]        - mu) / sigma
            target_norm = (test_raw[idx + lead] - mu) / sigma

            # Climatological ensemble
            train_raw = states[:TRAIN_END]
            clim_idx  = rng.integers(0, len(train_raw), size=(len(idx), N_ENS))
            clim_ens  = (train_raw[clim_idx] - mu) / sigma

            # DDPM ensemble
            cond_t = torch.from_numpy(cond_norm).float().to(device)
            with torch.no_grad():
                ddpm_ens = model.sample(cond_t, n_samples=N_ENS,
                                        sample_steps=SAMPLE_STEPS).cpu().numpy()

            # Per-sample scores
            crps_ddpm = crps_per_sample(target_norm, ddpm_ens)
            crps_clim = crps_per_sample(target_norm, clim_ens)
            vs_ddpm   = vs_per_sample(target_norm, ddpm_ens)
            vs_clim   = vs_per_sample(target_norm, clim_ens)

            boot_rng = np.random.default_rng(SEED + i * 100 + j)
            crps_dist[i, j], vs_dist[i, j], diff_dist[i, j] = bootstrap_paired(
                crps_ddpm, crps_clim, vs_ddpm, vs_clim, rng=boot_rng)

            cm = np.median(crps_dist[i, j])
            vm = np.median(vs_dist[i, j])
            dm = np.median(diff_dist[i, j])
            c2, c97 = np.percentile(crps_dist[i, j], [2.5, 97.5])
            v2, v97 = np.percentile(vs_dist[i, j],   [2.5, 97.5])
            d2, d97 = np.percentile(diff_dist[i, j], [2.5, 97.5])
            print(f"  CRPS skill {cm:+.4f} [{c2:+.4f}, {c97:+.4f}]   "
                  f"VS skill {vm:+.4f} [{v2:+.4f}, {v97:+.4f}]   "
                  f"diff {dm:+.4f} [{d2:+.4f}, {d97:+.4f}]", flush=True)

    np.savez(OUT_PATH,
             lead_values=LEAD_VALUES, n_train_values=N_TRAIN_VALUES,
             crps_skill_dist=crps_dist, vs_skill_dist=vs_dist,
             diff_skill_dist=diff_dist)
    print(f"\nSaved → {OUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.device)


if __name__ == "__main__":
    main()
