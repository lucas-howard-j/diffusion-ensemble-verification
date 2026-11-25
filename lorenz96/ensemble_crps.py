"""
CRPS-trained generative emulator ensemble for L96.

For each test IC x₀, generates M ensemble members by drawing independent noise
fields and forwarding each through the trained ConvGenerativeNet.  No IC
perturbation: all spread comes from the learned noise map.  Uses the same test
ICs, climatological baseline, seed and scoring as ensemble_deterministic.py, so
results are directly comparable to the DDPM / dynamical / det-emulator numbers.

Usage:
    python -m lorenz96.ensemble_crps \\
        [--ckpt CKPT] [--data PATH] [--n-eval N] [--n-ens M] [--out PATH]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lorenz96.config import TRAIN_END, TEST_START
from lorenz96.dataset import load_trajectory
from lorenz96.eval import crps_ensemble, variogram_score, bootstrap_skill_ci
from lorenz96.model import ConvGenerativeNet


def run_crps_ensemble(
    ckpt_path:  str   = "lorenz96/checkpoints/crps_emulator.pt",
    data_path:  str   = "lorenz96/l96_trajectory.npz",
    n_eval:     int   = 2_000,
    n_ens:      int   = 500,
    out_path:   str   = "lorenz96/ensemble_results/crps_results.npz",
    device:     str | None = None,
    seed:       int   = 42,
    chunk_size: int   = 100,
) -> dict:
    """
    Returns dict with keys: crps_skill, vs_skill, spread.
    Also writes an npz to out_path.
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    states, _ = load_trajectory(data_path)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    mu    = ckpt["mu"]
    sigma = ckpt["sigma"]
    lead  = ckpt["lead_steps"]
    cfg   = json.loads(ckpt["config"])
    N     = len(mu)

    model = ConvGenerativeNet(
        hidden_ch=cfg["hidden_ch"], n_res=cfg["n_res"],
        kernel_size=cfg["kernel"], noise_ch=cfg.get("noise_ch", 1), state_dim=N,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Test ICs (identical protocol to ensemble_deterministic.py)
    test_raw = states[TEST_START:]
    max_test = len(test_raw) - lead
    idx = rng.choice(max_test, size=min(n_eval, max_test), replace=False)
    idx.sort()

    cond_raw    = test_raw[idx]
    target_raw  = test_raw[idx + lead]
    target_norm = (target_raw - mu) / sigma

    # Climatological baseline
    train_raw = states[:TRAIN_END]
    clim_idx  = rng.integers(0, len(train_raw), size=(len(idx), n_ens))
    clim_ens_norm = (train_raw[clim_idx] - mu) / sigma

    n = len(idx)
    ens_norm = np.empty((n, n_ens, N), dtype=np.float32)

    print(f"Running {n} × {n_ens} members through CRPS emulator  "
          f"(device={device}) ...", flush=True)

    cond_norm_all = (cond_raw - mu) / sigma   # (n, N)

    with torch.no_grad():
        for i0 in range(0, n, chunk_size):
            i1 = min(i0 + chunk_size, n)
            nc = i1 - i0

            if i0 % (chunk_size * 10) == 0:
                print(f"  IC {i0}/{n}", flush=True)

            ic = torch.from_numpy(cond_norm_all[i0:i1].astype(np.float32)).to(device)
            cond_rep = ic.repeat_interleave(n_ens, dim=0)        # (nc*n_ens, N)
            z = model.sample_latent(nc * n_ens, device)
            pred = model(cond_rep, z).cpu().numpy()               # (nc*n_ens, N)
            ens_norm[i0:i1] = pred.reshape(nc, n_ens, N)

    print(f"  Done.", flush=True)

    # Scores
    crps_ens  = crps_ensemble(target_norm, ens_norm).mean(0)
    crps_clim = crps_ensemble(target_norm, clim_ens_norm).mean(0)
    crps_node = 1 - crps_ens / crps_clim
    # Ratio of means over ICs and nodes (the statistic bootstrap_skill_ci resamples)
    crps_skill = float(1 - crps_ens.mean() / crps_clim.mean())

    vs_ens  = variogram_score(target_norm, ens_norm)
    vs_clim = variogram_score(target_norm, clim_ens_norm)
    vs_skill = float(1 - vs_ens / vs_clim)

    spread = float(ens_norm.std(axis=1).mean())

    print(f"CRPS skill={crps_skill:+.4f}  VS skill={vs_skill:+.4f}  "
          f"VS/CRPS={vs_skill / crps_skill:.4f}  spread={spread:.4f}")

    print("Bootstrapping CIs ...", flush=True)
    ci = bootstrap_skill_ci(target_norm, ens_norm, clim_ens_norm)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        crps_skill=crps_skill, vs_skill=vs_skill,
        crps_node=crps_node, spread=spread,
        **{k: np.float64(v) for k, v in ci.items()},
    )
    print(f"Saved → {out_path}")
    return {"crps_skill": crps_skill, "vs_skill": vs_skill, "spread": spread}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="lorenz96/checkpoints/crps_emulator.pt")
    parser.add_argument("--data",       default="lorenz96/l96_trajectory.npz")
    parser.add_argument("--n-eval",     type=int, default=2_000)
    parser.add_argument("--n-ens",      type=int, default=500)
    parser.add_argument("--out",        default="lorenz96/ensemble_results/crps_results.npz")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    run_crps_ensemble(
        ckpt_path=args.ckpt, data_path=args.data,
        n_eval=args.n_eval, n_ens=args.n_ens,
        out_path=args.out, device=args.device, chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
