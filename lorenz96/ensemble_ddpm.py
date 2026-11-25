"""
DDPM ensemble for L96, scored on the shared comparison protocol.

Uses the same test ICs, climatological baseline, seed and scoring as
ensemble_crps.py / ensemble_deterministic.py / ensemble_dynamical.py
(seed 42: rng.choice for the ICs, then rng.integers for the climatology),
so the DDPM bar in Figure 6 is paired with the other four models.

Usage:
    python -m lorenz96.ensemble_ddpm \\
        [--ckpt CKPT] [--data PATH] [--n-eval N] [--n-ens M] \\
        [--sample-steps 200] [--out PATH]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lorenz96.config import TRAIN_END, TEST_START
from lorenz96.dataset import load_trajectory
from lorenz96.eval import crps_ensemble, variogram_score, bootstrap_skill_ci
from lorenz96.model import DDPM, ConvScoreNet


def run_ddpm_ensemble(
    ckpt_path:    str   = "lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt",
    data_path:    str   = "lorenz96/l96_trajectory.npz",
    n_eval:       int   = 2_000,
    n_ens:        int   = 500,
    sample_steps: int   = 200,
    out_path:     str   = "lorenz96/ensemble_results/ddpm_results.npz",
    device:       str | None = None,
    seed:         int   = 42,
    chunk_size:   int   = 20,
) -> dict:
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

    net = ConvScoreNet(
        hidden_ch=cfg["hidden_ch"], n_res=cfg["n_res"],
        emb_dim=cfg["emb_dim"], kernel_size=cfg["kernel"],
    )
    model = DDPM(net, T=cfg["T_diff"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Test ICs and climatology: identical protocol to ensemble_crps.py
    test_raw = states[TEST_START:]
    max_test = len(test_raw) - lead
    idx = rng.choice(max_test, size=min(n_eval, max_test), replace=False)
    idx.sort()

    cond_raw    = test_raw[idx]
    target_raw  = test_raw[idx + lead]
    target_norm = (target_raw - mu) / sigma

    train_raw = states[:TRAIN_END]
    clim_idx  = rng.integers(0, len(train_raw), size=(len(idx), n_ens))
    clim_ens_norm = (train_raw[clim_idx] - mu) / sigma

    n = len(idx)
    ens_norm = np.empty((n, n_ens, N), dtype=np.float32)
    cond_norm_all = (cond_raw - mu) / sigma

    print(f"Sampling DDPM: {n} × {n_ens} members, lead={lead}, "
          f"steps={sample_steps}  (device={device}) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n, chunk_size):
            i1 = min(i0 + chunk_size, n)
            ic = torch.from_numpy(cond_norm_all[i0:i1].astype(np.float32)).to(device)
            ens = model.sample(ic, n_samples=n_ens, sample_steps=sample_steps)
            ens_norm[i0:i1] = ens.cpu().numpy()
            if (i0 // chunk_size) % 10 == 0:
                print(f"  IC {i1}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    print("  Done.", flush=True)

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
        n_eval=n, n_ens=n_ens, sample_steps=sample_steps,
        **{k: np.float64(v) for k, v in ci.items()},
    )
    print(f"Saved → {out_path}")
    return {"crps_skill": crps_skill, "vs_skill": vs_skill, "spread": spread}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         default="lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt")
    parser.add_argument("--data",         default="lorenz96/l96_trajectory.npz")
    parser.add_argument("--n-eval",       type=int, default=2_000)
    parser.add_argument("--n-ens",        type=int, default=500)
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--out",          default="lorenz96/ensemble_results/ddpm_results.npz")
    parser.add_argument("--device",       default=None)
    parser.add_argument("--chunk-size",   type=int, default=20)
    args = parser.parse_args()
    run_ddpm_ensemble(
        ckpt_path=args.ckpt, data_path=args.data, n_eval=args.n_eval,
        n_ens=args.n_ens, sample_steps=args.sample_steps, out_path=args.out,
        device=args.device, chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
