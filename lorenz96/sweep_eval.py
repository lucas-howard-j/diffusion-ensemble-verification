"""
Single-pass evaluation of the training-volume sweep (Figure 4).

For each lead, one set of test ICs and one climatological ensemble are drawn
(seed 42: rng.choice for the ICs, then rng.integers for the climatology, the
same protocol as lorenz96/ensemble_*.py) and shared by all eight training
sizes, so the comparison across training volume is paired.  Point estimates
and bootstrap distributions come from the same ensemble draw.

Replaces the separate evaluations in lorenz96/sweep.py (points; 500 × 20) and
lorenz96/sweep/bootstrap_eval.py (CIs; 200 × 20, different ICs).

Scores are computed in one normalised space for every checkpoint (per-node
mean and SD of the full training set), not each checkpoint's own training-
subset statistics.  Skills are ratio of means over ICs of the node-mean CRPS
and of the VS,
1 − mean(score) / mean(clim score); the paired bootstrap resamples ICs with
the same indices for CRPS and VS.

Writes (same keys figure4.py reads, plus n_eval / n_ens / sample_steps):
    lorenz96/sweep/sweep_results.npz   crps_skill, vs_skill            (n_leads, n_sizes)
    lorenz96/sweep/sweep_errors.npz    {crps,vs,diff}_skill_dist       (n_leads, n_sizes, n_boot)

Usage:
    python -m lorenz96.sweep_eval [--n-eval 500] [--n-ens 500] [--device mps]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lorenz96.config import TRAIN_END, TEST_START
from lorenz96.dataset import load_trajectory
from lorenz96.eval import crps_ensemble, vs_per_ic
from lorenz96.model import DDPM, ConvScoreNet

LEAD_VALUES    = [20, 40, 60]
N_TRAIN_VALUES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 249_000]
CKPT_DIR       = Path("lorenz96/sweep/checkpoints")
OUT_DIR        = Path("lorenz96/sweep")


def _load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = json.loads(ckpt["config"])
    net  = ConvScoreNet(hidden_ch=cfg["hidden_ch"], n_res=cfg["n_res"],
                        emb_dim=cfg["emb_dim"], kernel_size=cfg["kernel"])
    model = DDPM(net, T=cfg["T_diff"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mu"], ckpt["sigma"], ckpt["lead_steps"]


def _sample(model, cond_norm, n_ens, sample_steps, device, chunk):
    n, N = cond_norm.shape
    ens = np.empty((n, n_ens, N), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            c  = torch.from_numpy(cond_norm[i0:i1].astype(np.float32)).to(device)
            ens[i0:i1] = model.sample(c, n_samples=n_ens, sample_steps=sample_steps).cpu().numpy()
    return ens


def run(n_eval=500, n_ens=500, sample_steps=200, n_boot=1000, seed=42,
        data_path="lorenz96/l96_trajectory.npz", out_dir=OUT_DIR,
        device=None, chunk=20, leads=LEAD_VALUES, n_trains=N_TRAIN_VALUES):
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(out_dir)
    states, _ = load_trajectory(data_path)
    test_raw, train_raw = states[TEST_START:], states[:TRAIN_END]
    mu_s, sd_s = train_raw.mean(0), train_raw.std(0)   # common scoring space

    shape = (len(leads), len(n_trains))
    crps_skill = np.full(shape, np.nan)
    vs_skill   = np.full(shape, np.nan)
    crps_dist  = np.full(shape + (n_boot,), np.nan)
    vs_dist    = np.full(shape + (n_boot,), np.nan)

    t0 = time.time()
    for i, lead in enumerate(leads):
        # Shared ICs + climatology for every training size at this lead
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(test_raw) - lead, size=n_eval, replace=False))
        clim_idx = rng.integers(0, len(train_raw), size=(n_eval, n_ens))

        for j, n_train in enumerate(n_trains):
            path = CKPT_DIR / f"ddpm_lead{lead}_{n_train}.pt"
            model, mu, sigma, ck_lead = _load_model(path, device)
            assert ck_lead == lead, f"{path}: lead {ck_lead} != {lead}"

            # The model sees inputs in its own normalisation (its training
            # subset's stats); scoring happens in one fixed space for every
            # checkpoint (full-training-set stats), so the verification
            # problem is identical across training sizes.
            cond_norm = (test_raw[idx] - mu) / sigma
            torch.manual_seed(seed + 1000 * i + j)
            ens_model = _sample(model, cond_norm, n_ens, sample_steps, device, chunk)
            ens = ((ens_model * sigma + mu) - mu_s) / sd_s

            target_norm = (test_raw[idx + lead] - mu_s) / sd_s
            clim_norm   = (train_raw[clim_idx] - mu_s) / sd_s

            c_ens  = crps_ensemble(target_norm, ens).mean(1)        # (n,)
            c_clim = crps_ensemble(target_norm, clim_norm).mean(1)
            v_ens  = vs_per_ic(target_norm, ens)
            v_clim = vs_per_ic(target_norm, clim_norm)

            crps_skill[i, j] = 1 - c_ens.mean() / c_clim.mean()
            vs_skill[i, j]   = 1 - v_ens.mean() / v_clim.mean()

            brng = np.random.default_rng(7)
            for b in range(n_boot):
                r = brng.integers(0, n_eval, size=n_eval)
                crps_dist[i, j, b] = 1 - c_ens[r].mean() / c_clim[r].mean()
                vs_dist[i, j, b]   = 1 - v_ens[r].mean() / v_clim[r].mean()

            print(f"lead={lead} n_train={n_train:>7,}  CRPS {crps_skill[i, j]:+.4f}  "
                  f"VS {vs_skill[i, j]:+.4f}  spread {ens.std(1).mean():.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    meta = dict(lead_values=np.array(leads), n_train_values=np.array(n_trains),
                n_eval=n_eval, n_ens=n_ens, sample_steps=sample_steps, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "sweep_results.npz", crps_skill=crps_skill, vs_skill=vs_skill, **meta)
    np.savez(out_dir / "sweep_errors.npz", crps_skill_dist=crps_dist, vs_skill_dist=vs_dist,
             diff_skill_dist=crps_dist - vs_dist, **meta)
    print(f"Saved {out_dir}/sweep_results.npz and sweep_errors.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--n-ens", type=int, default=500)
    p.add_argument("--sample-steps", type=int, default=200)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true", help="one lead, two sizes (for testing)")
    a = p.parse_args()
    kw = dict(leads=[20], n_trains=[1_000, 249_000]) if a.smoke else {}
    run(n_eval=a.n_eval, n_ens=a.n_ens, sample_steps=a.sample_steps,
        out_dir=a.out_dir, device=a.device, **kw)


if __name__ == "__main__":
    main()
