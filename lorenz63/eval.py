"""
Evaluate a trained L63 DDPM on the test set.

Produces a side-by-side figure:
  Left  — CRPS skill score per component (x, y, z), vs climatological draw
  Right — Variogram score skill (p=0.5), per component-pair and overall

Climatological baseline: random draw of M ensemble members from the
training pool (same M as DDPM ensemble size).

Usage:
    python -m lorenz63.eval [--ckpt PATH] [--n_eval N] [--n_ens M]
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from lorenz63.config import TEST_START, TRAIN_END, LEAD_STEPS
from lorenz63.dataset import load_trajectory
from lorenz63.model import DDPM, ScoreNet


# ---------------------------------------------------------------------------
# Score functions (operate on physical / normalized arrays)
# ---------------------------------------------------------------------------

def crps_ensemble(truth: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """
    CRPS for an ensemble forecast, per sample and per dimension.

    Parameters
    ----------
    truth : (N, D)
    ens   : (N, M, D)

    Returns
    -------
    crps  : (N, D)
    """
    M = ens.shape[1]
    mae  = np.abs(ens - truth[:, None, :]).mean(axis=1)          # (N, D)
    spread = np.abs(ens[:, :, None, :] - ens[:, None, :, :]).sum(axis=(1, 2)) / (2 * M * M)
    return mae - spread   # (N, D)


def variogram_score(truth: np.ndarray, ens: np.ndarray, p: float = 0.5) -> float:
    """
    Variogram score (Scheuerer & Hamill 2015), scalar averaged over N samples.

    Parameters
    ----------
    truth : (N, D)
    ens   : (N, M, D)

    Returns
    -------
    vs : float
    """
    N, M, D = ens.shape
    vs_total = 0.0
    n_pairs = 0
    for i in range(D):
        for j in range(i + 1, D):
            obs_diff  = np.abs(truth[:, i] - truth[:, j]) ** p          # (N,)
            ens_diff  = np.abs(ens[:, :, i] - ens[:, :, j]) ** p        # (N, M)
            ens_mean  = ens_diff.mean(axis=1)                            # (N,)
            vs_total += ((obs_diff - ens_mean) ** 2).mean()
            n_pairs  += 1
    return vs_total / n_pairs


def variogram_score_perpair(truth: np.ndarray, ens: np.ndarray, p: float = 0.5) -> dict:
    """VS per component-pair, averaged over N samples."""
    N, M, D = ens.shape
    labels = ["x", "y", "z"]
    out = {}
    for i in range(D):
        for j in range(i + 1, D):
            obs_diff = np.abs(truth[:, i] - truth[:, j]) ** p
            ens_diff = np.abs(ens[:, :, i] - ens[:, :, j]) ** p
            out[f"{labels[i]}-{labels[j]}"] = ((obs_diff - ens_diff.mean(1)) ** 2).mean()
    return out


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    ckpt_path: str = "lorenz63/checkpoints/ddpm.pt",
    data_path: str = "lorenz63/data/l63_trajectory.npz",
    n_eval: int = 5_000,
    n_ens: int = 50,
    out_path: str = "lorenz63/figures/skill_scores.pdf",
    device: str | None = None,
    seed: int = 0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    # --- load data ---
    states, data_cfg = load_trajectory(data_path)
    dt = data_cfg["dt"]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mu    = ckpt["mu"]
    sigma = ckpt["sigma"]
    lead  = ckpt["lead_steps"]
    n_train = ckpt["n_train"]

    # Test pairs: subsample n_eval from the test split
    test_raw = states[TEST_START:]
    max_test = len(test_raw) - lead
    idx = rng.choice(max_test, size=min(n_eval, max_test), replace=False)
    idx.sort()

    cond_raw   = test_raw[idx]                  # (n_eval, 3) physical
    target_raw = test_raw[idx + lead]           # (n_eval, 3) physical

    cond_norm   = (cond_raw   - mu) / sigma
    target_norm = (target_raw - mu) / sigma

    # --- climatological baseline: random draw from training pool ---
    train_raw = states[:TRAIN_END]
    clim_idx  = rng.integers(0, len(train_raw), size=(len(idx), n_ens))
    clim_ens_norm = (train_raw[clim_idx] - mu) / sigma   # (n_eval, n_ens, 3)

    # --- DDPM ensemble ---
    cfg_json = __import__("json").loads(ckpt["config"])
    net = ScoreNet(
        hidden_dim=cfg_json["hidden_dim"],
        n_layers=cfg_json["n_layers"],
        emb_dim=cfg_json["emb_dim"],
    )
    model = DDPM(net, T=cfg_json["T_diff"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Sampling {len(idx)} × {n_ens} ensemble members ...", flush=True)
    cond_t = torch.from_numpy(cond_norm).float().to(device)
    ddpm_ens_norm = model.sample(cond_t, n_samples=n_ens).cpu().numpy()  # (n_eval, n_ens, 3)

    truth_norm = target_norm   # (n_eval, 3)

    # --- CRPS ---
    crps_ddpm = crps_ensemble(truth_norm, ddpm_ens_norm).mean(0)   # (3,)
    crps_clim = crps_ensemble(truth_norm, clim_ens_norm).mean(0)   # (3,)
    crps_skill = 1 - crps_ddpm / crps_clim                         # (3,)

    # --- VS ---
    vs_ddpm_pairs = variogram_score_perpair(truth_norm, ddpm_ens_norm)
    vs_clim_pairs = variogram_score_perpair(truth_norm, clim_ens_norm)
    pair_labels = list(vs_ddpm_pairs.keys())
    vs_skill_pairs = {k: 1 - vs_ddpm_pairs[k] / vs_clim_pairs[k] for k in pair_labels}

    vs_ddpm  = variogram_score(truth_norm, ddpm_ens_norm)
    vs_clim  = variogram_score(truth_norm, clim_ens_norm)
    vs_skill = 1 - vs_ddpm / vs_clim

    # --- print summary ---
    print(f"\nn_train={n_train:,}  n_eval={len(idx)}  n_ens={n_ens}")
    print(f"CRPS skill   x={crps_skill[0]:+.3f}  y={crps_skill[1]:+.3f}  z={crps_skill[2]:+.3f}  mean={crps_skill.mean():+.3f}")
    print(f"VS skill     " + "  ".join(f"{k}={v:+.3f}" for k, v in vs_skill_pairs.items()) + f"  overall={vs_skill:+.3f}")

    # --- plot ---
    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: CRPS skill per component
    ax = axes[0]
    comp_labels = ["x", "y", "z"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(comp_labels, crps_skill, color=colors, width=0.5, edgecolor="k", linewidth=0.7)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Skill score  (1 − CRPS / CRPS$_{clim}$)")
    ax.set_title("CRPS skill")
    ax.set_ylim(min(-0.1, crps_skill.min() * 1.3), max(0.1, crps_skill.max() * 1.3))
    for bar, val in zip(bars, crps_skill):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005 * np.sign(val),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    # Right: VS skill per pair + overall
    ax = axes[1]
    all_labels = pair_labels + ["overall"]
    all_vals   = [vs_skill_pairs[k] for k in pair_labels] + [vs_skill]
    pair_colors = ["#C44E52", "#8172B2", "#937860", "#4C72B0"]
    bars = ax.bar(all_labels, all_vals, color=pair_colors[:len(all_labels)],
                  width=0.5, edgecolor="k", linewidth=0.7)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Skill score  (1 − VS / VS$_{clim}$)")
    ax.set_title("Variogram score skill  (p=0.5)")
    ax.set_ylim(min(-0.1, min(all_vals) * 1.3), max(0.1, max(all_vals) * 1.3))
    for bar, val in zip(bars, all_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005 * np.sign(val),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    fig.suptitle(f"L63 DDPM  |  n_train={n_train:,}  n_ens={n_ens}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")

    return {"crps_skill": crps_skill, "vs_skill": vs_skill, "vs_skill_pairs": vs_skill_pairs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="lorenz63/checkpoints/ddpm.pt")
    parser.add_argument("--data",   default="lorenz63/data/l63_trajectory.npz")
    parser.add_argument("--n_eval", type=int, default=5_000)
    parser.add_argument("--n_ens",  type=int, default=50)
    parser.add_argument("--out",    default="lorenz63/figures/skill_scores.pdf")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    evaluate(args.ckpt, args.data, args.n_eval, args.n_ens, args.out, args.device)


if __name__ == "__main__":
    main()
