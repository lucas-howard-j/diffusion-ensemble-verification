"""
Evaluate a trained L96 DDPM on the test set.

Produces a side-by-side figure:
  Left  — CRPS skill score (mean over N=40 nodes), vs climatological draw
  Right — Variogram score skill (p=0.5, all N*(N-1)/2 pairs), vs climatological draw

Scores are computed in normalised space (zero mean, unit variance per node).

VS is vectorised over all 780 pairs using index arrays to avoid materialising
an (n_eval, n_ens, N, N) tensor.  With n_eval=2000 and n_ens=50 peak memory
for the pair tensor is ~300 MB.

Usage:
    python -m lorenz96.eval [--ckpt PATH] [--n_eval N] [--n_ens M]
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from lorenz96.config import N, TEST_START, TRAIN_END, LEAD_STEPS
from lorenz96.dataset import load_trajectory
from lorenz96.model import DDPM, ConvScoreNet


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def crps_ensemble(truth: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """CRPS per sample × node.  truth: (n, D), ens: (n, M, D) → (n, D).

    Spread computed via the sorted-ensemble identity (O(M log M), O(n M D) memory)
    rather than the naive pairwise formula (O(M²), infeasible for large M).
    """
    M   = ens.shape[1]
    mae = np.abs(ens - truth[:, None, :]).mean(axis=1)   # (n, D)

    ens_sorted = np.sort(ens, axis=1)                     # (n, M, D)
    k       = np.arange(M, dtype=np.float64)              # 0-indexed
    weights = (2 * k - M + 1) / M**2                     # (M,)
    spread  = (ens_sorted * weights[None, :, None]).sum(axis=1)   # (n, D)

    return mae - spread


def vs_per_ic(
    truth: np.ndarray,
    ens:   np.ndarray,
    p:     float = 0.5,
    chunk: int   = 100,
) -> np.ndarray:
    """Per-IC variogram score contribution.  truth: (n, D), ens: (n, M, D) → (n,)."""
    D = truth.shape[1]
    i_idx, j_idx = np.triu_indices(D, k=1)
    n   = truth.shape[0]
    out = np.zeros(n)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        obs_diff = np.abs(truth[i0:i1, i_idx] - truth[i0:i1, j_idx]) ** p
        ens_diff = np.abs(ens[i0:i1, :, i_idx] - ens[i0:i1, :, j_idx]) ** p
        out[i0:i1] = ((obs_diff - ens_diff.mean(axis=1)) ** 2).mean(axis=1)
    return out


def bootstrap_skill_ci(
    target_norm:   np.ndarray,
    ens_norm:      np.ndarray,
    clim_ens_norm: np.ndarray,
    n_boot: int   = 1000,
    ci:     float = 95,
    seed:   int   = 7,
) -> dict:
    """
    Bootstrap 95% CIs for CRPS skill, VS skill, and VS/CRPS ratio by
    resampling test ICs with replacement.

    Returns dict with keys: crps_{lo,hi}, vs_{lo,hi}, ratio_{lo,hi}.
    """
    rng = np.random.default_rng(seed)
    n   = len(target_norm)

    crps_ic_ens  = crps_ensemble(target_norm, ens_norm).mean(1)    # (n,)
    crps_ic_clim = crps_ensemble(target_norm, clim_ens_norm).mean(1)
    vs_ic_ens    = vs_per_ic(target_norm, ens_norm)                 # (n,)
    vs_ic_clim   = vs_per_ic(target_norm, clim_ens_norm)

    crps_b = np.empty(n_boot)
    vs_b   = np.empty(n_boot)
    ratio_b = np.empty(n_boot)

    for b in range(n_boot):
        idx       = rng.integers(0, n, size=n)
        crps_b[b] = 1 - crps_ic_ens[idx].mean() / crps_ic_clim[idx].mean()
        vs_skill  = 1 - vs_ic_ens[idx].mean()   / vs_ic_clim[idx].mean()
        vs_b[b]   = vs_skill
        ratio_b[b] = vs_skill / crps_b[b] if crps_b[b] != 0 else np.nan

    alpha = (100 - ci) / 2
    return {
        "crps_lo":  float(np.percentile(crps_b,  alpha)),
        "crps_hi":  float(np.percentile(crps_b,  100 - alpha)),
        "vs_lo":    float(np.percentile(vs_b,    alpha)),
        "vs_hi":    float(np.percentile(vs_b,    100 - alpha)),
        "ratio_lo": float(np.nanpercentile(ratio_b, alpha)),
        "ratio_hi": float(np.nanpercentile(ratio_b, 100 - alpha)),
    }


def variogram_score(
    truth: np.ndarray,
    ens:   np.ndarray,
    p:     float = 0.5,
    chunk: int   = 100,
) -> float:
    """
    Variogram score (Scheuerer & Hamill 2015), mean over n samples and all
    D*(D-1)/2 pairs.

    truth : (n, D), ens : (n, M, D)

    Peak memory is O(chunk × M × n_pairs); default chunk=100 keeps this well
    under 1 GB for M=500, D=40.
    """
    D = truth.shape[1]
    i_idx, j_idx = np.triu_indices(D, k=1)   # (n_pairs,)
    n = truth.shape[0]

    total = 0.0
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        obs_diff = np.abs(truth[i0:i1, i_idx] - truth[i0:i1, j_idx]) ** p   # (c, n_pairs)
        ens_diff = np.abs(
            ens[i0:i1, :, i_idx] - ens[i0:i1, :, j_idx]
        ) ** p                                                                  # (c, M, n_pairs)
        ens_mean = ens_diff.mean(axis=1)                                        # (c, n_pairs)
        total   += float(((obs_diff - ens_mean) ** 2).sum())

    return total / (n * len(i_idx))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    ckpt_path:    str = "lorenz96/checkpoints/ddpm.pt",
    data_path:    str = "lorenz96/l96_trajectory.npz",
    n_eval:       int = 2_000,
    n_ens:        int = 50,
    out_path:     str = "lorenz96/figures/skill_scores.pdf",
    device:       str | None = None,
    seed:         int = 0,
    sample_steps: int | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    states, data_cfg = load_trajectory(data_path)
    dt = data_cfg["dt"]

    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    mu      = ckpt["mu"]
    sigma   = ckpt["sigma"]
    lead    = ckpt["lead_steps"]
    n_train = ckpt["n_train"]

    # Test pairs
    test_raw = states[TEST_START:]
    max_test = len(test_raw) - lead
    idx = rng.choice(max_test, size=min(n_eval, max_test), replace=False)
    idx.sort()

    cond_raw   = test_raw[idx]
    target_raw = test_raw[idx + lead]

    cond_norm   = (cond_raw   - mu) / sigma
    target_norm = (target_raw - mu) / sigma

    # Climatological baseline: random draw from training pool
    train_raw = states[:TRAIN_END]
    clim_idx  = rng.integers(0, len(train_raw), size=(len(idx), n_ens))
    clim_ens_norm = (train_raw[clim_idx] - mu) / sigma   # (n_eval, n_ens, N)

    # DDPM ensemble
    cfg_json = __import__("json").loads(ckpt["config"])
    net = ConvScoreNet(
        hidden_ch=cfg_json["hidden_ch"],
        n_res=cfg_json["n_res"],
        emb_dim=cfg_json["emb_dim"],
        kernel_size=cfg_json["kernel"],
    )
    model = DDPM(net, T=cfg_json["T_diff"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    steps_str = f"{sample_steps}" if sample_steps else "full"
    print(f"Sampling {len(idx)} × {n_ens} ensemble members  (steps={steps_str}) ...", flush=True)
    cond_t = torch.from_numpy(cond_norm).float().to(device)
    ddpm_ens_norm = model.sample(cond_t, n_samples=n_ens, sample_steps=sample_steps).cpu().numpy()

    truth_norm = target_norm

    # CRPS skill (mean over nodes)
    crps_ddpm  = crps_ensemble(truth_norm, ddpm_ens_norm).mean(0)   # (N,)
    crps_clim  = crps_ensemble(truth_norm, clim_ens_norm).mean(0)   # (N,)
    crps_skill_node = 1 - crps_ddpm / crps_clim                     # (N,)
    crps_skill_mean = float(crps_skill_node.mean())

    # VS skill
    vs_ddpm  = variogram_score(truth_norm, ddpm_ens_norm)
    vs_clim  = variogram_score(truth_norm, clim_ens_norm)
    vs_skill = 1 - vs_ddpm / vs_clim

    print(f"\nn_train={n_train:,}  n_eval={len(idx)}  n_ens={n_ens}")
    print(f"CRPS skill  mean={crps_skill_mean:+.4f}  "
          f"min={crps_skill_node.min():+.4f}  max={crps_skill_node.max():+.4f}")
    print(f"VS skill    overall={vs_skill:+.4f}")

    # --- plot ---
    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: CRPS skill per node
    ax = axes[0]
    ax.bar(np.arange(N), crps_skill_node, color="#4C72B0", edgecolor="none", width=1.0)
    ax.axhline(0,                    color="k",   linewidth=0.8, linestyle="--")
    ax.axhline(crps_skill_mean,      color="#DD8452", linewidth=1.2,
               linestyle="-", label=f"mean={crps_skill_mean:+.3f}")
    ax.set_xlabel("L96 node  $k$")
    ax.set_ylabel("Skill  (1 − CRPS / CRPS$_{clim}$)")
    ax.set_title("CRPS skill per node")
    ax.legend(fontsize=8)

    # Right: VS overall bar
    ax = axes[1]
    ax.bar(["DDPM", "clim"], [1 - vs_ddpm / vs_clim, 0.0],
           color=["#4C72B0", "#AAAAAA"], edgecolor="k", linewidth=0.7, width=0.4)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Skill  (1 − VS / VS$_{clim}$)")
    ax.set_title(f"VS skill  (p=0.5,  {N*(N-1)//2} pairs)")
    ax.text(0, vs_skill + 0.005 * np.sign(vs_skill), f"{vs_skill:+.4f}",
            ha="center", va="bottom" if vs_skill >= 0 else "top", fontsize=9)

    fig.suptitle(f"L96 DDPM  |  n_train={n_train:,}  lead={lead} steps ({lead*dt:.1f} TU)  "
                 f"n_ens={n_ens}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")

    return {"crps_skill": crps_skill_mean, "vs_skill": vs_skill,
            "crps_skill_node": crps_skill_node}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="lorenz96/checkpoints/ddpm.pt")
    parser.add_argument("--data",   default="lorenz96/l96_trajectory.npz")
    parser.add_argument("--n_eval", type=int, default=2_000)
    parser.add_argument("--n_ens",  type=int, default=50)
    parser.add_argument("--out",    default="lorenz96/figures/skill_scores.pdf")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    evaluate(args.ckpt, args.data, args.n_eval, args.n_ens, args.out, args.device)


if __name__ == "__main__":
    main()
