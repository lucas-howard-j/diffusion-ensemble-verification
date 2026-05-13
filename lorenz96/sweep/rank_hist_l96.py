"""
Rank histograms for L96 DDPM vs dynamical ensemble.

Marginal (per-node, pooled over all 40 nodes) and multivariate (all 40 nodes
jointly, average-distance pre-rank) rank histograms, for DDPM and dynamical
ensemble side-by-side, at three lead times (20, 40, 60 steps).

Both raw and bias-corrected versions are produced.

Figures (saved to figures/):
  rank_hist_l96_marginal.pdf/png       — 2×3: [DDPM, Dyn] × [L20, L40, L60]
  rank_hist_l96_marginal_bc.pdf/png    — same, bias-corrected
  rank_hist_l96_mv.pdf/png             — 2×3 multivariate
  rank_hist_l96_mv_bc.pdf/png          — same, bias-corrected

Usage:
    PYTHONPATH=. python lorenz96/sweep/rank_hist_l96.py
    PYTHONPATH=. python lorenz96/sweep/rank_hist_l96.py --n_train 200000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from lorenz96.config  import N, F, DT, TEST_START, TRAIN_END
from lorenz96.dataset import load_trajectory
from lorenz96.model   import DDPM, ConvScoreNet

CKPT_DIR   = Path(__file__).parent / "checkpoints"
DATA_PATH  = "lorenz96/data/l96_trajectory.npz"
OUT_DIR    = Path("figures/supplement")

LEAD_VALUES   = [20, 40, 60]
N_TRAIN       = 100_000    # which DDPM checkpoint to use (default)
N_ENS         = 20         # match sweep ensemble size
N_EVAL        = 200        # match sweep N_EVAL
N_PLOT_BINS   = 10         # 10 equal percentile bins
SAMPLE_STEPS  = 200        # DDIM steps for DDPM
SEED          = 42

# Dynamical ensemble parameters
SIGMA_IC_FRAC  = 0.10   # IC perturbation as fraction of sigma_clim
SIGMA_MODEL    = 0.05   # per-step additive noise (physical units)

# Row/column labels and colours
ROW_LABELS   = ["DDPM", "Dynamical"]
DDPM_COLOR   = "#4C72B0"
DYN_COLOR    = "#DD8452"


# ---------------------------------------------------------------------------
# L96 integrator (matches generate_data.py and dyn_eval.py)
# ---------------------------------------------------------------------------

def _l96_deriv(x: np.ndarray) -> np.ndarray:
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + F


def rk4_step(x: np.ndarray) -> np.ndarray:
    k1 = _l96_deriv(x)
    k2 = _l96_deriv(x + 0.5 * DT * k1)
    k3 = _l96_deriv(x + 0.5 * DT * k2)
    k4 = _l96_deriv(x + DT * k3)
    return x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_ensemble(
    x0: np.ndarray,
    n_steps: int,
    n_ens: int,
    sigma_ic: float,
    sigma_model: float,
    rng: np.random.Generator,
    sigma_bias: float = 0.0,
) -> np.ndarray:
    """
    Perturbed-IC ensemble; returns (n_ens, N) physical-space states.

    IC perturbation = shared bias + per-member noise, both N(0, sigma_ic²·I):
      b      ~ N(0, sigma_ic²·I)   shared across all members
      eps_m  ~ N(0, sigma_ic²·I)   i.i.d. per member
      x0_m   = x0 + b + eps_m

    This makes the truth exchangeable with any member at t=0: truth sits at
    displacement -b from the ensemble mean, same distribution as any member.

    sigma_bias > 0: additional per-step shared bias drawn once per call,
      shared across members and steps.
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
    return ens


# ---------------------------------------------------------------------------
# Rank functions
# ---------------------------------------------------------------------------

def marginal_rank(truth: np.ndarray, ens: np.ndarray) -> int:
    """Rank of scalar truth among 1-D members.  Result in {1, …, M+1}."""
    return int(np.sum(ens < truth)) + 1


def mv_rank(obs: np.ndarray, ens: np.ndarray) -> int:
    """
    Multivariate rank using average-distance pre-rank (energy-score family).
    obs : (d,)   ens : (M, d)   — both in normalised space.
    Returns rank in {1, …, M+1}.
    """
    M   = ens.shape[0]
    aug = np.concatenate([obs[None], ens], axis=0)   # (M+1, d)
    diff = aug[:, None, :] - aug[None, :, :]          # (M+1, M+1, d)
    D    = np.sqrt((diff ** 2).sum(-1))               # (M+1, M+1) Euclidean
    np.fill_diagonal(D, 0.0)
    s    = D.sum(1) / M                               # avg dist to others
    return int(np.sum(s[1:] < s[0])) + 1


def coarsen_counts(counts: np.ndarray, n_plot_bins: int) -> np.ndarray:
    n_fine = len(counts)
    edges = np.round(np.linspace(0, n_fine, n_plot_bins + 1)).astype(int)
    return np.array([counts[edges[i]:edges[i+1]].sum() for i in range(n_plot_bins)])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hist(
    ax: plt.Axes,
    counts: np.ndarray,
    title: str,
    color: str,
    ylabel: bool = True,
) -> None:
    coarse  = coarsen_counts(counts, N_PLOT_BINS)
    freq    = coarse / coarse.sum()
    uniform = 1.0 / N_PLOT_BINS

    bin_edges = np.linspace(0, 100, N_PLOT_BINS + 1)
    ax.bar(bin_edges[:-1], freq, width=10.0, align="edge",
           color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axhline(uniform, color="k", lw=1.0, ls="--")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(freq.max() * 1.3, uniform * 1.6))
    ax.set_xlabel("Percentile", fontsize=8)
    if ylabel:
        ax.set_ylabel("Relative frequency", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.annotate(f"n={int(counts.sum())}", xy=(0.97, 0.95),
                xycoords="axes fraction", ha="right", va="top", fontsize=7)


def make_figure(
    counts_ddpm: list,   # list of (N_FINE,) count arrays, one per lead
    counts_dyn:  list,
    n_leads: int,
    title: str,
    out_stem: str,
) -> None:
    lead_labels = [f"Lead = {l} steps ({l*DT:.1f} TU)" for l in LEAD_VALUES]

    fig, axes = plt.subplots(
        2, n_leads,
        figsize=(3.0 * n_leads, 5.0),
        constrained_layout=True,
    )

    for j in range(n_leads):
        show_ylabel = (j == 0)
        plot_hist(axes[0, j], counts_ddpm[j],
                  title=lead_labels[j], color=DDPM_COLOR, ylabel=show_ylabel)
        plot_hist(axes[1, j], counts_dyn[j],
                  title="", color=DYN_COLOR, ylabel=show_ylabel)

    axes[0, 0].set_ylabel("DDPM\nRelative frequency", fontsize=8)
    axes[1, 0].set_ylabel("Dynamical\nRelative frequency", fontsize=8)

    fig.suptitle(title, fontsize=9)

    out = Path(out_stem)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.pdf/.png")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def load_ddpm_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = json.loads(ckpt["config"])
    net  = ConvScoreNet(
        hidden_ch=cfg["hidden_ch"],
        n_res=cfg["n_res"],
        emb_dim=cfg["emb_dim"],
        kernel_size=cfg["kernel"],
    )
    model = DDPM(net, T=cfg["T_diff"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mu      = ckpt["mu"]       # (N,)
    sigma   = ckpt["sigma"]    # (N,)
    return model, mu, sigma


def run(n_train: int = N_TRAIN, sigma_ic_frac: float = SIGMA_IC_FRAC,
        sigma_model: float = SIGMA_MODEL, sigma_bias: float = 0.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_train={n_train:,}  N_ENS={N_ENS}  N_EVAL={N_EVAL}")

    states, _ = load_trajectory(DATA_PATH)
    train_pool = states[:TRAIN_END]
    mu_global    = train_pool.mean(axis=0)
    sigma_global = train_pool.std(axis=0)
    sigma_clim   = float(sigma_global.mean())
    sigma_ic  = sigma_ic_frac * sigma_clim
    sic_tag   = (f"sic{int(sigma_ic_frac * 100):02d}"
                 f"_sm{int(sigma_model * 100):02d}"
                 f"_sb{int(sigma_bias * 100):02d}")
    print(f"sigma_ic={sigma_ic:.4f} ({sigma_ic_frac*100:.0f}% of sigma_clim={sigma_clim:.4f})  "
          f"sigma_model={sigma_model}  sigma_bias={sigma_bias}")

    test_raw = states[TEST_START:]
    n_fine   = N_ENS + 1

    # Storage: counts per lead, shape (n_fine,) for marginal/MV
    marg_counts_ddpm = []
    marg_counts_dyn  = []
    mv_counts_ddpm   = []
    mv_counts_dyn    = []

    # Bias storage: per-lead
    bias_ddpm_list   = []
    bias_dyn_list    = []
    ens_ddpm_list    = []   # (N_EVAL, N_ENS, N) normalised
    ens_dyn_list     = []
    truth_list       = []   # (N_EVAL, N) normalised

    for i, lead in enumerate(LEAD_VALUES):
        rng = np.random.default_rng(SEED + i * 1000)

        # Test pairs: same selection as dyn_eval.py
        max_test = len(test_raw) - lead
        n_eval   = min(N_EVAL, max_test)
        idx      = rng.choice(max_test, size=n_eval, replace=False)
        idx.sort()

        cond_phys   = test_raw[idx]           # (n_eval, N) physical
        target_phys = test_raw[idx + lead]    # (n_eval, N) physical

        # Load DDPM checkpoint for this lead
        ckpt_path = str(CKPT_DIR / f"ddpm_lead{lead}_{n_train}.pt")
        print(f"\nLead={lead}  Loading {ckpt_path}")
        model, mu, sigma = load_ddpm_model(ckpt_path, device)

        target_norm = (target_phys - mu) / sigma     # (n_eval, N)
        cond_norm   = (cond_phys   - mu) / sigma     # (n_eval, N)

        # ── DDPM ensemble ─────────────────────────────────────────────────
        print(f"  Sampling DDPM ensemble ({n_eval}×{N_ENS})...")
        cond_t = torch.from_numpy(cond_norm).float().to(device)
        with torch.no_grad():
            ddpm_ens_norm = model.sample(
                cond_t, n_samples=N_ENS, sample_steps=SAMPLE_STEPS
            ).cpu().numpy()   # (n_eval, N_ENS, N)

        # ── Dynamical ensemble ────────────────────────────────────────────
        print(f"  Integrating dynamical ensemble ({n_eval}×{N_ENS})...")
        dyn_ens_norm = np.empty((n_eval, N_ENS, N), dtype=np.float64)
        for t in range(n_eval):
            if t % 50 == 0:
                print(f"    sample {t}/{n_eval}", flush=True)
            members_phys = integrate_ensemble(
                cond_phys[t], lead, N_ENS, sigma_ic, sigma_model,
                rng=np.random.default_rng(SEED + i * 1000 + t + 1),
                sigma_bias=sigma_bias,
            )
            dyn_ens_norm[t] = (members_phys - mu) / sigma

        ens_ddpm_list.append(ddpm_ens_norm)
        ens_dyn_list.append(dyn_ens_norm)
        truth_list.append(target_norm)

        # Bias per node (for bc variant)
        bias_ddpm_list.append((ddpm_ens_norm.mean(axis=1) - target_norm).mean(axis=0))
        bias_dyn_list.append( (dyn_ens_norm.mean(axis=1)  - target_norm).mean(axis=0))

    # ── Compute rank histograms ────────────────────────────────────────────
    for bc in [False, True]:
        marg_c_ddpm = []
        marg_c_dyn  = []
        mv_c_ddpm   = []
        mv_c_dyn    = []

        for i, lead in enumerate(LEAD_VALUES):
            target_norm  = truth_list[i]
            ens_ddpm_raw = ens_ddpm_list[i].copy()
            ens_dyn_raw  = ens_dyn_list[i].copy()

            if bc:
                ens_ddpm_raw = ens_ddpm_raw - bias_ddpm_list[i][None, None, :]
                ens_dyn_raw  = ens_dyn_raw  - bias_dyn_list[i][None, None, :]

            n_eval = target_norm.shape[0]
            marg_d = np.zeros(n_fine, dtype=np.int64)
            marg_y = np.zeros(n_fine, dtype=np.int64)
            mv_d   = np.zeros(n_fine, dtype=np.int64)
            mv_y   = np.zeros(n_fine, dtype=np.int64)

            for t in range(n_eval):
                # Marginal: rank each node separately, pool into counts
                for k in range(N):
                    r = marginal_rank(float(target_norm[t, k]), ens_ddpm_raw[t, :, k])
                    marg_d[r - 1] += 1
                    r = marginal_rank(float(target_norm[t, k]), ens_dyn_raw[t, :, k])
                    marg_y[r - 1] += 1
                # Multivariate
                r = mv_rank(target_norm[t], ens_ddpm_raw[t])
                mv_d[r - 1] += 1
                r = mv_rank(target_norm[t], ens_dyn_raw[t])
                mv_y[r - 1] += 1

            marg_c_ddpm.append(marg_d)
            marg_c_dyn.append(marg_y)
            mv_c_ddpm.append(mv_d)
            mv_c_dyn.append(mv_y)

            bc_tag = " (bias-corrected)" if bc else ""
            print(f"Lead={lead}{bc_tag}  "
                  f"MV DDPM low/high tail: {mv_d[:2].sum()}/{mv_d[-2:].sum()}  "
                  f"Dyn low/high: {mv_y[:2].sum()}/{mv_y[-2:].sum()}")

        bc_suffix = "_bc" if bc else ""
        bc_note   = " — bias-corrected" if bc else ""

        make_figure(
            marg_c_ddpm, marg_c_dyn, len(LEAD_VALUES),
            title=f"L96 marginal rank histograms (pooled over 40 nodes){bc_note}\n"
                  f"n_train={n_train:,}  N_ENS={N_ENS}  "
                  f"σ_ic={sigma_ic_frac*100:.0f}%  σ_model={sigma_model}  σ_bias={sigma_bias}",
            out_stem=str(OUT_DIR / f"rank_hist_l96_marginal_{sic_tag}{bc_suffix}"),
        )
        make_figure(
            mv_c_ddpm, mv_c_dyn, len(LEAD_VALUES),
            title=f"L96 multivariate rank histograms (avg-distance pre-rank, 40D){bc_note}\n"
                  f"n_train={n_train:,}  N_ENS={N_ENS}  "
                  f"σ_ic={sigma_ic_frac*100:.0f}%  σ_model={sigma_model}  σ_bias={sigma_bias}",
            out_stem=str(OUT_DIR / f"rank_hist_l96_mv_{sic_tag}{bc_suffix}"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",      type=int,   default=N_TRAIN,
                        help="n_train value for checkpoint selection (default %(default)s)")
    parser.add_argument("--sigma_ic_frac", type=float, default=SIGMA_IC_FRAC,
                        help="IC perturbation as fraction of sigma_clim (default %(default)s)")
    parser.add_argument("--sigma_model",   type=float, default=SIGMA_MODEL,
                        help="Per-step residual noise std, i.i.d. per member per step (default %(default)s)")
    parser.add_argument("--sigma_bias",    type=float, default=0.0,
                        help="Per-step shared bias std b~N(0,σ²I), constant across members/steps (default %(default)s)")
    parser.add_argument("--data",          type=str,   default=DATA_PATH)
    args = parser.parse_args()
    DATA_PATH = args.data
    run(args.n_train, args.sigma_ic_frac, args.sigma_model, args.sigma_bias)
