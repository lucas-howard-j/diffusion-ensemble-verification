"""
Weight perturbation sensitivity for the L96 DDPM.

Quantifies how CRPS (marginal) and ES / VS (joint) skill vs. climatology
degrade as filter-normalised Gaussian perturbations move the trained weights
away from their optimum.

Filter-normalised perturbation: each parameter tensor p gets elementwise
noise ~ N(0, (sigma_rel * ||p|| / sqrt(n))^2), so every layer is perturbed
by the same relative fraction of its own norm regardless of layer size.

Single-sigma-per-run design — parallelise over sigma on HPC:
    python -m lorenz96.weight_sensitivity --sigma 0.001 --n-draws 50 &
    python -m lorenz96.weight_sensitivity --sigma 0.01  --n-draws 50 &
    ...

After all sigmas are cached, build the summary figure:
    python -m lorenz96.weight_sensitivity --plot-only

Smoke test (does not need the full trajectory; needs --data):
    python -m lorenz96.weight_sensitivity --sigma 0.05 --smoke --data PATH

Data: generate with
    python -m lorenz96.generate_data --traj_len 400000   # ~3 min; covers all splits
    python -m lorenz96.generate_data                      # full 650k-step trajectory
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lorenz96.config import TEST_START, TRAIN_END
from lorenz96.dataset import load_trajectory
from lorenz96.model import DDPM, ConvScoreNet

SAMPLE_STEPS = 200   # DDIM steps — matches lorenz96/sweep.py
OUT_DIR      = Path("lorenz96/weight_sensitivity")
DEFAULT_CKPT = Path("lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt")
DEFAULT_DATA = Path("lorenz96/data/l96_trajectory.npz")


# ---------------------------------------------------------------------------
# Score functions (normalised space)
# ---------------------------------------------------------------------------

def _crps(truth: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """truth: (n, D), ens: (n, M, D) → (n, D)"""
    M      = ens.shape[1]
    mae    = np.abs(ens - truth[:, None, :]).mean(axis=1)
    spread = np.abs(ens[:, :, None, :] - ens[:, None, :, :]).sum(axis=(1, 2)) / (2 * M * M)
    return mae - spread


def _vs(truth: np.ndarray, ens: np.ndarray, p: float = 0.5) -> float:
    """Variogram score, mean over n samples and all D*(D-1)/2 pairs.
    truth: (n, D), ens: (n, M, D) → float"""
    D        = truth.shape[1]
    ii, jj   = np.triu_indices(D, k=1)
    obs_diff = np.abs(truth[:, ii] - truth[:, jj]) ** p
    ens_diff = np.abs(ens[:, :, ii] - ens[:, :, jj]) ** p
    return float(((obs_diff - ens_diff.mean(axis=1)) ** 2).mean())


def _es(truth: np.ndarray, ens: np.ndarray) -> float:
    """Energy score, mean over n samples.
    truth: (n, D), ens: (n, M, D) → float"""
    mae  = np.linalg.norm(ens - truth[:, None, :], axis=-1).mean()
    sq   = (ens ** 2).sum(-1)                             # (n, M)
    dot  = np.einsum("nmd,nkd->nmk", ens, ens)           # (n, M, M)
    dist = np.sqrt(np.maximum(sq[:, :, None] + sq[:, None, :] - 2 * dot, 0))
    return float(mae - 0.5 * dist.mean())


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

def _perturb_state_dict(
    state_dict: dict,
    sigma_rel: float,
    rng: np.random.Generator,
    only_keys: set | None = None,
) -> dict:
    """Filter-normalised Gaussian perturbation of a state dict.

    only_keys: if given, only perturb these keys (use to skip non-learnable
    buffers such as the DDPM schedule tensors betas/alphas/alpha_bar).
    """
    new_sd = {}
    for name, param in state_dict.items():
        if only_keys is not None and name not in only_keys:
            new_sd[name] = param
            continue
        p     = param.cpu().numpy().astype(np.float64)
        scale = sigma_rel * np.linalg.norm(p) / np.sqrt(p.size)
        new_sd[name] = torch.tensor(
            (p + rng.standard_normal(p.shape) * scale).astype(np.float32)
        )
    return new_sd


# ---------------------------------------------------------------------------
# Model loading and evaluation
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path, device: str) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = json.loads(ckpt["config"])
    net  = ConvScoreNet(
        hidden_ch=cfg["hidden_ch"], n_res=cfg["n_res"],
        emb_dim=cfg["emb_dim"], kernel_size=cfg["kernel"],
    )
    model = DDPM(net, T=cfg["T_diff"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _run_eval(
    model,
    cond_t:       torch.Tensor,
    truth_norm:   np.ndarray,
    clim_ens_norm: np.ndarray,
    n_ens:        int,
) -> tuple[float, float, float]:
    """One forward pass → (crps_skill, es_skill, vs_skill) vs climatology."""
    with torch.no_grad():
        ens_norm = model.sample(cond_t, n_samples=n_ens, sample_steps=SAMPLE_STEPS).cpu().numpy()

    # CRPS: per-node ratio-of-means, then mean over nodes (matches eval.py)
    crps_d  = _crps(truth_norm, ens_norm).mean(0)     # (D,)
    crps_r  = _crps(truth_norm, clim_ens_norm).mean(0)
    crps_sk = float((1 - crps_d / crps_r).mean())

    es_sk = 1 - _es(truth_norm, ens_norm) / _es(truth_norm, clim_ens_norm)
    vs_sk = 1 - _vs(truth_norm, ens_norm) / _vs(truth_norm, clim_ens_norm)
    return crps_sk, es_sk, vs_sk


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_convergence(
    crps_draws: np.ndarray,
    es_draws: np.ndarray,
    vs_draws: np.ndarray,
    crps_base: float,
    es_base: float,
    vs_base: float,
    sigma: float,
    figures_dir: Path,
) -> None:
    n  = len(crps_draws)
    xs = np.arange(1, n + 1)

    def cumstats(arr):
        mean = np.array([arr[:k].mean() for k in xs])
        se   = np.array([
            arr[:k].std(ddof=1) / np.sqrt(k) if k > 1 else 0.0
            for k in xs
        ])
        return mean, se

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)
    for ax, arr, label, base in zip(
        axes,
        [crps_draws, es_draws, vs_draws],
        ["CRPS (marginal)", "ES (joint)", "VS (joint/dep)"],
        [crps_base, es_base, vs_base],
    ):
        mean, se = cumstats(arr)
        t_stat   = arr.mean() / (arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        ax.axhline(0, color="tomato", lw=1.2, zorder=1)
        ax.scatter(xs, arr, s=8, color="steelblue", alpha=0.35, zorder=2)
        ax.plot(xs, mean, color="steelblue", lw=1.5, label="Cumulative mean", zorder=3)
        ax.fill_between(xs, mean - 1.96 * se, mean + 1.96 * se,
                        color="steelblue", alpha=0.2, label="±1.96 SE")
        ax.set_xlabel("Draw number", fontsize=8)
        ax.set_ylabel("Δ skill (perturbed − baseline)", fontsize=8)
        ax.set_title(f"{label}  (base {base:+.4f}  t={t_stat:.2f})", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2, lw=0.5)

    fig.suptitle(f"L96 DDPM weight sensitivity  |  σ_rel={sigma}  |  N={n} draws", fontsize=8)
    fig.tight_layout()
    tag  = f"{sigma:.4f}".replace(".", "p")
    path = figures_dir / f"convergence_sigma{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_summary(results_dir: Path, figures_dir: Path) -> None:
    """Aggregate all per-sigma caches and plot mean Δskill vs sigma."""
    files = sorted(results_dir.glob("sigma*_n[0-9]*.npz"))
    if not files:
        print("No cached results found in", results_dir)
        return

    rows = []
    for f in files:
        stem   = f.stem                                  # e.g. sigma0p0100_n50
        s_part = stem.split("_")[0].replace("sigma", "").replace("p", ".")
        try:
            sigma = float(s_part)
        except ValueError:
            continue
        d = np.load(f)
        rows.append({
            "sigma":      sigma,
            "crps_mean":  d["crps_draws"].mean(),
            "crps_se":    d["crps_draws"].std(ddof=1) / np.sqrt(len(d["crps_draws"])),
            "es_mean":    d["es_draws"].mean(),
            "es_se":      d["es_draws"].std(ddof=1) / np.sqrt(len(d["es_draws"])),
            "vs_mean":    d["vs_draws"].mean(),
            "vs_se":      d["vs_draws"].std(ddof=1) / np.sqrt(len(d["vs_draws"])),
            "crps_base":  float(d["crps_base"]),
            "es_base":    float(d["es_base"]),
            "vs_base":    float(d["vs_base"]),
        })

    rows.sort(key=lambda r: r["sigma"])
    sigmas = np.array([r["sigma"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in [
        ("crps", "CRPS (marginal)", "#4C72B0"),
        ("es",   "ES (joint)",      "#DD8452"),
        ("vs",   "VS (joint/dep)",  "#55A868"),
    ]:
        means = np.array([r[f"{key}_mean"] for r in rows])
        ses   = np.array([r[f"{key}_se"]   for r in rows])
        ax.errorbar(sigmas, means, yerr=1.96 * ses,
                    marker="o", label=label, color=color, capsize=3, lw=1.5, ms=5)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("σ_rel (filter-normalised perturbation scale)", fontsize=9)
    ax.set_ylabel("Mean Δ skill (perturbed − baseline)", fontsize=9)
    ax.set_title("L96 DDPM: marginal vs joint skill sensitivity to weight perturbation",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5, which="both")
    fig.tight_layout()

    path = figures_dir / "sensitivity_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Print table
    print(f"\n{'sigma':>10}  {'ΔCRPS':>10}  {'ΔES':>10}  {'ΔVS':>10}  "
          f"{'CRPS_base':>10}  {'ES_base':>10}  {'VS_base':>10}")
    for r in rows:
        print(f"{r['sigma']:10.4f}  {r['crps_mean']:+10.4f}  {r['es_mean']:+10.4f}  "
              f"{r['vs_mean']:+10.4f}  {r['crps_base']:+10.4f}  "
              f"{r['es_base']:+10.4f}  {r['vs_base']:+10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight perturbation sensitivity for the L96 DDPM."
    )
    parser.add_argument("--ckpt",      type=Path,  default=DEFAULT_CKPT)
    parser.add_argument("--data",      type=Path,  default=DEFAULT_DATA)
    parser.add_argument("--sigma",     type=float, default=0.01,
                        help="Relative perturbation scale (fraction of each layer norm).")
    parser.add_argument("--n-draws",   type=int,   default=50)
    parser.add_argument("--n-eval",    type=int,   default=2000,
                        help="Number of test samples (default 2000).")
    parser.add_argument("--n-ens",     type=int,   default=50,
                        help="Ensemble size per forecast.")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--rerun",     action="store_true")
    parser.add_argument("--smoke",     action="store_true",
                        help="Quick smoke test: n_eval=50, n_draws=3.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference; just rebuild the summary figure.")
    parser.add_argument("--device",    default=None)
    args = parser.parse_args()

    if args.smoke:
        args.n_eval  = 50
        args.n_draws = 3

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    results_dir = OUT_DIR / "results"
    figures_dir = OUT_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        _plot_summary(results_dir, figures_dir)
        return

    smoke_tag  = "_smoke" if args.smoke else ""
    sigma_tag  = f"{args.sigma:.4f}".replace(".", "p")
    cache_path = results_dir / f"sigma{sigma_tag}_n{args.n_draws}{smoke_tag}.npz"

    if cache_path.exists() and not args.rerun:
        print(f"Loading cached results from {cache_path}")
        d = np.load(cache_path)
        crps_draws = d["crps_draws"]
        es_draws   = d["es_draws"]
        vs_draws   = d["vs_draws"]
        crps_base  = float(d["crps_base"])
        es_base    = float(d["es_base"])
        vs_base    = float(d["vs_base"])
    else:
        model, ckpt = _load_model(args.ckpt, args.device)
        mu, sigma   = ckpt["mu"], ckpt["sigma"]
        lead        = int(ckpt["lead_steps"])

        states, _ = load_trajectory(str(args.data))
        test_raw  = states[TEST_START:]
        rng_data  = np.random.default_rng(args.seed)
        n_avail   = len(test_raw) - lead
        idx       = rng_data.choice(n_avail, size=min(args.n_eval, n_avail), replace=False)
        idx.sort()

        cond_norm   = (test_raw[idx]        - mu) / sigma
        target_norm = (test_raw[idx + lead] - mu) / sigma

        # Climatological reference — fixed for all draws
        train_raw = states[:TRAIN_END]
        rng_clim  = np.random.default_rng(args.seed + 1)
        clim_idx  = rng_clim.integers(0, len(train_raw), size=(len(idx), args.n_ens))
        clim_ens_norm = (train_raw[clim_idx] - mu) / sigma

        cond_t = torch.from_numpy(cond_norm).float().to(args.device)

        print(f"Device: {args.device}  |  σ_rel: {args.sigma}  |  "
              f"n_draws: {args.n_draws}  |  n_eval: {len(idx)}  |  "
              f"n_ens: {args.n_ens}", flush=True)

        # Baseline (unperturbed)
        print("\nRunning baseline ...", flush=True)
        t0 = time.perf_counter()
        crps_base, es_base, vs_base = _run_eval(
            model, cond_t, target_norm, clim_ens_norm, args.n_ens
        )
        print(f"  {time.perf_counter()-t0:.1f}s  "
              f"CRPS={crps_base:+.4f}  ES={es_base:+.4f}  VS={vs_base:+.4f}", flush=True)

        base_state  = {k: v.clone() for k, v in model.state_dict().items()}
        param_names = {n for n, _ in model.named_parameters()}
        rng_perturb = np.random.default_rng(args.seed + 2)

        crps_draws = np.full(args.n_draws, np.nan)
        es_draws   = np.full(args.n_draws, np.nan)
        vs_draws   = np.full(args.n_draws, np.nan)

        for i in range(args.n_draws):
            t0 = time.perf_counter()
            perturbed = _perturb_state_dict(base_state, args.sigma, rng_perturb, only_keys=param_names)
            model.load_state_dict({k: v.to(args.device) for k, v in perturbed.items()})
            crps_sk, es_sk, vs_sk = _run_eval(
                model, cond_t, target_norm, clim_ens_norm, args.n_ens
            )
            crps_draws[i] = crps_sk - crps_base
            es_draws[i]   = es_sk   - es_base
            vs_draws[i]   = vs_sk   - vs_base
            print(f"  Draw {i+1}/{args.n_draws}  {time.perf_counter()-t0:.1f}s  "
                  f"ΔCRPS={crps_draws[i]:+.4f}  ΔES={es_draws[i]:+.4f}  "
                  f"ΔVS={vs_draws[i]:+.4f}", flush=True)

        model.load_state_dict(base_state)

        np.savez(
            cache_path,
            crps_draws=crps_draws, es_draws=es_draws, vs_draws=vs_draws,
            crps_base=crps_base,   es_base=es_base,   vs_base=vs_base,
        )
        print(f"\nSaved {cache_path}", flush=True)

    # Convergence plot for this sigma
    _plot_convergence(
        crps_draws, es_draws, vs_draws,
        crps_base, es_base, vs_base,
        args.sigma, figures_dir,
    )

    # Summary stats
    n = len(crps_draws)
    print(f"\nσ_rel={args.sigma}  N={n}")
    print(f"  Baseline  CRPS={crps_base:+.4f}  ES={es_base:+.4f}  VS={vs_base:+.4f}")
    for name, arr in [("CRPS", crps_draws), ("ES", es_draws), ("VS", vs_draws)]:
        t = arr.mean() / (arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        print(f"  Δ{name}: mean={arr.mean():+.4f}  std={arr.std():.4f}  t={t:.2f}")


if __name__ == "__main__":
    main()
