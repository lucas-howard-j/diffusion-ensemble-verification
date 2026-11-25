"""
Compare DDPM, perfect-model dynamical, and deterministic-emulator ensembles.

Workflow:
  1. Evaluate the DDPM ensemble → CRPS skill, VS skill, spread
  2. Calibrate σ_ic so the dynamical ensemble spread matches the DDPM's
     (optional; skip with --sigma-ic to fix it manually)
  3. Evaluate the dynamical ensemble with calibrated σ_ic
  4. Evaluate the deterministic-emulator ensemble with the same σ_ic
  5. Plot a 2-panel comparison (CRPS skill, VS skill) and print a summary table

Pre-computed results can be loaded instead of re-running:
  --dyn-results PATH   load dynamical npz from this path
  --det-results PATH   load det-emulator npz from this path

Usage:
    python -m lorenz96.compare_ensembles \\
        --ddpm-ckpt lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt \\
        --det-ckpt  lorenz96/checkpoints/det_emulator.pt \\
        [--data PATH] [--n-eval 2000] [--n-ens 500] \\
        [--sigma-ic 0.01 | --match-spread] \\
        [--out-dir lorenz96/ensemble_results]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lorenz96.config import TRAIN_END, TEST_START
from lorenz96.dataset import load_trajectory
from lorenz96.eval import crps_ensemble, variogram_score
from lorenz96.model import DDPM, ConvScoreNet
from lorenz96.ensemble_dynamical import run_dynamical_ensemble, integrate_batch
from lorenz96.ensemble_deterministic import run_det_ensemble


# ---------------------------------------------------------------------------
# Shared test-IC selection (same seed → identical ICs across all methods)
# ---------------------------------------------------------------------------

def _select_test_ics(states, lead, n_eval, seed=42):
    test_raw = states[TEST_START:]
    max_test = len(test_raw) - lead
    rng = np.random.default_rng(seed)
    idx = rng.choice(max_test, size=min(n_eval, max_test), replace=False)
    idx.sort()
    return idx, rng


def _clim_baseline(states, idx, n_ens, mu, sigma, rng):
    """Climatological draws from the IC-selection rng, continuing the same
    stream as ensemble_{dynamical,deterministic,crps,ddpm}.py (seed 42)."""
    train_raw = states[:TRAIN_END]
    clim_idx  = rng.integers(0, len(train_raw), size=(len(idx), n_ens))
    return (train_raw[clim_idx] - mu) / sigma   # (n, M, N)


# ---------------------------------------------------------------------------
# DDPM evaluation  (returns normalised ensemble)
# ---------------------------------------------------------------------------

def _eval_ddpm(ckpt_path, states, idx, n_ens, device, sample_steps=200):
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    mu     = ckpt["mu"]
    sigma  = ckpt["sigma"]
    lead   = ckpt["lead_steps"]
    cfg    = json.loads(ckpt["config"])

    net = ConvScoreNet(
        hidden_ch=cfg["hidden_ch"], n_res=cfg["n_res"],
        emb_dim=cfg["emb_dim"], kernel_size=cfg["kernel"],
    )
    model = DDPM(net, T=cfg["T_diff"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_raw  = states[TEST_START:]
    cond_raw  = test_raw[idx]
    target_raw = test_raw[idx + lead]

    cond_norm   = (cond_raw   - mu) / sigma
    target_norm = (target_raw - mu) / sigma

    print(f"Sampling DDPM: {len(idx)} × {n_ens} members (steps={sample_steps}) ...",
          flush=True)
    cond_t = torch.from_numpy(cond_norm.astype(np.float32)).to(device)
    with torch.no_grad():
        ens = model.sample(cond_t, n_samples=n_ens, sample_steps=sample_steps)
    ens_norm = ens.cpu().numpy()   # (n, M, N)

    spread = float(ens_norm.std(axis=1).mean())
    print(f"  spread={spread:.4f}", flush=True)

    return {
        "ens_norm":   ens_norm,
        "target_norm": target_norm,
        "mu": mu, "sigma": sigma, "lead": lead, "spread": spread,
    }


# ---------------------------------------------------------------------------
# σ_ic calibration via quick grid search
# ---------------------------------------------------------------------------

def _calibrate_sigma_ic(
    target_spread, states, ckpt_path, data_cfg,
    n_calib=200, n_ens_calib=100, seed=99, chunk_size=200,
):
    """
    Find σ_ic such that the dynamical ensemble spread ≈ target_spread.
    Uses a coarse log-grid, then linear interpolation.
    """
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mu    = ckpt["mu"]
    sigma = ckpt["sigma"]
    lead  = ckpt["lead_steps"]
    dt    = data_cfg["dt"]
    N     = len(mu)

    rng = np.random.default_rng(seed)
    test_raw = states[TEST_START:]
    max_test = len(test_raw) - lead
    idx = rng.choice(max_test, size=min(n_calib, max_test), replace=False)
    idx.sort()
    cond_raw = test_raw[idx]

    grid = np.array([0.001, 0.003, 0.007, 0.015, 0.03, 0.07, 0.15, 0.3])
    spreads = []

    print(f"Calibrating σ_ic to match DDPM spread={target_spread:.4f} ...", flush=True)
    for sig in grid:
        ic_norm = (cond_raw - mu) / sigma          # (n_calib, N)
        eps     = rng.standard_normal((n_calib, n_ens_calib, N)) * sig
        ic_pert = (ic_norm[:, None, :] + eps) * sigma[None, None, :] + mu[None, None, :]
        batch   = ic_pert.reshape(n_calib * n_ens_calib, N)
        final   = integrate_batch(batch, lead, dt)
        final_norm = (final - mu) / sigma
        sp = float(final_norm.reshape(n_calib, n_ens_calib, N).std(axis=1).mean())
        spreads.append(sp)
        print(f"  σ_ic={sig:.3f}  spread={sp:.4f}", flush=True)

    spreads = np.array(spreads)

    # Linear interpolation in log space
    log_grid    = np.log(grid)
    log_spreads = np.log(spreads)
    log_target  = np.log(target_spread)

    sigma_ic = float(np.exp(np.interp(log_target, log_spreads, log_grid)))
    sigma_ic = float(np.clip(sigma_ic, grid[0], grid[-1]))
    print(f"  → calibrated σ_ic = {sigma_ic:.4f}", flush=True)
    return sigma_ic


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _score(target_norm, ens_norm, clim_ens_norm):
    crps_ens  = crps_ensemble(target_norm, ens_norm).mean(0)
    crps_clim = crps_ensemble(target_norm, clim_ens_norm).mean(0)
    crps_node = 1 - crps_ens / crps_clim
    # Ratio of means over ICs and nodes (the statistic bootstrap_skill_ci resamples)
    crps_skill = float(1 - crps_ens.mean() / crps_clim.mean())

    vs_ens   = variogram_score(target_norm, ens_norm)
    vs_clim  = variogram_score(target_norm, clim_ens_norm)
    vs_skill = float(1 - vs_ens / vs_clim)
    spread   = float(ens_norm.std(axis=1).mean())

    return {"crps_skill": crps_skill, "vs_skill": vs_skill,
            "crps_node": crps_node, "spread": spread}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(results, out_dir, sigma_ic):
    labels  = ["DDPM\n(one-shot)", "Dynamical\n(perfect model)", "Det. emulator\n(IC perturb)"]
    colors  = ["#4C72B0", "#55A868", "#DD8452"]
    methods = ["ddpm", "dyn", "det"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=False)

    for ax, metric, ylabel in zip(
        axes,
        ["crps_skill", "vs_skill"],
        ["CRPS skill  (1 − CRPS / CRPS$_{clim}$)",
         "VS skill  (1 − VS / VS$_{clim}$)"],
    ):
        vals = [results[m][metric] for m in methods]
        bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=0.7, width=0.5)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        for bar, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            offset = 0.004 if v >= 0 else -0.004
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:+.4f}", ha="center", va=va, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    spreads = "  ".join(f"{m}={results[m]['spread']:.3f}" for m in methods)
    fig.suptitle(
        f"L96 DDPM vs dynamical vs det. emulator  |  n_ens=500  |  σ_ic={sigma_ic:.4f}\n"
        f"spread: {spreads}",
        fontsize=9,
    )
    fig.tight_layout()

    out = Path(out_dir) / "compare_ensembles.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out}")

    # Per-node CRPS skill comparison
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    N = len(results["ddpm"]["crps_node"])
    x = np.arange(N)
    w = 0.27
    for i, (m, label, color) in enumerate(zip(methods, labels, colors)):
        ax2.bar(x + (i - 1) * w, results[m]["crps_node"], width=w,
                label=label.replace("\n", " "), color=color, edgecolor="none", alpha=0.85)
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("L96 node  $k$")
    ax2.set_ylabel("CRPS skill per node")
    ax2.legend(fontsize=8)
    ax2.set_title(f"Per-node CRPS skill  |  σ_ic={sigma_ic:.4f}")
    fig2.tight_layout()
    out2 = Path(out_dir) / "compare_crps_node.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Per-node figure saved → {out2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm-ckpt",  default="lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt")
    parser.add_argument("--det-ckpt",   default="lorenz96/checkpoints/det_emulator.pt")
    parser.add_argument("--data",       default="lorenz96/l96_trajectory.npz")
    parser.add_argument("--n-eval",     type=int, default=2_000)
    parser.add_argument("--n-ens",      type=int, default=500)
    parser.add_argument("--sigma-ic",   type=float, default=None,
                        help="Fix σ_ic; if omitted, calibrate to match DDPM spread")
    parser.add_argument("--match-spread", action="store_true",
                        help="Force spread-matching calibration (default when --sigma-ic omitted)")
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--out-dir",    default="lorenz96/ensemble_results")
    parser.add_argument("--dyn-results",  default=None,
                        help="Load pre-computed dynamical results from this npz")
    parser.add_argument("--det-results",  default=None,
                        help="Load pre-computed det-emulator results from this npz")
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states, data_cfg = load_trajectory(args.data)
    lead = int(torch.load(args.ddpm_ckpt, map_location="cpu", weights_only=False)["lead_steps"])
    idx, ic_rng = _select_test_ics(states, lead, args.n_eval)

    # ---- DDPM ----
    print("\n" + "=" * 60)
    print("DDPM evaluation")
    print("=" * 60)
    ddpm_out = _eval_ddpm(
        args.ddpm_ckpt, states, idx, args.n_ens, args.device, args.sample_steps
    )
    mu    = ddpm_out["mu"]
    sigma = ddpm_out["sigma"]
    lead  = ddpm_out["lead"]

    target_norm     = ddpm_out["target_norm"]
    ddpm_ens_norm   = ddpm_out["ens_norm"]
    clim_ens_norm   = _clim_baseline(states, idx, args.n_ens, mu, sigma, ic_rng)

    ddpm_scores = _score(target_norm, ddpm_ens_norm, clim_ens_norm)
    ddpm_scores["spread"] = ddpm_out["spread"]
    print(f"DDPM  CRPS={ddpm_scores['crps_skill']:+.4f}  "
          f"VS={ddpm_scores['vs_skill']:+.4f}  spread={ddpm_scores['spread']:.4f}")

    # ---- σ_ic calibration ----
    if args.sigma_ic is not None:
        sigma_ic = args.sigma_ic
        print(f"\nUsing fixed σ_ic={sigma_ic:.4f}")
    else:
        print("\n" + "=" * 60)
        print("σ_ic calibration")
        print("=" * 60)
        sigma_ic = _calibrate_sigma_ic(
            ddpm_out["spread"], states, args.ddpm_ckpt, data_cfg
        )

    # ---- Dynamical ----
    print("\n" + "=" * 60)
    print("Dynamical ensemble")
    print("=" * 60)

    if args.dyn_results:
        print(f"Loading pre-computed results from {args.dyn_results}")
        d = np.load(args.dyn_results)
        dyn_scores = {k: float(d[k]) for k in ("crps_skill", "vs_skill", "spread")}
        dyn_scores["crps_node"] = d["crps_node"]
    else:
        dyn_out = run_dynamical_ensemble(
            ckpt_path=args.ddpm_ckpt,   # borrow μ, σ, lead from DDPM ckpt
            data_path=args.data,
            sigma_ic=sigma_ic,
            n_eval=args.n_eval,
            n_ens=args.n_ens,
            out_path=str(out_dir / "dyn_results.npz"),
        )
        d = np.load(str(out_dir / "dyn_results.npz"))
        dyn_scores = {k: float(d[k]) for k in ("crps_skill", "vs_skill", "spread")}
        dyn_scores["crps_node"] = d["crps_node"]

    print(f"Dyn   CRPS={dyn_scores['crps_skill']:+.4f}  "
          f"VS={dyn_scores['vs_skill']:+.4f}  spread={dyn_scores['spread']:.4f}")

    # ---- Deterministic emulator ----
    print("\n" + "=" * 60)
    print("Deterministic emulator ensemble")
    print("=" * 60)

    if args.det_results:
        print(f"Loading pre-computed results from {args.det_results}")
        d = np.load(args.det_results)
        det_scores = {k: float(d[k]) for k in ("crps_skill", "vs_skill", "spread")}
        det_scores["crps_node"] = d["crps_node"]
    else:
        run_det_ensemble(
            ckpt_path=args.det_ckpt,
            data_path=args.data,
            sigma_ic=sigma_ic,
            n_eval=args.n_eval,
            n_ens=args.n_ens,
            out_path=str(out_dir / "det_results.npz"),
            device=args.device,
        )
        d = np.load(str(out_dir / "det_results.npz"))
        det_scores = {k: float(d[k]) for k in ("crps_skill", "vs_skill", "spread")}
        det_scores["crps_node"] = d["crps_node"]

    print(f"Det   CRPS={det_scores['crps_skill']:+.4f}  "
          f"VS={det_scores['vs_skill']:+.4f}  spread={det_scores['spread']:.4f}")

    # ---- Summary ----
    results = {"ddpm": ddpm_scores, "dyn": dyn_scores, "det": det_scores}

    print("\n" + "=" * 60)
    print(f"{'Method':<22}  {'CRPS skill':>12}  {'VS skill':>10}  {'VS/CRPS':>8}  {'spread':>8}")
    print("-" * 70)
    for label, key in [("DDPM (one-shot)", "ddpm"),
                        ("Dynamical (perfect)", "dyn"),
                        ("Det. emulator", "det")]:
        r = results[key]
        ratio = r["vs_skill"] / r["crps_skill"] if r["crps_skill"] != 0 else float("nan")
        print(f"{label:<22}  {r['crps_skill']:>+12.4f}  {r['vs_skill']:>+10.4f}  "
              f"{ratio:>8.3f}  {r['spread']:>8.4f}")
    print("=" * 60)

    _plot(results, out_dir, sigma_ic)


if __name__ == "__main__":
    main()
