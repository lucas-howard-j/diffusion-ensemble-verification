"""
Fig 1 skills across DDPM training seeds.

Runs scripts/figure1.py's inference (500 members) for each checkpoint and
reports the Fig 1 quantities per seed plus the across-seed mean and range:
CRPS skill per lead, ES and VS skill per lead, and the pooled shuffle deltas.

Run:
    PYTHONPATH=.:scripts python scripts/ddpm_seed_spread.py [--split test] [--rerun]

Per-seed scores are cached to results/figure1_scores_{split}_{tag}.npz; the
primary model reuses figure1.py's own cache (results/figure3_scores_{split}.npz).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import figure1
from src.config import CHECKPOINTS
from src.evaluation.metrics import ratio_of_means_skill

CKPTS = {
    "s0": CHECKPOINTS / "ddpm_best_build2025.pt",      # primary model (unseeded run)
    "s1": CHECKPOINTS / "ddpm_best_build2025_s1.pt",
    "s2": CHECKPOINTS / "ddpm_best_build2025_s2.pt",
    "s3": CHECKPOINTS / "ddpm_best_build2025_s3.pt",
}


def _cache_path(tag: str, split: str) -> Path:
    if tag == "s0":
        return figure1.CACHE_DIR / f"figure3_scores_{split}.npz"
    return figure1.CACHE_DIR / f"figure1_scores_{split}_{tag}.npz"


def scores_for(tag: str, split: str, rerun: bool) -> dict:
    path = _cache_path(tag, split)
    if path.exists() and not rerun:
        return dict(np.load(path, allow_pickle=True))
    s = figure1.run_inference(CKPTS[tag], split)
    np.savez(path, **s)
    return s


def skills(s: dict) -> dict:
    """Point estimates matching figure1.summarise."""
    out = {"crps": (1 - s["crps_ddpm"].mean(0) / s["crps_ref"].mean(0)).mean(-1)}
    for name in ["es", "vs"]:
        a = s[name][np.all(np.isfinite(s[name]), axis=(1, 2))]
        sk = ratio_of_means_skill(a)                     # (n_leads, 4), as figure1.py
        out[name] = sk[:, 0]
        d_ddpm, d_clim = sk[:, 0] - sk[:, 1], sk[:, 2] - sk[:, 3]
        out[f"{name}_delta"] = np.array([d_ddpm.mean(), d_clim.mean(), (d_ddpm - d_clim).mean()])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    res = {tag: skills(scores_for(tag, args.split, args.rerun)) for tag in CKPTS}
    leads = "  ".join(f"{l:>6}" for l in ["2wk", "4wk", "8wk", "12wk"])
    for key, label in [("crps", "CRPS skill"), ("es", "ES skill"), ("vs", "VS skill")]:
        print(f"\n{label} ({args.split})        {leads}")
        vals = np.stack([res[t][key] for t in CKPTS])
        for t, v in zip(CKPTS, vals):
            print(f"  {t:4s}                  " + "  ".join(f"{x:+.3f}" for x in v))
        print("  mean                  " + "  ".join(f"{x:+.3f}" for x in vals.mean(0)))
        print("  range (max−min)       " + "  ".join(f"{x:.3f}" for x in np.ptp(vals, 0)))
    for key in ["es_delta", "vs_delta"]:
        print(f"\n{key.upper()[:2]} pooled deltas      Δ_DDPM   Δ_GLORYS   Δ−Δ_clim")
        vals = np.stack([res[t][key] for t in CKPTS])
        for t, v in zip(CKPTS, vals):
            print(f"  {t:4s}                  " + "  ".join(f"{x:+.4f}" for x in v))
        print("  mean                  " + "  ".join(f"{x:+.4f}" for x in vals.mean(0)))


if __name__ == "__main__":
    main()
