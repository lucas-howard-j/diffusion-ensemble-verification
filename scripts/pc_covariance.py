"""
Within-ensemble PC covariance vs observed PC covariance.

Tests whether DDPM ensemble members reproduce the joint distribution of SSH PCs
or sample each PC dimension independently (diagonal within-ensemble cov).
No EOF reconstruction — avoids the spatial-coherence confound entirely.

Run:
    conda run -n ssh_ml python -m scripts.pc_covariance
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import LEAD_WEEKS
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS = 25
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def _plot_cov(ax, mat, title, vmax):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC index")
    ax.set_ylabel("PC index")
    return im


def main() -> None:
    model, schedule, ckpt, device = load_checkpoint()
    target_dim = int(ckpt["target_dim"])
    print(f"Device: {device}  |  target_dim: {target_dim}")

    ds = SSHDiffusionDataset("val")
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx

    leads = sorted(LEAD_WEEKS)
    valid_ts = sorted(
        t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads)
    )
    print(f"Val time steps: {len(valid_ts)}")

    # ── Observed PC covariance (z-scored; should be ~identity by construction) ──
    # Collect true target PCs across all val (t, lead) pairs
    all_true: dict[int, list[np.ndarray]] = {l: [] for l in leads}
    for t_val in valid_ts:
        for lead in leads:
            _, target = ds[t_to_lead_idx[t_val][lead]]
            all_true[lead].append(target.numpy())

    obs_cov: dict[int, np.ndarray] = {}
    for l in leads:
        X = np.stack(all_true[l])           # (n_times, target_dim)
        X -= X.mean(axis=0)
        obs_cov[l] = (X.T @ X) / (len(X) - 1)

    # ── Within-ensemble PC covariance ────────────────────────────────────────
    within_cov_sum = {l: np.zeros((target_dim, target_dim)) for l in leads}
    within_cov_n   = {l: 0 for l in leads}

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(valid_ts):
            if i % 20 == 0:
                print(f"  t {i}/{len(valid_ts)}")

            conds = []
            for lead in leads:
                cond, _ = ds[t_to_lead_idx[t_val][lead]]
                conds.extend([cond] * N_MEMBERS)

            cond_batch = torch.stack(conds).to(device)
            x = torch.randn(len(cond_batch), target_dim, device=device)
            for step in reversed(range(schedule.T)):
                x = schedule.p_sample(model, x, step, cond_batch)

            pcs = x.cpu().numpy()   # (n_leads * N_MEMBERS, target_dim)

            for j, lead in enumerate(leads):
                block = pcs[j * N_MEMBERS : (j + 1) * N_MEMBERS]   # (25, target_dim)
                centered = block - block.mean(axis=0, keepdims=True)
                within_cov_sum[lead] += (centered.T @ centered) / (N_MEMBERS - 1)
                within_cov_n[lead]   += 1

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\nPC covariance diagnostics (z-scored space; obs cov ≈ identity)")
    print(f"{'Lead':>6}  {'obs diag mean':>14}  {'ens diag mean':>14}  "
          f"{'obs offdiag mean':>17}  {'ens offdiag mean':>17}")

    mask_off = ~np.eye(target_dim, dtype=bool)
    for l in leads:
        wc = within_cov_sum[l] / within_cov_n[l]
        oc = obs_cov[l]
        print(f"  {l:>2}wk"
              f"    {np.diag(oc).mean():>+.4f}"
              f"            {np.diag(wc).mean():>+.4f}"
              f"             {oc[mask_off].mean():>+.4f}"
              f"                {wc[mask_off].mean():>+.4f}")

    # ── Figure: observed vs within-ensemble PC cov per lead ──────────────────
    ncols = 2 * len(leads)
    fig, axes = plt.subplots(2, len(leads), figsize=(3.5 * len(leads), 7))

    for j, l in enumerate(leads):
        wc = within_cov_sum[l] / within_cov_n[l]
        oc = obs_cov[l]
        vmax = max(np.abs(oc).max(), np.abs(wc).max())

        im = _plot_cov(axes[0, j], oc, f"Observed {l}wk", vmax)
        fig.colorbar(im, ax=axes[0, j], shrink=0.7)

        im = _plot_cov(axes[1, j], wc, f"Within-ens {l}wk", vmax)
        fig.colorbar(im, ax=axes[1, j], shrink=0.7)

    axes[0, 0].set_ylabel("Observed\nPC index", fontsize=9)
    axes[1, 0].set_ylabel("Within-ensemble\nPC index", fontsize=9)
    fig.suptitle("SSH PC covariance — observed vs within-ensemble (val set)", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "pc_covariance_val.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
