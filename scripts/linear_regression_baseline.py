"""
Linear regression baseline using the exact same conditioning vector as the
diffusion model (106 features, z-scored).

Fits Ridge regression separately per lead time on the training set, then
evaluates on both train and val sets. Reports R² and RMSE per station × lead.

Run: PYTHONPATH=. python scripts/linear_regression_baseline.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from src.model.dataset import SSHDiffusionDataset
from src.config import LEAD_WEEKS, TIDE_GAUGES

import os
os.makedirs("figures", exist_ok=True)

STATION_NAMES = [g["name"].replace("_", " ") for g in TIDE_GAUGES]
LEADS = list(LEAD_WEEKS)          # [2, 4, 8, 12]
ALPHAS = np.logspace(-2, 4, 20)   # Ridge regularisation grid


# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading train dataset ...")
train_ds = SSHDiffusionDataset(split="train")
print("Loading val dataset ...")
val_ds   = SSHDiffusionDataset(split="val")


def extract_by_lead(ds: SSHDiffusionDataset, lead: int):
    """Return (X, Y) arrays for a single lead time."""
    idx = [i for i, (_, l) in enumerate(ds.pairs) if l == lead]
    X = np.stack([ds[i][0].numpy() for i in idx])   # (N, cond_dim)
    Y = np.stack([ds[i][1].numpy() for i in idx])   # (N, n_gauges)
    return X, Y


# ── Fit and score ─────────────────────────────────────────────────────────────
n_stations = len(TIDE_GAUGES)
n_leads    = len(LEADS)

r2_train  = np.full((n_leads, n_stations), np.nan)
r2_val    = np.full((n_leads, n_stations), np.nan)
rmse_train = np.full((n_leads, n_stations), np.nan)
rmse_val   = np.full((n_leads, n_stations), np.nan)
best_alpha = np.full((n_leads, n_stations), np.nan)

for li, lead in enumerate(LEADS):
    print(f"\nLead {lead}wk ...")
    X_tr, Y_tr = extract_by_lead(train_ds, lead)
    X_va, Y_va = extract_by_lead(val_ds,   lead)
    print(f"  Train N={len(X_tr)}, Val N={len(X_va)}")

    for si in range(n_stations):
        y_tr = Y_tr[:, si]
        y_va = Y_va[:, si]

        # RidgeCV selects alpha via leave-one-out on training set
        reg = RidgeCV(alphas=ALPHAS, fit_intercept=True)
        reg.fit(X_tr, y_tr)
        best_alpha[li, si] = reg.alpha_

        pred_tr = reg.predict(X_tr)
        pred_va = reg.predict(X_va)

        r2_train[li, si]   = r2_score(y_tr, pred_tr)
        r2_val[li, si]     = r2_score(y_va, pred_va)
        rmse_train[li, si] = np.sqrt(np.mean((y_tr - pred_tr) ** 2))
        rmse_val[li, si]   = np.sqrt(np.mean((y_va - pred_va) ** 2))

    print(f"  Train R²: {r2_train[li].round(3)}")
    print(f"  Val   R²: {r2_val[li].round(3)}")


# ── Print summary table ───────────────────────────────────────────────────────
short = [g["name"].split("_")[0] for g in TIDE_GAUGES]
header = f"{'Lead':>6}  " + "  ".join(f"{s:>10}" for s in short)
print("\n── Train R² ──")
print(header)
for li, lead in enumerate(LEADS):
    row = f"{lead:>4}wk  " + "  ".join(f"{r2_train[li,si]:>10.3f}" for si in range(n_stations))
    print(row)

print("\n── Val R² ──")
print(header)
for li, lead in enumerate(LEADS):
    row = f"{lead:>4}wk  " + "  ".join(f"{r2_val[li,si]:>10.3f}" for si in range(n_stations))
    print(row)


# ── Figure 1: R² heatmaps (train vs val side by side) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
fig.suptitle("Ridge regression R² — same conditioning vector as DDPM", fontsize=11, fontweight="bold")

lead_labels = [f"{l}wk" for l in LEADS]

for ax, data, title in zip(axes,
                            [r2_train, r2_val],
                            ["Train (1993–2020)", "Val (2021–2023)"]):
    vmax = max(0.05, np.nanmax(np.abs(data)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(n_stations))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_leads))
    ax.set_yticklabels(lead_labels, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Lead time", fontsize=8)
    # Annotate cells
    for li in range(n_leads):
        for si in range(n_stations):
            val = data[li, si]
            ax.text(si, li, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.3 else "black")
    plt.colorbar(im, ax=ax, label="R²", shrink=0.85)

fig.savefig("figures/linreg_r2_heatmap.pdf")
plt.close()
print("\nSaved: figures/linreg_r2_heatmap.pdf")


# ── Figure 2: R² vs lead per station (train + val lines) ─────────────────────
fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), constrained_layout=True, sharey=False)
fig.suptitle("Ridge regression R² vs lead — per station", fontsize=11, fontweight="bold")

for si, ax in enumerate(axes.flat):
    ax.plot(LEADS, r2_train[:, si], "o-", color="#4878CF", lw=1.5, ms=5, label="Train")
    ax.plot(LEADS, r2_val[:, si],   "s--", color="#C44E52", lw=1.5, ms=5, label="Val")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_title(STATION_NAMES[si], fontsize=8)
    ax.set_xlabel("Lead (wk)", fontsize=7.5)
    ax.set_ylabel("R²", fontsize=7.5)
    ax.set_xticks(LEADS)
    ax.tick_params(labelsize=7)
    if si == 0:
        ax.legend(fontsize=7)

fig.savefig("figures/linreg_r2_vs_lead.pdf")
plt.close()
print("Saved: figures/linreg_r2_vs_lead.pdf")


# ── Figure 3: RMSE vs lead (z-score units) ───────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), constrained_layout=True, sharey=False)
fig.suptitle("Ridge regression RMSE vs lead (z-score units) — per station",
             fontsize=11, fontweight="bold")

for si, ax in enumerate(axes.flat):
    ax.plot(LEADS, rmse_train[:, si], "o-", color="#4878CF", lw=1.5, ms=5, label="Train")
    ax.plot(LEADS, rmse_val[:, si],   "s--", color="#C44E52", lw=1.5, ms=5, label="Val")
    ax.axhline(1.0, color="k", lw=0.7, ls=":", label="σ=1 (no skill)")
    ax.set_title(STATION_NAMES[si], fontsize=8)
    ax.set_xlabel("Lead (wk)", fontsize=7.5)
    ax.set_ylabel("RMSE (z)", fontsize=7.5)
    ax.set_xticks(LEADS)
    ax.tick_params(labelsize=7)
    if si == 0:
        ax.legend(fontsize=7)

fig.savefig("figures/linreg_rmse_vs_lead.pdf")
plt.close()
print("Saved: figures/linreg_rmse_vs_lead.pdf")

print("\nDone.")
