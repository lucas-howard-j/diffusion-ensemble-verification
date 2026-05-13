"""
Diagnostic histograms of SSH PCs, SST PCs, NAO, and ENSO index for the
training set (1993–2020) — both raw (pre-z-score) and z-scored values.

Figures saved as PDF per project convention.
Run: PYTHONPATH=. python scripts/input_normalization_histograms.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from src.eof.compute_eofs import load_eofs
from src.indices.nao import load_nao
from src.indices.enso import load_enso
from src.config import (
    N_SSH_MODES, N_SST_MODES,
    TRAIN_YEARS,
)

OUT_DIR = "figures"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
HIST_KW  = dict(bins=40, density=True, alpha=0.65, color="#4878CF", edgecolor="none")
FIT_KW   = dict(color="#C44E52", lw=1.5, label="N(μ,σ) fit")
NORM_KW  = dict(color="k", lw=1.2, ls="--", label="N(0,1)")
TITLE_FS = 8
LABEL_FS = 7

def gaussian_fit_line(data, x):
    """Return PDF of fitted Gaussian over x."""
    mu, sigma = data.mean(), data.std()
    return stats.norm.pdf(x, mu, sigma)

def add_stats(ax, data, fontsize=6.5):
    """Annotate axis with μ, σ, skew, kurt."""
    sk = stats.skew(data)
    ku = stats.kurtosis(data)   # excess kurtosis
    ax.text(0.97, 0.95,
            f"μ={data.mean():.2f}\nσ={data.std():.2f}\nskew={sk:.2f}\nkurt={ku:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=fontsize, family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SSH PCs  (raw + z-scored, side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pcs(var: str, n_modes: int, tag: str):
    eof = load_eofs(var)
    pcs_all  = eof["pcs"][:, :n_modes].astype(np.float64)   # (T, n_modes)
    time     = pd.to_datetime(eof["time"])
    train_mask = time.year.isin(list(TRAIN_YEARS))

    pcs_train = pcs_all[train_mask]                          # raw training

    mu  = pcs_train.mean(axis=0)
    std = pcs_train.std(axis=0) + 1e-8
    pcs_z = (pcs_train - mu) / std                           # z-scored

    ncols = 5
    nrows = int(np.ceil(n_modes / ncols))

    # ── raw ──────────────────────────────────────────────────────────────────
    fig_raw, axes_raw = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.4, nrows * 1.9),
        constrained_layout=True,
    )
    fig_raw.suptitle(
        f"{tag.upper()} PCs — raw (training 1993–2020, N={train_mask.sum()})",
        fontsize=10, fontweight="bold",
    )

    # ── z-scored ─────────────────────────────────────────────────────────────
    fig_z, axes_z = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.4, nrows * 1.9),
        constrained_layout=True,
    )
    fig_z.suptitle(
        f"{tag.upper()} PCs — z-scored (training 1993–2020)",
        fontsize=10, fontweight="bold",
    )

    for i in range(n_modes):
        ax_r = axes_raw.flat[i]
        ax_z = axes_z.flat[i]

        raw_i = pcs_train[:, i]
        z_i   = pcs_z[:, i]

        # ── raw panel ──
        ax_r.hist(raw_i, **HIST_KW)
        x = np.linspace(raw_i.min(), raw_i.max(), 300)
        ax_r.plot(x, gaussian_fit_line(raw_i, x), **FIT_KW)
        add_stats(ax_r, raw_i)
        ax_r.set_title(f"Mode {i+1}", fontsize=TITLE_FS)
        ax_r.set_xlabel("PC score", fontsize=LABEL_FS)
        ax_r.set_ylabel("Density", fontsize=LABEL_FS)
        ax_r.tick_params(labelsize=LABEL_FS)

        # ── z-scored panel ──
        ax_z.hist(z_i, **HIST_KW)
        x = np.linspace(z_i.min(), z_i.max(), 300)
        ax_z.plot(x, gaussian_fit_line(z_i, x), **FIT_KW)
        # Reference N(0,1)
        xn = np.linspace(-4, 4, 300)
        ax_z.plot(xn, stats.norm.pdf(xn), **NORM_KW)
        add_stats(ax_z, z_i)
        ax_z.set_title(f"Mode {i+1}", fontsize=TITLE_FS)
        ax_z.set_xlabel("z-score", fontsize=LABEL_FS)
        ax_z.set_ylabel("Density", fontsize=LABEL_FS)
        ax_z.tick_params(labelsize=LABEL_FS)

    # Hide unused axes
    for j in range(n_modes, len(axes_raw.flat)):
        axes_raw.flat[j].set_visible(False)
        axes_z.flat[j].set_visible(False)

    # Shared legend
    for fig in (fig_raw, fig_z):
        handles, labels = fig.axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=LABEL_FS,
                       framealpha=0.9, ncol=2)

    path_raw = f"{OUT_DIR}/{var}_pcs_raw_histograms.pdf"
    path_z   = f"{OUT_DIR}/{var}_pcs_zscored_histograms.pdf"
    fig_raw.savefig(path_raw)
    fig_z.savefig(path_z)
    plt.close("all")
    print(f"  Saved: {path_raw}")
    print(f"  Saved: {path_z}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NAO + ENSO  (already standardised in their loaders)
# ─────────────────────────────────────────────────────────────────────────────
def plot_indices():
    nao_series  = load_nao()   # pd.Series, weekly, already z-scored
    enso_series = load_enso()  # pd.Series, weekly, already z-scored

    # Align to a common weekly index and filter to training years
    nao_train  = nao_series[nao_series.index.year.isin(list(TRAIN_YEARS))].dropna()
    enso_train = enso_series[enso_series.index.year.isin(list(TRAIN_YEARS))].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    fig.suptitle(
        "Climate indices — training set (1993–2020, already z-scored by loaders)",
        fontsize=10, fontweight="bold",
    )

    datasets = [
        (axes[0], nao_train.values,  "NAO index"),
        (axes[1], enso_train.values, "ENSO (Niño3.4) index"),
    ]

    for ax, data, title in datasets:
        ax.hist(data, **HIST_KW)

        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, gaussian_fit_line(data, x), **FIT_KW)

        xn = np.linspace(-4, 4, 300)
        ax.plot(xn, stats.norm.pdf(xn), **NORM_KW)

        add_stats(ax, data, fontsize=8)
        ax.set_title(f"{title}  (N={len(data)})", fontsize=9)
        ax.set_xlabel("Index value (standardised)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8,
               framealpha=0.9, ncol=2)

    path = f"{OUT_DIR}/nao_enso_histograms.pdf"
    fig.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Summary QQ-plots: one per variable group
# ─────────────────────────────────────────────────────────────────────────────
def plot_qq_summary():
    """
    QQ-plots vs N(0,1) after z-scoring for SSH PCs, SST PCs, NAO, ENSO.
    Overlays all modes on one axis per variable group (helps spot outlier tails).
    """
    sst_eof  = load_eofs("sst")
    ssh_eof  = load_eofs("ssh")

    sst_time = pd.to_datetime(sst_eof["time"])
    ssh_time = pd.to_datetime(ssh_eof["time"])
    train_sst = sst_time.year.isin(list(TRAIN_YEARS))
    train_ssh = ssh_time.year.isin(list(TRAIN_YEARS))

    def zscore_train(pcs_all, mask):
        tr = pcs_all[mask]
        mu = tr.mean(0); sd = tr.std(0) + 1e-8
        return (tr - mu) / sd

    sst_z = zscore_train(sst_eof["pcs"][:, :N_SST_MODES], train_sst)
    ssh_z = zscore_train(ssh_eof["pcs"][:, :N_SSH_MODES], train_ssh)

    nao_series  = load_nao()
    enso_series = load_enso()
    nao_raw  = nao_series[nao_series.index.year.isin(list(TRAIN_YEARS))].dropna().values
    enso_raw = enso_series[enso_series.index.year.isin(list(TRAIN_YEARS))].dropna().values
    # z-score the already-standardised indices relative to their training mean/std
    nao_z  = (nao_raw  - nao_raw.mean())  / (nao_raw.std()  + 1e-8)
    enso_z = (enso_raw - enso_raw.mean()) / (enso_raw.std() + 1e-8)

    groups = [
        ("SSH PCs (modes 1–15)", ssh_z),
        ("SST PCs (modes 1–15)", sst_z),
        ("NAO index",            nao_z[:, None]),
        ("ENSO index",           enso_z[:, None]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    fig.suptitle("QQ-plots vs N(0,1) after z-scoring — training set",
                 fontsize=10, fontweight="bold")

    ref_line = np.array([-3.5, 3.5])

    for ax, (title, data) in zip(axes, groups):
        n_modes = data.shape[1]
        for m in range(n_modes):
            col = data[:, m]
            osm, osr = stats.probplot(col, dist="norm", fit=False)
            alpha = max(0.2, 1.0 / n_modes)
            ax.plot(osm, osr, ".", ms=1.5, alpha=alpha, color="#4878CF")

        ax.plot(ref_line, ref_line, "r-", lw=1.2, label="y = x")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        ax.set_ylabel("Sample quantiles", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    path = f"{OUT_DIR}/input_qq_plots.pdf"
    fig.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SSH PCs ...")
    plot_pcs("ssh", N_SSH_MODES, "ssh")

    print("SST PCs ...")
    plot_pcs("sst", N_SST_MODES, "sst")

    print("NAO + ENSO indices ...")
    plot_indices()

    print("QQ-plots ...")
    plot_qq_summary()

    print("\nDone. All figures in figures/")
