"""Visualization utilities for validation results."""
from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TIDE_GAUGES


def plot_skill_map(skill_by_gauge: dict[str, float], title: str = "Pearson r",
                   ax=None):
    """Scatter skill values at gauge locations on a coastline map."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-85, -65, 23, 46], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True)

    for g in TIDE_GAUGES:
        r = skill_by_gauge.get(g["name"], np.nan)
        if np.isnan(r):
            continue
        sc = ax.scatter(g["lon"], g["lat"], c=[r], cmap="RdBu_r",
                        vmin=-1, vmax=1, s=120, edgecolor="k",
                        transform=ccrs.PlateCarree(), zorder=5)
    plt.colorbar(sc, ax=ax, label=title, shrink=0.6)
    ax.set_title(title)
    return ax


def plot_correlation_matrices(obs: pd.DataFrame, pred: pd.DataFrame,
                              titles: tuple[str, str] = ("Observed", "Model")):
    """Plot observed and model pairwise station correlation matrices side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in zip(axes, [obs, pred], titles):
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_yticks(range(len(mat.index)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_lead_time_skill(skill_table: pd.DataFrame, ax=None):
    """
    Plot Pearson r vs. lead time for each gauge.

    skill_table : DataFrame with columns = station names, index = lead weeks
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    for col in skill_table.columns:
        ax.plot(skill_table.index, skill_table[col], marker="o", label=col)
    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("Pearson r")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(-0.5, 1.0)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Forecast skill vs. lead time")
    return ax


def plot_spread_error(spread: np.ndarray, error: np.ndarray, ax=None):
    """Scatter ensemble spread vs. absolute error for calibration check."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(spread, error, alpha=0.5, s=12)
    m = max(spread.max(), error.max())
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.set_xlabel("Ensemble spread (std)")
    ax.set_ylabel("|Ensemble mean − obs|")
    ax.set_title("Spread-error calibration")
    return ax
