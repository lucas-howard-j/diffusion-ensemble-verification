"""Skill metrics for SSH predictions."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def block_bootstrap_ci(
    x: np.ndarray,
    block_size: int = 8,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving-block bootstrap confidence interval for the mean of a time series.

    Accounts for temporal autocorrelation by resampling contiguous blocks of
    observations rather than individual time steps (IID assumption).

    Parameters
    ----------
    x : (N, ...) array
        Time series along axis 0; any trailing shape is preserved.
    block_size : int
        Length of each contiguous block in time steps.  For weekly
        initializations a value of 8 (≈ 2-month blocks) is conservative;
        test sensitivity with 4 and 12.
    n_boot : int
        Number of bootstrap resamples (1 000 is standard).
    alpha : float
        Two-tailed significance level.  0.05 → 95 % CI.
    rng : numpy Generator, optional
        Pass a seeded generator for reproducibility.

    Returns
    -------
    lo, hi : arrays with the same shape as x[0]
        Lower and upper bounds of the (1−alpha) CI on the mean.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x)
    N = x.shape[0]

    # Number of blocks needed to cover N time steps
    n_blocks = int(np.ceil(N / block_size))
    max_start = N - block_size  # last valid block start

    boot_means = np.empty((n_boot,) + x.shape[1:])
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        idx = idx[:N]  # trim to original length
        boot_means[b] = x[idx].mean(axis=0)

    lo = np.percentile(boot_means, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def pearson_r(obs: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation ignoring NaN pairs."""
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 3:
        return np.nan
    r, _ = pearsonr(obs[mask], pred[mask])
    return float(r)


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 1:
        return np.nan
    return float(np.sqrt(np.mean((obs[mask] - pred[mask]) ** 2)))


def anomaly_correlation(obs: np.ndarray, pred: np.ndarray) -> float:
    """Correlation of anomalies relative to each series' mean."""
    obs_a  = obs  - np.nanmean(obs)
    pred_a = pred - np.nanmean(pred)
    return pearson_r(obs_a, pred_a)


def pairwise_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Station-by-station correlation matrix (ignores NaN pairs per-pair)."""
    return df.corr(method="pearson", min_periods=12)


def upper_triangle(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def pattern_correlation(obs_mat: pd.DataFrame, pred_mat: pd.DataFrame) -> float:
    """Pearson r between upper triangles of two station-correlation matrices."""
    # Align columns
    cols = obs_mat.columns.intersection(pred_mat.columns)
    a = obs_mat.loc[cols, cols].values
    b = pred_mat.loc[cols, cols].values
    return pearson_r(upper_triangle(a), upper_triangle(b))
