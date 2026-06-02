"""Proper-score functions for probabilistic ensemble verification.

Public API
----------
crps(members, obs)                      → np.ndarray  shape (d,)
energy_score(members, obs)              → float
variogram_score(members, obs, p=0.5)    → float

All functions accept members: (M, d) and obs: (d,).
Stations where obs is NaN are handled gracefully:
  - crps: returns NaN at those stations, finite elsewhere
  - energy_score / variogram_score: drops NaN stations before scoring;
    returns NaN when fewer than 2 valid stations remain.
"""
from __future__ import annotations

import numpy as np


def crps(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score per station.

    Parameters
    ----------
    members : (M, d)  ensemble members
    obs     : (d,)    observations

    Returns
    -------
    (d,) CRPS at each station; NaN where obs is NaN.
    """
    members = np.asarray(members, dtype=float)
    obs = np.asarray(obs, dtype=float)
    M = members.shape[0]
    term1 = np.mean(np.abs(members - obs[None, :]), axis=0)
    diff = np.abs(members[:, None, :] - members[None, :, :])   # (M, M, d)
    off = ~np.eye(M, dtype=bool)
    term2 = diff[off].reshape(M * (M - 1), -1).mean(axis=0) / 2
    result = term1 - term2
    result[~np.isfinite(obs)] = np.nan
    return result


def energy_score(members: np.ndarray, obs: np.ndarray) -> float:
    """Multivariate Energy Score, dropping NaN stations.

    Parameters
    ----------
    members : (M, d)
    obs     : (d,)

    Returns
    -------
    float — ES value; np.nan if fewer than 2 valid stations.
    """
    members = np.asarray(members, dtype=float)
    obs = np.asarray(obs, dtype=float)
    valid = np.isfinite(obs)
    if valid.sum() < 2:
        return float(np.nan)
    m, y = members[:, valid], obs[valid]
    M = m.shape[0]
    term1 = np.linalg.norm(m - y[None, :], axis=1).mean()
    diffs = m[:, None, :] - m[None, :, :]          # (M, M, d_valid)
    pair_norms = np.linalg.norm(diffs, axis=2)      # (M, M)
    off = ~np.eye(M, dtype=bool)
    term2 = pair_norms[off].mean() / 2
    return float(term1 - term2)


def variogram_score(
    members: np.ndarray, obs: np.ndarray, p: float = 0.5,
    reduce: bool = True,
) -> "float | np.ndarray":
    """Variogram Score (Scheuerer & Hamill 2015), equal weights.

    Parameters
    ----------
    members : (M, d)
    obs     : (d,)
    p       : power (default 0.5 per paper convention)
    reduce  : if True (default) return a scalar mean over all valid pairs;
              if False return shape (d*(d-1)//2,) with NaN for pairs where
              either obs value is NaN — preserving original pair ordering so
              index k corresponds to np.triu_indices(d, k=1) pair k.

    Returns
    -------
    float (reduce=True)  — VS value; np.nan if fewer than 1 valid pair.
    np.ndarray (reduce=False) — per-pair squared terms, shape (d*(d-1)//2,).
    """
    members = np.asarray(members, dtype=float)
    obs     = np.asarray(obs,     dtype=float)
    d       = obs.shape[0]
    n_pairs = d * (d - 1) // 2
    out     = np.full(n_pairs, np.nan)
    k = 0
    for i in range(d):
        for j in range(i + 1, d):
            if np.isfinite(obs[i]) and np.isfinite(obs[j]):
                obs_diff  = abs(float(obs[i]) - float(obs[j])) ** p
                ens_diffs = np.abs(members[:, i] - members[:, j]) ** p
                out[k]    = (obs_diff - ens_diffs.mean()) ** 2
            k += 1
    if not reduce:
        return out
    valid = out[np.isfinite(out)]
    return float(valid.mean()) if len(valid) >= 1 else float(np.nan)
