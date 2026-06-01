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
    members: np.ndarray, obs: np.ndarray, p: float = 0.5
) -> float:
    """Variogram Score (Scheuerer & Hamill 2015), equal weights.

    Parameters
    ----------
    members : (M, d)
    obs     : (d,)
    p       : power (default 0.5 per paper convention)

    Returns
    -------
    float — VS value; np.nan if fewer than 2 valid stations.
    """
    members = np.asarray(members, dtype=float)
    obs = np.asarray(obs, dtype=float)
    valid = np.isfinite(obs)
    if valid.sum() < 2:
        return float(np.nan)
    m, y = members[:, valid], obs[valid]
    d = y.shape[0]
    vs = 0.0
    n_pairs = 0
    for i in range(d):
        for j in range(i + 1, d):
            obs_diff = abs(float(y[i]) - float(y[j])) ** p
            ens_diffs = np.abs(m[:, i] - m[:, j]) ** p
            vs += (obs_diff - ens_diffs.mean()) ** 2
            n_pairs += 1
    return float(vs / n_pairs) if n_pairs > 0 else float(np.nan)
