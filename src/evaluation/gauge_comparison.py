"""Interpolate SSH fields to tide gauge locations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TIDE_GAUGES
from src.eof.compute_eofs import load_eofs


def _nearest_ocean(lat: float, lon: float,
                   lats: np.ndarray, lons: np.ndarray,
                   ocean_mask: np.ndarray) -> tuple[int, int]:
    """Nearest ocean grid point (simple search over small domain)."""
    # Precompute squared-distance matrix
    lat_diff = lats[:, None] - lat
    lon_diff = lons[None, :] - lon
    dist2 = lat_diff ** 2 + lon_diff ** 2
    dist2 = np.where(ocean_mask, dist2, np.inf)
    i, j = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(i), int(j)


def interp_ssh_to_gauges(ssh_fields: np.ndarray) -> pd.DataFrame:
    """
    Map an SSH time series onto each tide gauge location using the
    nearest valid ocean grid point. (Gauges typically sit in sheltered
    harbors where the adjacent ocean grid cell is the most defensible
    comparison point; bilinear interpolation would pull in land NaNs.)

    Parameters
    ----------
    ssh_fields : (n_time, n_lat, n_lon) ndarray  — reconstructed SSH anomaly

    Returns
    -------
    DataFrame with columns = station names, index = 0..n_time-1
    (caller attaches a time index).
    """
    eof_data = load_eofs("ssh")
    lats = eof_data["lat"]
    lons = eof_data["lon"]
    ocean_mask = eof_data["ocean_mask"]

    out = {}
    for g in TIDE_GAUGES:
        # Bilinear over (lat, lon). RegularGridInterpolator needs strictly increasing coords.
        # Note: for land points the grid is NaN; we fall back to nearest ocean.
        lat_g, lon_g = g["lat"], g["lon"]
        if not (lats.min() <= lat_g <= lats.max() and lons.min() <= lon_g <= lons.max()):
            out[g["name"]] = np.full(ssh_fields.shape[0], np.nan)
            continue

        i, j = _nearest_ocean(lat_g, lon_g, lats, lons, ocean_mask)
        out[g["name"]] = ssh_fields[:, i, j]

    return pd.DataFrame(out)


def build_gauge_series(ssh_fields: np.ndarray, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    df = interp_ssh_to_gauges(ssh_fields)
    df.index = time_index
    return df
