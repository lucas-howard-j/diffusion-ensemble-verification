"""Reconstruct spatial field from PC amplitudes."""
import numpy as np

from src.eof.compute_eofs import load_eofs


def reconstruct_field(pcs: np.ndarray, variable: str) -> np.ndarray:
    """
    Reconstruct anomaly field on full lat/lon grid from PC amplitudes.

    Parameters
    ----------
    pcs      : (n_samples, n_modes) PC amplitudes
    variable : "sst" or "ssh"

    Returns
    -------
    field : (n_samples, n_lat, n_lon) ndarray
            Land points are NaN; ocean points are in original (un-weighted) units.
    """
    eof_data = load_eofs(variable, need_patterns=True)
    eofs_modes = eof_data["eofs"]        # (n_modes, n_ocean_points)
    ocean_mask = eof_data["ocean_mask"]  # (n_lat, n_lon)
    lats       = eof_data["lat"]
    lons       = eof_data["lon"]
    weights    = eof_data["weights"]     # 1-D along lat

    n_samples = pcs.shape[0]
    n_modes_saved = eofs_modes.shape[0]
    n_modes_used  = min(pcs.shape[1], n_modes_saved)

    # Reconstruction: pcs @ eofs gives back weighted field
    weighted_flat = pcs[:, :n_modes_used] @ eofs_modes[:n_modes_used]   # (n_samples, n_ocean)

    # Put back into 2-D grid with NaN on land
    nlat, nlon = len(lats), len(lons)
    out = np.full((n_samples, nlat * nlon), np.nan)
    out[:, ocean_mask.ravel()] = weighted_flat
    out = out.reshape(n_samples, nlat, nlon)

    # Undo area weighting: divide by sqrt(cos(lat))
    w2d = weights[:, np.newaxis] * np.ones_like(out[0])
    # Avoid div-by-zero near the poles (won't happen in this domain)
    w2d = np.where(w2d > 0, w2d, 1.0)
    out = out / w2d[np.newaxis, :, :]

    return out
