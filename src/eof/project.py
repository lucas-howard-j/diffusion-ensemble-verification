"""Project new data onto pre-computed EOF modes."""
import numpy as np
import xarray as xr

from src.eof.compute_eofs import load_eofs
from src.preprocessing.weights import area_weights


def project_onto_eofs(da: xr.DataArray, variable: str) -> np.ndarray:
    """
    Project a DataArray of anomalies onto saved EOF modes.

    Parameters
    ----------
    da       : DataArray with dims (time, lat, lon); anomalies already removed.
    variable : "sst" or "ssh"

    Returns
    -------
    pcs : ndarray of shape (n_time, n_modes)
    """
    eof_data = load_eofs(variable, need_patterns=True)
    lats      = eof_data["lat"]
    ocean_idx = eof_data["ocean_mask"].ravel()
    eofs_modes = eof_data["eofs"]      # (n_modes, n_ocean_points)

    w1d = area_weights(lats)

    weighted = da.values * w1d[np.newaxis, :, np.newaxis]
    nt = weighted.shape[0]
    flat = weighted.reshape(nt, -1)[:, ocean_idx]
    flat = np.nan_to_num(flat, nan=0.0)

    return flat @ eofs_modes.T   # (n_time, n_modes)
