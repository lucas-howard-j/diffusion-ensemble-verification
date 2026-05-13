"""
Compute EOF decomposition of weekly SST or SSH anomalies.

Run as a script:
    python -m src.eof.compute_eofs --variable sst
    python -m src.eof.compute_eofs --variable ssh
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
from eofs.standard import Eof

from src.config import (
    EOF_SST, EOF_SSH,
    SST_ANOM_WEEKLY, SSH_ANOM_WEEKLY,
    N_EOFS_MAX, VARIANCE_THRESHOLD,
    TRAIN_YEARS,
)
from src.preprocessing.weights import area_weights


def _load_weekly_anom(variable: str) -> xr.DataArray:
    path = SST_ANOM_WEEKLY if variable == "sst" else SSH_ANOM_WEEKLY
    return xr.open_dataarray(str(path))


def compute_eofs(variable: str, save: bool = True, n_eofs_max: int | None = None,
                 n_modes_override: int | None = None) -> dict:
    """
    Run EOF analysis on weekly anomaly data.

    Returns dict with keys:
        eofs       : (n_modes, n_ocean_points) ndarray
        pcs        : (n_time, n_modes) ndarray
        variance_fraction : (n_modes,) ndarray
        n_modes    : int (modes retained at VARIANCE_THRESHOLD)
        lat        : 1-D ndarray
        lon        : 1-D ndarray
        ocean_mask : 2-D boolean array (True = valid ocean point)
        weights    : 1-D sqrt(cos-lat) array
        time       : 1-D array of weekly timestamps
    """
    out_path = EOF_SST if variable == "sst" else EOF_SSH

    n_eofs_max = n_eofs_max or N_EOFS_MAX

    print(f"Loading weekly {variable.upper()} anomalies ...")
    da = _load_weekly_anom(variable)

    # Identify lat/lon dim names
    lat_dim = [d for d in da.dims if "lat" in d][0]
    lon_dim = [d for d in da.dims if "lon" in d][0]

    lats = da[lat_dim].values
    lons = da[lon_dim].values

    # Ocean mask: grid points that are non-NaN for at least one time step
    ocean_mask = np.isfinite(da.isel(time=0).values)

    # Area weights (1-D along lat)
    w1d = area_weights(lats)

    # Apply area weighting (broadcast along lon)
    weighted = da.values * w1d[np.newaxis, :, np.newaxis]   # (time, lat, lon)

    # Flatten and mask to ocean points only (avoids allocating full land+ocean array)
    nt = weighted.shape[0]
    flat_ocean = weighted.reshape(nt, -1)[:, ocean_mask.ravel()]

    # Replace any remaining NaN with 0 (shouldn't happen after masking)
    flat_ocean = np.nan_to_num(flat_ocean, nan=0.0)

    # Use only training period for EOF computation
    time_vals = da.time.values
    years = pd.to_datetime(time_vals).year
    train_mask = np.isin(years, list(TRAIN_YEARS))
    flat_train = flat_ocean[train_mask]

    print(f"Running EOF on {flat_train.shape[0]} training samples, {flat_train.shape[1]} ocean points ...")
    solver = Eof(flat_train, center=True)
    eofs_modes     = solver.eofs(neofs=n_eofs_max)
    var_frac       = solver.varianceFraction(n_eofs_max)
    cumvar         = np.cumsum(var_frac)

    # Number of modes to retain
    if n_modes_override is not None:
        n_modes = min(n_modes_override, n_eofs_max)
    else:
        n_modes = int(np.searchsorted(cumvar, VARIANCE_THRESHOLD)) + 1
        n_modes = min(n_modes, n_eofs_max)
    print(f"  Retaining {n_modes} modes ({100*cumvar[n_modes-1]:.1f}% variance explained)")

    # Project ALL time steps onto training EOFs
    pcs_all = flat_ocean @ eofs_modes[:n_modes].T   # (n_time, n_modes)

    result = dict(
        eofs=eofs_modes[:n_modes],
        pcs=pcs_all,
        variance_fraction=var_frac[:n_modes],
        n_modes=n_modes,
        lat=lats,
        lon=lons,
        ocean_mask=ocean_mask,
        weights=w1d,
        time=time_vals,
    )

    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_path), **result)
        print(f"Saved → {out_path}")

    return result


def load_eofs(variable: str) -> dict:
    """Load pre-computed EOF decomposition."""
    path = EOF_SST if variable == "sst" else EOF_SSH
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", choices=["sst", "ssh"], required=True)
    parser.add_argument("--n_eofs", type=int, default=None)
    parser.add_argument("--n_modes", type=int, default=None,
                        help="Force exactly this many modes, bypassing variance threshold")
    args = parser.parse_args()
    compute_eofs(args.variable, n_eofs_max=args.n_eofs, n_modes_override=args.n_modes)
