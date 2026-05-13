"""
Remove a linear trend (per grid point) computed over the training period.

The slope is estimated on training data only and then extrapolated to all
periods — this avoids data leakage while handling the full record cleanly.

Usage (typically called from anomalies.py, after seasonal cycle removal):
    from src.preprocessing.detrend import fit_trend, remove_trend
"""
from pathlib import Path

import numpy as np
import xarray as xr

from src.config import PROC, TRAIN_YEARS


def _time_coord(da: xr.DataArray) -> np.ndarray:
    """Return time in decimal weeks since the first time step."""
    t0 = da.time.values[0]
    dt = (da.time.values - t0).astype("timedelta64[D]").astype(float)
    return dt / 7.0  # weeks


def fit_trend(da: xr.DataArray) -> xr.Dataset:
    """
    Fit a per-grid-point linear trend over the training period.

    Returns an xr.Dataset with:
        slope     : (lat, lon) — units per week
        intercept : (lat, lon) — value at t=0 (first training sample)
        t0        : scalar — first time step of the full record (for alignment)
    """
    train = da.sel(time=da.time.dt.year.isin(list(TRAIN_YEARS)))

    t = _time_coord(train)           # (n_train,)
    t_mean = float(t.mean())
    t_anom = t - t_mean
    denom  = float((t_anom ** 2).sum())

    # Keep computation lazy: use xarray DataArray for t so Dask can chunk spatially
    t_da = xr.DataArray(t_anom, coords={"time": train.time}, dims=["time"])

    slope     = (t_da * train).sum("time") / denom
    intercept = train.mean("time") - slope * t_mean

    ds = xr.Dataset({"slope": slope, "intercept": intercept})
    ds.attrs["t0"] = str(da.time.values[0])
    return ds


def remove_trend(da: xr.DataArray, trend: xr.Dataset) -> xr.DataArray:
    """Subtract the fitted linear trend from all time steps."""
    t  = _time_coord(da)                                        # (n_time,)
    tf = xr.DataArray(t, coords={"time": da.time}, dims=["time"])
    fitted = trend["intercept"] + trend["slope"] * tf           # (n_time, lat, lon)
    return da - fitted


def save_trend(trend: xr.Dataset, variable: str) -> None:
    path = PROC / f"trend_{variable}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    trend.to_netcdf(str(path))


def load_trend(variable: str) -> xr.Dataset:
    return xr.open_dataset(str(PROC / f"trend_{variable}.nc"))
