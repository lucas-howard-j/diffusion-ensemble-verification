"""
Compute EOF decomposition of weekly ERA5 10m wind anomalies.

u10 and v10 are treated jointly by stacking them into a single EOF analysis
(concatenated along the spatial axis). This preserves co-variability between
the two components and keeps the number of PC time series manageable.

Run as a script:
    python -m src.eof.compute_wind_eofs

Saves:
    data/processed/eof_wind.npz
    data/processed/wind_anom_weekly.nc
"""
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from sklearn.utils.extmath import randomized_svd

from src.config import PROC, N_EOFS_MAX, TRAIN_YEARS, WIND_VARIANCE_THRESHOLD
from src.data.era5_loader import load_era5_wind
from src.preprocessing.temporal import daily_to_weekly
from src.preprocessing.weights import area_weights
from src.preprocessing.detrend import fit_trend, remove_trend, save_trend


EOF_WIND = PROC / "eof_wind.npz"
WIND_ANOM_WEEKLY = PROC / "wind_anom_weekly.nc"

# Retain this many wind modes; wind has broader spatial scales than SSH,
# so fewer modes needed to capture the relevant variance.
N_WIND_MODES_MAX = 20


def _climatology(da: xr.DataArray) -> xr.DataArray:
    train = da.sel(time=da.time.dt.year.isin(list(TRAIN_YEARS)))
    return train.groupby("time.dayofyear").mean("time")


def compute_wind_eofs(save: bool = True) -> dict:
    """
    Compute joint EOF of detrended weekly u10/v10 anomalies.

    Returns dict with same structure as compute_eofs():
        eofs, pcs, variance_fraction, n_modes, lat, lon, time, weights
    Plus:
        n_ocean  : n_u + n_v spatial points (concatenated)
    """
    print("Loading ERA5 wind ...")
    ds = load_era5_wind()
    u10 = ds["u10"]
    v10 = ds["v10"]

    lat_dim = [d for d in u10.dims if "lat" in d][0]
    lon_dim = [d for d in u10.dims if "lon" in d][0]
    lats = u10[lat_dim].values
    lons = u10[lon_dim].values

    print("Removing seasonal cycle and aggregating to weekly ...")
    u_anom = daily_to_weekly(u10.groupby("time.dayofyear") - _climatology(u10))
    v_anom = daily_to_weekly(v10.groupby("time.dayofyear") - _climatology(v10))

    print("Removing linear trend ...")
    u_trend = fit_trend(u_anom)
    v_trend = fit_trend(v_anom)
    u_dt = remove_trend(u_anom, u_trend)
    v_dt = remove_trend(v_anom, v_trend)

    if save:
        save_trend(u_trend, "u10")
        save_trend(v_trend, "v10")
        ds_out = xr.Dataset({"u10_anom": u_dt, "v10_anom": v_dt})
        WIND_ANOM_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 4, "chunksizes": (52, 90, 96)}
               for v in ds_out.data_vars}
        with ProgressBar():
            ds_out.to_netcdf(str(WIND_ANOM_WEEKLY), encoding=enc, compute=True)

    w1d = area_weights(lats)

    # Apply area weighting, flatten, and concatenate u+v for joint EOF
    u_weighted = u_dt.values * w1d[np.newaxis, :, np.newaxis]
    v_weighted = v_dt.values * w1d[np.newaxis, :, np.newaxis]
    nt = u_weighted.shape[0]
    nu = u_weighted.reshape(nt, -1)
    nv = v_weighted.reshape(nt, -1)
    data2d = np.nan_to_num(np.concatenate([nu, nv], axis=1), nan=0.0)

    # Centre and fit on training period only
    time_vals = u_dt.time.values
    years = pd.to_datetime(time_vals).year
    train_mask = np.isin(years, list(TRAIN_YEARS))
    data_train = data2d[train_mask]
    col_mean = data_train.mean(axis=0)
    data_train_c = data_train - col_mean
    data2d_c = data2d - col_mean

    print(f"Running joint u10/v10 truncated SVD (k={N_WIND_MODES_MAX}) on {data_train_c.shape[0]} training samples ...")
    _, s, Vt = randomized_svd(data_train_c, n_components=N_WIND_MODES_MAX, random_state=42)

    eofs_modes = Vt  # (N_WIND_MODES_MAX, n_space)
    total_var = np.sum(data_train_c ** 2)
    var_frac = (s ** 2) / total_var
    cumvar = np.cumsum(var_frac)

    n_modes = int(np.searchsorted(cumvar, WIND_VARIANCE_THRESHOLD)) + 1
    n_modes = min(n_modes, N_WIND_MODES_MAX)
    print(f"  Retaining {n_modes} wind modes ({100*cumvar[n_modes-1]:.1f}% variance)")

    pcs_all = data2d_c @ eofs_modes[:n_modes].T

    result = dict(
        eofs=eofs_modes[:n_modes],
        pcs=pcs_all,
        variance_fraction=var_frac[:n_modes],
        n_modes=n_modes,
        lat=lats,
        lon=lons,
        time=time_vals,
        weights=w1d,
        n_u=nu.shape[1],  # spatial length of u portion; v portion starts here
    )

    if save:
        np.savez(str(EOF_WIND), **result)
        print(f"Saved → {EOF_WIND}")

    return result


def load_wind_eofs() -> dict:
    data = np.load(str(EOF_WIND), allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    compute_wind_eofs(save=True)
