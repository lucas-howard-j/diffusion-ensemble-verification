"""
Compute weekly NAO index from ERA5 mean sea level pressure via EOF.

The NAO is defined as PC1 of the leading EOF of North Atlantic MSLP anomalies
(20-80N, 90W-40E), following the EOF-based convention.

Run as a script:
    python -m src.indices.nao
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.utils.extmath import randomized_svd

from src.config import ERA5_DIR, NAO_WEEKLY, PROC, TRAIN_YEARS

NAO_EOF1 = PROC / "nao_eof1.npz"
from src.preprocessing.weights import area_weights
from src.preprocessing.temporal import daily_to_weekly


def _load_era5_mslp() -> xr.DataArray:
    pattern = str(ERA5_DIR / "era5_mslp_*.nc")
    # No explicit chunks — avoid misalignment overhead; xarray streams file-by-file
    ds = xr.open_mfdataset(pattern, combine="by_coords", parallel=False)
    varname = [v for v in ds.data_vars if "msl" in v.lower()][0]
    da = ds[varname]
    # Normalise time dim name (newer CDS uses valid_time)
    time_dim = [d for d in da.dims if "time" in d][0]
    if time_dim != "time":
        da = da.rename({time_dim: "time"})
    return da


def _training_climatology(da: xr.DataArray) -> xr.DataArray:
    train = da.sel(time=da.time.dt.year.isin(list(TRAIN_YEARS)))
    return train.groupby("time.dayofyear").mean("time")


def compute_nao(save: bool = True) -> pd.Series:
    """
    Compute weekly NAO index and optionally save to data/processed/nao_weekly.csv.

    Returns
    -------
    pandas Series indexed by weekly timestamp, values = NAO PC1 (standardized).
    """
    print("Loading ERA5 MSLP ...")
    mslp = _load_era5_mslp()

    print("Computing anomalies ...")
    clim = _training_climatology(mslp)
    anom = mslp.groupby("time.dayofyear") - clim

    # Determine lat/lon dim names
    lat_dim = [d for d in anom.dims if "lat" in d][0]
    lon_dim = [d for d in anom.dims if "lon" in d][0]

    lats = anom[lat_dim].values
    weights = area_weights(lats)  # 1-D array along lat

    # Area-weight: broadcast over (time, lat, lon)
    weighted = anom * weights[np.newaxis, :, np.newaxis]

    # Flatten to (time, space); replace NaN with 0 (MSLP has no ocean mask)
    nt = weighted.shape[0]
    data2d = np.nan_to_num(weighted.values.reshape(nt, -1), nan=0.0)

    # SVD fit on training period only
    time_vals = anom.time.values
    years = pd.to_datetime(time_vals).year
    train_mask = np.isin(years, list(TRAIN_YEARS))
    train2d = data2d[train_mask]

    print("Running truncated SVD (k=1) ...")
    _, _, Vt = randomized_svd(train2d, n_components=1, random_state=42)
    eof1_flat = Vt[0]  # (n_space,)

    # Enforce sign: positive NAO = positive loading over Azores (~40°N, 25°W)
    nlat = len(anom[lat_dim])
    nlon = len(anom[lon_dim])
    eof1 = eof1_flat.reshape(nlat, nlon)
    azores_lat_idx = int(np.argmin(np.abs(lats - 40.0)))
    azores_lon_idx = int(np.argmin(np.abs(anom[lon_dim].values - (-25.0))))
    if eof1[azores_lat_idx, azores_lon_idx] < 0:
        eof1_flat = -eof1_flat

    # Project all time steps onto EOF1
    pcs = data2d @ eof1_flat  # (n_time,)

    nao_daily = pd.Series(pcs, index=pd.to_datetime(anom.time.values), name="nao")

    print("Aggregating to weekly ...")
    nao_weekly = daily_to_weekly(nao_daily)

    # Standardize over training period
    train_mask = nao_weekly.index.year.isin(list(TRAIN_YEARS))
    mu  = nao_weekly[train_mask].mean()
    std = nao_weekly[train_mask].std()
    nao_weekly = (nao_weekly - mu) / std

    if save:
        NAO_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
        nao_weekly.to_csv(NAO_WEEKLY, header=True)
        print(f"Saved → {NAO_WEEKLY}")
        np.savez(str(NAO_EOF1),
                 eof1=eof1_flat.reshape(nlat, nlon),
                 lat=lats,
                 lon=anom[lon_dim].values)
        print(f"Saved → {NAO_EOF1}")

    return nao_weekly


def load_nao_eof1() -> dict:
    """Load cached NAO EOF1 spatial pattern."""
    data = np.load(str(NAO_EOF1))
    return {k: data[k] for k in data.files}


def load_nao() -> pd.Series:
    """Load pre-computed weekly NAO index."""
    df = pd.read_csv(NAO_WEEKLY, index_col=0, parse_dates=True)
    return df.iloc[:, 0].rename("nao")


if __name__ == "__main__":
    compute_nao(save=True)
