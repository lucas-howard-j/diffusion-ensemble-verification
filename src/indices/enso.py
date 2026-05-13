"""
Compute weekly ENSO index (Niño 3.4) from GLORYS SST.

Niño 3.4 = area-averaged SST anomaly over 5S-5N, 170W-120W.

Run as a script:
    python -m src.indices.enso
"""
import numpy as np
import pandas as pd
import xarray as xr

from src.config import ENSO_WEEKLY, TRAIN_YEARS
from src.data.glorys_loader import load_nino34_sst
from src.preprocessing.weights import area_weights
from src.preprocessing.temporal import daily_to_weekly


def _training_climatology(da: xr.DataArray) -> xr.DataArray:
    train = da.sel(time=da.time.dt.year.isin(list(TRAIN_YEARS)))
    return train.groupby("time.dayofyear").mean("time")


def compute_enso(save: bool = True) -> pd.Series:
    """
    Compute weekly Niño 3.4 index and optionally save to data/processed/enso_nino34_weekly.csv.

    Returns
    -------
    pandas Series indexed by weekly timestamp, values = Niño3.4 anomaly (standardized).
    """
    print("Loading GLORYS Niño 3.4 SST ...")
    sst = load_nino34_sst().load()

    print("Computing anomalies ...")
    clim = _training_climatology(sst)
    anom = sst.groupby("time.dayofyear") - clim

    # Determine lat dim name
    lat_dim = [d for d in anom.dims if "lat" in d][0]
    lon_dim = [d for d in anom.dims if "lon" in d][0]

    lats = anom[lat_dim].values
    weights = area_weights(lats)  # cos-lat weights

    # Area-weighted mean over the Niño 3.4 box
    w_da = xr.DataArray(weights, coords={lat_dim: anom[lat_dim]}, dims=[lat_dim])
    nino34_daily = (anom * w_da).sum(dim=[lat_dim, lon_dim]) / w_da.sum()
    nino34_daily = nino34_daily.values

    ts = pd.Series(
        nino34_daily,
        index=pd.to_datetime(anom.time.values),
        name="enso_nino34",
    )

    print("Aggregating to weekly ...")
    enso_weekly = daily_to_weekly(ts)

    # Standardize over training period
    train_mask = enso_weekly.index.year.isin(list(TRAIN_YEARS))
    mu  = enso_weekly[train_mask].mean()
    std = enso_weekly[train_mask].std()
    enso_weekly = (enso_weekly - mu) / std

    if save:
        ENSO_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
        enso_weekly.to_csv(ENSO_WEEKLY, header=True)
        print(f"Saved → {ENSO_WEEKLY}")

    return enso_weekly


def load_enso() -> pd.Series:
    """Load pre-computed weekly ENSO index."""
    df = pd.read_csv(ENSO_WEEKLY, index_col=0, parse_dates=True)
    return df.iloc[:, 0].rename("enso_nino34")


if __name__ == "__main__":
    compute_enso(save=True)
