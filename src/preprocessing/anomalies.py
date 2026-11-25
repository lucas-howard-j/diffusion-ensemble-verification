"""
Compute seasonal anomalies from GLORYS daily data, remove linear trend,
and aggregate to weekly means.

Pipeline per variable:
    1. Subtract daily seasonal climatology (training period only)
    2. Aggregate daily → weekly
    3. Fit linear trend over training weeks (per grid point)
    4. Subtract trend from all weeks
    5. Save detrended weekly anomalies + trend coefficients

Run as a script:
    python -m src.preprocessing.anomalies
"""
import xarray as xr
from dask.diagnostics import ProgressBar

from src.config import SST_ANOM_WEEKLY, SSH_ANOM_WEEKLY, TRAIN_YEARS
from src.data.glorys_loader import load_sst, load_ssh
from src.preprocessing.temporal import daily_to_weekly
from src.preprocessing.detrend import fit_trend, remove_trend, save_trend


def _climatology(da: xr.DataArray) -> xr.DataArray:
    """Daily climatology over the training period only."""
    train = da.sel(time=da.time.dt.year.isin(list(TRAIN_YEARS)))
    return train.groupby("time.dayofyear").mean("time")


def compute_anomalies(da: xr.DataArray) -> xr.DataArray:
    """Subtract daily seasonal climatology (training period)."""
    return da.groupby("time.dayofyear") - _climatology(da)


def process_and_save(variable: str = "sst") -> xr.DataArray:
    """
    Full pipeline: load GLORYS → seasonal anomaly → weekly → detrend → save.

    Saves:
        data/processed/{variable}_anom_weekly.nc  — detrended weekly anomalies
        data/processed/trend_{variable}.nc        — trend coefficients (slope, intercept)
    """
    out_path = SST_ANOM_WEEKLY if variable == "sst" else SSH_ANOM_WEEKLY

    print(f"Loading GLORYS {variable.upper()} ...")
    da = load_sst() if variable == "sst" else load_ssh()

    print("Removing seasonal cycle ...")
    anom = compute_anomalies(da)

    print("Aggregating daily → weekly ...")
    weekly = daily_to_weekly(anom)
    weekly.name = variable + "_anom"

    print("Fitting and removing linear trend (training period only) ...")
    trend = fit_trend(weekly)
    weekly_dt = remove_trend(weekly, trend)
    weekly_dt.name = variable + "_anom"

    print(f"Saving anomalies → {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lat_dim = [d for d in weekly_dt.dims if "lat" in d][0]
    lon_dim = [d for d in weekly_dt.dims if "lon" in d][0]
    enc = {weekly_dt.name: {"zlib": True, "complevel": 4,
                             "chunksizes": (52, 90, 96)}}
    with ProgressBar():
        weekly_dt.to_netcdf(str(out_path), encoding=enc, compute=True)

    print(f"Saving trend coefficients → trend_{variable}.nc ...")
    save_trend(trend, variable)

    if variable == "ssh":
        # The dataset caches the gauge-interpolated SSH; drop it so the next
        # load rebuilds it from the new anomalies instead of using stale targets.
        gauge_cache = out_path.parent / "gauge_ssh_interp.npy"
        if gauge_cache.exists():
            gauge_cache.unlink()
            print(f"Removed stale {gauge_cache.name} (rebuilt on next dataset load)")

    print("Done.")
    return weekly_dt


if __name__ == "__main__":
    process_and_save("sst")
    process_and_save("ssh")
