"""Load and concatenate yearly GLORYS NetCDF files into a single xarray Dataset."""
from pathlib import Path

import xarray as xr

from src.config import GLORYS_DIR, GLORYS_NINO34_DIR


def load_glorys(variable: str = "sst", region: str = "east_coast") -> xr.DataArray:
    """
    Return a lazily-loaded DataArray for the requested variable.

    Parameters
    ----------
    variable : "sst" or "ssh"
    region   : "east_coast" or "nino34"

    Returns
    -------
    DataArray with dims (time, latitude, longitude), chunked by year.
    """
    if region == "east_coast":
        pattern = str(GLORYS_DIR / f"glorys_{variable}_*_east_coast.nc")
        varname = "thetao" if variable == "sst" else "zos"
    elif region == "nino34":
        pattern = str(GLORYS_NINO34_DIR / "glorys_nino34_*_sst.nc")
        varname = "thetao"
    else:
        raise ValueError(f"Unknown region: {region}")

    ds = xr.open_mfdataset(
        pattern,
        combine="by_coords",
        parallel=True,
        chunks={"time": 365},
    )

    da = ds[varname]

    # Drop depth dimension if present (SST is surface only)
    if "depth" in da.dims:
        da = da.isel(depth=0).drop_vars("depth")

    return da


def load_sst() -> xr.DataArray:
    return load_glorys("sst", "east_coast")


def load_ssh() -> xr.DataArray:
    return load_glorys("ssh", "east_coast")


def load_nino34_sst() -> xr.DataArray:
    return load_glorys("sst", "nino34")
