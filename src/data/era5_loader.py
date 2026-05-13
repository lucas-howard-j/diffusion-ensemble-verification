"""Load ERA5 wind (u10/v10) fields over the East Coast domain."""
import xarray as xr

from src.config import ERA5_DIR


def load_era5_wind() -> xr.Dataset:
    """
    Return a lazily-loaded Dataset with u10 and v10 for the East Coast domain.

    The CDS download stores both components in one file per year.
    Variable names in ERA5 netCDF: 'u10' and 'v10'.
    """
    pattern = str(ERA5_DIR / "era5_wind_east_coast_*.nc")
    ds = xr.open_mfdataset(pattern, combine="by_coords", parallel=False)
    # Normalise time dim name (newer CDS uses valid_time)
    time_dim = [d for d in ds.dims if "time" in d][0]
    if time_dim != "time":
        ds = ds.rename({time_dim: "time"})
    return ds
