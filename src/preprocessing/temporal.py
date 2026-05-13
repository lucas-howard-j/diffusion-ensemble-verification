"""Temporal aggregation utilities."""
import pandas as pd
import xarray as xr


def daily_to_weekly(data: pd.Series | pd.DataFrame | xr.DataArray) -> pd.Series | pd.DataFrame | xr.DataArray:
    """
    Aggregate daily data to ISO-week means (Monday-anchored).

    Works on pandas Series/DataFrame and xarray DataArray.
    """
    if isinstance(data, xr.DataArray):
        # resample along the time dimension
        return data.resample(time="1W-MON").mean()
    return data.resample("1W-MON").mean()


def weekly_time_index(start: str, end: str) -> pd.DatetimeIndex:
    """Return a Monday-anchored weekly DatetimeIndex between start and end."""
    return pd.date_range(start=start, end=end, freq="1W-MON")
