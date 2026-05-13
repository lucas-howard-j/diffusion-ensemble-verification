"""Load and align NOAA CO-OPS tide gauge monthly mean sea level data."""
import pandas as pd

from src.config import TIDE_GAUGE_DIR, TIDE_GAUGES, TRAIN_YEARS


def _monthly_climatology(series: pd.Series) -> pd.Series:
    """Monthly climatology computed over the training period only."""
    train = series[series.index.year.isin(list(TRAIN_YEARS))]
    return train.groupby(train.index.month).mean()


def load_gauges(time_index: pd.DatetimeIndex | None = None,
                remove_seasonal: bool = True) -> pd.DataFrame:
    """
    Load all station CSVs and return a wide DataFrame indexed by month.

    Processing:
        1. Remove long-term mean per station
        2. Optionally remove monthly seasonal cycle (training-period climatology)
           so that anomalies are comparable to detrended model SSH anomalies.

    Parameters
    ----------
    time_index      : optional DatetimeIndex to reindex output
    remove_seasonal : if True (default), subtract the monthly climatology

    Returns
    -------
    DataFrame with columns = station names, monthly DatetimeIndex.
    """
    frames = {}
    for gauge in TIDE_GAUGES:
        csv = TIDE_GAUGE_DIR / f"{gauge['id']}_{gauge['name']}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, parse_dates=["date"], index_col="date")
        series = df["msl_m"].rename(gauge["name"])

        # Remove long-term mean
        series = series - series.mean()

        if remove_seasonal:
            clim = _monthly_climatology(series)
            series = series - series.index.month.map(clim)

        frames[gauge["name"]] = series

    out = pd.DataFrame(frames)

    if time_index is not None:
        out = out.reindex(time_index)

    return out


def gauge_locations() -> pd.DataFrame:
    """Return a DataFrame with columns [lat, lon] indexed by station name."""
    return pd.DataFrame(TIDE_GAUGES).set_index("name")[["lat", "lon"]]
