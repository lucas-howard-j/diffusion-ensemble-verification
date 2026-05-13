import numpy as np, xarray as xr
from src.config import TIDE_GAUGES, SSH_ANOM_WEEKLY
from src.model.dataset import _interp_to_gauges
da = xr.open_dataarray(str(SSH_ANOM_WEEKLY))
gauge_ssh = _interp_to_gauges(da)
for i, g in enumerate(TIDE_GAUGES):
    col = gauge_ssh[:, i]
    nan_frac = np.isnan(col).mean()
    print(f"{g['name']:25s}  nan={nan_frac:.2f}  mean={np.nanmean(col):.4f}")
