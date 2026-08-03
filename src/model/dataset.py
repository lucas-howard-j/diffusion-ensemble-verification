"""
PyTorch Dataset building (X_cond, Y) pairs for diffusion model training.

Condition vector layout:
    SST_PCs  at t+lag for each lag in SST_LAGS   → N_SST_MODES  × len(SST_LAGS)
    SSH_PCs  at t-lag for each lag in SSH_LAGS    → N_SSH_MODES  × len(SSH_LAGS)  (includes t=0)
    Wind_PCs at t                                 → N_WIND_MODES × 1
    NAO      at t+lag for each lag in NAO_LAGS    → 1            × len(NAO_LAGS)
    ENSO     at t-lag for each lag in ENSO_LAGS   → 1            × len(ENSO_LAGS)
    lead_norm (scalar in [0,1])                   → 1

Target:
    SSH anomaly at each tide gauge location (t + lead)  → len(TIDE_GAUGES)

Valid samples require t >= MAX_LAG and t + max(LEAD_WEEKS) < T.
Train/val/test splits are by the year of t, with a MAX_LAG-week buffer at
each boundary to prevent temporal leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset

from src.config import (
    LEAD_WEEKS,
    MAX_LAG,
    N_SSH_MODES, N_SST_MODES, N_WIND_MODES,
    SSH_LAGS, SST_LAGS, ENSO_LAGS, NAO_LAGS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    TIDE_GAUGES, SSH_ANOM_WEEKLY,
    normalise_lead,
)
from src.eof.compute_eofs import load_eofs
from src.eof.compute_wind_eofs import load_wind_eofs
from src.indices.nao import load_nao
from src.indices.enso import load_enso


SPLITS = {
    "train": list(TRAIN_YEARS),
    "val":   list(VAL_YEARS),
    "test":  list(TEST_YEARS),
}



def _align_to(series: pd.Series, time_index: pd.DatetimeIndex) -> np.ndarray:
    """Align a Series to time_index via nearest-neighbor, return float32 array."""
    return series.reindex(time_index, method="nearest").values.astype(np.float32)


def _interp_to_gauges(da: xr.DataArray) -> np.ndarray:
    """
    Nearest non-NaN ocean point interpolation to tide gauge locations.

    The nearest GLORYS grid cell is often on land. We find the closest valid
    ocean point using great-circle distance on the ocean mask.

    Returns (n_time, n_gauges) float32 array.
    """
    lat_dim = [d for d in da.dims if "lat" in d][0]
    lon_dim = [d for d in da.dims if "lon" in d][0]
    lats = da[lat_dim].values
    lons = da[lon_dim].values
    ocean_mask = np.isfinite(da.isel(time=0).values)   # (n_lat, n_lon)

    # Grid of lat/lon for all ocean points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    ocean_lats = lat_grid[ocean_mask]
    ocean_lons = lon_grid[ocean_mask]

    # Precompute flat ocean index → (ilat, ilon)
    ocean_idx = np.argwhere(ocean_mask)

    cols = []
    for g in TIDE_GAUGES:
        glat, glon = g["lat"], g["lon"]
        # Great-circle distance (approx) to all ocean points
        dlat = np.radians(ocean_lats - glat)
        dlon = np.radians(ocean_lons - glon)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(glat)) * np.cos(np.radians(ocean_lats)) * np.sin(dlon / 2)**2
        dist = 2 * np.arcsin(np.sqrt(a))
        nearest = np.argmin(dist)
        ilat, ilon = ocean_idx[nearest]
        col = da.values[:, ilat, ilon]
        cols.append(col)
    return np.stack(cols, axis=-1).astype(np.float32)   # (n_time, n_gauges)


class SSHDiffusionDataset(Dataset):
    """
    Dataset of (X_cond, Y) pairs for every valid (t, lead) combination.

    Parameters
    ----------
    split : "train", "val", or "test"
    """

    def __init__(self, split: str = "train"):
        self.split  = split
        self.leads  = list(LEAD_WEEKS)
        self.max_lead = max(self.leads)

        # ── Load EOF PCs (conditioning) ───────────────────────────────────────
        sst_eof  = load_eofs("sst")
        ssh_eof  = load_eofs("ssh")
        wind_eof = load_wind_eofs()

        if sst_eof["pcs"].shape[1] < N_SST_MODES:
            raise ValueError(f"SST EOF has {sst_eof['pcs'].shape[1]} modes but N_SST_MODES={N_SST_MODES}")
        if ssh_eof["pcs"].shape[1] < N_SSH_MODES:
            raise ValueError(f"SSH EOF has {ssh_eof['pcs'].shape[1]} modes but N_SSH_MODES={N_SSH_MODES}")
        if wind_eof["pcs"].shape[1] < N_WIND_MODES:
            raise ValueError(f"Wind EOF has {wind_eof['pcs'].shape[1]} modes but N_WIND_MODES={N_WIND_MODES}")

        sst_pcs  = sst_eof["pcs"][:, :N_SST_MODES].astype(np.float32)
        ssh_pcs  = ssh_eof["pcs"][:, :N_SSH_MODES].astype(np.float32)
        wind_pcs = wind_eof["pcs"][:, :N_WIND_MODES].astype(np.float32)

        time_axis = pd.to_datetime(ssh_eof["time"])

        if not np.array_equal(ssh_eof["time"], sst_eof["time"]):
            raise ValueError("SST and SSH EOF time axes do not match.")

        # ── Load SSH anomaly field and interpolate to gauge locations ─────────
        print("Interpolating SSH anomaly field to gauge locations ...")
        da_ssh = xr.open_dataarray(str(SSH_ANOM_WEEKLY))
        gauge_ssh = _interp_to_gauges(da_ssh)   # (n_time, n_gauges)
        da_ssh.close()

        # ── Standardise on training period ────────────────────────────────────
        train_mask = time_axis.year.isin(list(TRAIN_YEARS))

        def _std(arr):
            mu  = arr[train_mask].mean(axis=0)
            std = arr[train_mask].std(axis=0) + 1e-8
            return (arr - mu) / std, mu, std

        sst_pcs_z,  self.sst_mu,      self.sst_std      = _std(sst_pcs)
        ssh_pcs_z,  self.ssh_mu,      self.ssh_std      = _std(ssh_pcs)
        wind_pcs_z, self.wind_mu,     self.wind_std     = _std(wind_pcs)
        gauge_z,    self.gauge_mu,    self.gauge_std     = _std(gauge_ssh)

        # ── Align climate indices ─────────────────────────────────────────────
        nao_arr  = _align_to(load_nao(),  time_axis)
        enso_arr = _align_to(load_enso(), time_axis)

        # ── Align wind PCs to SSH time axis ───────────────────────────────────
        wind_time = pd.to_datetime(wind_eof["time"])
        if not wind_time.equals(time_axis):
            wind_df = pd.DataFrame(wind_pcs_z, index=wind_time)
            wind_pcs_z = wind_df.reindex(time_axis, method="nearest").values.astype(np.float32)
            if np.isnan(wind_pcs_z).any():
                raise ValueError(
                    "Wind PCs contain NaN after reindex — wind time coverage does not span SSH time axis."
                )

        # Store on self for __getitem__
        self.sst_pcs   = sst_pcs_z
        self.ssh_pcs   = ssh_pcs_z
        self.wind_pcs  = wind_pcs_z
        self.gauge_ssh = gauge_z       # (n_time, n_gauges) — target
        self.nao       = nao_arr
        self.enso      = enso_arr
        self.time      = time_axis

        # ── Build valid (t, lead) index ───────────────────────────────────────
        split_years = SPLITS[split]
        T = len(time_axis)
        valid_pairs = []
        for t in range(MAX_LAG, T - self.max_lead):
            year_t = time_axis[t].year
            if year_t not in split_years:
                continue
            min_idx = t - max(SSH_LAGS + SST_LAGS + ENSO_LAGS + NAO_LAGS)
            if time_axis[min_idx].year not in split_years:
                continue
            for lead in self.leads:
                if t + lead < T and time_axis[t + lead].year in split_years:
                    valid_pairs.append((t, lead))

        self.pairs = valid_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t, lead = self.pairs[idx]
        lead_norm = normalise_lead(lead, self.leads)

        parts = []

        # SST PCs at t, t-1, t-2
        for lag in SST_LAGS:
            parts.append(self.sst_pcs[t - lag])

        # SSH PCs at t, t-1, t-2, t-4  (init state + lagged memory; no leakage since lead >= 2wk)
        for lag in SSH_LAGS:
            parts.append(self.ssh_pcs[t - lag])

        # Wind PCs at current time t
        parts.append(self.wind_pcs[t])

        # NAO at t
        for lag in NAO_LAGS:
            parts.append(np.array([self.nao[t - lag]], dtype=np.float32))

        # ENSO at t, t-4, t-8, t-13
        for lag in ENSO_LAGS:
            parts.append(np.array([self.enso[t - lag]], dtype=np.float32))

        # Gauge SSH at t (initialization-time local state; lead >= 2wk, no leakage)
        parts.append(self.gauge_ssh[t].astype(np.float32))

        # Lead time (scalar)
        parts.append(np.array([lead_norm], dtype=np.float32))

        cond   = np.concatenate(parts).astype(np.float32)
        target = self.gauge_ssh[t + lead].astype(np.float32)   # (n_gauges,)

        return torch.from_numpy(cond), torch.from_numpy(target)

    @property
    def cond_dim(self) -> int:
        return (N_SST_MODES * len(SST_LAGS)
                + N_SSH_MODES * len(SSH_LAGS)
                + N_WIND_MODES
                + len(NAO_LAGS)
                + len(ENSO_LAGS)
                + len(TIDE_GAUGES)   # gauge SSH at t
                + 1)   # lead

    @property
    def target_dim(self) -> int:
        return len(TIDE_GAUGES)

    def denormalize_gauge(self, gauge_z: np.ndarray) -> np.ndarray:
        """Invert the z-score on gauge SSH anomalies."""
        return gauge_z * self.gauge_std + self.gauge_mu
