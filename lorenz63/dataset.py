"""
L63 dataset: extracts (cond, target) pairs from a trajectory file.

cond   = state at time t           (3,)
target = state at time t+lead_steps (3,)

States are standardized to zero mean, unit variance using statistics
computed from the training portion only.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from lorenz63.config import LEAD_STEPS


class L63Dataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,       # (N, 3) raw trajectory slice
        lead_steps: int = LEAD_STEPS,
        mu: np.ndarray | None = None,   # if provided, use these for normalization
        sigma: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        states     : (N, 3) trajectory — pass the training slice here
        lead_steps : prediction horizon in integration steps
        mu, sigma  : normalization stats; computed from states if not given
        """
        if mu is None:
            mu    = states.mean(axis=0)
            sigma = states.std(axis=0)
        self.mu    = mu
        self.sigma = sigma

        normed = (states - mu) / sigma
        n_pairs = len(normed) - lead_steps

        self.cond   = torch.from_numpy(normed[:n_pairs]).float()
        self.target = torch.from_numpy(normed[lead_steps:]).float()

    def __len__(self):
        return len(self.cond)

    def __getitem__(self, idx):
        return self.cond[idx], self.target[idx]


def load_trajectory(path: str) -> tuple[np.ndarray, dict]:
    """Load trajectory npz, return (states, config_dict)."""
    d = np.load(path)
    cfg = json.loads(str(d["config"]))
    return d["states"], cfg
