"""
L96 dataset: extracts (cond, target) pairs from a trajectory file.

cond   = state at time t            (N,) = (40,)
target = state at time t+lead_steps (N,) = (40,)

States are standardized to zero mean, unit variance using per-node statistics
computed from the training portion only.  For L96 all nodes are statistically
identical under ergodicity, so this is mostly a convenience normalization.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from lorenz96.config import LEAD_STEPS


class L96Dataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        lead_steps: int = LEAD_STEPS,
        mu: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
    ):
        if mu is None:
            mu    = states.mean(axis=0)   # (N,)
            sigma = states.std(axis=0)    # (N,)
        self.mu    = mu
        self.sigma = sigma

        normed  = (states - mu) / sigma
        n_pairs = len(normed) - lead_steps

        self.cond   = torch.from_numpy(normed[:n_pairs]).float()
        self.target = torch.from_numpy(normed[lead_steps:]).float()

    def __len__(self):
        return len(self.cond)

    def __getitem__(self, idx):
        return self.cond[idx], self.target[idx]


def load_trajectory(path: str) -> tuple[np.ndarray, dict]:
    d   = np.load(path)
    cfg = json.loads(str(d["config"]))
    return d["states"], cfg
