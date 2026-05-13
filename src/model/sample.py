"""Conditional sampling from a trained DDPM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import CHECKPOINTS
from src.model.diffusion import DDPMSchedule
from src.model.network import MLPDenoiser


def load_checkpoint(path: Path | None = None, device: str | None = None):
    """Load the EMA weights for inference."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = path or (CHECKPOINTS / "ddpm_best_large.pt")
    ckpt = torch.load(path, map_location=device)

    model = MLPDenoiser(
        target_dim=int(ckpt["target_dim"]),
        cond_dim=int(ckpt["cond_dim"]),
        hidden=tuple(ckpt.get("hidden_dims", (256, 256, 128))),
        dropout=float(ckpt.get("dropout", 0.1)),
        use_beta_silu=bool(ckpt.get("use_beta_silu", False)),
    ).to(device)
    # EMA requires many steps to diverge from random init (half-life ~6.9k steps at
    # decay=0.9999). If the EMA norm is within 1% of a fresh init, fall back to the
    # regular model weights.
    import copy
    fresh_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
    ema_candidate = copy.deepcopy(model)
    ema_candidate.load_state_dict(ckpt["ema"])
    ema_norm = sum(p.norm().item() ** 2 for p in ema_candidate.parameters()) ** 0.5
    if abs(ema_norm - fresh_norm) / (fresh_norm + 1e-8) < 0.01:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["ema"])
    model.eval()
    schedule = DDPMSchedule(schedule=ckpt.get("beta_schedule", "linear")).to(device)
    return model, schedule, ckpt, device


@torch.no_grad()
def sample_ensemble(condition: np.ndarray, n_members: int = 100,
                    ckpt_path: Path | None = None) -> np.ndarray:
    """
    Draw an ensemble of SSH PC predictions for a single condition vector.

    Parameters
    ----------
    condition : 1-D ndarray of shape (cond_dim,)
    n_members : number of ensemble members

    Returns
    -------
    pcs_z : (n_members, target_dim) ndarray  — z-scored SSH PCs
    """
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    target_dim = int(ckpt["target_dim"])

    cond = torch.from_numpy(condition.astype(np.float32)).to(device)
    cond = cond.unsqueeze(0).expand(n_members, -1)

    pcs = schedule.sample(model, cond, target_dim)
    return pcs.cpu().numpy()
