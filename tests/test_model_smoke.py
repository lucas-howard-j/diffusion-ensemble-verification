"""
Integration smoke test: sanity-check trained DDPM ensemble outputs.

Checks that for a zero (mean-condition) conditioning vector:
  1. Per-station ensemble mean is within 5 climatological std deviations of clim mean
  2. Per-station ensemble std is less than 150% of climatological std

These are generous bounds that catch runaway model behaviour (NaN, infinite
amplification, complete distributional collapse) without being sensitive to
calibration quality.

Requires: checkpoints/ddpm_best_gauge_init.pt
Skipped automatically if the checkpoint is absent.

Run:
    PYTHONPATH=. pytest tests/test_model_smoke.py -v
Exclude from fast suite:
    PYTHONPATH=. pytest tests/ -v -m "not integration"
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

CKPT_PATH = Path(__file__).parent.parent / "checkpoints" / "ddpm_best_gauge_init.pt"
N_MEMBERS = 20

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not CKPT_PATH.exists(), reason="checkpoint not found")
class TestModelOutputSanity:

    @pytest.fixture(scope="class")
    def ensemble_output(self):
        """Run inference once; shared across both tests in this class."""
        from src.model.sample import load_checkpoint

        model, schedule, ckpt, device = load_checkpoint(CKPT_PATH)

        target_dim = int(ckpt["target_dim"])
        cond_dim   = int(ckpt["cond_dim"])
        gauge_mu   = np.asarray(ckpt["gauge_mu"], dtype=float)   # (n_st,)
        gauge_std  = np.asarray(ckpt["gauge_std"], dtype=float)  # (n_st,)

        # Zero = mean climate state (all conditioning inputs are z-scored during training)
        cond  = torch.zeros(N_MEMBERS, cond_dim, device=device)
        preds = schedule.sample(model, cond, target_dim)          # (N, n_st), normalised
        members = preds.cpu().numpy() * gauge_std + gauge_mu      # physical units (cm)

        return members, gauge_mu, gauge_std

    def test_ensemble_mean_within_5_clim_stds(self, ensemble_output):
        members, mu, std = ensemble_output
        deviation = np.abs(members.mean(axis=0) - mu) / std      # (n_st,)
        worst_st  = int(deviation.argmax())
        assert deviation.max() < 5.0, (
            f"Station {worst_st}: ensemble mean {deviation[worst_st]:.2f}σ "
            f"from climatology (limit 5σ)"
        )

    def test_ensemble_std_less_than_150pct_clim_std(self, ensemble_output):
        members, mu, std = ensemble_output
        ratio    = members.std(axis=0) / std                      # (n_st,)
        worst_st = int(ratio.argmax())
        assert ratio.max() < 1.5, (
            f"Station {worst_st}: ensemble spread {ratio[worst_st]:.2f}× "
            f"climatological std (limit 1.50×)"
        )
