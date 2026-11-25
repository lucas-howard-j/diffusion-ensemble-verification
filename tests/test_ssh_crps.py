"""Tests for the CRPS-trained SSH generator, its loss, checkpoint loading, and the
calibrated-dependence reference used in scripts/ssh_crps_comparison.py.

No data loading or real checkpoints required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.scores import crps, energy_score
from src.model.dataset import SSHDiffusionDataset
from src.model.network import CRPSGenerator
from src.model.sample import load_checkpoint
from src.model.train_crps import fair_crps_loss, fair_es_loss, sample_with_fixed_noise

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ssh_crps_comparison import error_correlations, recolour  # noqa: E402

_ds = object.__new__(SSHDiffusionDataset)
TARGET_DIM = _ds.target_dim
COND_DIM = _ds.cond_dim
BATCH = 4


@pytest.fixture
def gen():
    torch.manual_seed(0)
    return CRPSGenerator(TARGET_DIM, COND_DIM, noise_dim=8, hidden=(32, 32), use_beta_silu=True)


class TestCRPSGenerator:

    def test_sample_shape(self, gen):
        assert gen.sample(torch.randn(BATCH, COND_DIM), 3).shape == (BATCH, 3, TARGET_DIM)

    def test_members_differ(self, gen):
        ens = gen.sample(torch.randn(BATCH, COND_DIM), 5)
        assert ens.std(dim=1).min() > 0

    def test_gradient_flows(self, gen):
        loss = fair_crps_loss(gen.sample(torch.randn(BATCH, COND_DIM), 2),
                              torch.randn(BATCH, TARGET_DIM))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in gen.parameters())

    def test_checkpoint_roundtrip(self, gen, tmp_path):
        path = tmp_path / "gen.pt"
        torch.save({"model_type": "crps_generator", "model": gen.state_dict(),
                    "target_dim": TARGET_DIM, "cond_dim": COND_DIM, "noise_dim": 8,
                    "hidden_dims": (32, 32), "dropout": 0.0, "use_beta_silu": True}, path)
        model, schedule, _, _ = load_checkpoint(path, device="cpu")
        assert schedule is None
        cond, z = torch.randn(BATCH, COND_DIM), torch.randn(BATCH, 8)
        gen.eval()
        assert torch.allclose(model(cond, z), gen(cond, z))


class TestFairCRPSLoss:

    def test_matches_scores_crps_summed_over_gauges(self):
        rng = np.random.default_rng(0)
        ens = rng.standard_normal((BATCH, 6, TARGET_DIM))
        y = rng.standard_normal((BATCH, TARGET_DIM))
        expected = np.mean([crps(ens[b], y[b]).sum() for b in range(BATCH)])
        got = fair_crps_loss(torch.from_numpy(ens), torch.from_numpy(y)).item()
        assert got == pytest.approx(expected, rel=1e-10)

    def test_m2_closed_form(self):
        x = torch.tensor([[[0.0], [2.0]]])
        y = torch.tensor([[0.5]])
        assert fair_crps_loss(x, y).item() == pytest.approx((0.5 + 1.5) / 2 - 2.0 / 2)


class TestValidationNoise:

    def test_fixed_noise_is_repeatable(self, gen):
        cond = torch.randn(BATCH, COND_DIM)
        a = sample_with_fixed_noise(gen, cond, 4, 123)
        b = sample_with_fixed_noise(gen, cond, 4, 123)
        assert torch.equal(a, b)

    def test_global_rng_stream_untouched(self, gen):
        """Validation sampling must not reset the training RNG stream."""
        cond = torch.randn(BATCH, COND_DIM)
        torch.manual_seed(0)
        expected = torch.randperm(100)
        torch.manual_seed(0)
        sample_with_fixed_noise(gen, cond, 4, 123)
        assert torch.equal(torch.randperm(100), expected)

    def test_epochs_get_fresh_draws(self, gen):
        """Two 'epochs' separated by validation see different training permutations."""
        cond = torch.randn(BATCH, COND_DIM)
        torch.manual_seed(0)
        p1 = torch.randperm(100)
        sample_with_fixed_noise(gen, cond, 4, 123)
        p2 = torch.randperm(100)
        assert not torch.equal(p1, p2)


class TestFairESLoss:

    def test_matches_scores_energy_score(self):
        rng = np.random.default_rng(0)
        ens = rng.standard_normal((BATCH, 6, TARGET_DIM))
        y = rng.standard_normal((BATCH, TARGET_DIM))
        expected = np.mean([energy_score(ens[b], y[b]) for b in range(BATCH)])
        got = fair_es_loss(torch.from_numpy(ens), torch.from_numpy(y)).item()
        assert got == pytest.approx(expected, rel=1e-8)

    def test_one_dim_equals_crps(self):
        x = torch.tensor([[[0.0], [2.0], [3.5]]], dtype=torch.float64)
        y = torch.tensor([[-1.0]], dtype=torch.float64)
        # (1 + 3 + 4.5)/3 − (2 + 3.5 + 1.5)·2 / (2·3·2) = 8.5/3 − 7/6
        expected = 8.5 / 3 - 7 / 6
        assert fair_crps_loss(x, y).item() == pytest.approx(expected)
        assert fair_es_loss(x, y).item() == pytest.approx(expected, abs=1e-6)

    def test_gradient_finite_with_coincident_members(self, gen):
        cond = torch.randn(BATCH, COND_DIM)
        ens = gen.sample(cond, 2)
        ens = torch.cat([ens[:, :1], ens[:, :1]], dim=1)     # identical members
        fair_es_loss(ens, torch.randn(BATCH, TARGET_DIM)).backward()
        grads = [p.grad for p in gen.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


class TestRecolour:

    @pytest.fixture
    def members_and_R(self, rng):
        A = rng.standard_normal((TARGET_DIM, TARGET_DIM))
        members = rng.standard_normal((500, TARGET_DIM)) @ A.T * 3 + 10
        B = rng.standard_normal((TARGET_DIM, TARGET_DIM))
        C = B @ B.T + TARGET_DIM * np.eye(TARGET_DIM)
        R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
        return members, R

    def test_preserves_mean_and_spread(self, members_and_R):
        x, R = members_and_R
        ref = recolour(x, R)
        np.testing.assert_allclose(ref.mean(0), x.mean(0), atol=1e-10)
        np.testing.assert_allclose(ref.std(0, ddof=1), x.std(0, ddof=1), rtol=1e-10)

    def test_sets_correlation(self, members_and_R):
        x, R = members_and_R
        np.testing.assert_allclose(np.corrcoef(recolour(x, R), rowvar=False), R, atol=1e-10)

    def test_error_correlations_leave_year_out(self, rng):
        N, n_leads, M = 30, 2, 20
        d = dict(members=rng.standard_normal((N, n_leads, M, TARGET_DIM)),
                 truth=rng.standard_normal((N, n_leads, TARGET_DIM)),
                 year=np.repeat([2019, 2020, 2021], 10))
        R = error_correlations(d)
        assert R.shape == (N, n_leads, TARGET_DIM, TARGET_DIM)
        err = d["truth"] - d["members"].mean(2)
        keep = d["year"] != 2019
        np.testing.assert_allclose(R[0, 1], np.corrcoef(err[keep, 1], rowvar=False))
