"""Tests for the CRPS-trained L96 emulator — ConvGenerativeNet and fair_crps_loss.

No data loading or checkpoints required.
"""

import numpy as np
import pytest
import torch

from lorenz96.config import N, HIDDEN_CH, N_RES, KERNEL
from lorenz96.eval import crps_ensemble
from lorenz96.model import ConvGenerativeNet
from lorenz96.train_crps import fair_crps_loss, fair_es_loss, generate_ensemble

BATCH = 4


class TestConvGenerativeNet:

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return ConvGenerativeNet(hidden_ch=HIDDEN_CH, n_res=N_RES, kernel_size=KERNEL)

    def test_output_shape(self, model):
        cond = torch.randn(BATCH, N)
        assert model(cond, model.sample_latent(BATCH, "cpu")).shape == (BATCH, N)

    def test_members_differ(self, model):
        ens = generate_ensemble(model, torch.randn(BATCH, N), 3, "cpu")
        assert ens.shape == (BATCH, 3, N)
        assert ens.std(dim=1).min() > 0

    def test_gradient_flows(self, model):
        loss = fair_crps_loss(generate_ensemble(model, torch.randn(BATCH, N), 2, "cpu"),
                              torch.randn(BATCH, N))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())


class TestFairCRPS:

    def test_matches_eval_crps_with_fair_correction(self):
        """Fair CRPS = MAE − M/(M−1) · (plug-in spread term of eval.crps_ensemble)."""
        rng = np.random.default_rng(0)
        M = 5
        ens   = rng.standard_normal((BATCH, M, N))
        truth = rng.standard_normal((BATCH, N))
        mae    = np.abs(ens - truth[:, None]).mean(axis=1)
        spread = mae - crps_ensemble(truth, ens)
        expected = (mae - M / (M - 1) * spread).mean()
        got = fair_crps_loss(torch.from_numpy(ens), torch.from_numpy(truth)).item()
        assert got == pytest.approx(expected, rel=1e-10)

    def test_m2_closed_form(self):
        """M=2: (|x1−y| + |x2−y|)/2 − |x1−x2|/2."""
        x = torch.tensor([[[0.0], [2.0]]])
        y = torch.tensor([[0.5]])
        assert fair_crps_loss(x, y).item() == pytest.approx((0.5 + 1.5) / 2 - 2.0 / 2)


class TestFairES:

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        M = 5
        ens   = rng.standard_normal((BATCH, M, N))
        truth = rng.standard_normal((BATCH, N))
        expected = np.mean([
            np.linalg.norm(ens[b] - truth[b], axis=1).mean()
            - sum(np.linalg.norm(ens[b, i] - ens[b, j]) for i in range(M) for j in range(M) if i != j)
            / (2 * M * (M - 1))
            for b in range(BATCH)])
        got = fair_es_loss(torch.from_numpy(ens), torch.from_numpy(truth)).item()
        assert got == pytest.approx(expected, rel=1e-8)

    def test_one_dim_equals_crps(self):
        x = torch.tensor([[[0.0], [2.0], [3.5]]], dtype=torch.float64)
        y = torch.tensor([[-1.0]], dtype=torch.float64)
        expected = 8.5 / 3 - 7 / 6
        assert fair_crps_loss(x, y).item() == pytest.approx(expected)
        assert fair_es_loss(x, y).item() == pytest.approx(expected, abs=1e-6)

    def test_gradient_flows(self):
        torch.manual_seed(0)
        model = ConvGenerativeNet(hidden_ch=HIDDEN_CH, n_res=N_RES, kernel_size=KERNEL)
        fair_es_loss(generate_ensemble(model, torch.randn(BATCH, N), 2, "cpu"),
                     torch.randn(BATCH, N)).backward()
        grads = [p.grad for p in model.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
