"""Tests for src.model.diffusion — DDPMSchedule math.

All tests use small synthetic schedules (T=10 or T=100).
No data files or checkpoints are loaded.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.diffusion import DDPMSchedule, _cosine_alpha_bars
from src.config import BETA_START, BETA_END, BETA_SCHEDULE

T_SMALL = 10
T_MED = 100


class TestDDPMScheduleLinear:

    @pytest.fixture(scope="class")
    def sched(self):
        return DDPMSchedule(T=T_SMALL, beta_start=BETA_START, beta_end=BETA_END, schedule=BETA_SCHEDULE)

    def test_alpha_bar_shape(self, sched):
        assert sched.alpha_bars.shape == (T_SMALL,)

    def test_alpha_bar_monotone_decreasing(self, sched):
        ab = sched.alpha_bars.numpy()
        assert np.all(np.diff(ab) < 0)

    def test_alpha_bar_bounds(self, sched):
        ab = sched.alpha_bars.numpy()
        assert ab[0] < 1.0
        assert ab[-1] > 0.0
        assert ab[-1] < ab[0]

    def test_alpha_bars_product_identity(self, sched):
        """alpha_bar_t = prod_{s=0}^{t} alpha_s (cumprod definition)."""
        alphas = sched.alphas.numpy()
        expected = np.cumprod(alphas)
        np.testing.assert_allclose(sched.alpha_bars.numpy(), expected, atol=1e-6)

    def test_sqrt_alpha_bar_squared_equals_alpha_bar(self, sched):
        sab = sched.sqrt_alpha_bars.numpy()
        ab = sched.alpha_bars.numpy()
        np.testing.assert_allclose(sab ** 2, ab, atol=1e-6)

    def test_sqrt_one_minus_alpha_bar_squared(self, sched):
        somb = sched.sqrt_one_minus_alpha_bars.numpy()
        ab = sched.alpha_bars.numpy()
        np.testing.assert_allclose(somb ** 2, 1.0 - ab, atol=1e-6)

    def test_q_sample_formula(self, sched):
        """x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*noise."""
        torch.manual_seed(0)
        batch, d = 4, 8
        x0 = torch.randn(batch, d)
        noise = torch.randn(batch, d)
        t = torch.tensor([0, 2, 5, 9])

        x_t = sched.q_sample(x0, t, noise)

        sab = sched.sqrt_alpha_bars[t].view(-1, 1)
        somb = sched.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        expected = sab * x0 + somb * noise

        torch.testing.assert_close(x_t, expected)

    def test_q_sample_at_t0_mostly_signal(self, sched):
        """At t=0, sqrt(alpha_bar) should be close to 1."""
        assert float(sched.sqrt_alpha_bars[0]) > 0.99

    def test_q_sample_at_tT_mostly_noise(self, sched):
        """At t=T-1, sqrt(1-alpha_bar) should be non-negligible.

        For T=10 with beta_end=0.02, the cumulative noise is moderate (~0.31).
        The key is that it's substantially larger than t=0 (~0.01).
        """
        somb_T = float(sched.sqrt_one_minus_alpha_bars[-1])
        somb_0 = float(sched.sqrt_one_minus_alpha_bars[0])
        assert somb_T > 0.25
        assert somb_T > somb_0 * 5

    def test_betas_in_valid_range(self, sched):
        betas = sched.betas.numpy()
        assert np.all(betas > 0)
        assert np.all(betas < 1)

    def test_t_attribute(self, sched):
        assert sched.T == T_SMALL


class TestDDPMScheduleCosine:

    @pytest.fixture(scope="class")
    def sched(self):
        return DDPMSchedule(T=T_MED, schedule="cosine")

    def test_cosine_alpha_bars_decreasing(self, sched):
        ab = sched.alpha_bars.numpy()
        assert np.all(np.diff(ab) < 0)

    def test_cosine_alpha_bar_starts_near_one(self, sched):
        assert float(sched.alpha_bars[0]) > 0.99

    def test_cosine_betas_clamped(self, sched):
        betas = sched.betas.numpy()
        assert np.all(betas >= 0)
        assert np.all(betas <= 0.999)


class TestCosineAlphaBarsHelper:

    def test_shape(self):
        ab = _cosine_alpha_bars(50).numpy()
        assert ab.shape == (50,)

    def test_values_in_open_01(self):
        ab = _cosine_alpha_bars(50).numpy()
        assert np.all(ab > 0)
        assert np.all(ab < 1.0)

    def test_monotone_decreasing(self):
        ab = _cosine_alpha_bars(50).numpy()
        assert np.all(np.diff(ab) < 0)


class _ZeroNoisePredictor(nn.Module):
    """Always predicts zero noise — lets p_sample's mean formula be verified exactly."""
    def forward(self, x_t, t, cond):
        return torch.zeros_like(x_t)


class TestReverseDiffusion:
    """Tests for p_sample (single denoising step) and sample (full reverse chain).

    Uses a zero-noise predictor so the deterministic mean formula can be
    checked analytically without a trained checkpoint.
    """

    @pytest.fixture(scope="class")
    def sched(self):
        return DDPMSchedule(T=T_SMALL, beta_start=BETA_START, beta_end=BETA_END,
                            schedule=BETA_SCHEDULE)

    @pytest.fixture(scope="class")
    def zero_model(self):
        return _ZeroNoisePredictor()

    def test_p_sample_output_shape(self, sched, zero_model):
        x_t = torch.randn(3, 4)
        cond = torch.zeros(3, 1)
        out = sched.p_sample(zero_model, x_t, t=0, cond=cond)
        assert out.shape == x_t.shape

    def test_p_sample_t0_deterministic(self, sched, zero_model):
        """At t=0 no noise is injected — two identical calls must return the same tensor."""
        x_t = torch.randn(2, 4)
        cond = torch.zeros(2, 1)
        out1 = sched.p_sample(zero_model, x_t, t=0, cond=cond)
        out2 = sched.p_sample(zero_model, x_t, t=0, cond=cond)
        torch.testing.assert_close(out1, out2)

    def test_p_sample_t_gt_0_stochastic(self, sched, zero_model):
        """At t>0, different torch seeds must produce different outputs."""
        x_t = torch.randn(2, 4)
        cond = torch.zeros(2, 1)
        torch.manual_seed(0)
        out1 = sched.p_sample(zero_model, x_t, t=5, cond=cond)
        torch.manual_seed(1)
        out2 = sched.p_sample(zero_model, x_t, t=5, cond=cond)
        assert not torch.allclose(out1, out2)

    def test_p_sample_zero_model_mean_formula(self, sched, zero_model):
        """With eps=0, output at t=0 equals x_t / sqrt(alpha_0) (no noise term).

        From the DDPM mean formula:
            mean = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
        With eps=0 and t=0 (no stochastic term): output = x_t / sqrt(alpha_0).
        """
        torch.manual_seed(0)
        x_t = torch.randn(2, 4)
        cond = torch.zeros(2, 1)
        out = sched.p_sample(zero_model, x_t, t=0, cond=cond)
        expected = x_t / sched.alphas[0].sqrt()
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)

    def test_p_sample_nonzero_eps_mean_formula(self, sched):
        """Full mean formula exercised with a constant non-zero eps predictor at t=0.

        mean = (x_t - beta_0/sqrt(1-alpha_bar_0) * eps) / sqrt(alpha_0)

        This catches bugs in the eps coefficient (wrong sign, using alpha instead
        of beta, wrong schedule index) that the eps=0 test cannot detect.
        """
        class _ConstantEps(nn.Module):
            def forward(self, x_t, t, cond):
                return torch.full_like(x_t, 0.5)

        model = _ConstantEps()
        torch.manual_seed(0)
        x_t  = torch.randn(2, 4)
        cond = torch.zeros(2, 1)
        t    = 0

        out = sched.p_sample(model, x_t, t=t, cond=cond)

        beta = sched.betas[t]
        ab   = sched.alpha_bars[t]
        eps  = torch.full_like(x_t, 0.5)
        expected = (x_t - beta / (1.0 - ab).sqrt() * eps) / sched.alphas[t].sqrt()
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)

    def test_p_sample_output_finite(self, sched, zero_model):
        x_t = torch.randn(4, 8)
        cond = torch.zeros(4, 1)
        for t in [0, T_SMALL // 2, T_SMALL - 1]:
            assert torch.all(torch.isfinite(sched.p_sample(zero_model, x_t, t, cond)))

    def test_sample_output_shape(self, sched, zero_model):
        batch, d = 5, 8
        cond = torch.zeros(batch, 1)
        out = sched.sample(zero_model, cond, target_dim=d)
        assert out.shape == (batch, d)

    def test_sample_output_finite(self, sched, zero_model):
        cond = torch.zeros(3, 1)
        out = sched.sample(zero_model, cond, target_dim=4)
        assert torch.all(torch.isfinite(out))

    def test_sample_reproducible_with_seed(self, sched, zero_model):
        """Two sample() calls with the same manual seed produce identical output."""
        cond = torch.zeros(3, 1)
        torch.manual_seed(7)
        out1 = sched.sample(zero_model, cond, target_dim=4)
        torch.manual_seed(7)
        out2 = sched.sample(zero_model, cond, target_dim=4)
        torch.testing.assert_close(out1, out2)


class TestQSampleStatistics:
    """Statistical tests verifying the forward process distribution.

    q(x_t | x_0) = N(sqrt(ab_t)*x_0, (1-ab_t)*I)
    """

    def test_q_sample_mean(self):
        """E[x_t | x_0] ≈ sqrt(ab_t) * x_0."""
        sched = DDPMSchedule(T=T_SMALL)
        torch.manual_seed(1)
        batch, d = 5000, 4
        x0 = torch.ones(1, d).expand(batch, -1)
        t_val = 5
        t = torch.full((batch,), t_val, dtype=torch.long)
        noise = torch.randn(batch, d)

        x_t = sched.q_sample(x0, t, noise)
        empirical_mean = x_t.mean(dim=0).numpy()
        expected_mean = float(sched.sqrt_alpha_bars[t_val]) * np.ones(d)
        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.05)

    def test_q_sample_variance(self):
        """Var[x_t | x_0=0] ≈ 1 - ab_t per dimension."""
        sched = DDPMSchedule(T=T_SMALL)
        torch.manual_seed(2)
        batch, d = 5000, 4
        x0 = torch.zeros(batch, d)
        t_val = 7
        t = torch.full((batch,), t_val, dtype=torch.long)
        noise = torch.randn(batch, d)

        x_t = sched.q_sample(x0, t, noise)
        empirical_var = x_t.var(dim=0).numpy()
        expected_var = 1.0 - float(sched.alpha_bars[t_val])
        np.testing.assert_allclose(empirical_var, np.full(d, expected_var), atol=0.05)
