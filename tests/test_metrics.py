"""Tests for src.evaluation.metrics — bootstrap CI and correlation utilities.

The block_bootstrap_ci is used for all confidence intervals in the paper.
No data files are loaded.
"""
import numpy as np
import pytest

from src.evaluation.metrics import block_bootstrap_ci, pearson_r, rmse

RNG = np.random.default_rng(0)


class TestBlockBootstrapCI:

    def test_output_shapes(self):
        x = RNG.standard_normal((50, 8))
        lo, hi = block_bootstrap_ci(x, block_size=5, n_boot=200, alpha=0.05)
        assert lo.shape == (8,)
        assert hi.shape == (8,)

    def test_lo_le_hi_everywhere(self):
        x = RNG.standard_normal((50, 8))
        lo, hi = block_bootstrap_ci(x, block_size=5, n_boot=200, alpha=0.05)
        assert np.all(lo <= hi)

    def test_ci_contains_true_mean(self):
        """95% CI should cover the true mean in ≥80% of independent trials.

        Uses AR(1) data (phi=0.5) to mimic temporal autocorrelation in scores.
        Conservative threshold (80%) keeps the test fast and non-flaky.
        """
        true_mean = 2.0
        n_trials = 200
        N = 40
        covered = 0
        rng_trial = np.random.default_rng(123)

        for _ in range(n_trials):
            x = np.zeros(N)
            x[0] = rng_trial.standard_normal()
            for i in range(1, N):
                x[i] = 0.5 * x[i - 1] + rng_trial.standard_normal()
            x += true_mean

            lo, hi = block_bootstrap_ci(
                x[:, None], block_size=4, n_boot=500, alpha=0.05,
                rng=rng_trial,
            )
            if lo[0] <= true_mean <= hi[0]:
                covered += 1

        assert covered / n_trials >= 0.80, (
            f"Bootstrap CI coverage {covered/n_trials:.2%} < 80%"
        )

    def test_ci_width_grows_with_confidence(self):
        """95% CI is wider than 80% CI."""
        x = RNG.standard_normal((60, 4))
        lo_95, hi_95 = block_bootstrap_ci(x, block_size=5, n_boot=500, alpha=0.05)
        lo_80, hi_80 = block_bootstrap_ci(x, block_size=5, n_boot=500, alpha=0.20)
        assert np.all(hi_95 - lo_95 >= hi_80 - lo_80 - 1e-10)

    def test_seeded_rng_reproducible(self):
        x = RNG.standard_normal((40, 3))
        lo1, hi1 = block_bootstrap_ci(x, block_size=4, n_boot=100, alpha=0.05,
                                       rng=np.random.default_rng(7))
        lo2, hi2 = block_bootstrap_ci(x, block_size=4, n_boot=100, alpha=0.05,
                                       rng=np.random.default_rng(7))
        np.testing.assert_array_equal(lo1, lo2)
        np.testing.assert_array_equal(hi1, hi2)

    def test_sample_mean_inside_ci(self):
        """The sample mean must lie within its own bootstrap CI."""
        x = RNG.standard_normal((50, 8))
        lo, hi = block_bootstrap_ci(x, block_size=5, n_boot=500, alpha=0.05)
        mean = x.mean(axis=0)
        assert np.all(lo <= mean + 1e-10)
        assert np.all(mean - 1e-10 <= hi)


class TestPearsonR:

    def test_perfect_correlation(self):
        x = np.arange(10, dtype=float)
        assert abs(pearson_r(x, x) - 1.0) < 1e-12

    def test_perfect_anticorrelation(self):
        x = np.arange(10, dtype=float)
        assert abs(pearson_r(x, -x) - (-1.0)) < 1e-12

    def test_nan_when_fewer_than_3_valid_pairs(self):
        obs = np.array([1.0, np.nan, np.nan, np.nan])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(pearson_r(obs, pred))

    def test_nan_pairs_ignored(self):
        """NaN pair is excluded; remaining perfectly-correlated pairs → r=1."""
        obs = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(pearson_r(obs, pred) - 1.0) < 1e-10

    def test_range(self):
        rng_l = np.random.default_rng(5)
        for _ in range(20):
            obs = rng_l.standard_normal(30)
            pred = rng_l.standard_normal(30)
            r = pearson_r(obs, pred)
            assert -1.0 - 1e-10 <= r <= 1.0 + 1e-10


class TestRMSE:

    def test_zero_for_perfect_forecast(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(rmse(x, x)) < 1e-12

    def test_known_value(self):
        """obs=[0,0], pred=[3,4] → RMSE = sqrt((9+16)/2) = sqrt(12.5)."""
        obs = np.array([0.0, 0.0])
        pred = np.array([3.0, 4.0])
        np.testing.assert_allclose(rmse(obs, pred), np.sqrt(12.5), atol=1e-12)

    def test_nan_pair_excluded(self):
        obs = np.array([0.0, np.nan])
        pred = np.array([0.0, 99.0])
        assert abs(rmse(obs, pred)) < 1e-12
