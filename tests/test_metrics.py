"""Tests for src.evaluation.metrics — bootstrap CI and correlation utilities.

The block_bootstrap_ci is used for all confidence intervals in the paper.
No data files are loaded.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    block_bootstrap_ci, pearson_r, rmse,
    anomaly_correlation, upper_triangle, pairwise_corr_matrix, pattern_correlation,
)


class TestBlockBootstrapCI:
    RNG = np.random.default_rng(0)

    def test_output_shapes(self):
        x = self.RNG.standard_normal((50, 8))
        lo, hi = block_bootstrap_ci(x, block_size=5, n_boot=200, alpha=0.05)
        assert lo.shape == (8,)
        assert hi.shape == (8,)

    def test_lo_le_hi_everywhere(self):
        x = self.RNG.standard_normal((50, 8))
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
        x = self.RNG.standard_normal((60, 4))
        lo_95, hi_95 = block_bootstrap_ci(x, block_size=5, n_boot=500, alpha=0.05)
        lo_80, hi_80 = block_bootstrap_ci(x, block_size=5, n_boot=500, alpha=0.20)
        assert np.all(hi_95 - lo_95 >= hi_80 - lo_80 - 1e-10)

    def test_seeded_rng_reproducible(self):
        x = self.RNG.standard_normal((40, 3))
        lo1, hi1 = block_bootstrap_ci(x, block_size=4, n_boot=100, alpha=0.05,
                                       rng=np.random.default_rng(7))
        lo2, hi2 = block_bootstrap_ci(x, block_size=4, n_boot=100, alpha=0.05,
                                       rng=np.random.default_rng(7))
        np.testing.assert_array_equal(lo1, lo2)
        np.testing.assert_array_equal(hi1, hi2)

    def test_sample_mean_inside_ci(self):
        """The sample mean must lie within its own bootstrap CI."""
        x = self.RNG.standard_normal((50, 8))
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


class TestAnomalyCorrelation:
    # NOTE: Pearson r is location-invariant, so subtracting nanmean from both
    # series is a mathematical no-op — anomaly_correlation(obs, pred) is always
    # equal to pearson_r(obs, pred). The tests below are correct but cannot
    # distinguish the two implementations. Flagged so the function's intent
    # (climatological anomaly removal) is reviewed if the implementation changes.

    def test_mean_shift_ignored(self):
        """Series differing only by a constant offset have identical anomalies → r=1."""
        x = np.arange(10, dtype=float)
        assert abs(anomaly_correlation(x, x + 5.0) - 1.0) < 1e-10

    def test_sign_flip_gives_minus_one(self):
        x = np.arange(1, 11, dtype=float)
        assert abs(anomaly_correlation(x, -x) - (-1.0)) < 1e-10

    def test_nan_when_too_few_valid_pairs(self):
        obs  = np.array([1.0, np.nan, np.nan, np.nan])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(anomaly_correlation(obs, pred))

    def test_returns_float_in_range(self):
        rng = np.random.default_rng(1)
        r = anomaly_correlation(rng.standard_normal(20), rng.standard_normal(20))
        assert isinstance(r, float)
        assert -1.0 - 1e-10 <= r <= 1.0 + 1e-10


class TestUpperTriangle:

    def test_known_3x3(self):
        """triu_indices(3, k=1) picks positions (0,1), (0,2), (1,2)."""
        mat = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=float)
        np.testing.assert_array_equal(upper_triangle(mat), [2.0, 3.0, 6.0])

    def test_length(self):
        n = 8
        assert len(upper_triangle(np.ones((n, n)))) == n * (n - 1) // 2


class TestPairwiseCorrMatrix:

    @staticmethod
    def _df(seed=10, n=50, cols="ABCD"):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(rng.standard_normal((n, len(cols))), columns=list(cols))

    def test_output_shape(self):
        result = pairwise_corr_matrix(self._df())
        assert result.shape == (4, 4)

    def test_diagonal_is_one(self):
        result = pairwise_corr_matrix(self._df())
        np.testing.assert_allclose(np.diag(result.values), 1.0, atol=1e-10)

    def test_symmetric(self):
        result = pairwise_corr_matrix(self._df())
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-15)

    def test_nan_below_min_periods(self):
        """min_periods=12: fewer than 12 overlapping finite values → NaN."""
        df = pd.DataFrame({
            "A": np.concatenate([np.arange(5, dtype=float), np.full(45, np.nan)]),
            "B": np.arange(50, dtype=float),
        })
        result = pairwise_corr_matrix(df)
        assert np.isnan(result.loc["A", "B"])


class TestPatternCorrelation:

    @staticmethod
    def _corr(seed, n=50, cols="ABCD"):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=list(cols))
        return pairwise_corr_matrix(df)

    def test_identical_matrices_give_r1(self):
        mat = self._corr(20)
        assert abs(pattern_correlation(mat, mat) - 1.0) < 1e-10

    def test_returns_float_in_range(self):
        r = pattern_correlation(self._corr(21), self._corr(22))
        assert isinstance(r, float)
        assert -1.0 - 1e-10 <= r <= 1.0 + 1e-10

    def test_column_intersection_matches_direct_computation(self):
        """Result with mismatched columns equals restricting to shared columns first."""
        rng = np.random.default_rng(23)
        df_abcd = pd.DataFrame(rng.standard_normal((50, 4)), columns=list("ABCD"))
        df_abc  = pd.DataFrame(rng.standard_normal((50, 3)), columns=list("ABC"))

        m_abcd = pairwise_corr_matrix(df_abcd)
        m_abc  = pairwise_corr_matrix(df_abc)

        # Ground truth: recompute mat1 restricted to shared columns, then correlate
        m_abc_from_abcd = pairwise_corr_matrix(df_abcd[list("ABC")])
        r_expected = pattern_correlation(m_abc_from_abcd, m_abc)

        r_intersection = pattern_correlation(m_abcd, m_abc)
        assert abs(r_intersection - r_expected) < 1e-10


class TestJointSkill:

    def test_ratio_of_means_not_mean_of_ratios(self):
        from src.evaluation.metrics import ratio_of_means_skill
        # cases (model, clim) = (1, 1) and (3, 4): ratio of means 1 − 4/5 = 0.2,
        # whereas the mean of per-case skills would be (0 + 0.25)/2 = 0.125
        s = np.array([[1.0, 0, 1.0, 0], [3.0, 0, 4.0, 0]])
        sk = ratio_of_means_skill(s)
        assert sk[0] == pytest.approx(0.2)
        assert sk[2] == pytest.approx(0.0)

    def test_summary_point_and_ci(self):
        from src.evaluation.metrics import joint_skill_summary
        rng = np.random.default_rng(0)
        N, L = 60, 3
        clim = rng.uniform(1, 2, (N, L))
        s = np.stack([0.8 * clim, 0.9 * clim, clim, 1.1 * clim], axis=-1)
        out = joint_skill_summary(s, block_size=3, n_boot=200, rng=rng)
        np.testing.assert_allclose(out["skill"][:, 0], 0.2)
        np.testing.assert_allclose(out["delta"], [0.1, 0.1, 0.0], atol=1e-12)
        np.testing.assert_allclose(out["skill_lo"][:, 0], 0.2)   # exact proportionality → no spread
        assert out["skill_lo"].shape == (L, 4) and out["delta_hi"].shape == (3,)
