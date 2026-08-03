"""Tests for src.evaluation.scores — centralized scoring functions.

Every test here guards a mathematical claim in the paper.
No real data or checkpoint files are loaded.
"""
import numpy as np
import pytest

from src.evaluation.scores import crps, energy_score, variogram_score

M = 50
D = 8


# ── CRPS ─────────────────────────────────────────────────────────────────────

class TestCRPS:
    RNG = np.random.default_rng(42)

    def test_deterministic_ensemble_equals_abs_error(self):
        """M identical members → CRPS = |forecast − obs|."""
        f = np.array([1.0, 2.0, -0.5, 0.3, 1.7, -1.0, 0.0, 0.8])
        obs = np.array([0.5, 1.5, -1.0, 0.5, 2.0, -0.5, 0.1, 0.9])
        members = np.tile(f, (M, 1))
        np.testing.assert_allclose(crps(members, obs), np.abs(f - obs), atol=1e-12)

    def test_perfect_forecast_is_zero(self):
        """Members all equal obs → CRPS = 0."""
        obs = np.array([1.0, -2.0, 0.5, 3.1, -0.7, 1.2, 0.0, -1.5])
        members = np.tile(obs, (M, 1))
        np.testing.assert_allclose(crps(members, obs), 0.0, atol=1e-12)

    def test_2member_unbiased_estimator_arithmetic(self):
        """M=2 arithmetic check for the unbiased CRPS estimator.

        term1 = mean(|0−1|, |2−1|) = 1
        term2 = mean(off-diagonal |diffs|) / 2 = mean(2, 2) / 2 = 1
        CRPS = 1 − 1 = 0

        NOTE: CRPS=0 here is an artefact of M=2 with the unbiased estimator,
        not a general consequence of obs being centred in the ensemble.
        With M=50 members uniformly spread around obs the score is positive.
        """
        members = np.array([[0.0], [2.0]])
        obs = np.array([1.0])
        np.testing.assert_allclose(crps(members, obs), [0.0], atol=1e-12)

    def test_2member_obs_outside(self):
        """members=[0,2], obs=5 → CRPS=3 (distance to nearest member).

        term1 = mean(5, 3) = 4
        term2 = 2 / 2 = 1
        CRPS = 4 − 1 = 3
        """
        members = np.array([[0.0], [2.0]])
        obs = np.array([5.0])
        np.testing.assert_allclose(crps(members, obs), [3.0], atol=1e-12)

    def test_returns_shape_d(self):
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        assert crps(members, obs).shape == (D,)

    def test_nonnegative(self):
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        assert np.all(crps(members, obs) >= -1e-12)

    def test_nan_where_obs_nan(self):
        """NaN in obs propagates to that station only."""
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        obs[3] = np.nan
        result = crps(members, obs)
        assert np.isnan(result[3])
        other = np.arange(D) != 3
        assert np.all(np.isfinite(result[other]))

    def test_multidim_equals_marginal(self):
        """Joint (M,d) call == per-column (M,1) calls (catches reshape bugs)."""
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        joint = crps(members, obs)
        for k in range(D):
            per_station = crps(members[:, k:k + 1], obs[k:k + 1])
            np.testing.assert_allclose(joint[k], per_station[0], atol=1e-10)


# ── Energy Score ──────────────────────────────────────────────────────────────

class TestEnergyScore:
    RNG = np.random.default_rng(43)

    def test_perfect_ensemble_is_zero(self):
        """All members = obs → ES = 0."""
        obs = self.RNG.standard_normal((D,))
        members = np.tile(obs, (M, 1))
        assert abs(energy_score(members, obs)) < 1e-10

    def test_nonnegative(self):
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        assert energy_score(members, obs) >= -1e-10

    def test_returns_float(self):
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        assert isinstance(energy_score(members, obs), float)

    def test_nan_when_fewer_than_2_valid_stations(self):
        members = self.RNG.standard_normal((M, D))
        obs = np.full(D, np.nan)
        obs[0] = 1.0
        assert np.isnan(energy_score(members, obs))

    def test_nan_when_all_obs_nan(self):
        members = self.RNG.standard_normal((M, D))
        obs = np.full(D, np.nan)
        assert np.isnan(energy_score(members, obs))

    def test_calibrated_ensemble_beats_bad_ensemble(self):
        """A well-calibrated ensemble scores lower ES than a bad one.

        Good ensemble: narrow, centred on obs.
        Bad ensemble: wide, unrelated to obs.
        """
        rng_l = np.random.default_rng(99)
        obs = rng_l.standard_normal(D)
        good = obs[None, :] + 0.05 * rng_l.standard_normal((M, D))
        bad = rng_l.standard_normal((M, D)) * 10
        assert energy_score(good, obs) < energy_score(bad, obs)


# ── Variogram Score ───────────────────────────────────────────────────────────

class TestVariogramScore:
    RNG = np.random.default_rng(44)

    def test_perfect_ensemble_is_zero(self):
        """Members all equal obs → all obs_diff == all ens_diff → VS=0.

        Uses non-constant obs so both the obs-term and ensemble-term are
        non-trivially exercised (unlike the all-constant degenerate case).
        """
        obs = np.arange(D, dtype=float)
        members = np.tile(obs, (M, 1))
        assert abs(variogram_score(members, obs)) < 1e-12

    def test_2station_members_match_variogram(self):
        """obs=[0,1], members=[[0,1],[0,1]] → VS=0.

        ens_mean(|m0−m1|^0.5) = |0−1|^0.5 = 1 = obs_diff → term = 0.
        """
        obs = np.array([0.0, 1.0])
        members = np.array([[0.0, 1.0], [0.0, 1.0]])
        assert abs(variogram_score(members, obs, p=0.5)) < 1e-12

    def test_2station_zero_ensemble(self):
        """obs=[0,4], members all zeros, p=0.5 → VS=4.0.

        obs_diff = |0−4|^0.5 = 2; ens mean = 0; (2−0)^2 = 4; n_pairs=1.
        """
        obs = np.array([0.0, 4.0])
        members = np.zeros((M, 2))
        np.testing.assert_allclose(variogram_score(members, obs, p=0.5), 4.0, atol=1e-10)

    def test_nonnegative(self):
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        assert variogram_score(members, obs) >= -1e-12

    def test_returns_float(self):
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        assert isinstance(variogram_score(members, obs), float)

    def test_nan_when_fewer_than_2_valid_stations(self):
        members = self.RNG.standard_normal((M, D))
        obs = np.full(D, np.nan)
        obs[0] = 1.0
        assert np.isnan(variogram_score(members, obs))

    def test_p_parameter_changes_score(self):
        """Different p values produce different scores."""
        members = self.RNG.standard_normal((M, D))
        obs = self.RNG.standard_normal((D,))
        vs_half = variogram_score(members, obs, p=0.5)
        vs_one = variogram_score(members, obs, p=1.0)
        vs_two = variogram_score(members, obs, p=2.0)
        assert not (abs(vs_half - vs_one) < 1e-12 and abs(vs_one - vs_two) < 1e-12)

    def test_calibrated_ensemble_beats_bad_ensemble(self):
        """Ensemble matching obs variogram has lower VS than one that doesn't.

        Good: all members ≈ obs (variogram matches exactly).
        Bad: all members = zero (variogram always 0, wrong for any non-flat obs).
        """
        obs = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        good = np.tile(obs, (M, 1)) + 0.01 * np.random.default_rng(5).standard_normal((M, D))
        bad = np.zeros((M, D))
        assert variogram_score(good, obs) < variogram_score(bad, obs)

    def test_reduce_false_returns_array_of_correct_shape(self):
        """reduce=False → shape (d*(d-1)//2,), matching np.triu_indices ordering."""
        members = self.RNG.standard_normal((M, D))
        obs     = self.RNG.standard_normal((D,))
        out     = variogram_score(members, obs, reduce=False)
        assert isinstance(out, np.ndarray)
        assert out.shape == (D * (D - 1) // 2,)

    def test_reduce_false_nan_where_obs_nan(self):
        """Pairs where either station obs is NaN appear as NaN in the output."""
        members = self.RNG.standard_normal((M, D))
        obs     = self.RNG.standard_normal((D,))
        obs[0]  = np.nan                             # station 0 invalid
        out     = variogram_score(members, obs, reduce=False)
        triu_i, triu_j = np.triu_indices(D, k=1)
        nan_mask = (triu_i == 0) | (triu_j == 0)    # all pairs involving station 0
        assert np.all(np.isnan(out[nan_mask]))
        assert np.all(np.isfinite(out[~nan_mask]))

    def test_reduce_false_nanmean_matches_reduce_true(self):
        """nanmean of reduce=False output equals the reduce=True scalar."""
        members = self.RNG.standard_normal((M, D))
        obs     = self.RNG.standard_normal((D,))
        per_pair = variogram_score(members, obs, reduce=False)
        scalar   = variogram_score(members, obs, reduce=True)
        np.testing.assert_allclose(np.nanmean(per_pair), scalar, rtol=1e-10)

    def test_reduce_false_ordering_matches_triu_indices(self):
        """Index k in reduce=False output corresponds to np.triu_indices(d, k=1) pair k.

        All members = 1 so ens_diff = 0. obs[i] = i so obs_diff = |i-j|^0.5.
        For pair (2, 5): term = (|2-5|^0.5 - 0)^2 = 3.0.
        """
        members = np.ones((M, D))
        obs     = np.arange(D, dtype=float)
        triu_i, triu_j = np.triu_indices(D, k=1)
        k25 = int(np.where((triu_i == 2) & (triu_j == 5))[0][0])
        out = variogram_score(members, obs, reduce=False)
        np.testing.assert_allclose(out[k25], 3.0, rtol=1e-10)
