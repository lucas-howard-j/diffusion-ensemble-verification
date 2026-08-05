"""Tests for lorenz96.eval — crps_ensemble, variogram_score, vs_per_ic, bootstrap_skill_ci.

All tests use synthetic arrays; no data loading or checkpoints required.
"""

import numpy as np
import pytest

from lorenz96.eval import (
    bootstrap_skill_ci,
    crps_ensemble,
    variogram_score,
    vs_per_ic,
)

RNG = np.random.default_rng(0)

N_IC = 40    # number of test ICs
M    = 20    # ensemble size
D    = 8     # state dimension (keep small for speed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def perfect_ens():
    """Ensemble where every member equals the truth."""
    truth = RNG.standard_normal((N_IC, D)).astype(np.float32)
    ens   = np.broadcast_to(truth[:, None, :], (N_IC, M, D)).copy()
    return truth, ens


@pytest.fixture(scope="module")
def calibrated_ens():
    """Truth and ensemble drawn from the same distribution (well-calibrated)."""
    truth = RNG.standard_normal((N_IC, D)).astype(np.float32)
    ens   = RNG.standard_normal((N_IC, M, D)).astype(np.float32)
    return truth, ens


@pytest.fixture(scope="module")
def clim_ens():
    """Independent climatological draw (same shape as calibrated_ens)."""
    return RNG.standard_normal((N_IC, M, D)).astype(np.float32)


# ---------------------------------------------------------------------------
# crps_ensemble
# ---------------------------------------------------------------------------

class TestCrpsEnsemble:

    def _naive(self, truth, ens):
        """Original O(M²) formula — used as reference for the rewrite."""
        M_ = ens.shape[1]
        mae    = np.abs(ens - truth[:, None, :]).mean(axis=1)
        spread = np.abs(ens[:, :, None, :] - ens[:, None, :, :]).sum(axis=(1, 2)) / (2 * M_ * M_)
        return mae - spread

    def test_output_shape(self, calibrated_ens):
        truth, ens = calibrated_ens
        assert crps_ensemble(truth, ens).shape == (N_IC, D)

    def test_non_negative(self, calibrated_ens):
        truth, ens = calibrated_ens
        assert np.all(crps_ensemble(truth, ens) >= -1e-9)

    def test_perfect_forecast_is_zero(self, perfect_ens):
        truth, ens = perfect_ens
        np.testing.assert_allclose(crps_ensemble(truth, ens), 0.0, atol=1e-6)

    def test_matches_naive_formula(self, calibrated_ens):
        """O(M log M) sorted-identity must equal the naive O(M²) formula."""
        truth, ens = calibrated_ens
        np.testing.assert_allclose(
            crps_ensemble(truth, ens),
            self._naive(truth, ens),
            rtol=1e-5, atol=1e-6,
        )

    def test_known_two_member_case(self):
        """For M=2, CRPS = (|y-a| + |y-b|)/2 - |a-b|/4 (closed form)."""
        truth = np.array([[0.0, 0.0]])           # (1, 2)
        a, b  = 1.0, 3.0
        ens   = np.array([[[a, a], [b, b]]])     # (1, 2, 2)
        expected = (np.abs(np.array([a, b])).mean() - np.abs(a - b) / 4)
        np.testing.assert_allclose(
            crps_ensemble(truth, ens),
            expected,
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# variogram_score
# ---------------------------------------------------------------------------

class TestVariogramScore:

    def test_returns_float(self, calibrated_ens):
        truth, ens = calibrated_ens
        assert isinstance(variogram_score(truth, ens), float)

    def test_non_negative(self, calibrated_ens):
        truth, ens = calibrated_ens
        assert variogram_score(truth, ens) >= 0.0

    def test_chunk_invariant(self, calibrated_ens):
        """Different chunk sizes must give the same result."""
        truth, ens = calibrated_ens
        ref = variogram_score(truth, ens, chunk=N_IC)
        np.testing.assert_allclose(variogram_score(truth, ens, chunk=7),  ref, rtol=1e-6)
        np.testing.assert_allclose(variogram_score(truth, ens, chunk=1),  ref, rtol=1e-6)

    def test_known_two_node_single_ic_single_member(self):
        """D=2, n=1, M=1: VS = (|y0-y1|^p - |e0-e1|^p)^2."""
        p = 0.5
        truth = np.array([[0.0, 4.0]])          # (1, 2)
        ens   = np.array([[[1.0, 3.0]]])        # (1, 1, 2)
        expected = (abs(0.0 - 4.0) ** p - abs(1.0 - 3.0) ** p) ** 2
        np.testing.assert_allclose(variogram_score(truth, ens, p=p), expected, rtol=1e-6)

    def test_calibrated_beats_noise(self, calibrated_ens, clim_ens):
        """A calibrated ensemble should score lower than independent noise."""
        truth, ens = calibrated_ens
        vs_cal  = variogram_score(truth, ens)
        vs_clim = variogram_score(truth, clim_ens)
        assert vs_cal < vs_clim


# ---------------------------------------------------------------------------
# vs_per_ic
# ---------------------------------------------------------------------------

class TestVsPerIc:

    def test_output_shape(self, calibrated_ens):
        truth, ens = calibrated_ens
        out = vs_per_ic(truth, ens)
        assert out.shape == (N_IC,)

    def test_mean_matches_variogram_score(self, calibrated_ens):
        """vs_per_ic().mean() must equal variogram_score() exactly."""
        truth, ens = calibrated_ens
        np.testing.assert_allclose(
            vs_per_ic(truth, ens).mean(),
            variogram_score(truth, ens),
            rtol=1e-5,
        )

    def test_chunk_invariant(self, calibrated_ens):
        truth, ens = calibrated_ens
        ref = vs_per_ic(truth, ens, chunk=N_IC)
        np.testing.assert_allclose(vs_per_ic(truth, ens, chunk=7), ref, rtol=1e-6)
        np.testing.assert_allclose(vs_per_ic(truth, ens, chunk=1), ref, rtol=1e-6)

    def test_non_negative(self, calibrated_ens):
        truth, ens = calibrated_ens
        assert np.all(vs_per_ic(truth, ens) >= 0.0)


# ---------------------------------------------------------------------------
# bootstrap_skill_ci
# ---------------------------------------------------------------------------

class TestBootstrapSkillCi:

    EXPECTED_KEYS = {"crps_lo", "crps_hi", "vs_lo", "vs_hi", "ratio_lo", "ratio_hi"}

    @pytest.fixture(scope="class")
    def ci_result(self, calibrated_ens, clim_ens):
        truth, ens = calibrated_ens
        return bootstrap_skill_ci(truth, ens, clim_ens, n_boot=200, seed=42)

    def test_keys_present(self, ci_result):
        assert set(ci_result.keys()) == self.EXPECTED_KEYS

    def test_lo_le_hi(self, ci_result):
        assert ci_result["crps_lo"]  <= ci_result["crps_hi"]
        assert ci_result["vs_lo"]    <= ci_result["vs_hi"]
        assert ci_result["ratio_lo"] <= ci_result["ratio_hi"]

    def test_seeded_reproducible(self, calibrated_ens, clim_ens):
        truth, ens = calibrated_ens
        r1 = bootstrap_skill_ci(truth, ens, clim_ens, n_boot=100, seed=7)
        r2 = bootstrap_skill_ci(truth, ens, clim_ens, n_boot=100, seed=7)
        assert r1 == r2

    def test_different_seeds_differ(self, calibrated_ens, clim_ens):
        truth, ens = calibrated_ens
        r1 = bootstrap_skill_ci(truth, ens, clim_ens, n_boot=100, seed=1)
        r2 = bootstrap_skill_ci(truth, ens, clim_ens, n_boot=100, seed=2)
        assert r1["crps_lo"] != r2["crps_lo"]

    def test_returns_floats(self, ci_result):
        for v in ci_result.values():
            assert isinstance(v, float)
