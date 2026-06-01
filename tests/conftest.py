"""Shared fixtures for the SSH diffusion test suite.

All fixtures produce synthetic data with fixed random seeds.
No real data files or model checkpoints are required.
"""
import numpy as np
import pytest

N_MEMBERS = 50
N_STATIONS = 8


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def ensemble_8st(rng):
    """(M, 8) members and (8,) obs — full station count."""
    members = rng.standard_normal((N_MEMBERS, N_STATIONS))
    obs = rng.standard_normal((N_STATIONS,))
    return members, obs


@pytest.fixture
def perfect_ensemble(rng):
    """All members identical to obs; all scores should be zero."""
    obs = rng.standard_normal((N_STATIONS,))
    members = np.tile(obs, (N_MEMBERS, 1))
    return members, obs
