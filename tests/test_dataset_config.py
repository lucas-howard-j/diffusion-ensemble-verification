"""Config consistency tests — no data loading.

These tests verify that src/config.py constants are internally consistent
and that the conditioning-vector dimension formula matches the property on
SSHDiffusionDataset. A mismatch silently breaks inference at load time.
"""
import pytest
import numpy as np


class TestConfigConstants:

    def test_lead_weeks(self):
        from src.config import LEAD_WEEKS
        assert sorted(LEAD_WEEKS) == [2, 4, 8, 12]
        assert len(LEAD_WEEKS) == 4

    def test_ssh_lags(self):
        from src.config import SSH_LAGS
        assert sorted(SSH_LAGS) == [0, 1, 2, 4]
        assert len(SSH_LAGS) == 4

    def test_sst_lags(self):
        from src.config import SST_LAGS
        assert sorted(SST_LAGS) == [0, 1, 2]
        assert len(SST_LAGS) == 3

    def test_enso_lags(self):
        from src.config import ENSO_LAGS
        assert sorted(ENSO_LAGS) == [0, 4, 8, 13]
        assert len(ENSO_LAGS) == 4

    def test_nao_lags(self):
        from src.config import NAO_LAGS
        assert sorted(NAO_LAGS) == [0, 1, 2]
        assert len(NAO_LAGS) == 3

    def test_tide_gauge_count(self):
        from src.config import TIDE_GAUGES
        assert len(TIDE_GAUGES) == 8

    def test_tide_gauge_ids_unique(self):
        from src.config import TIDE_GAUGES
        ids = [g["id"] for g in TIDE_GAUGES]
        assert len(ids) == len(set(ids))

    def test_max_lag_equals_13(self):
        from src.config import MAX_LAG, SSH_LAGS, SST_LAGS, ENSO_LAGS, NAO_LAGS
        all_lags = SSH_LAGS + SST_LAGS + ENSO_LAGS + NAO_LAGS
        assert MAX_LAG == max(all_lags)
        assert MAX_LAG == 13

    def test_train_val_test_nonoverlapping(self):
        from src.config import TRAIN_YEARS, VAL_YEARS, TEST_YEARS
        train = set(TRAIN_YEARS)
        val = set(VAL_YEARS)
        test = set(TEST_YEARS)
        assert len(train & val) == 0, "Train/val overlap"
        assert len(train & test) == 0, "Train/test overlap"
        assert len(val & test) == 0, "Val/test overlap"

    def test_n_diffusion_steps(self):
        from src.config import N_DIFFUSION_STEPS
        assert N_DIFFUSION_STEPS == 1000

    def test_mode_counts(self):
        from src.config import N_SSH_MODES, N_SST_MODES, N_WIND_MODES
        assert N_SSH_MODES == 15
        assert N_SST_MODES == 15
        assert N_WIND_MODES == 10


class TestCondDimFormula:

    def test_formula_equals_131(self):
        """Hand-verify: 15*3 + 15*4 + 10 + 3 + 4 + 8 + 1 = 131."""
        from src.config import (
            N_SSH_MODES, N_SST_MODES, N_WIND_MODES,
            SSH_LAGS, SST_LAGS, NAO_LAGS, ENSO_LAGS, TIDE_GAUGES,
        )
        computed = (
            N_SST_MODES * len(SST_LAGS)    # 45
            + N_SSH_MODES * len(SSH_LAGS)  # 60
            + N_WIND_MODES                 # 10
            + len(NAO_LAGS)                # 3
            + len(ENSO_LAGS)               # 4
            + len(TIDE_GAUGES)             # 8
            + 1                            # lead scalar
        )
        assert computed == 131

    def test_cond_dim_property_matches_formula(self):
        """SSHDiffusionDataset.cond_dim property returns 131.

        Called on an uninitialized instance — safe because cond_dim only
        reads from src.config constants, not from self.
        """
        from src.config import (
            N_SSH_MODES, N_SST_MODES, N_WIND_MODES,
            SSH_LAGS, SST_LAGS, NAO_LAGS, ENSO_LAGS, TIDE_GAUGES,
        )
        from src.model.dataset import SSHDiffusionDataset

        expected = (
            N_SST_MODES * len(SST_LAGS)
            + N_SSH_MODES * len(SSH_LAGS)
            + N_WIND_MODES
            + len(NAO_LAGS)
            + len(ENSO_LAGS)
            + len(TIDE_GAUGES)
            + 1
        )
        obj = object.__new__(SSHDiffusionDataset)
        assert obj.cond_dim == expected

    def test_target_dim_property_equals_8(self):
        from src.model.dataset import SSHDiffusionDataset
        obj = object.__new__(SSHDiffusionDataset)
        assert obj.target_dim == 8

    def test_lead_norm_range(self):
        """normalise_lead maps every LEAD_WEEKS value to [0, 1]."""
        from src.config import LEAD_WEEKS, normalise_lead

        for lead in LEAD_WEEKS:
            norm = normalise_lead(lead, LEAD_WEEKS)
            assert 0.0 <= norm <= 1.0, f"lead_norm out of range for lead={lead}"

    def test_lead_norm_min_is_zero(self):
        """The smallest lead normalises to 0."""
        from src.config import LEAD_WEEKS, normalise_lead

        assert abs(normalise_lead(min(LEAD_WEEKS), LEAD_WEEKS)) < 1e-15
