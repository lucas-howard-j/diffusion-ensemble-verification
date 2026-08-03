"""Tests for src.model.network — shape and basic behavior of the MLP denoiser.

No data loading. All tests use synthetic tensors.
"""
import pytest
import torch
import torch.nn as nn

from src.model.network import MLPDenoiser, SinusoidalTimeEmbedding, BetaSiLU
from src.model.dataset import SSHDiffusionDataset

_ds = object.__new__(SSHDiffusionDataset)
TARGET_DIM = _ds.target_dim
COND_DIM = _ds.cond_dim
TIME_EMBED_DIM = 128
BATCH = 4


class TestSinusoidalTimeEmbedding:

    @pytest.fixture
    def emb(self):
        return SinusoidalTimeEmbedding(dim=TIME_EMBED_DIM)

    def test_output_shape(self, emb):
        t = torch.randint(0, 1000, (BATCH,))
        assert emb(t).shape == (BATCH, TIME_EMBED_DIM)

    def test_values_bounded(self, emb):
        """Output is pure sin/cos concatenation — all values must be in [-1, 1] exactly."""
        t = torch.arange(0, 1000, 100)
        out = emb(t)
        assert out.abs().max().item() <= 1.0

    def test_output_finite(self, emb):
        t = torch.randint(0, 1000, (BATCH,))
        assert torch.all(torch.isfinite(emb(t)))

    def test_different_t_gives_different_output(self, emb):
        e0 = emb(torch.tensor([0]))
        e500 = emb(torch.tensor([500]))
        assert not torch.allclose(e0, e500)

    def test_deterministic(self, emb):
        t = torch.tensor([42])
        assert torch.allclose(emb(t), emb(t))


class TestBetaSiLU:

    def test_output_shape(self):
        act = BetaSiLU()
        x = torch.randn(BATCH, 64)
        assert act(x).shape == (BATCH, 64)

    def test_exactly_one_trainable_param(self):
        """Scalar β — exactly 1 trainable parameter."""
        act = BetaSiLU()
        params = list(act.parameters())
        assert len(params) == 1
        assert params[0].shape == ()

    def test_beta_1_matches_silu(self):
        """At default init (β=1), BetaSiLU(x) == SiLU(x)."""
        act = BetaSiLU()
        silu = nn.SiLU()
        x = torch.randn(BATCH, 64)
        torch.testing.assert_close(act(x), silu(x), atol=1e-5, rtol=1e-5)

    def test_output_finite(self):
        act = BetaSiLU()
        x = torch.randn(BATCH, 64) * 10
        assert torch.all(torch.isfinite(act(x)))


class TestMLPDenoiser:

    @pytest.fixture(scope="class")
    def model(self):
        return MLPDenoiser(
            target_dim=TARGET_DIM,
            cond_dim=COND_DIM,
            time_embed_dim=TIME_EMBED_DIM,
            hidden=(64, 64),
            dropout=0.0,
        )

    def _inputs(self, batch=BATCH):
        x_t = torch.randn(batch, TARGET_DIM)
        t = torch.randint(0, 1000, (batch,))
        cond = torch.randn(batch, COND_DIM)
        return x_t, t, cond

    def test_output_shape(self, model):
        x_t, t, cond = self._inputs()
        assert model(x_t, t, cond).shape == (BATCH, TARGET_DIM)

    def test_batch_size_1(self, model):
        x_t, t, cond = self._inputs(batch=1)
        assert model(x_t, t, cond).shape == (1, TARGET_DIM)

    def test_output_finite(self, model):
        assert torch.all(torch.isfinite(model(*self._inputs())))

    def test_gradient_flows(self):
        """Backward pass produces finite gradients for all parameters."""
        m = MLPDenoiser(
            target_dim=TARGET_DIM, cond_dim=COND_DIM, hidden=(64,), dropout=0.0
        )
        x_t = torch.randn(BATCH, TARGET_DIM)
        t = torch.randint(0, 100, (BATCH,))
        cond = torch.randn(BATCH, COND_DIM)
        loss = m(x_t, t, cond).mean()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.all(torch.isfinite(p.grad)), f"Inf/NaN gradient for {name}"

    def test_with_beta_silu(self):
        m = MLPDenoiser(
            target_dim=TARGET_DIM, cond_dim=COND_DIM, hidden=(64,),
            dropout=0.0, use_beta_silu=True
        )
        x_t, t, cond = torch.randn(BATCH, TARGET_DIM), torch.randint(0, 100, (BATCH,)), torch.randn(BATCH, COND_DIM)
        assert m(x_t, t, cond).shape == (BATCH, TARGET_DIM)

    def test_wrong_cond_dim_raises(self, model):
        """Mismatched cond_dim should raise a RuntimeError from the linear layer."""
        x_t = torch.randn(BATCH, TARGET_DIM)
        t = torch.randint(0, 100, (BATCH,))
        bad_cond = torch.randn(BATCH, COND_DIM + 1)
        with pytest.raises(RuntimeError):
            model(x_t, t, bad_cond)
