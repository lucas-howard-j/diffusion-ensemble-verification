"""Tests for lorenz96.model — DetResBlock and ConvDeterministicNet.

No data loading or checkpoints required.  Mirrors test_network.py style.
"""

import pytest
import torch

from lorenz96.model import ConvDeterministicNet, DetResBlock
from lorenz96.config import N, HIDDEN_CH, N_RES, KERNEL

BATCH = 4


# ---------------------------------------------------------------------------
# DetResBlock
# ---------------------------------------------------------------------------

class TestDetResBlock:

    @pytest.fixture
    def block(self):
        return DetResBlock(hidden_ch=HIDDEN_CH, kernel_size=KERNEL)

    def test_output_shape(self, block):
        x = torch.randn(BATCH, HIDDEN_CH, N)
        assert block(x).shape == (BATCH, HIDDEN_CH, N)

    def test_output_finite(self, block):
        x = torch.randn(BATCH, HIDDEN_CH, N)
        assert torch.all(torch.isfinite(block(x)))

    def test_gradient_flows(self, block):
        x = torch.randn(BATCH, HIDDEN_CH, N, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# ---------------------------------------------------------------------------
# ConvDeterministicNet
# ---------------------------------------------------------------------------

class TestConvDeterministicNet:

    @pytest.fixture
    def model(self):
        return ConvDeterministicNet(hidden_ch=HIDDEN_CH, n_res=N_RES, kernel_size=KERNEL)

    def test_output_shape(self, model):
        cond = torch.randn(BATCH, N)
        assert model(cond).shape == (BATCH, N)

    def test_batch_size_1(self, model):
        cond = torch.randn(1, N)
        assert model(cond).shape == (1, N)

    def test_output_finite(self, model):
        cond = torch.randn(BATCH, N)
        assert torch.all(torch.isfinite(model(cond)))

    def test_gradient_flows(self, model):
        cond = torch.randn(BATCH, N, requires_grad=True)
        model(cond).sum().backward()
        assert cond.grad is not None
        assert torch.all(torch.isfinite(cond.grad))

    def test_deterministic_given_same_input(self, model):
        model.eval()
        cond = torch.randn(BATCH, N)
        with torch.no_grad():
            out1 = model(cond)
            out2 = model(cond)
        assert torch.allclose(out1, out2)

    def test_different_inputs_give_different_outputs(self, model):
        model.eval()
        with torch.no_grad():
            out1 = model(torch.randn(BATCH, N))
            out2 = model(torch.randn(BATCH, N))
        assert not torch.allclose(out1, out2)
