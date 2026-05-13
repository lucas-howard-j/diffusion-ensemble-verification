"""MLP denoiser for DDPM in PC space."""
import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding of the diffusion timestep."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) integer timesteps
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class BetaSiLU(nn.Module):
    """
    SiLU with a single trainable scalar β per layer:
        f(x) = x * sigmoid(β * x)
    β=1 recovers standard SiLU. One parameter per layer instance.
    """

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class MLPDenoiser(nn.Module):
    """
    MLP that predicts the noise ε added to x_0 at diffusion step t.

    Input  : (x_t, t, condition)
    Output : predicted noise of same shape as x_t

    Parameters
    ----------
    use_beta_silu : if True, replace fixed SiLU with trainable BetaSiLU
                    (one β per hidden channel, init=1).
    """

    def __init__(self,
                 target_dim: int,
                 cond_dim: int,
                 time_embed_dim: int = 128,
                 hidden: tuple[int, ...] = (256, 256, 128),
                 dropout: float = 0.1,
                 use_beta_silu: bool = False):
        super().__init__()
        self.use_beta_silu = use_beta_silu
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        in_dim = target_dim + time_embed_dim + cond_dim
        layers = []
        prev = in_dim
        for h in hidden:
            act = BetaSiLU() if use_beta_silu else nn.SiLU()
            layers.extend([
                nn.Linear(prev, h),
                act,
                nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        h = torch.cat([x_t, te, cond], dim=-1)
        return self.net(h)
