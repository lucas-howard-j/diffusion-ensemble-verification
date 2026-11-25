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


class CRPSGenerator(nn.Module):
    """
    Noise-conditioned MLP generator trained with the fair CRPS (no diffusion).

    Input  : (condition, z) with z ~ N(0, I_noise_dim)
    Output : one ensemble member of shape (target_dim,)

    Same hidden widths / activation / LayerNorm as MLPDenoiser so the two
    SSH emulators differ only in how the distribution is learned.  Dropout
    defaults to 0: each training member is a separate forward pass, so
    dropout masks would act as extra, train-only noise inside the CRPS
    spread term and leave the eval-time ensemble under-dispersed.
    """

    def __init__(self,
                 target_dim: int,
                 cond_dim: int,
                 noise_dim: int = 8,
                 hidden: tuple[int, ...] = (256, 256, 128),
                 dropout: float = 0.0,
                 use_beta_silu: bool = False,
                 linear_skip: bool = False):
        super().__init__()
        self.noise_dim = noise_dim
        # optional linear map cond → output added to the MLP (a ridge-like mean path)
        self.skip = nn.Linear(cond_dim, target_dim) if linear_skip else None
        layers = []
        prev = cond_dim + noise_dim
        for h in hidden:
            act = BetaSiLU() if use_beta_silu else nn.SiLU()
            layers.extend([nn.Linear(prev, h), act, nn.LayerNorm(h), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([cond, z], dim=-1))
        return out if self.skip is None else out + self.skip(cond)

    def sample(self, cond: torch.Tensor, n_members: int) -> torch.Tensor:
        """cond (B, cond_dim) → members (B, n_members, target_dim)."""
        B = cond.shape[0]
        cond_rep = cond.repeat_interleave(n_members, dim=0)
        z = torch.randn(B * n_members, self.noise_dim, device=cond.device)
        return self.forward(cond_rep, z).view(B, n_members, -1)
