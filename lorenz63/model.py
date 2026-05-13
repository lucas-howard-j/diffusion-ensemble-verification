"""
DDPM for Lorenz-63 state prediction.

Architecture
------------
ScoreNet: MLP that predicts the noise added to the target state.
  input  = concat(x_noisy[3], cond[3], t_emb[emb_dim])
  output = predicted noise (3,)

DDPM: wraps ScoreNet with the forward/reverse diffusion process.
  - Linear beta schedule
  - Epsilon (noise) parametrization
  - DDPM ancestral sampling
"""

import math
import torch
import torch.nn as nn

from lorenz63.config import (
    T_DIFF, BETA_START, BETA_END, HIDDEN_DIM, N_LAYERS, EMB_DIM,
)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = EMB_DIM):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer diffusion timestep
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        emb_dim: int = EMB_DIM,
    ):
        super().__init__()
        self.t_emb = SinusoidalEmbedding(emb_dim)

        in_dim = state_dim + state_dim + emb_dim   # x_noisy + cond + t_emb
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_noisy: torch.Tensor,   # (B, 3)
        t: torch.Tensor,          # (B,) int diffusion step
        cond: torch.Tensor,       # (B, 3)
    ) -> torch.Tensor:
        h = torch.cat([x_noisy, cond, self.t_emb(t)], dim=-1)
        return self.net(h)


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    def __init__(self, score_net: ScoreNet, T: int = T_DIFF):
        super().__init__()
        self.score_net = score_net
        self.T = T

        betas = torch.linspace(BETA_START, BETA_END, T)          # (T,)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",      betas)
        self.register_buffer("alphas",     alphas)
        self.register_buffer("alpha_bar",  alpha_bar)
        self.register_buffer("sqrt_ab",    alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab",  (1 - alpha_bar).sqrt())

    # -- training ------------------------------------------------------------

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Epsilon-prediction MSE loss on a batch."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self.sqrt_ab[t, None] * x0 + self.sqrt_1mab[t, None] * eps
        eps_pred = self.score_net(x_t, t, cond)
        return (eps_pred - eps).pow(2).mean()

    # -- sampling ------------------------------------------------------------

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """
        Draw n_samples ensemble members for each conditioning state in cond.

        Parameters
        ----------
        cond      : (B, 3) or (3,) conditioning states
        n_samples : ensemble size

        Returns
        -------
        samples : (B, n_samples, 3)
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        B = cond.shape[0]
        device = cond.device

        # Expand cond to (B*n_samples, 3)
        cond_rep = cond.repeat_interleave(n_samples, dim=0)

        x = torch.randn(B * n_samples, cond.shape[-1], device=device)

        for t_val in reversed(range(self.T)):
            t_tensor = torch.full((B * n_samples,), t_val, device=device, dtype=torch.long)
            eps_pred = self.score_net(x, t_tensor, cond_rep)

            beta_t     = self.betas[t_val]
            alpha_t    = self.alphas[t_val]
            alpha_bar_t = self.alpha_bar[t_val]

            # DDPM reverse step
            mean = (1 / alpha_t.sqrt()) * (
                x - beta_t / (1 - alpha_bar_t).sqrt() * eps_pred
            )
            if t_val > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean

        return x.view(B, n_samples, -1)   # (B, n_samples, 3)
