"""DDPM noise schedule and forward/reverse process utilities."""
import math

import torch
import torch.nn as nn

from src.config import N_DIFFUSION_STEPS, BETA_START, BETA_END, BETA_SCHEDULE


def _cosine_alpha_bars(T: int, s: float = 0.008) -> torch.Tensor:
    """ᾱ_t via cosine schedule (Nichol & Dhariwal 2021, eq. 17)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    return alpha_bars[1:].float()


class DDPMSchedule(nn.Module):
    """DDPM noise schedule. Supports 'linear' and 'cosine' beta schedules."""

    def __init__(self, T: int = N_DIFFUSION_STEPS,
                 beta_start: float = BETA_START, beta_end: float = BETA_END,
                 schedule: str = BETA_SCHEDULE):
        super().__init__()
        if schedule == "cosine":
            alpha_bars = _cosine_alpha_bars(T)
            alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
            betas = torch.clamp(1.0 - alpha_bars / alpha_bars_prev, 0.0, 0.999)
            alphas = 1.0 - betas
        else:
            betas = torch.linspace(beta_start, beta_end, T)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)

        self.T = T
        self.register_buffer("betas",      betas)
        self.register_buffer("alphas",     alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars",   torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward noise: x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε"""
        sa  = self.sqrt_alpha_bars[t].view(-1, 1)
        soa = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sa * x0 + soa * noise

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: int,
                 cond: torch.Tensor) -> torch.Tensor:
        """Single reverse step from x_t → x_{t-1}."""
        device = x_t.device
        t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        eps = model(x_t, t_tensor, cond)

        beta_t      = self.betas[t]
        alpha_t     = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps
        )
        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * noise
        return mean

    @torch.no_grad()
    def sample(self, model, cond: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Full reverse diffusion: start from noise, end at x_0."""
        device = cond.device
        x = torch.randn(cond.shape[0], target_dim, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond)
        return x
