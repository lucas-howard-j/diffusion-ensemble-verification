"""
DDPM for Lorenz-96 state prediction.

Architecture
------------
ConvScoreNet: 1-D convolutional network that predicts the noise added to the
target state.  Circular (cyclic) padding is used throughout to respect the
periodic boundary conditions of L96.

  Input channels : 2  (x_noisy stacked with cond along channel dim)
  Spatial dim    : N  (40 L96 grid points)

  in_proj  : CircularConv1d(2, hidden_ch, kernel)
  blocks   : N_RES × ResBlock(hidden_ch, emb_dim, kernel)
               each block injects the sinusoidal time embedding additively
  out_proj : SiLU → Conv1d(hidden_ch, 1, kernel_size=1)  [pointwise]

DDPM: wraps ConvScoreNet with linear-schedule forward/reverse diffusion.
  - Epsilon (noise) parametrisation
  - DDPM ancestral sampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lorenz96.config import (
    N, T_DIFF, BETA_START, BETA_END, HIDDEN_CH, N_RES, EMB_DIM, KERNEL,
)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding  (identical to L63 version)
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = EMB_DIM):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


# ---------------------------------------------------------------------------
# Circular convolution
# ---------------------------------------------------------------------------

class CircularConv1d(nn.Module):
    """Conv1d with circular padding to respect L96 periodic boundaries."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.pad  = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        x = F.pad(x, (self.pad, self.pad), mode="circular")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, hidden_ch: int, emb_dim: int, kernel_size: int):
        super().__init__()
        self.conv1  = CircularConv1d(hidden_ch, hidden_ch, kernel_size)
        self.t_proj = nn.Linear(emb_dim, hidden_ch)
        self.conv2  = CircularConv1d(hidden_ch, hidden_ch, kernel_size)
        self.act    = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, hidden_ch, N);  t_emb: (B, emb_dim)
        h = self.conv1(self.act(x))
        h = h + self.t_proj(t_emb).unsqueeze(-1)   # broadcast over spatial dim
        h = self.conv2(self.act(h))
        return x + h


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------

class ConvScoreNet(nn.Module):
    def __init__(
        self,
        state_dim:  int = N,
        hidden_ch:  int = HIDDEN_CH,
        n_res:      int = N_RES,
        emb_dim:    int = EMB_DIM,
        kernel_size: int = KERNEL,
    ):
        super().__init__()
        self.t_emb  = SinusoidalEmbedding(emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.in_proj  = CircularConv1d(2, hidden_ch, kernel_size)
        self.blocks   = nn.ModuleList(
            [ResBlock(hidden_ch, emb_dim, kernel_size) for _ in range(n_res)]
        )
        self.out_proj = nn.Sequential(nn.SiLU(), nn.Conv1d(hidden_ch, 1, 1))

    def forward(
        self,
        x_noisy: torch.Tensor,   # (B, N)
        t:       torch.Tensor,   # (B,) int diffusion step
        cond:    torch.Tensor,   # (B, N)
    ) -> torch.Tensor:
        x     = torch.stack([x_noisy, cond], dim=1)   # (B, 2, N)
        t_emb = self.t_proj(self.t_emb(t))             # (B, emb_dim)
        h     = self.in_proj(x)                        # (B, hidden_ch, N)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h).squeeze(1)             # (B, N)


# ---------------------------------------------------------------------------
# DDPM  (identical logic to L63 version; operates on (B, N) tensors)
# ---------------------------------------------------------------------------

def make_betas(T: int, schedule: str = "linear") -> torch.Tensor:
    """Noise schedule.  'linear' is the default used for all paper models;
    'cosine' (Nichol & Dhariwal 2021) is an ablation option."""
    if schedule == "linear":
        return torch.linspace(BETA_START, BETA_END, T)
    if schedule == "cosine":
        s = 0.008
        t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        return (1 - ab[1:] / ab[:-1]).clamp(max=0.999).float()
    raise ValueError(f"unknown beta schedule {schedule!r}")


class DDPM(nn.Module):
    def __init__(self, score_net: ConvScoreNet, T: int = T_DIFF, beta_schedule: str = "linear"):
        super().__init__()
        self.score_net = score_net
        self.T = T
        # The cosine schedule ends at alpha_bar_T ≈ 2e-9, so the first DDIM step
        # amplifies eps error by 1/sqrt(alpha_bar_T) ≈ 2e4 (linear: ≈ 160).
        # Clip the x0 estimate (normalised L96 states lie within ≈ ±3 SD) for
        # cosine only; linear-schedule (paper) models sample exactly as before.
        self.clip_x0 = 5.0 if beta_schedule == "cosine" else None

        betas     = make_betas(T, beta_schedule)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab",   alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab", (1 - alpha_bar).sqrt())

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B  = x0.shape[0]
        t  = torch.randint(0, self.T, (B,), device=x0.device)
        eps    = torch.randn_like(x0)
        x_t    = self.sqrt_ab[t, None] * x0 + self.sqrt_1mab[t, None] * eps
        eps_pred = self.score_net(x_t, t, cond)
        return (eps_pred - eps).pow(2).mean()

    @torch.no_grad()
    def sample(
        self,
        cond:         torch.Tensor,
        n_samples:    int = 50,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cond         : (B, N)
        n_samples    : ensemble size
        sample_steps : number of reverse diffusion steps (default: full T).
                       Values < T use DDIM deterministic updates, which give
                       good samples in 100–200 steps with no retraining.

        Returns
        -------
        samples : (B, n_samples, N)
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        B, state_dim = cond.shape
        device = cond.device

        # Build descending timestep sequence
        if sample_steps is None or sample_steps >= self.T:
            t_seq = list(range(self.T - 1, -1, -1))
        else:
            # evenly-spaced indices into [0, T-1], then reversed
            idx   = torch.linspace(0, self.T - 1, sample_steps + 1).long()
            t_seq = idx.flip(0).tolist()   # length sample_steps+1; last entry is 0

        cond_rep = cond.repeat_interleave(n_samples, dim=0)   # (B*M, N)
        x = torch.randn(B * n_samples, state_dim, device=device)

        for i, t_val in enumerate(t_seq[:-1]):
            t_prev   = t_seq[i + 1]
            t_tensor = torch.full((B * n_samples,), t_val, device=device, dtype=torch.long)
            eps_pred = self.score_net(x, t_tensor, cond_rep)

            ab_t    = self.alpha_bar[t_val]
            ab_prev = self.alpha_bar[t_prev]

            # DDIM update: x0 estimate → re-noise to t_prev level
            x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            if self.clip_x0 is not None:
                x0_pred = x0_pred.clamp(-self.clip_x0, self.clip_x0)
                eps_pred = (x - ab_t.sqrt() * x0_pred) / (1 - ab_t).sqrt()
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred

        return x.view(B, n_samples, state_dim)   # (B, M, N)


# ---------------------------------------------------------------------------
# Deterministic emulator (no diffusion, MSE-trained direct map)
# ---------------------------------------------------------------------------

class DetResBlock(nn.Module):
    """Residual block without timestep embedding."""
    def __init__(self, hidden_ch: int, kernel_size: int):
        super().__init__()
        self.conv1 = CircularConv1d(hidden_ch, hidden_ch, kernel_size)
        self.conv2 = CircularConv1d(hidden_ch, hidden_ch, kernel_size)
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(x))
        h = self.conv2(self.act(h))
        return x + h


class ConvDeterministicNet(nn.Module):
    """Direct cond → target map; no noise or timestep embedding."""
    def __init__(
        self,
        hidden_ch:   int = HIDDEN_CH,
        n_res:       int = N_RES,
        kernel_size: int = KERNEL,
    ):
        super().__init__()
        self.in_proj  = CircularConv1d(1, hidden_ch, kernel_size)
        self.blocks   = nn.ModuleList(
            [DetResBlock(hidden_ch, kernel_size) for _ in range(n_res)]
        )
        self.out_proj = nn.Sequential(nn.SiLU(), nn.Conv1d(hidden_ch, 1, 1))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = cond.unsqueeze(1)               # (B, 1, N)
        h = self.in_proj(x)                  # (B, hidden_ch, N)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h).squeeze(1)   # (B, N)


class ConvGenerativeNet(nn.Module):
    """Noise-conditioned generator: (cond, z) → ensemble member.

    Same trunk as ConvDeterministicNet, with a full-field noise channel
    z ~ N(0, I) concatenated to the conditioning state.  Each member is an
    independent draw of z.  Trained by the CRPS (see train_crps.py) rather
    than score matching or MSE.
    """
    def __init__(
        self,
        hidden_ch:   int = HIDDEN_CH,
        n_res:       int = N_RES,
        kernel_size: int = KERNEL,
        noise_ch:    int = 1,
        state_dim:   int = N,
    ):
        super().__init__()
        self.noise_ch  = noise_ch
        self.state_dim = state_dim
        self.in_proj  = CircularConv1d(1 + noise_ch, hidden_ch, kernel_size)
        self.blocks   = nn.ModuleList(
            [DetResBlock(hidden_ch, kernel_size) for _ in range(n_res)]
        )
        self.out_proj = nn.Sequential(nn.SiLU(), nn.Conv1d(hidden_ch, 1, 1))

    def sample_latent(self, n: int, device) -> torch.Tensor:
        """Draw noise for n members: (n, noise_ch, N)."""
        return torch.randn(n, self.noise_ch, self.state_dim, device=device)

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # cond: (B, N);  z: (B, noise_ch, N)
        x = torch.cat([cond.unsqueeze(1), z], dim=1)   # (B, 1+noise_ch, N)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h).squeeze(1)             # (B, N)
