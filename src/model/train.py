"""
Train the DDPM denoiser on (condition, SSH_PCs) pairs.

Run as a script:
    python -m src.model.train
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE,
    GRAD_CLIP, EMA_DECAY, CHECKPOINTS, HIDDEN_DIMS, DROPOUT, BETA_SCHEDULE,
)
from src.model.dataset import SSHDiffusionDataset
from src.model.diffusion import DDPMSchedule
from src.model.network import MLPDenoiser


def _beta_stats(model: nn.Module) -> list[float]:
    """Return scalar β for each BetaSiLU layer in order."""
    from src.model.network import BetaSiLU
    return [m.beta.item() for m in model.modules() if isinstance(m, BetaSiLU)]


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float = EMA_DECAY) -> None:
    with torch.no_grad():
        for pe, p in zip(ema_model.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(device: str | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds = SSHDiffusionDataset(split="train")
    val_ds   = SSHDiffusionDataset(split="val")
    print(f"Train pairs: {len(train_ds)}  |  Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = MLPDenoiser(
        target_dim=train_ds.target_dim,
        cond_dim=train_ds.cond_dim,
        hidden=HIDDEN_DIMS,
        dropout=DROPOUT,
        use_beta_silu=True,
    ).to(device)

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    schedule = DDPMSchedule().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=50, min_lr=1e-7
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS / "ddpm_best_sched50.pt"

    # Beta logging
    beta_log_path = Path("logs/beta_history.csv")
    beta_log_path.parent.mkdir(exist_ok=True)
    n_beta_layers = len([m for m in model.modules()
                         if m.__class__.__name__ == "BetaSiLU"])
    beta_log_file = beta_log_path.open("w", newline="")
    beta_writer = csv.writer(beta_log_file)
    header = ["epoch"] + [f"layer{i}_beta" for i in range(n_beta_layers)]
    beta_writer.writerow(header)

    t0_total = time.time()

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        train_losses = []
        for cond, target in train_loader:
            cond, target = cond.to(device), target.to(device)
            noise = torch.randn_like(target)
            t = torch.randint(0, schedule.T, (target.shape[0],), device=device)
            x_t = schedule.q_sample(target, t, noise)

            pred = model(x_t, t, cond)
            loss = loss_fn(pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            _update_ema(ema_model, model)
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for cond, target in val_loader:
                cond, target = cond.to(device), target.to(device)
                noise = torch.randn_like(target)
                t = torch.randint(0, schedule.T, (target.shape[0],), device=device)
                x_t = schedule.q_sample(target, t, noise)
                pred = model(x_t, t, cond)
                val_losses.append(loss_fn(pred, noise).item())

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        scheduler.step(va)

        # Log beta stats
        beta_writer.writerow([epoch] + [round(b, 6) for b in _beta_stats(model)])
        epoch_secs = time.time() - t0
        if epoch % 10 == 0 or epoch < 5:
            elapsed = time.time() - t0_total
            eta = (elapsed / (epoch + 1)) * (MAX_EPOCHS - epoch - 1)
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}  train={tr:.4f}  val={va:.4f}  "
                  f"lr={lr:.2e}  {epoch_secs:.1f}s/ep  ETA {eta/60:.0f}m")

        if va < best_val:
            best_val = va
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema":   ema_model.state_dict(),
                "val_loss": va,
                "sst_mu":    train_ds.sst_mu,    "sst_std":    train_ds.sst_std,
                "ssh_mu":    train_ds.ssh_mu,    "ssh_std":    train_ds.ssh_std,
                "gauge_mu":  train_ds.gauge_mu,  "gauge_std":  train_ds.gauge_std,
                "cond_dim":      train_ds.cond_dim,
                "target_dim":    train_ds.target_dim,
                "hidden_dims":    HIDDEN_DIMS,
                "dropout":        DROPOUT,
                "beta_schedule":  BETA_SCHEDULE,
                "use_beta_silu":  True,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    beta_log_file.close()
    print(f"Best val loss: {best_val:.4f}  →  {ckpt_path}")
    print(f"Beta history  →  {beta_log_path}")


if __name__ == "__main__":
    train()
