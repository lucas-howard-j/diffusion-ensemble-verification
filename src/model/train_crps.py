"""
Train a CRPS-optimised generator on the same (condition, gauge SSH) pairs as
the DDPM, so the two SSH emulators differ only in how the distribution is learned.

Model: CRPSGenerator — MLP f(cond, z) (hyperparameters: CRPS_* in src/config.py), z ~ N(0, I); each member is a forward
pass with an independent z.  Loss: fair CRPS (Ferro 2014), summed over the 8
gauges and averaged over the batch, with M members per training sample:

    CRPS_fair = (1/M) Σ_i |x_i − y|  −  1/(2M(M−1)) Σ_{i≠j} |x_i − x_j|

This is a purely marginal (per-gauge) objective: nothing in it constrains the
dependence between gauges.  Mirrors lorenz96/train_crps.py.

With --loss es the same generator is trained on the fair energy score instead,
a multivariate objective over the 8-gauge vector (Gneiting & Raftery 2007):

    ES_fair = (1/M) Σ_i ||x_i − y||  −  1/(2M(M−1)) Σ_{i≠j} ||x_i − x_j||

Everything else (data, network, optimiser, early stopping) is shared, so the
CRPS- and ES-trained generators differ only in the training score.

Run as a script:
    PYTHONPATH=. python -m src.model.train_crps --seed 0
    PYTHONPATH=. python -m src.model.train_crps --seed 0 --loss es
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from src.config import (
    BATCH_SIZE, WEIGHT_DECAY, MAX_EPOCHS, GRAD_CLIP, CHECKPOINTS,
    CRPS_N_MEMBERS, CRPS_NOISE_DIM, CRPS_HIDDEN, CRPS_LR, CRPS_DROPOUT, CRPS_COND_NOISE, CRPS_EMA_DECAY, CRPS_LINEAR_SKIP,
    CRPS_PATIENCE,
)
from src.model.dataset import SSHDiffusionDataset
from src.model.network import CRPSGenerator
from src.model.train import _update_ema

N_VAL_MEMBERS = 16   # more members at validation for a lower-variance estimate


def fair_crps_loss(members: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fair CRPS summed over gauges, mean over batch.  members (B, M, d), target (B, d)."""
    M = members.shape[1]
    assert M >= 2, "fair CRPS requires M >= 2"
    term1 = (members - target.unsqueeze(1)).abs().mean(dim=1)                    # (B, d)
    pair  = (members.unsqueeze(2) - members.unsqueeze(1)).abs().sum(dim=(1, 2))  # (B, d)
    return (term1 - pair / (2 * M * (M - 1))).sum(-1).mean()


def fair_es_loss(members: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Fair energy score over the gauge vector, mean over batch.  members (B, M, d), target (B, d).

    eps keeps the norm's gradient finite when two vectors coincide."""
    M = members.shape[1]
    assert M >= 2, "fair ES requires M >= 2"
    def norm(v):
        return torch.sqrt((v ** 2).sum(-1) + eps)
    term1 = norm(members - target.unsqueeze(1)).mean(dim=1)                     # (B,)
    i, j = torch.triu_indices(M, M, offset=1)
    pair = norm(members[:, i] - members[:, j]).sum(dim=1)                       # (B,) each unordered pair once
    return (term1 - pair / (M * (M - 1))).mean()


LOSSES = {"crps": fair_crps_loss, "es": fair_es_loss}


def sample_with_fixed_noise(model, cond: torch.Tensor, n_members: int, seed: int) -> torch.Tensor:
    """Draw members with a fixed seed (same validation noise every epoch) without
    touching the global RNG stream that drives training shuffles and noise.

    Reseeding the global RNG here instead would make every training epoch replay
    the same permutation and noise draws."""
    devices = [] if cond.device.type == "cpu" else None   # None: fork all CUDA devices too
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return model.sample(cond, n_members)


def _stack(ds: SSHDiffusionDataset) -> tuple[torch.Tensor, torch.Tensor]:
    conds, targets = zip(*(ds[i] for i in range(len(ds))))
    return torch.stack(conds), torch.stack(targets)


def train(seed: int = 0, n_members: int = CRPS_N_MEMBERS, device: str = "cpu",
          out_path=None, lr: float = CRPS_LR, dropout: float = CRPS_DROPOUT,
          hidden: tuple[int, ...] = CRPS_HIDDEN, weight_decay: float = WEIGHT_DECAY,
          cond_noise: float = CRPS_COND_NOISE, ema_decay: float = CRPS_EMA_DECAY,
          linear_skip: bool = CRPS_LINEAR_SKIP, loss: str = "crps") -> None:
    """cond_noise: SD of Gaussian noise added to the (z-scored) condition during
    training only — input-noise regularisation against memorising the ~1300
    distinct initial states.  ema_decay > 0 validates and saves EMA weights
    (as the DDPM does); 0 uses the raw weights.  loss: "crps" or "es" — the
    training and early-stopping score; val fair CRPS is logged either way."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loss_fn = LOSSES[loss]
    out_path = out_path or CHECKPOINTS / f"{loss}_gen_seed{seed}.pt"

    train_ds = SSHDiffusionDataset(split="train")
    val_ds   = SSHDiffusionDataset(split="val")
    Xtr, Ytr = _stack(train_ds)
    Xva, Yva = _stack(val_ds)
    Xva, Yva = Xva.to(device), Yva.to(device)
    print(f"Train pairs: {len(Xtr)}  |  Val pairs: {len(Xva)}  |  M={n_members}  seed={seed}  loss={loss}")

    model = CRPSGenerator(
        target_dim=train_ds.target_dim,
        cond_dim=train_ds.cond_dim,
        noise_dim=CRPS_NOISE_DIM,
        hidden=hidden,
        dropout=dropout,
        use_beta_silu=True,
        linear_skip=linear_skip,
    ).to(device)
    ema_model = copy.deepcopy(model).eval() if ema_decay > 0 else model
    for p in ema_model.parameters() if ema_decay > 0 else []:
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    best_val, patience_counter, best_state, best_epoch = float("inf"), 0, None, -1
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        train_losses = []
        for k in range(0, len(perm) - BATCH_SIZE + 1, BATCH_SIZE):   # drop_last
            idx = perm[k:k + BATCH_SIZE]
            cond, target = Xtr[idx].to(device), Ytr[idx].to(device)
            if cond_noise > 0:
                cond = cond + cond_noise * torch.randn_like(cond)
            batch_loss = loss_fn(model.sample(cond, n_members), target)
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if ema_decay > 0:
                _update_ema(ema_model, model, ema_decay)
            train_losses.append(batch_loss.item())

        model.eval()
        with torch.no_grad():
            va_ens = sample_with_fixed_noise(ema_model, Xva, N_VAL_MEMBERS, 10_000 + seed)
            va = loss_fn(va_ens, Yva).item()
            va_crps = fair_crps_loss(va_ens, Yva).item()
        tr = float(np.mean(train_losses))
        scheduler.step(va)

        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:4d}  train={tr:.4f}  val={va:.4f}  val_crps={va_crps:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)", flush=True)

        if va < best_val:
            best_val, best_val_crps, patience_counter, best_epoch = va, va_crps, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CRPS_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type":  "crps_generator",   # loader key; the training score is in "loss"
        "loss":        loss,
        "epoch":       best_epoch,
        "model":       best_state,
        "val_loss":    best_val,
        "val_crps":    best_val_crps,
        "seed":        seed,
        "n_members":   n_members,
        "noise_dim":   CRPS_NOISE_DIM,
        "sst_mu":    train_ds.sst_mu,    "sst_std":    train_ds.sst_std,
        "ssh_mu":    train_ds.ssh_mu,    "ssh_std":    train_ds.ssh_std,
        "gauge_mu":  train_ds.gauge_mu,  "gauge_std":  train_ds.gauge_std,
        "cond_dim":      train_ds.cond_dim,
        "target_dim":    train_ds.target_dim,
        "hidden_dims":   hidden,
        "weight_decay":  weight_decay,
        "cond_noise":    cond_noise,
        "ema_decay":     ema_decay,
        "linear_skip":   linear_skip,
        "dropout":       dropout,
        "lr":            lr,
        "use_beta_silu": True,
    }, out_path)
    print(f"Best val fair {loss.upper()}: {best_val:.4f}  (val fair CRPS {best_val_crps:.4f}, "
          f"epoch {best_epoch})  →  {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--n-members", type=int, default=CRPS_N_MEMBERS)
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--lr",        type=float, default=CRPS_LR)
    parser.add_argument("--dropout",   type=float, default=CRPS_DROPOUT)
    parser.add_argument("--hidden",    type=int, nargs="+", default=list(CRPS_HIDDEN))
    parser.add_argument("--wd",        type=float, default=WEIGHT_DECAY)
    parser.add_argument("--cond-noise", type=float, default=CRPS_COND_NOISE)
    parser.add_argument("--ema",       type=float, default=CRPS_EMA_DECAY)
    parser.add_argument("--linear-skip", action=argparse.BooleanOptionalAction,
                        default=CRPS_LINEAR_SKIP)
    parser.add_argument("--loss",      choices=sorted(LOSSES), default="crps")
    parser.add_argument("--out",       type=Path, default=None)
    args = parser.parse_args()
    train(args.seed, args.n_members, args.device, args.out, args.lr, args.dropout,
          tuple(args.hidden), args.wd, args.cond_noise, args.ema, args.linear_skip, args.loss)


if __name__ == "__main__":
    main()
