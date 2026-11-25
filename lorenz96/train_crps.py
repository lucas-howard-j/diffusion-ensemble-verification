"""
Train a CRPS-optimised L96 generative emulator.

Architecture: ConvGenerativeNet — same circular conv + residual trunk as
ConvDeterministicNet, with a full-field Gaussian noise channel as extra input.
Each ensemble member is a forward pass with an independent noise draw.

Loss: fair CRPS (Ferro 2014) averaged over nodes and batch, using M members
per training sample:

    CRPS_fair = (1/M) Σ_i |x_i − y|  −  1/(2M(M−1)) Σ_{i≠j} |x_i − x_j|

The fair estimator is unbiased for the population CRPS at finite M; the plug-in
1/(2M²) form under-rewards spread by (M−1)/M and gives under-dispersed
ensembles at small M.  This is a purely marginal (per-node) objective, the
standard choice for CRPS-trained weather emulators.

With --loss es the same generator is trained on the fair energy score over the
full 40-node state instead (the w=40 case of emulator-geometry's patch-ES sweep):

    ES_fair = (1/M) Σ_i ||x_i − y||  −  1/(2M(M−1)) Σ_{i≠j} ||x_i − x_j||

Everything else (data, network, optimiser, early stopping) is shared.

Usage:
    python -m lorenz96.train_crps [--data PATH] [--n-members M] [--loss crps|es] [--out CKPT]
        [--seed S] [--hidden-ch C] [--n-res R] [--kernel K] [--noise-ch Z]
        [--lr LR] [--batch-size B] [--max-epochs E]

The architecture/optimiser flags default to lorenz96/config.py and exist for
the one-at-a-time ablation (scripts/run_l96_ablation.sh).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from lorenz96.config import (
    LEAD_STEPS, BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE,
    HIDDEN_CH, N_RES, KERNEL,
    TRAIN_END, VAL_START, VAL_END,
)
from lorenz96.dataset import L96Dataset, load_trajectory
from lorenz96.model import ConvGenerativeNet


def fair_crps_loss(members: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean fair CRPS.  members: (B, M, N), target: (B, N) → scalar."""
    M = members.shape[1]
    assert M >= 2, "fair CRPS requires M >= 2"
    term1 = (members - target.unsqueeze(1)).abs().mean(dim=1)                  # (B, N)
    pair  = (members.unsqueeze(2) - members.unsqueeze(1)).abs().sum(dim=(1, 2))  # (B, N)
    term2 = pair / (2 * M * (M - 1))
    return (term1 - term2).mean()


def fair_es_loss(members: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean fair energy score over the full state.  members: (B, M, N), target: (B, N) → scalar.

    eps keeps the norm's gradient finite when two vectors coincide."""
    M = members.shape[1]
    assert M >= 2, "fair ES requires M >= 2"
    def norm(v):
        return torch.sqrt((v ** 2).sum(-1) + eps)
    term1 = norm(members - target.unsqueeze(1)).mean(dim=1)               # (B,)
    i, j = torch.triu_indices(M, M, offset=1)
    pair = norm(members[:, i] - members[:, j]).sum(dim=1)                 # (B,) each unordered pair once
    return (term1 - pair / (M * (M - 1))).mean()


LOSSES = {"crps": fair_crps_loss, "es": fair_es_loss}


def generate_ensemble(model, cond, n_members, device):
    """cond: (B, N) → members: (B, n_members, N)."""
    B, N = cond.shape
    cond_rep = cond.repeat_interleave(n_members, dim=0)   # (B*M, N)
    z = model.sample_latent(B * n_members, device)
    return model(cond_rep, z).view(B, n_members, N)


def train_crps(
    data_path:  str = "lorenz96/l96_trajectory.npz",
    n_members:  int = 2,
    n_train:    int | None = None,
    lead_steps: int = LEAD_STEPS,
    out_path:   str | None = None,    # default: lorenz96/checkpoints/{loss}_emulator.pt
    device:     str | None = None,
    seed:       int = 0,
    loss:       str = "crps",
    hidden_ch:  int = HIDDEN_CH,
    n_res:      int = N_RES,
    kernel:     int = KERNEL,
    lr:         float = LR,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    noise_ch:   int = 1,
):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    torch.manual_seed(seed)
    loss_fn = LOSSES[loss]
    out_path = out_path or f"lorenz96/checkpoints/{loss}_emulator.pt"

    states, data_cfg = load_trajectory(data_path)
    dt = data_cfg["dt"]
    print(f"Trajectory: {len(states):,} steps ({len(states)*dt:.0f} TU)  N={data_cfg['N']}")

    pool = states[:TRAIN_END]
    val  = states[VAL_START:VAL_END]

    max_train_pairs = len(pool) - lead_steps
    if n_train is None:
        n_train = max_train_pairs
    assert n_train <= max_train_pairs, \
        f"n_train={n_train} exceeds pool capacity {max_train_pairs}"

    train_states = pool[:n_train + lead_steps]
    train_ds     = L96Dataset(train_states, lead_steps=lead_steps)
    val_ds_full  = L96Dataset(val, lead_steps=lead_steps, mu=train_ds.mu, sigma=train_ds.sigma)

    val_sub_idx = np.random.default_rng(0).choice(
        len(val_ds_full), size=min(5_000, len(val_ds_full)), replace=False
    )
    val_ds = torch.utils.data.Subset(val_ds_full, val_sub_idx)

    print(f"n_train={n_train:,}  n_val={len(val_ds)}  "
          f"lead={lead_steps} steps ({lead_steps*dt:.2f} TU)  M={n_members}  loss={loss}  device={device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = ConvGenerativeNet(
        hidden_ch=hidden_ch, n_res=n_res, kernel_size=kernel,
        noise_ch=noise_ch, state_dim=train_ds.cond.shape[1],
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, patience_ct, best_state = float("inf"), 0, None
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for cond, target in train_loader:
            cond, target = cond.to(device), target.to(device)
            batch_loss = loss_fn(generate_ensemble(model, cond, n_members, device), target)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item() * len(cond)
        train_loss /= len(train_ds)

        # Validation uses more members for a lower-variance estimate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond, target in val_loader:
                cond, target = cond.to(device), target.to(device)
                members = generate_ensemble(model, cond, max(n_members, 8), device)
                val_loss += loss_fn(members, target).item() * len(cond)
        val_loss /= len(val_ds)

        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

        if val_loss < best_val:
            best_val    = val_loss
            patience_ct = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"Early stop at epoch {epoch}  best val={best_val:.5f}", flush=True)
                break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "model_state": best_state,
        "mu":          train_ds.mu,
        "sigma":       train_ds.sigma,
        "n_train":     n_train,
        "lead_steps":  lead_steps,
        "val_loss":    best_val,
        "n_members":   n_members,
        "loss":        loss,
        "seed":        seed,
        "epochs":      epoch,
        "config": json.dumps({"hidden_ch": hidden_ch, "n_res": n_res, "kernel": kernel,
                              "noise_ch": noise_ch, "lr": lr, "batch_size": batch_size}),
    }, out_path)
    print(f"Saved → {out_path}  (best val={best_val:.5f})")
    return model, train_ds.mu, train_ds.sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="lorenz96/l96_trajectory.npz")
    parser.add_argument("--n-members", type=int, default=2)
    parser.add_argument("--n_train",   type=int, default=None)
    parser.add_argument("--lead",      type=int, default=LEAD_STEPS)
    parser.add_argument("--loss",      choices=sorted(LOSSES), default="crps")
    parser.add_argument("--out",       default=None,
                        help="default: lorenz96/checkpoints/{loss}_emulator.pt")
    parser.add_argument("--device",    default=None)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--noise-ch",   type=int,   default=1)
    parser.add_argument("--hidden-ch",  type=int,   default=HIDDEN_CH)
    parser.add_argument("--n-res",      type=int,   default=N_RES)
    parser.add_argument("--kernel",     type=int,   default=KERNEL)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int,   default=MAX_EPOCHS)
    args = parser.parse_args()

    train_crps(args.data, args.n_members, args.n_train, args.lead,
               args.out, args.device, args.seed, args.loss,
               noise_ch=args.noise_ch, hidden_ch=args.hidden_ch, n_res=args.n_res, kernel=args.kernel,
        lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
