"""
Train a deterministic (MSE) L96 emulator.

Architecture: ConvDeterministicNet — same circular conv + residual structure
as ConvScoreNet but without timestep embedding (direct cond → target map).

Usage:
    python -m lorenz96.train_deterministic [--data PATH] [--n_train N] [--lead L] [--out CKPT]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lorenz96.config import (
    LEAD_STEPS, BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE,
    HIDDEN_CH, N_RES, KERNEL,
    TRAIN_END, VAL_START, VAL_END,
)
from lorenz96.dataset import L96Dataset, load_trajectory
from lorenz96.model import ConvDeterministicNet


def train_deterministic(
    data_path:  str = "lorenz96/data/l96_trajectory.npz",
    n_train:    int | None = None,
    lead_steps: int = LEAD_STEPS,
    out_path:   str = "lorenz96/checkpoints/det_emulator.pt",
    device:     str | None = None,
):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

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
          f"lead={lead_steps} steps ({lead_steps*dt:.2f} TU)  device={device}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ConvDeterministicNet(hidden_ch=HIDDEN_CH, n_res=N_RES, kernel_size=KERNEL).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, patience_ct, best_state = float("inf"), 0, None
    t0 = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for cond, target in train_loader:
            cond, target = cond.to(device), target.to(device)
            loss = F.mse_loss(model(cond), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(cond)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond, target in val_loader:
                cond, target = cond.to(device), target.to(device)
                val_loss += F.mse_loss(model(cond), target).item() * len(cond)
        val_loss /= len(val_ds)

        if epoch % 20 == 0:
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
        "config": json.dumps({"hidden_ch": HIDDEN_CH, "n_res": N_RES, "kernel": KERNEL}),
    }, out_path)
    print(f"Saved → {out_path}  (best val={best_val:.5f})")
    return model, train_ds.mu, train_ds.sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="lorenz96/data/l96_trajectory.npz")
    parser.add_argument("--n_train", type=int,   default=None)
    parser.add_argument("--lead",    type=int,   default=LEAD_STEPS)
    parser.add_argument("--out",     default="lorenz96/checkpoints/det_emulator.pt")
    parser.add_argument("--device",  default=None)
    args = parser.parse_args()

    train_deterministic(args.data, args.n_train, args.lead, args.out, args.device)


if __name__ == "__main__":
    main()
