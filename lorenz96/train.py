"""
Train the L96 DDPM.

Usage:
    python -m lorenz96.train [--data PATH] [--n_train N] [--lead L] [--out CKPT]
                             [--seed S] [--hidden-ch C] [--n-res R] [--kernel K]
                             [--emb-dim E] [--lr LR] [--batch-size B]
                             [--beta-schedule linear|cosine] [--max-epochs E]

--n_train controls training set size, enabling the data-volume sweep.
The architecture/optimiser flags default to lorenz96/config.py and exist for
the one-at-a-time ablation (scripts/run_l96_ablation.sh).

Fixed split (from config):
    Training pool : steps 0          – TRAIN_END-1
    gap           : TRAIN_END        – VAL_START-1
    Val           : VAL_START        – VAL_END-1
    gap           : VAL_END          – TEST_START-1
    Test          : TEST_START       – end
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
    HIDDEN_CH, N_RES, EMB_DIM, KERNEL, T_DIFF,
    TRAIN_END, VAL_START, VAL_END, TEST_START,
)
from lorenz96.dataset import L96Dataset, load_trajectory
from lorenz96.model import DDPM, ConvScoreNet


def train(
    data_path:  str = "lorenz96/l96_trajectory.npz",
    n_train:    int | None = None,
    lead_steps: int = LEAD_STEPS,
    out_path:   str = "lorenz96/checkpoints/ddpm.pt",
    device:     str | None = None,
    seed:       int | None = None,
    hidden_ch:  int = HIDDEN_CH,
    n_res:      int = N_RES,
    kernel:     int = KERNEL,
    emb_dim:    int = EMB_DIM,
    lr:         float = LR,
    batch_size: int = BATCH_SIZE,
    beta_schedule: str = "linear",
    max_epochs: int = MAX_EPOCHS,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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

    train_ds    = L96Dataset(train_states, lead_steps=lead_steps)
    val_ds_full = L96Dataset(val, lead_steps=lead_steps, mu=train_ds.mu, sigma=train_ds.sigma)

    val_sub_idx = np.random.default_rng(0).choice(
        len(val_ds_full), size=min(5_000, len(val_ds_full)), replace=False
    )
    val_ds = torch.utils.data.Subset(val_ds_full, val_sub_idx)

    print(f"n_train={n_train:,}  n_val_stop={len(val_ds)}  n_val_full={len(val_ds_full):,}")
    print(f"lead={lead_steps} steps ({lead_steps*dt:.2f} TU)  device={device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    net   = ConvScoreNet(hidden_ch=hidden_ch, n_res=n_res, emb_dim=emb_dim, kernel_size=kernel)
    model = DDPM(net, T=T_DIFF, beta_schedule=beta_schedule).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, patience_ct, best_state = float("inf"), 0, None
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for cond, target in train_loader:
            cond, target = cond.to(device), target.to(device)
            loss = model.loss(target, cond)
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
                val_loss += model.loss(target, cond).item() * len(cond)
        val_loss /= len(val_ds)

        if epoch % 20 == 0:
            print(f"epoch {epoch:4d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

        if val_loss < best_val:
            best_val    = val_loss
            patience_ct = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"Early stop at epoch {epoch}  best val={best_val:.4f}", flush=True)
                break

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "model_state": best_state,
        "mu":          train_ds.mu,
        "sigma":       train_ds.sigma,
        "n_train":     n_train,
        "lead_steps":  lead_steps,
        "val_loss":    best_val,
        "epochs":      epoch,
        "seed":        seed,
        "config": json.dumps({
            "hidden_ch": hidden_ch, "n_res": n_res,
            "emb_dim": emb_dim, "kernel": kernel, "T_diff": T_DIFF,
            "beta_schedule": beta_schedule, "lr": lr, "batch_size": batch_size,
        }),
    }, out_path)
    print(f"Saved → {out_path}  (best val={best_val:.4f})")
    return model, train_ds.mu, train_ds.sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="lorenz96/l96_trajectory.npz")
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--lead",    type=int, default=LEAD_STEPS)
    parser.add_argument("--out",     default="lorenz96/checkpoints/ddpm.pt")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--hidden-ch",  type=int,   default=HIDDEN_CH)
    parser.add_argument("--n-res",      type=int,   default=N_RES)
    parser.add_argument("--kernel",     type=int,   default=KERNEL)
    parser.add_argument("--emb-dim",    type=int,   default=EMB_DIM)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--beta-schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--max-epochs", type=int,   default=MAX_EPOCHS)
    args = parser.parse_args()

    train(args.data, args.n_train, args.lead, args.out, args.device,
          seed=args.seed, hidden_ch=args.hidden_ch, n_res=args.n_res,
          kernel=args.kernel, emb_dim=args.emb_dim, lr=args.lr,
          batch_size=args.batch_size, beta_schedule=args.beta_schedule,
          max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
