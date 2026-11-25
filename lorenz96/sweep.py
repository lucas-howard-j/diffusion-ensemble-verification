"""
Data-volume × lead-time sensitivity sweep for the L96 DDPM (Figure 4): training.

Trains one DDPM per (lead, n_train) pair; existing checkpoints are skipped, so
the sweep can be resumed or extended.  Lead values span comparable
Lyapunov-time multiples:
  20 steps = 1.0 TU ≈ 1.7 Lyapunov times
  40 steps = 2.0 TU ≈ 3.3 Lyapunov times
  60 steps = 3.0 TU ≈ 5.0 Lyapunov times

Evaluation is done separately, in a single pass over all checkpoints with
shared test ICs and climatology per lead:
    python -m lorenz96.sweep_eval

Output:
    lorenz96/sweep/checkpoints/ddpm_lead{lead}_{n_train}.pt

Usage:
    python -m lorenz96.sweep [--data PATH] [--out_dir DIR] [--device DEVICE] [--evaluate]
"""

import argparse
import os

from lorenz96.train import train
from lorenz96.sweep_eval import LEAD_VALUES, N_TRAIN_VALUES, run as run_eval


def run_sweep(
    data_path: str = "lorenz96/l96_trajectory.npz",
    out_dir:   str = "lorenz96/sweep",
    device:    str | None = None,
    evaluate:  bool = False,
):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for lead in LEAD_VALUES:
        for n_train in N_TRAIN_VALUES:
            ckpt_path = os.path.join(ckpt_dir, f"ddpm_lead{lead}_{n_train}.pt")
            if os.path.exists(ckpt_path):
                print(f"Exists  lead={lead} n_train={n_train:,} — skipping")
                continue
            print(f"\n{'=' * 60}\nlead={lead} steps ({lead * 0.05:.1f} TU)   "
                  f"n_train={n_train:,}\n{'=' * 60}")
            train(data_path=data_path, n_train=n_train, lead_steps=lead,
                  out_path=ckpt_path, device=device)

    if evaluate:
        run_eval(data_path=data_path, out_dir=out_dir, device=device)
    else:
        print("\nTraining done. Evaluate with:  python -m lorenz96.sweep_eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="lorenz96/l96_trajectory.npz")
    parser.add_argument("--out_dir",  default="lorenz96/sweep")
    parser.add_argument("--device",   default=None)
    parser.add_argument("--evaluate", action="store_true",
                        help="Run lorenz96.sweep_eval after training")
    args = parser.parse_args()
    run_sweep(args.data, args.out_dir, args.device, args.evaluate)


if __name__ == "__main__":
    main()
