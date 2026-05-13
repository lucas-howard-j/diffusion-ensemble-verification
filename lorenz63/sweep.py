"""
Data-volume × lead-time sensitivity sweep for L63 DDPM.

For each (lead, n_train) combination: trains a model, evaluates on the
fixed test set, records CRPS skill and VS skill vs climatology.

Plot: two panels (CRPS skill, VS skill), one line per lead, log x-axis.

Usage:
    python -m lorenz63.sweep [--out_dir DIR]
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from lorenz63.train import train
from lorenz63.eval import evaluate

LEAD_VALUES   = [300, 500, 700]          # steps  (3, 5, 7 TU)
N_TRAIN_VALUES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_EVAL  = 5_000
N_ENS   = 50


def run_sweep(
    data_path: str = "lorenz63/data/l63_trajectory.npz",
    out_dir:   str = "lorenz63/sweep_T100",
    device:    str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # results[lead_steps][n_train] = {"crps_skill": float, "vs_skill": float}
    results = {lead: {} for lead in LEAD_VALUES}

    for lead in LEAD_VALUES:
        for n_train in N_TRAIN_VALUES:
            print(f"\n{'='*60}")
            print(f"lead={lead} steps ({lead/100:.0f} TU)   n_train={n_train:,}")
            print(f"{'='*60}")

            ckpt_path = os.path.join(ckpt_dir, f"ddpm_lead{lead}_{n_train}.pt")

            train(
                data_path=data_path,
                n_train=n_train,
                lead_steps=lead,
                out_path=ckpt_path,
                device=device,
            )

            metrics = evaluate(
                ckpt_path=ckpt_path,
                data_path=data_path,
                n_eval=N_EVAL,
                n_ens=N_ENS,
                out_path=os.path.join(out_dir, f"skill_lead{lead}_{n_train}.pdf"),
                device=device,
            )

            results[lead][n_train] = {
                "crps_skill": float(metrics["crps_skill"].mean()),
                "vs_skill":   float(metrics["vs_skill"]),
            }

    _save(results, out_dir)
    _plot(results, out_dir)
    return results


def _save(results, out_dir):
    n_leads  = len(LEAD_VALUES)
    n_sizes  = len(N_TRAIN_VALUES)
    crps_arr = np.full((n_leads, n_sizes), np.nan)
    vs_arr   = np.full((n_leads, n_sizes), np.nan)
    for i, lead in enumerate(LEAD_VALUES):
        for j, n in enumerate(N_TRAIN_VALUES):
            crps_arr[i, j] = results[lead][n]["crps_skill"]
            vs_arr[i, j]   = results[lead][n]["vs_skill"]

    path = os.path.join(out_dir, "sweep_results.npz")
    np.savez(path,
             lead_values=LEAD_VALUES,
             n_train_values=N_TRAIN_VALUES,
             crps_skill=crps_arr,
             vs_skill=vs_arr)
    print(f"\nResults saved → {path}")


def _plot(results, out_dir):
    n_train = np.array(N_TRAIN_VALUES)
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    for ax, metric, ylabel in zip(
        axes,
        ["crps_skill", "vs_skill"],
        ["CRPS skill  (1 − CRPS / CRPS$_{clim}$)",
         "VS skill  (1 − VS / VS$_{clim}$)"],
    ):
        for i, lead in enumerate(LEAD_VALUES):
            vals = np.array([results[lead][n][metric] for n in N_TRAIN_VALUES])
            ax.semilogx(n_train, vals, marker=markers[i], color=colors[i],
                        label=f"{lead//100} TU", linewidth=1.5, markersize=5)

        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Training set size  $N$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(n_train)
        ax.set_xticklabels([f"{n:,}" for n in n_train], rotation=30, ha="right", fontsize=8)
        ax.legend(title="Lead", framealpha=0.9)

    axes[0].set_title("CRPS skill vs training size")
    axes[1].set_title("Variogram score skill vs training size")
    fig.suptitle(f"L63 DDPM  |  $n_{{ens}}$={N_ENS}", fontsize=10)
    fig.tight_layout()

    path = os.path.join(out_dir, "skill_vs_ntrain.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Plot saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="lorenz63/data/l63_trajectory.npz")
    parser.add_argument("--out_dir", default="lorenz63/sweep_T100")
    parser.add_argument("--device",  default=None)
    args = parser.parse_args()

    run_sweep(args.data, args.out_dir, args.device)


if __name__ == "__main__":
    main()
