"""
Data-volume × lead-time sensitivity sweep for L96 DDPM.

Lead values span comparable Lyapunov-time multiples to the L63 testbed:
  20 steps = 1.0 TU ≈ 1.7 Lyapunov times
  40 steps = 2.0 TU ≈ 3.3 Lyapunov times
  60 steps = 3.0 TU ≈ 5.0 Lyapunov times

Resuming: if sweep_results.npz already exists in out_dir, any (lead, n_train)
pair recorded there is loaded from cache and skipped entirely (no retrain, no
re-eval).  This makes it safe to add new N_TRAIN_VALUES and rerun.

Output:
    sweep/sweep_results.npz   — arrays indexed [lead, n_train]
    sweep/skill_vs_ntrain.pdf — two-panel plot matching L63 sweep figure

Usage:
    python -m lorenz96.sweep [--data PATH] [--out_dir DIR] [--device DEVICE]
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from lorenz96.train import train
from lorenz96.eval  import evaluate

LEAD_VALUES    = [20, 40, 60]
N_TRAIN_VALUES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 249_000]
N_EVAL         = 500
N_ENS          = 20
SAMPLE_STEPS   = 200


def _load_cache(out_dir: str) -> dict:
    """Load any (lead, n_train) results already saved in sweep_results.npz."""
    path = os.path.join(out_dir, "sweep_results.npz")
    if not os.path.exists(path):
        return {}
    d = np.load(path)
    cached = {}
    for i, lead in enumerate(d["lead_values"].tolist()):
        for j, n in enumerate(d["n_train_values"].tolist()):
            cs = float(d["crps_skill"][i, j])
            vs = float(d["vs_skill"][i, j])
            if not np.isnan(cs):
                cached[(int(lead), int(n))] = {"crps_skill": cs, "vs_skill": vs}
    return cached


def run_sweep(
    data_path: str = "lorenz96/data/l96_trajectory.npz",
    out_dir:   str = "lorenz96/sweep",
    device:    str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    cached  = _load_cache(out_dir)
    results = {lead: {} for lead in LEAD_VALUES}

    for lead in LEAD_VALUES:
        for n_train in N_TRAIN_VALUES:
            if (lead, n_train) in cached:
                print(f"Cached  lead={lead} n_train={n_train:,} — skipping")
                results[lead][n_train] = cached[(lead, n_train)]
                continue

            print(f"\n{'='*60}")
            print(f"lead={lead} steps ({lead*0.05:.1f} TU)   n_train={n_train:,}")
            print(f"{'='*60}")

            ckpt_path = os.path.join(ckpt_dir, f"ddpm_lead{lead}_{n_train}.pt")

            if not os.path.exists(ckpt_path):
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
                sample_steps=SAMPLE_STEPS,
            )

            results[lead][n_train] = {
                "crps_skill": float(metrics["crps_skill"]),
                "vs_skill":   float(metrics["vs_skill"]),
            }

    _save(results, out_dir)
    _plot(results, out_dir)
    return results


def _save(results, out_dir):
    n_leads = len(LEAD_VALUES)
    n_sizes = len(N_TRAIN_VALUES)
    crps_arr = np.full((n_leads, n_sizes), np.nan)
    vs_arr   = np.full((n_leads, n_sizes), np.nan)
    for i, lead in enumerate(LEAD_VALUES):
        for j, n in enumerate(N_TRAIN_VALUES):
            if n in results.get(lead, {}):
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
    dt      = 0.05
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
            vals = np.array([results[lead].get(n, {}).get(metric, np.nan)
                             for n in N_TRAIN_VALUES])
            label = f"{lead*dt:.0f} TU  ({lead} steps)"
            ax.semilogx(n_train, vals, marker=markers[i], color=colors[i],
                        label=label, linewidth=1.5, markersize=5)

        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Training set size  $N$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(n_train)
        ax.set_xticklabels([f"{n:,}" for n in n_train], rotation=30, ha="right", fontsize=8)
        ax.legend(title="Lead", framealpha=0.9, fontsize=8)

    axes[0].set_title("CRPS skill vs training size  [L96, N=40, F=8]")
    axes[1].set_title("Variogram score skill vs training size  [L96]")
    fig.suptitle(f"L96 DDPM  |  $n_{{ens}}$={N_ENS}", fontsize=10)
    fig.tight_layout()

    path = os.path.join(out_dir, "skill_vs_ntrain.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Plot saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="lorenz96/data/l96_trajectory.npz")
    parser.add_argument("--out_dir", default="lorenz96/sweep")
    parser.add_argument("--device",  default=None)
    args = parser.parse_args()

    run_sweep(args.data, args.out_dir, args.device)


if __name__ == "__main__":
    main()
