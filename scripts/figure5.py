"""
Figure 5 — Lorenz-96: does chaos alone produce the CRPS–VS gap?

At lead 20 steps (1.0 TU), ensembles formed by perturbing the initial condition
with Gaussian noise of size σ_IC (as % of the climatological SD) and integrating
either the true L96 dynamics (perfect-model dynamical ensemble) or the MSE-trained
deterministic emulator.

(a) Dynamical ensemble: CRPS and VS skill vs σ_IC
(b) Deterministic emulator: CRPS and VS skill vs σ_IC
(c) VS/CRPS ratio vs σ_IC for both.
    The dashed vertical line marks σ_IC = 15.8 %, which matches the DDPM's ensemble
    spread; the matched-spread runs and the DDPM are compared in Figure 6.

Shading and error bars: 95 % bootstrap CIs over test initial conditions.

Data:
    lorenz96/sweep/{dyn,det}_results_sic{01,02,05,10,20,40}.npz   σ_IC sweeps
    lorenz96/ensemble_results/{dyn,det}_results.npz               matched-spread runs
    lorenz96/ensemble_results/ddpm_results.npz                    DDPM (lead 20, n_train 249k), printed only

Run:
    PYTHONPATH=. python scripts/figure5.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LEAD = 20
FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
SWEEP = Path("lorenz96/sweep")
MATCHED = Path("lorenz96/ensemble_results")
MATCHED_FRAC = 0.1578          # σ_IC matched to the DDPM spread (Figure 6)
OUT_DIR = Path("figures/main")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS_TICK, FS_LABEL, FS_TITLE, FS_LEG = 8, 9, 9, 7      # match figures 2–3
# Blue is reserved for the DDPM across figures 2–4, so CRPS/VS use black/orange
CRPS_COLOR, VS_COLOR = "#252525", "#DD8452"
DYN_COLOR, DET_COLOR, DDPM_COLOR = "#1b7837", "#762a83", "#2166ac"
DYN_LABEL, DET_LABEL = "Dynamical", "Deterministic emulator"   # "perfect model" in caption


def _q(dist):
    return float(np.nanmedian(dist)), float(np.nanpercentile(dist, 2.5)), float(np.nanpercentile(dist, 97.5))


def load_sweep(kind: str) -> dict:
    """Median and 95 % CI of CRPS skill, VS skill and VS/CRPS at LEAD for each σ_IC."""
    out = {k: [] for k in ("x", "crps", "vs", "ratio")}
    for f in FRACS:
        d = np.load(SWEEP / f"{kind}_results_sic{int(round(f * 100)):02d}.npz")
        cd, vd = d["crps_skill_dist"], d["vs_skill_dist"]
        if cd.ndim == 2:                                    # dynamical: (n_leads, n_boot)
            i = d["lead_values"].tolist().index(LEAD)
            cd, vd = cd[i], vd[i]
        out["x"].append(100 * f)
        out["crps"].append(_q(cd))
        out["vs"].append(_q(vd))
        out["ratio"].append(_q(vd / cd))                    # paired bootstrap draws
    return {k: np.array(v) for k, v in out.items()}


def load_matched(kind: str) -> dict:
    d = np.load(MATCHED / f"{kind}_results.npz")
    c, v = float(d["crps_skill"]), float(d["vs_skill"])
    return dict(crps=(c, float(d["crps_lo"]), float(d["crps_hi"])),
                vs=(v, float(d["vs_lo"]), float(d["vs_hi"])),
                ratio=(v / c, float(d["ratio_lo"]), float(d["ratio_hi"])))


def _style(ax, title, ylabel):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.axvline(100 * MATCHED_FRAC, color="0.3", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontweight="bold", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.2, lw=0.5)


def _skill_panel(ax, sw, title, ylabel=""):
    for key, color, mk, ls, label in [("crps", CRPS_COLOR, "o", "-", "CRPS skill"),
                                      ("vs", VS_COLOR, "s", "--", "VS skill")]:
        ax.plot(sw["x"], sw[key][:, 0], ls + mk, color=color, lw=1.2, ms=3, label=label)
        ax.fill_between(sw["x"], sw[key][:, 1], sw[key][:, 2], color=color, alpha=0.18)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    _style(ax, title, ylabel)


def make_figure() -> None:
    dyn, det = load_sweep("dyn"), load_sweep("det")
    dyn_m, det_m = load_matched("dyn"), load_matched("det")
    dd = load_matched("ddpm")["ratio"]   # same ICs/protocol as the matched runs

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw=dict(wspace=0.38))
    _skill_panel(axes[0], dyn, f"(a) {DYN_LABEL}", "Skill Score")
    _skill_panel(axes[1], det, f"(b) {DET_LABEL}")
    axes[1].set_ylim(bottom=min(axes[1].get_ylim()[0], -0.02))
    for a in axes[:2]:
        a.legend(fontsize=FS_LEG, loc="lower left", bbox_to_anchor=(0, 0.05), frameon=False)

    ax = axes[2]
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    # Matched-spread values (and the DDPM) are compared in figure 6; the dashed
    # vertical line marks that σ_IC here.
    for sw, color, mk, label in [(dyn, DYN_COLOR, "o", DYN_LABEL),
                                 (det, DET_COLOR, "D", DET_LABEL)]:
        ax.plot(sw["x"], sw["ratio"][:, 0], "-" + mk, color=color, lw=1.2, ms=3, label=label)
        ax.fill_between(sw["x"], sw["ratio"][:, 1], sw["ratio"][:, 2], color=color, alpha=0.18)
    _style(ax, "(c) Comparison", "VS / CRPS skill ratio")
    ax.set_ylim(0, 1.12)
    # opaque frame so the matched-spread line doesn't run through the entries
    leg = ax.legend(fontsize=FS_LEG, loc="lower left", frameon=True, framealpha=1,
                    facecolor="white", edgecolor="none", borderaxespad=0.3,
                    handlelength=1.6, handletextpad=0.5)
    leg.set_zorder(2.4)          # above the data lines, below the axes spines

    # one shared x-label: the three panels use the same σ_IC axis
    fig.supxlabel(r"Initial-condition perturbation $\sigma_\mathrm{IC}$ (% of climatological SD)",
                  fontsize=FS_LABEL, y=-0.02)

    fig.savefig(OUT_DIR / "figure5.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure5.png", bbox_inches="tight", dpi=300)
    print("Saved figures/main/figure5.pdf and figures/main/figure5.png")

    print(f"\nLead {LEAD}: VS/CRPS (median [95% CI])")
    print(f"  DDPM            {dd[0]:.3f} [{dd[1]:.3f}, {dd[2]:.3f}]")
    for name, sw, mt in [("Dynamical", dyn, dyn_m), ("Det. emulator", det, det_m)]:
        row = "  ".join(f"{x:g}%:{r[0]:.2f}" for x, r in zip(sw["x"], sw["ratio"]))
        print(f"  {name:<15} {row}   matched {mt['ratio'][0]:.3f} [{mt['ratio'][1]:.3f}, {mt['ratio'][2]:.3f}]")


if __name__ == "__main__":
    make_figure()
