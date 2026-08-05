"""
Supplementary figure: CRPS and VS skill vs IC perturbation size (σ_ic)
for the deterministic emulator (lead=20 only).

Style matches figure_dyn_sweep.py — panels (a) and (d) equivalent.

Output:
    figures/supplement/figure_det_sweep.pdf
    figures/supplement/figure_det_sweep.png

Run:
    PYTHONPATH=. python scripts/figure_det_sweep.py
"""

from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DET_FILES = {
    0.01: Path("lorenz96/sweep/det_results_sic01.npz"),
    0.02: Path("lorenz96/sweep/det_results_sic02.npz"),
    0.05: Path("lorenz96/sweep/det_results_sic05.npz"),
    0.10: Path("lorenz96/sweep/det_results_sic10.npz"),
    0.20: Path("lorenz96/sweep/det_results_sic20.npz"),
    0.40: Path("lorenz96/sweep/det_results_sic40.npz"),
}
OUT_DIR = Path("figures/supplement")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CI_PCT = 95
LEAD   = 20

# Style — matches figure_dyn_sweep.py
FS_TICK  = 6
FS_LABEL = 7
FS_TITLE = 7
FS_LEG   = 6.5

CRPS_COLOR = "#4C72B0"
VS_COLOR   = "#DD8452"
DIFF_COLOR = "#555555"


def load_results(ci_pct: float) -> dict:
    alpha  = (100 - ci_pct) / 2
    fracs  = sorted(DET_FILES)
    x, crps_med, crps_lo, crps_hi = [], [], [], []
    vs_med, vs_lo, vs_hi = [], [], []
    diff_dist, diff_sig  = [], []

    for frac in fracs:
        path = DET_FILES[frac]
        if not path.exists():
            continue
        d  = np.load(path)
        cd = d["crps_skill_dist"]   # (n_boot,)
        vd = d["vs_skill_dist"]

        x.append(frac * 100)
        crps_med.append(np.nanmedian(cd))
        crps_lo.append(np.nanpercentile(cd, alpha))
        crps_hi.append(np.nanpercentile(cd, 100 - alpha))
        vs_med.append(np.nanmedian(vd))
        vs_lo.append(np.nanpercentile(vd, alpha))
        vs_hi.append(np.nanpercentile(vd, 100 - alpha))

        dd = cd - vd
        diff_dist.append(dd)
        d_lo = np.nanpercentile(dd, alpha)
        d_hi = np.nanpercentile(dd, 100 - alpha)
        diff_sig.append((d_lo > 0) or (d_hi < 0))

    return dict(
        x         = np.array(x),
        crps_med  = np.array(crps_med),
        crps_lo   = np.array(crps_lo),
        crps_hi   = np.array(crps_hi),
        vs_med    = np.array(vs_med),
        vs_lo     = np.array(vs_lo),
        vs_hi     = np.array(vs_hi),
        diff_dist = diff_dist,
        diff_sig  = np.array(diff_sig, dtype=bool),
    )


def plot() -> None:
    r = load_results(CI_PCT)
    x = r["x"]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(2.6, 4.0), sharex=False
    )

    # ── Top panel (a): skill vs σ_ic ─────────────────────────────────────────
    kw_line = dict(linewidth=1.2, markersize=3.0)
    kw_ci   = dict(linewidth=0, alpha=0.18)

    hc, = ax_top.plot(x, r["crps_med"], "o-", color=CRPS_COLOR, **kw_line)
    ax_top.fill_between(x, r["crps_lo"], r["crps_hi"], color=CRPS_COLOR, **kw_ci)

    hv, = ax_top.plot(x, r["vs_med"], "s--", color=VS_COLOR, **kw_line)
    ax_top.fill_between(x, r["vs_lo"], r["vs_hi"], color=VS_COLOR, **kw_ci)

    ax_top.axhline(0, color="k", linewidth=0.6, linestyle=":", zorder=1)

    ax_top.set_xscale("log")
    ax_top.set_xlim(x.min() * 0.7, x.max() * 1.4)
    ax_top.set_ylim(bottom=-0.05)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("Skill score", fontsize=FS_LABEL)
    ax_top.set_title("(a)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
    ax_top.tick_params(axis="both", labelsize=FS_TICK)
    ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))

    lead_tu = LEAD * 0.05
    lead_days = int(round(lead_tu * 5))
    fig.text(
        0.5, 1.01,
        f"Lead = {lead_tu:.1f} TU / {lead_days} days",
        ha="center", va="bottom",
        fontsize=FS_TITLE, fontweight="bold",
        transform=ax_top.transAxes,
    )

    # ── Bottom panel (d): CRPS − VS violins ──────────────────────────────────
    log_x   = np.log10(x)
    log_gap = np.diff(log_x).mean() if len(log_x) > 1 else 0.3
    widths  = 10 ** (log_x + 0.35 * log_gap) - 10 ** (log_x - 0.35 * log_gap)

    ax_bot.set_xscale("log")
    vp = ax_bot.violinplot(r["diff_dist"], positions=x, widths=widths,
                           showmedians=True, showextrema=False)
    for body, sig in zip(vp["bodies"], r["diff_sig"]):
        body.set_facecolor(DIFF_COLOR if sig else "lightgrey")
        body.set_alpha(0.65)
        body.set_edgecolor("none")
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(1.2)

    ax_bot.axhline(0, color="k", linewidth=0.7, linestyle="--")
    ax_bot.set_xlim(ax_top.get_xlim())
    ax_bot.set_xticks(x)
    ax_bot.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax_bot.tick_params(axis="x", labelsize=FS_TICK, rotation=35)
    ax_bot.tick_params(axis="y", labelsize=FS_TICK)
    ax_bot.set_ylabel("CRPS − VS skill", fontsize=FS_LABEL)
    ax_bot.set_xlabel(r"$\sigma_\mathrm{IC}$  (% of $\sigma_\mathrm{clim}$)",
                      fontsize=FS_LABEL)
    ax_bot.set_title("(d)", fontsize=FS_TITLE, fontweight="bold", loc="left", pad=2)

    # ── Legend ────────────────────────────────────────────────────────────────
    sig_patch   = mpatches.Patch(color=DIFF_COLOR,  alpha=0.65)
    nosig_patch = mpatches.Patch(color="lightgrey", alpha=0.65)
    handles = [hc, hv, sig_patch, nosig_patch]
    labels  = ["CRPS skill", "VS skill",
               f"Sig. ≠ 0 ({CI_PCT}%)", "Not sig."]

    fig.legend(handles, labels,
               loc="center", bbox_to_anchor=(0.5, 0.495),
               ncol=2, fontsize=FS_LEG, frameon=True, framealpha=0.95,
               handlelength=2.0, handletextpad=0.4, columnspacing=0.8,
               borderpad=0.5)

    fig.tight_layout(h_pad=1.8)
    fig.subplots_adjust(hspace=0.38)

    out_pdf = OUT_DIR / "figure_det_sweep.pdf"
    out_png = OUT_DIR / "figure_det_sweep.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    plot()
