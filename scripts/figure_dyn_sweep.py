"""
Supplementary figure: CRPS and VS skill vs IC perturbation size (σ_ic)
for the dynamical ensemble baseline.

Layout mirrors Figure 4 (L96 training-volume sweep); main-text Figure 5 shows lead 20 only:
  Top row (a/b/c):    CRPS and VS skill vs σ_ic (log x-axis), 95% CIs shaded
  Bottom row (d/e/f): Violin distributions of CRPS−VS skill diff; dark = CI
                      excludes zero (significant), light grey = not significant

Demonstrates that a well-calibrated dynamical ensemble achieves VS ≥ CRPS
at short leads (gap inverts), whereas the DDPM shows a persistent CRPS > VS
gap regardless of training volume.

Output:
    figures/figure_dyn_sweep.pdf
    figures/figure_dyn_sweep.png

Run:
    PYTHONPATH=. python scripts/figure_dyn_sweep.py
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DYN_FILES = {
    0.01: Path("lorenz96/sweep/dyn_results_sic01.npz"),
    0.02: Path("lorenz96/sweep/dyn_results_sic02.npz"),
    0.05: Path("lorenz96/sweep/dyn_results_sic05.npz"),
    0.10: Path("lorenz96/sweep/dyn_results_sic10.npz"),
    0.20: Path("lorenz96/sweep/dyn_results_sic20.npz"),
    0.40: Path("lorenz96/sweep/dyn_results_sic40.npz"),
}
OUT_DIR   = Path("figures/supplement")
OUT_DIR.mkdir(exist_ok=True)

LEAD_VALUES  = [20, 40, 60]
DT           = 0.05
CI_PCT       = 95
SIGMA_CLIM   = 3.6380   # training-period mean per-node std

# Style — match figure3/figure4 hierarchy
FS_TICK  = 6
FS_LABEL = 7
FS_TITLE = 7
FS_LEG   = 6.5

CRPS_COLOR = "#4C72B0"
VS_COLOR   = "#DD8452"
DIFF_COLOR = "#555555"

LEAD_LABELS = {
    20: "1.0 TU / 5 days",
    40: "2.0 TU / 10 days",
    60: "3.0 TU / 15 days",
}
PANEL_LABELS_TOP = ["(a)", "(b)", "(c)"]
PANEL_LABELS_BOT = ["(d)", "(e)", "(f)"]


def load_results(frac_to_path: dict, lead_values: list, ci_pct: float):
    """
    Returns dict keyed by lead index with arrays for top-row lines and
    bottom-row violins.
      x          : σ_ic values as % of σ_clim
      crps_med/lo/hi, vs_med/lo/hi : (n_sic,) point estimates + CI bounds
      diff_dist  : list of (n_boot,) arrays — one per σ_ic value
      diff_sig   : (n_sic,) bool — CI excludes zero
    """
    alpha = (100 - ci_pct) / 2
    fracs  = sorted(frac_to_path)
    paths  = [frac_to_path[f] for f in fracs]

    fracs = [f for f, p in zip(fracs, paths) if p.exists()]
    paths = [p for p in paths if p.exists()]

    results = {}
    for li, lead in enumerate(lead_values):
        crps_med, crps_lo, crps_hi = [], [], []
        vs_med,   vs_lo,   vs_hi   = [], [], []
        diff_dist = []
        diff_sig  = []
        x = []
        for frac, path in zip(fracs, paths):
            d = np.load(path)
            leads_in_file = d["lead_values"].tolist()
            if lead not in leads_in_file:
                continue
            i = leads_in_file.index(lead)
            x.append(frac * 100)
            cd = d["crps_skill_dist"][i]
            vd = d["vs_skill_dist"][i]
            crps_med.append(np.nanmedian(cd))
            crps_lo.append( np.nanpercentile(cd, alpha))
            crps_hi.append( np.nanpercentile(cd, 100 - alpha))
            vs_med.append(  np.nanmedian(vd))
            vs_lo.append(   np.nanpercentile(vd, alpha))
            vs_hi.append(   np.nanpercentile(vd, 100 - alpha))
            dd = cd - vd   # paired CRPS−VS bootstrap diff
            diff_dist.append(dd)
            d_lo = np.nanpercentile(dd, alpha)
            d_hi = np.nanpercentile(dd, 100 - alpha)
            diff_sig.append((d_lo > 0) or (d_hi < 0))

        results[li] = dict(
            x         = np.array(x),
            crps_med  = np.array(crps_med),
            crps_lo   = np.array(crps_lo),
            crps_hi   = np.array(crps_hi),
            vs_med    = np.array(vs_med),
            vs_lo     = np.array(vs_lo),
            vs_hi     = np.array(vs_hi),
            diff_dist = diff_dist,          # list of (n_boot,) arrays
            diff_sig  = np.array(diff_sig, dtype=bool),
        )
    return results


def plot():
    results = load_results(DYN_FILES, LEAD_VALUES, CI_PCT)

    # σ_ic values used in the main-text ensemble comparison (Fig. 4), in % of σ_clim
    FIG4_SIGMA = [1.0, 15.78]
    FIG4_LABELS = [r"$\sigma_{ic}=0.01$", r"$\sigma_{ic}=0.16$"]

    # shared y-limits for bottom row
    bot_ymin, bot_ymax = np.inf, -np.inf
    for li in range(len(LEAD_VALUES)):
        for dd in results[li]["diff_dist"]:
            if len(dd):
                bot_ymin = min(bot_ymin, np.nanpercentile(dd, 1))
                bot_ymax = max(bot_ymax, np.nanpercentile(dd, 99))
    if np.isfinite(bot_ymin):
        bot_pad  = (bot_ymax - bot_ymin) * 0.12
        bot_ylim = (bot_ymin - bot_pad, bot_ymax + bot_pad)
    else:
        bot_ylim = None

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8),
                             sharey=False, sharex=False)
    top_axes = axes[0]
    bot_axes = axes[1]

    fig_leg_handles = []
    fig_leg_labels  = []

    # ── Top row ───────────────────────────────────────────────────────────────
    for col, (ax, li, plbl) in enumerate(
        zip(top_axes, range(len(LEAD_VALUES)), PANEL_LABELS_TOP)
    ):
        r = results[li]
        x = r["x"]

        kw_line = dict(linewidth=1.2, markersize=3.0)
        kw_ci   = dict(linewidth=0, alpha=0.18)

        hc, = ax.plot(x, r["crps_med"], "o-", color=CRPS_COLOR, **kw_line)
        ax.fill_between(x, r["crps_lo"], r["crps_hi"], color=CRPS_COLOR, **kw_ci)

        hv, = ax.plot(x, r["vs_med"], "s--", color=VS_COLOR, **kw_line)
        ax.fill_between(x, r["vs_lo"], r["vs_hi"], color=VS_COLOR, **kw_ci)

        ax.axhline(0, color="k", linewidth=0.6, linestyle=":", zorder=1)

        # shade inverted region (VS > CRPS)
        if len(x):
            inv = r["vs_med"] > r["crps_med"]
            if inv.any():
                ax.fill_between(x, r["crps_med"], r["vs_med"],
                                where=inv, color=VS_COLOR, alpha=0.12,
                                linewidth=0)

        ax.set_xscale("log")
        ax.set_xlim(x.min() * 0.7, x.max() * 1.4)
        cur_bot, cur_top = ax.get_ylim()
        ax.set_ylim(bottom=max(cur_bot, -0.05), top=min(cur_top * 1.05, 1.05))

        # vertical lines for Fig. 4 σ_ic values
        for sv, slbl in zip(FIG4_SIGMA, FIG4_LABELS):
            ax.axvline(sv, color="k", lw=0.7, ls=(0, (3, 2)), zorder=0, alpha=0.55)
            if col == 0:
                ax.text(sv * 1.08, 0.07, slbl, transform=ax.get_xaxis_transform(),
                        fontsize=FS_TICK - 0.5, ha="left", va="bottom",
                        color="k", alpha=0.70)

        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Skill score", fontsize=FS_LABEL)

        ax.set_title(plbl, fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))


        if col == 0:
            fig_leg_handles = [hc, hv]
            fig_leg_labels  = ["CRPS skill", "VS skill"]

    # ── Bottom row ────────────────────────────────────────────────────────────
    for col, (top_ax, ax, li, plbl) in enumerate(
        zip(top_axes, bot_axes, range(len(LEAD_VALUES)), PANEL_LABELS_BOT)
    ):
        r  = results[li]
        x  = r["x"]
        dd = r["diff_dist"]
        sg = r["diff_sig"]

        if not len(dd):
            ax.set_visible(False)
            continue

        log_x   = np.log10(x)
        log_gap = np.diff(log_x).mean() if len(log_x) > 1 else 0.3
        widths  = 10 ** (log_x + 0.35 * log_gap) - 10 ** (log_x - 0.35 * log_gap)

        ax.set_xscale("log")
        vp = ax.violinplot(dd, positions=x, widths=widths,
                           showmedians=True, showextrema=False)
        for body, s in zip(vp["bodies"], sg):
            body.set_facecolor(DIFF_COLOR if s else "lightgrey")
            body.set_alpha(0.65)
            body.set_edgecolor("none")
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(1.2)

        ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
        for sv in FIG4_SIGMA:
            ax.axvline(sv, color="k", lw=0.7, ls=(0, (3, 2)), zorder=0, alpha=0.55)
        ax.set_xlim(top_ax.get_xlim())
        if bot_ylim is not None:
            ax.set_ylim(bot_ylim)

        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.tick_params(axis="x", labelsize=FS_TICK, rotation=35)
        ax.tick_params(axis="y", labelsize=FS_TICK)

        if col == 0:
            ax.set_ylabel("CRPS − VS skill", fontsize=FS_LABEL)
            sig_patch   = mpatches.Patch(color=DIFF_COLOR,  alpha=0.65)
            nosig_patch = mpatches.Patch(color="lightgrey", alpha=0.65)
            fig_leg_handles += [sig_patch, nosig_patch]
            fig_leg_labels  += [f"Sig. ≠ 0 ({CI_PCT}%)", "Not sig."]

        if col == 1:
            ax.set_xlabel(r"$\sigma_\mathrm{IC}$  (% of $\sigma_\mathrm{clim}$)",
                          fontsize=FS_LABEL)

        ax.set_title(plbl, fontsize=FS_TITLE, fontweight="bold", loc="left", pad=2)

    # ── Layout, column headers, figure legend ─────────────────────────────────
    fig.tight_layout(h_pad=1.8, w_pad=0.9)
    fig.canvas.draw()

    # column header labels centered above each top panel
    for col, lead in enumerate(LEAD_VALUES):
        ax = top_axes[col]
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.025,
            f"Lead = {LEAD_LABELS[lead]}",
            ha="center", va="bottom",
            fontsize=FS_TITLE, fontweight="bold",
            transform=fig.transFigure,
        )

    # figure legend centered between the two rows
    top_row_bot = min(ax.get_position().y0 for ax in top_axes)
    bot_row_top = max(ax.get_position().y1 for ax in bot_axes)
    legend_y    = (top_row_bot + bot_row_top) / 2

    fig.legend(fig_leg_handles, fig_leg_labels,
               loc="center", bbox_to_anchor=(0.5, legend_y),
               ncol=len(fig_leg_handles),
               fontsize=FS_LEG, frameon=True, framealpha=0.95,
               handlelength=2.2, handletextpad=0.4, columnspacing=0.9,
               borderpad=0.5)

    out_pdf = OUT_DIR / "figure_dyn_sweep.pdf"
    out_png = OUT_DIR / "figure_dyn_sweep.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    plot()
