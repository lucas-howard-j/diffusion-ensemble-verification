"""
Figure 4 — L96 sweep: CRPS/VS skill vs training volume and CRPS−VS decoupling.

One column per lead (header above each column); panels lettered down columns.

  Top row (a/c/e):    CRPS and VS skill vs equivalent training record (log scale), 95% CIs
  Bottom row (b/d/f): Violin distributions of CRPS−VS skill diff; dark = CI excludes zero

Data:
    lorenz96/sweep/sweep_results.npz   — point estimates
    lorenz96/sweep/sweep_errors.npz    — full bootstrap distributions (1 000 draws)

Output:
    figures/main/figure4.pdf   (vector, main)
    figures/main/figure4.png   (dpi=300)

Run:
    PYTHONPATH=. python scripts/figure4.py
"""

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

SWEEP_NPZ  = Path("lorenz96/sweep/sweep_results.npz")
ERRORS_NPZ = Path("lorenz96/sweep/sweep_errors.npz")
OUT_DIR    = Path("figures/main")
OUT_DIR.mkdir(exist_ok=True)

DT          = 0.05
DAYS_PER_TU = 5.0
DAYS_PER_YR = 365.25
ERA5_YEARS  = 86.0

# ── Style — shared font hierarchy across main figures ─────────────────────────────────────
FS_TICK  = 8
FS_LABEL = 9
FS_TITLE = 9
FS_LEG   = 7

# Blue is reserved for the DDPM across the main figures, so CRPS/VS use black/orange
CRPS_COLOR  = "#252525"
VS_COLOR    = "#DD8452"
DIFF_COLOR  = "#555555"
ERA5_COLOR  = "#2ca02c"
CI_PCT      = 95


# ── Helpers ───────────────────────────────────────────────────────────────────

def n_to_years(n):
    return n * DT * DAYS_PER_TU / DAYS_PER_YR


def fmt_years(yr):
    if yr <= 0:
        return ""
    mag     = 10 ** math.floor(math.log10(yr))
    rounded = round(yr / mag) * mag
    return f"{int(rounded)} yr" if rounded >= 2 else f"{rounded:.0f} yr"


def fmt_ntrain(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}k" if v == int(v) else f"{v:.1f}k"
    return str(int(n))


def _panel_letter(ax, lbl):
    ax.text(0.03, 0.97, lbl, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_TITLE, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))


def _deduplicate(x, labels, min_log_gap=0.45):
    keep = [True] * len(x)
    for k in range(1, len(x)):
        prev = next((i for i in range(k - 1, -1, -1) if keep[i]), None)
        if prev is not None and math.log10(x[k]) - math.log10(x[prev]) < min_log_gap:
            keep[k] = False
    return [lb if keep[k] else "" for k, lb in enumerate(labels)]


LEAD_LABELS  = {
    20: "1.0 TU (5 days)",
    40: "2.0 TU (10 days)",
    60: "3.0 TU (15 days)",
}
PANEL_LABELS_TOP = ["(a)", "(c)", "(e)"]
PANEL_LABELS_BOT = ["(b)", "(d)", "(f)"]


# ── Main ──────────────────────────────────────────────────────────────────────

def plot():
    d       = np.load(SWEEP_NPZ)
    leads   = d["lead_values"].tolist()

    n_train = d["n_train_values"]
    n_list  = n_train.tolist()

    has_errors = ERRORS_NPZ.exists()
    if has_errors:
        e     = np.load(ERRORS_NPZ)
        err_n = e["n_train_values"].tolist()
        alpha = (100 - CI_PCT) / 2

        def _align_dist(arr_3d, i):
            out = np.full((len(n_list), arr_3d.shape[2]), np.nan)
            for k, n in enumerate(err_n):
                if n in n_list:
                    out[n_list.index(n)] = arr_3d[i, k]
            return out

        def _ci(dist_2d):
            med = np.nanmedian(dist_2d, axis=1)
            lo  = np.nanpercentile(dist_2d, alpha,       axis=1)
            hi  = np.nanpercentile(dist_2d, 100 - alpha, axis=1)
            return med, med - lo, hi - med

        crps_skill_all, vs_skill_all = [], []
        crps_err_all, vs_err_all     = [], []
        diff_dist_all, diff_sig_all  = [], []
        for i in range(len(leads)):
            cm, clo, chi = _ci(_align_dist(e["crps_skill_dist"], i))
            vm, vlo, vhi = _ci(_align_dist(e["vs_skill_dist"],   i))
            crps_skill_all.append(cm);  vs_skill_all.append(vm)
            crps_err_all.append(np.vstack([clo, chi]))
            vs_err_all.append(np.vstack([vlo, vhi]))
            dd   = _align_dist(e["diff_skill_dist"], i)
            d_lo = np.nanpercentile(dd, alpha,       axis=1)
            d_hi = np.nanpercentile(dd, 100 - alpha, axis=1)
            diff_dist_all.append(dd)
            diff_sig_all.append((d_lo > 0) | (d_hi < 0))
        crps_skill_all = np.vstack(crps_skill_all)
        vs_skill_all   = np.vstack(vs_skill_all)
    else:
        crps_skill_all = d["crps_skill"]
        vs_skill_all   = d["vs_skill"]
        crps_err_all   = [np.zeros((2, crps_skill_all.shape[1])) for _ in leads]
        vs_err_all     = [np.zeros((2, vs_skill_all.shape[1]))   for _ in leads]
        diff_dist_all  = [None] * len(leads)
        diff_sig_all   = [np.zeros(crps_skill_all.shape[1], dtype=bool) for _ in leads]

    # ── Shared y-limits for bottom row ────────────────────────────────────────
    bot_ymin, bot_ymax = np.inf, -np.inf
    for diff_dist in diff_dist_all:
        if diff_dist is None:
            continue
        valid = ~np.all(np.isnan(diff_dist), axis=1)
        for j in np.where(valid)[0]:
            vals = diff_dist[j]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                bot_ymin = min(bot_ymin, np.percentile(vals, 1))
                bot_ymax = max(bot_ymax, np.percentile(vals, 99))
    if np.isfinite(bot_ymin):
        bot_pad  = (bot_ymax - bot_ymin) * 0.12
        bot_ylim = (bot_ymin - bot_pad, bot_ymax + bot_pad)
    else:
        bot_ylim = None

    # ── Axes grid — extra wspace leaves room between column groups ────────────
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.8), sharey=False, sharex=False)
    top_axes = axes[0]
    bot_axes = axes[1]
    ax2_list = []   # secondary axes refs — needed for bbox calculation

    fig_leg_handles = []   # collected during col==0, placed as fig.legend after layout
    fig_leg_labels  = []

    # ── Top row ───────────────────────────────────────────────────────────────
    for col, (ax, lead, crps_row, vs_row, crps_e, vs_e, top_lbl) in enumerate(zip(
        top_axes, leads, crps_skill_all, vs_skill_all,
        crps_err_all, vs_err_all, PANEL_LABELS_TOP,
    )):
        valid    = ~np.isnan(crps_row)
        x_n      = n_train[valid]
        x_yr     = n_to_years(x_n)
        yr_ticks = x_yr

        kw = dict(linewidth=1.2, markersize=2.5, capsize=2, capthick=0.8, elinewidth=0.8)
        h_crps_d = ax.errorbar(x_yr, crps_row[valid], yerr=crps_e[:, valid],
                               fmt="o-", color=CRPS_COLOR, **kw)
        h_vs_d   = ax.errorbar(x_yr, vs_row[valid], yerr=vs_e[:, valid],
                               fmt="s--", color=VS_COLOR, **kw)

        ax.set_xscale("log")
        h_era5 = ax.axvline(ERA5_YEARS, color=ERA5_COLOR, linewidth=1.0, linestyle="-",
                            zorder=3)
        ax.axhline(0, color="k", linewidth=0.6, linestyle=":", zorder=2)

        if col == 0:
            ax.set_ylabel("Skill Score", fontsize=FS_LABEL)
            # legend is filled column-major: DDPM scores, then ERA5, then sig.
            blank = Line2D([], [], linestyle="none")
            fig_leg_handles = [h_crps_d, h_vs_d, h_era5, blank]
            fig_leg_labels  = ["CRPS skill (DDPM)", "VS skill (DDPM)",
                               "ERA5 record (1940–present)", ""]

        _panel_letter(ax, top_lbl)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.set_xticks(yr_ticks)
        ax.set_xticklabels([])

        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(ax.get_xlim())
        n_labels = _deduplicate(x_yr, [fmt_ntrain(n) for n in x_n])
        ax2.set_xticks(yr_ticks)
        ax2.set_xticklabels(n_labels, rotation=35, ha="left", fontsize=FS_TICK)
        ax2.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        if col == 1:
            ax2.set_xlabel("Training samples  $N$", fontsize=FS_LABEL, labelpad=3)
        ax2_list.append(ax2)

    # ── Bottom row ────────────────────────────────────────────────────────────
    for col, (top_ax, ax, lead, diff_dist, diff_sig, bot_lbl) in enumerate(zip(
        top_axes, bot_axes, leads, diff_dist_all, diff_sig_all, PANEL_LABELS_BOT,
    )):
        if diff_dist is None:
            ax.set_visible(False)
            continue

        valid_b = ~np.all(np.isnan(diff_dist), axis=1)
        x_n_b   = n_train[valid_b]
        x_yr_b  = n_to_years(x_n_b)
        data    = [diff_dist[j] for j in np.where(valid_b)[0]]
        sig     = diff_sig[valid_b]

        # per-violin width from the nearest neighbour so closely spaced N don't overlap
        log_x = np.log10(x_yr_b)
        gaps  = np.diff(log_x) if len(log_x) > 1 else np.array([0.6])
        near  = np.minimum(np.r_[gaps[0], gaps], np.r_[gaps, gaps[-1]])
        half  = 0.45 * np.minimum(near, 0.35)
        widths = 10 ** (log_x + half) - 10 ** (log_x - half)

        ax.set_xscale("log")
        vp = ax.violinplot(data, positions=x_yr_b, widths=widths,
                           showmedians=True, showextrema=False)
        for body, s in zip(vp["bodies"], sig):
            body.set_facecolor(DIFF_COLOR if s else "lightgrey")
            body.set_alpha(0.65)
            body.set_edgecolor("none")
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(1.2)

        ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
        ax.axvline(ERA5_YEARS, color=ERA5_COLOR, linewidth=1.0, linestyle="-", zorder=3)
        ax.set_xlim(top_ax.get_xlim())
        yr_labels = _deduplicate(x_yr_b, [fmt_years(y) for y in x_yr_b])
        ax.set_xticks(x_yr_b)
        ax.set_xticklabels(yr_labels, rotation=35, ha="right", fontsize=FS_TICK)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="y", labelsize=FS_TICK)

        if bot_ylim is not None:
            ax.set_ylim(bot_ylim)

        if col == 0:
            ax.set_ylabel("CRPS − VS skill", fontsize=FS_LABEL)
            sig_patch   = mpatches.Patch(color=DIFF_COLOR,  alpha=0.65)
            nosig_patch = mpatches.Patch(color="lightgrey", alpha=0.65)
            fig_leg_handles += [sig_patch, nosig_patch]
            fig_leg_labels  += [f"Sig. ≠ 0 ({CI_PCT}%)", "Not sig."]

        if col == 1:
            ax.set_xlabel("Equivalent training record (yr)", fontsize=FS_LABEL)

        _panel_letter(ax, bot_lbl)

    # ── Tight layout then compute column bounding boxes ───────────────────────
    fig.tight_layout(h_pad=1.8, w_pad=0.9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Column headers centred over each column, just above the top-axis N labels
    hdr_y = max(ax2.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y1
                for ax2 in ax2_list) + 0.012
    for col, ax in enumerate(top_axes):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, hdr_y, f"Lead = {LEAD_LABELS[leads[col]]}",
                 ha="center", va="bottom", fontsize=FS_TITLE, fontweight="bold")

    # ── Figure-level legend below bottom row ─────────────────────────────────
    fig.legend(fig_leg_handles, fig_leg_labels,
               loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=3,
               fontsize=FS_LEG, frameon=True, framealpha=0.95,
               handlelength=2.2, handletextpad=0.4, columnspacing=0.9,
               borderpad=0.5)

    out_pdf = OUT_DIR / "figure4.pdf"
    out_png = OUT_DIR / "figure4.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    plot()
