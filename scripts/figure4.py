"""
Figure 4 — L96 sweep: CRPS/VS skill vs training volume and CRPS−VS decoupling.

Three column groups (a,d) (b,e) (c,f) are visually boxed by background patches
sized after layout so all labels stay inside the box.

  Top row (a/b/c):    CRPS and VS skill vs equivalent training record (log scale), 95% CIs
  Bottom row (d/e/f): Violin distributions of CRPS−VS skill diff; dark = CI excludes zero

Data:
    lorenz96/sweep/sweep_results.npz   — point estimates
    lorenz96/sweep/sweep_errors.npz    — full bootstrap distributions (1 000 draws)

Output:
    figures/figure4.pdf   (vector, main)
    figures/figure4.png   (dpi=150, for quick inspection)

Run:
    PYTHONPATH=. python scripts/figure4.py
"""

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from pathlib import Path

SWEEP_NPZ  = Path("lorenz96/sweep/sweep_results.npz")
ERRORS_NPZ = Path("lorenz96/sweep/sweep_errors.npz")
MVN_NPZ    = Path("lorenz96/sweep/mvn_results.npz")
DYN_NPZ    = Path("lorenz96/sweep/dyn_results.npz")
OUT_DIR    = Path("figures/main")
OUT_DIR.mkdir(exist_ok=True)

DT          = 0.05
DAYS_PER_TU = 5.0
DAYS_PER_YR = 365.25
ERA5_YEARS  = 86.0

# ── Style — match figure3 font hierarchy ─────────────────────────────────────
FS_TICK  = 6
FS_LABEL = 7
FS_TITLE = 7
FS_LEG   = 6.5

CRPS_COLOR  = "#4C72B0"
VS_COLOR    = "#DD8452"
DIFF_COLOR  = "#555555"
ERA5_COLOR  = "#2ca02c"
BOX_COLOR   = "#eef0f4"
BOX_EDGE    = "#b8bcc8"
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


def _deduplicate(x, labels, min_log_gap=0.12):
    keep = [True] * len(x)
    for k in range(1, len(x)):
        prev = next((i for i in range(k - 1, -1, -1) if keep[i]), None)
        if prev is not None and math.log10(x[k]) - math.log10(x[prev]) < min_log_gap:
            keep[k] = False
    return [lb if keep[k] else "" for k, lb in enumerate(labels)]


LEAD_LABELS  = {
    20: "1.0 TU / 5 days",
    40: "2.0 TU / 10 days",
    60: "3.0 TU / 15 days",
}
PANEL_LABELS_TOP = ["(a)", "(c)", "(e)"]
PANEL_LABELS_BOT = ["(b)", "(d)", "(f)"]


# ── Main ──────────────────────────────────────────────────────────────────────

def plot():
    d       = np.load(SWEEP_NPZ)
    leads   = d["lead_values"].tolist()
    n_train = d["n_train_values"]
    n_list  = n_train.tolist()

    # MVN baseline — point estimates + bootstrap CIs
    has_mvn = MVN_NPZ.exists()
    if has_mvn:
        m = np.load(MVN_NPZ)
        mvn_n = m["n_train_values"].tolist()
        alpha_mvn = (100 - CI_PCT) / 2

        def _align_mvn(arr_2d):
            out = np.full((len(leads), len(n_list)), np.nan)
            for ii, lead in enumerate(leads):
                if lead not in m["lead_values"].tolist():
                    continue
                li = m["lead_values"].tolist().index(lead)
                for jj, n in enumerate(n_list):
                    if n in mvn_n:
                        out[ii, jj] = arr_2d[li, mvn_n.index(n)]
            return out

        def _align_mvn_ci(arr_3d):
            """Returns (lo, hi) each (n_leads, n_list) in data units."""
            lo = np.full((len(leads), len(n_list)), np.nan)
            hi = np.full((len(leads), len(n_list)), np.nan)
            for ii, lead in enumerate(leads):
                if lead not in m["lead_values"].tolist():
                    continue
                li = m["lead_values"].tolist().index(lead)
                for jj, n in enumerate(n_list):
                    if n in mvn_n:
                        dist = arr_3d[li, mvn_n.index(n)]
                        lo[ii, jj] = np.nanpercentile(dist, alpha_mvn)
                        hi[ii, jj] = np.nanpercentile(dist, 100 - alpha_mvn)
            return lo, hi

        mvn_crps_all = _align_mvn(m["crps_skill"])
        mvn_vs_all   = _align_mvn(m["vs_skill"])
        has_mvn_ci   = "crps_skill_dist" in m
        if has_mvn_ci:
            mvn_crps_lo, mvn_crps_hi = _align_mvn_ci(m["crps_skill_dist"])
            mvn_vs_lo,   mvn_vs_hi   = _align_mvn_ci(m["vs_skill_dist"])

    # ── Dynamical baseline ────────────────────────────────────────────────────
    has_dyn = DYN_NPZ.exists()
    if has_dyn:
        dn        = np.load(DYN_NPZ)
        dyn_leads = dn["lead_values"].tolist()
        dyn_crps  = dn["crps_skill"]   # (n_leads,)
        dyn_vs    = dn["vs_skill"]     # (n_leads,)
        alpha_dyn = (100 - CI_PCT) / 2
        dyn_crps_lo = np.nanpercentile(dn["crps_skill_dist"], alpha_dyn,       axis=1)
        dyn_crps_hi = np.nanpercentile(dn["crps_skill_dist"], 100 - alpha_dyn, axis=1)
        dyn_vs_lo   = np.nanpercentile(dn["vs_skill_dist"],   alpha_dyn,       axis=1)
        dyn_vs_hi   = np.nanpercentile(dn["vs_skill_dist"],   100 - alpha_dyn, axis=1)

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
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.4), sharey=False, sharex=False)
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

        h_crps_m = h_vs_m = None
        if has_mvn:
            mvn_c  = mvn_crps_all[col]
            mvn_v  = mvn_vs_all[col]
            mvn_ok = ~np.isnan(mvn_c) & (n_train >= 5_000)   # drop n=1k (OLS poorly conditioned)
            if mvn_ok.any():
                x_mvn = n_to_years(n_train[mvn_ok])
                mkw = dict(linewidth=0.85, alpha=0.55)
                h_crps_m, = ax.plot(x_mvn, mvn_c[mvn_ok], ":", color=CRPS_COLOR, **mkw)
                h_vs_m,   = ax.plot(x_mvn, mvn_v[mvn_ok], ":", color=VS_COLOR,   **mkw)
                if has_mvn_ci:
                    ax.fill_between(x_mvn,
                                    mvn_crps_lo[col][mvn_ok], mvn_crps_hi[col][mvn_ok],
                                    color=CRPS_COLOR, alpha=0.07, linewidth=0)
                    ax.fill_between(x_mvn,
                                    mvn_vs_lo[col][mvn_ok], mvn_vs_hi[col][mvn_ok],
                                    color=VS_COLOR,   alpha=0.07, linewidth=0)

        # ── Dynamical baseline horizontal lines ───────────────────────────
        h_crps_dyn = h_vs_dyn = None
        if has_dyn and lead in dyn_leads:
            di  = dyn_leads.index(lead)
            dkw = dict(linewidth=1.0, alpha=0.80, linestyle="-.")
            h_crps_dyn = ax.axhline(dyn_crps[di], color=CRPS_COLOR, **dkw)
            h_vs_dyn   = ax.axhline(dyn_vs[di],   color=VS_COLOR,   **dkw)
            ax.axhspan(dyn_crps_lo[di], dyn_crps_hi[di],
                       color=CRPS_COLOR, alpha=0.07, linewidth=0)
            ax.axhspan(dyn_vs_lo[di],   dyn_vs_hi[di],
                       color=VS_COLOR,   alpha=0.07, linewidth=0)

        ax.set_xscale("log")
        h_era5 = ax.axvline(ERA5_YEARS, color=ERA5_COLOR, linewidth=1.0, linestyle="-",
                            zorder=3)
        ax.axhline(0, color="k", linewidth=0.6, linestyle=":", zorder=2)

        # Clip y bottom so n=1k MVN outliers don't squish the interesting range
        cur_bot, cur_top = ax.get_ylim()
        ax.set_ylim(bottom=max(cur_bot, -0.08), top=cur_top)

        if col == 0:
            ax.set_ylabel("Skill score", fontsize=FS_LABEL)
            # Save handles for figure-level legend (placed between rows after layout)
            fig_leg_handles = [h_crps_d, h_vs_d, h_era5]
            fig_leg_labels  = ["CRPS skill (DDPM)", "VS skill (DDPM)", "ERA5  (1940–present)"]
            if h_crps_m is not None and h_crps_dyn is not None:
                fig_leg_handles = [h_crps_d, h_crps_m, h_crps_dyn,
                                   h_vs_d,   h_vs_m,   h_vs_dyn,   h_era5]
                fig_leg_labels  = ["CRPS (DDPM)", "CRPS (MVN)", "CRPS (Dyn)",
                                   "VS (DDPM)",   "VS (MVN)",   "VS (Dyn)",
                                   "ERA5  (1940–present)"]
            elif h_crps_m is not None:
                fig_leg_handles = [h_crps_d, h_crps_m, h_vs_d, h_vs_m, h_era5]
                fig_leg_labels  = ["CRPS skill (DDPM)", "CRPS skill (MVN)",
                                   "VS skill (DDPM)",   "VS skill (MVN)",
                                   "ERA5  (1940–present)"]
            elif h_crps_dyn is not None:
                fig_leg_handles = [h_crps_d, h_crps_dyn, h_vs_d, h_vs_dyn, h_era5]
                fig_leg_labels  = ["CRPS (DDPM)", "CRPS (Dyn)",
                                   "VS (DDPM)",   "VS (Dyn)",
                                   "ERA5  (1940–present)"]

        ax.set_title(top_lbl, fontsize=FS_TITLE, fontweight="bold", loc="left", pad=3)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.set_xticks(yr_ticks)
        ax.set_xticklabels([])

        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(ax.get_xlim())
        n_labels = _deduplicate(x_yr, [fmt_ntrain(n) for n in x_n])
        ax2.set_xticks(yr_ticks)
        ax2.set_xticklabels(n_labels, rotation=35, ha="left", fontsize=FS_TICK)
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

        log_x   = np.log10(x_yr_b)
        log_gap = np.diff(log_x).mean() if len(log_x) > 1 else 0.3
        widths  = 10 ** (log_x + 0.35 * log_gap) - 10 ** (log_x - 0.35 * log_gap)

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
        ax.set_xlim(top_ax.get_xlim())
        yr_labels = _deduplicate(x_yr_b, [fmt_years(y) for y in x_yr_b])
        ax.set_xticks(x_yr_b)
        ax.set_xticklabels(yr_labels, rotation=35, ha="right", fontsize=FS_TICK)
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

        ax.set_title(bot_lbl, fontsize=FS_TITLE, fontweight="bold",
                     loc="left", pad=2)

    # ── Tight layout then compute column bounding boxes ───────────────────────
    fig.tight_layout(h_pad=1.8, w_pad=0.9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    PAD        = 0.010   # figure-fraction padding around each column box
    HDR_OFFSET = 0.008   # figure-fraction gap between box top and header text

    # First pass: collect per-column bboxes and compute shared vertical extent
    col_bboxes = []
    for col in range(3):
        bbs = [bb for bb in (top_axes[col].get_tightbbox(renderer),
                              bot_axes[col].get_tightbbox(renderer),
                              ax2_list[col].get_tightbbox(renderer))
               if bb is not None]
        col_bboxes.append(Bbox.union(bbs).transformed(fig.transFigure.inverted()))

    shared_y0 = min(bb.y0 for bb in col_bboxes) - PAD
    shared_y1 = max(bb.y1 for bb in col_bboxes) + PAD + 0.030  # room for header

    # Second pass: draw aligned boxes and headers
    for col, union_fig in enumerate(col_bboxes):
        x0 = union_fig.x0 - PAD
        x1 = union_fig.x1 + PAD

        rect = plt.Rectangle(
            (x0, shared_y0), x1 - x0, shared_y1 - shared_y0,
            transform=fig.transFigure, clip_on=False,
            facecolor="none", edgecolor=BOX_EDGE, linewidth=0.8,
            linestyle="--", zorder=0,
        )
        fig.add_artist(rect)

        fig.text(
            (x0 + x1) / 2,
            max(bb.y1 for bb in col_bboxes) + PAD + HDR_OFFSET,
            f"Lead = {LEAD_LABELS[leads[col]]}",
            ha="center", va="bottom",
            fontsize=FS_TITLE, fontweight="bold",
            transform=fig.transFigure,
        )

    # ── Figure-level legend centered between the two rows ────────────────────
    top_row_bot = min(ax.get_position().y0 for ax in top_axes)
    bot_row_top = max(ax.get_position().y1 for ax in bot_axes)
    legend_y    = (top_row_bot + bot_row_top) / 2

    fig.legend(fig_leg_handles, fig_leg_labels,
               loc="center", bbox_to_anchor=(0.5, legend_y),
               ncol=len(fig_leg_handles),
               fontsize=FS_LEG, frameon=True, framealpha=0.95,
               handlelength=2.2, handletextpad=0.4, columnspacing=0.9,
               borderpad=0.5)

    out_pdf = OUT_DIR / "figure4.pdf"
    out_png = OUT_DIR / "figure4.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    plot()
