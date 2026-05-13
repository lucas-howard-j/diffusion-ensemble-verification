"""
Sweep figure: top row = CRPS/VS skill vs n_train (3 leads);
              bottom row = violin distributions of CRPS−VS difference.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

SWEEP_NPZ  = os.path.join(os.path.dirname(__file__), "sweep_results.npz")
ERRORS_NPZ = os.path.join(os.path.dirname(__file__), "sweep_errors.npz")
MVN_NPZ    = os.path.join(os.path.dirname(__file__), "mvn_results.npz")
OUT_PATH   = os.path.join(os.path.dirname(__file__), "sweep_figure.pdf")

DT          = 0.05
DAYS_PER_TU = 5.0
DAYS_PER_YR = 365.25
ERA5_YEARS  = 86.0

def n_to_years(n):
    return n * DT * DAYS_PER_TU / DAYS_PER_YR

def fmt_years(yr):
    if yr <= 0:
        return ""
    mag = 10 ** math.floor(math.log10(yr))
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

LEAD_LABELS = {
    20: "1.0 TU  /  5 days  (≈1.7 Lyapunov)",
    40: "2.0 TU  /  10 days  (≈3.3 Lyapunov)",
    60: "3.0 TU  /  15 days  (≈5.0 Lyapunov)",
}
CRPS_COLOR     = "#4C72B0"
VS_COLOR       = "#DD8452"
DIFF_COLOR     = "#555555"
ERA5_COLOR     = "#2ca02c"
MVN_CRPS_COLOR = "#4C72B0"   # same blue, distinguished by style
MVN_VS_COLOR   = "#DD8452"   # same orange, distinguished by style
CI_PCT     = 95   # confidence interval percentage; change without re-running bootstrap


def plot():
    d = np.load(SWEEP_NPZ)
    leads   = d["lead_values"].tolist()
    n_train = d["n_train_values"]
    n_list  = n_train.tolist()

    # MVN baseline (optional — omitted from plot if file absent)
    has_mvn = os.path.exists(MVN_NPZ)
    if has_mvn:
        m       = np.load(MVN_NPZ)
        mvn_n   = m["n_train_values"].tolist()
        # Align MVN arrays to DDPM n_list (in case grids differ)
        def _align_mvn(arr_2d):
            out = np.full((len(leads), len(n_list)), np.nan)
            for ii, lead in enumerate(leads):
                li = list(m["lead_values"]).index(lead) if lead in m["lead_values"] else None
                if li is None:
                    continue
                for jj, n in enumerate(n_list):
                    if n in mvn_n:
                        out[ii, jj] = arr_2d[li, mvn_n.index(n)]
            return out
        mvn_crps = _align_mvn(m["crps_skill"])
        mvn_vs   = _align_mvn(m["vs_skill"])

    has_errors = os.path.exists(ERRORS_NPZ)
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
            dd   = _align_dist(e["diff_skill_dist"], i)   # (n_list, N_BOOT)
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

    nrows = 2 if has_errors else 1
    fig, axes_all = plt.subplots(nrows, 3, figsize=(11, 4.4 * nrows),
                                 sharey=False, sharex=False)
    if nrows == 1:
        top_axes = axes_all
        bot_axes = [None] * 3
    else:
        top_axes = axes_all[0]
        bot_axes = axes_all[1]

    # ── Top row: CRPS / VS skill with CIs ──────────────────────────────────
    for ax, lead, crps_row, vs_row, crps_e, vs_e, diff_sig in zip(
        top_axes, leads, crps_skill_all, vs_skill_all,
        crps_err_all, vs_err_all, diff_sig_all
    ):
        valid = ~np.isnan(crps_row)
        x_n  = n_train[valid]
        x_yr = n_to_years(x_n)

        ax.errorbar(x_yr, crps_row[valid], yerr=crps_e[:, valid],
                    fmt="o-", color=CRPS_COLOR, linewidth=1.5, markersize=3,
                    capsize=3, capthick=1.0, elinewidth=1.0, label="CRPS skill (DDPM)")
        ax.errorbar(x_yr, vs_row[valid], yerr=vs_e[:, valid],
                    fmt="s--", color=VS_COLOR, linewidth=1.5, markersize=3,
                    capsize=3, capthick=1.0, elinewidth=1.0, label="VS skill (DDPM)")

        if has_mvn:
            li        = leads.index(lead)
            mvn_c     = mvn_crps[li]   # (n_sizes,) aligned to n_list
            mvn_v     = mvn_vs[li]
            mvn_ok    = ~np.isnan(mvn_c)
            mvn_x     = n_to_years(n_train[mvn_ok])
            if mvn_ok.any():
                ax.plot(mvn_x, mvn_c[mvn_ok],
                        "^:", color=MVN_CRPS_COLOR, linewidth=1.2, markersize=3,
                        alpha=0.6, label="CRPS skill (MVN)")
                ax.plot(mvn_x, mvn_v[mvn_ok],
                        "v:", color=MVN_VS_COLOR, linewidth=1.2, markersize=3,
                        alpha=0.6, label="VS skill (MVN)")
        ax.set_xscale("log")
        ax.axvline(ERA5_YEARS, color=ERA5_COLOR, linewidth=1.4,
                   linestyle="--", zorder=3, label="ERA5 (1940–present)")
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Skill score" if lead == leads[0] else "", fontsize=9)
        ax.set_title(f"Lead = {LEAD_LABELS[lead]}", fontsize=9)

        yr_labels = _deduplicate(x_yr, [fmt_years(y) for y in x_yr])
        ax.set_xticks(x_yr)
        ax.set_xticklabels([] if nrows > 1 else yr_labels, rotation=35, ha="right", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=8)

        ax2 = ax.twiny()
        ax2.set_xscale("log");  ax2.set_xlim(ax.get_xlim())
        n_labels = _deduplicate(x_yr, [fmt_ntrain(n) for n in x_n])
        ax2.set_xticks(x_yr);  ax2.set_xticklabels(n_labels, rotation=35, ha="left", fontsize=7.5)
        ax2.set_xlabel("Training samples  $N$", fontsize=9, labelpad=6)
        ax.legend(fontsize=8, framealpha=0.9)

    # ── Bottom row: violin distributions of CRPS − VS ──────────────────────
    for top_ax, ax, lead, diff_dist, diff_sig in zip(
        top_axes, bot_axes, leads, diff_dist_all, diff_sig_all
    ):
        if ax is None or diff_dist is None:
            continue

        valid = ~np.all(np.isnan(diff_dist), axis=1)
        x_n   = n_train[valid]
        x_yr  = n_to_years(x_n)
        data  = [diff_dist[j] for j in np.where(valid)[0]]
        sig   = diff_sig[valid]

        # Violin widths proportional to x so they appear equal in log space
        log_x   = np.log10(x_yr)
        log_gap = np.diff(log_x).mean() if len(log_x) > 1 else 0.3
        widths  = 10 ** (log_x + 0.35 * log_gap) - 10 ** (log_x - 0.35 * log_gap)

        ax.set_xscale("log")
        vp = ax.violinplot(data, positions=x_yr, widths=widths,
                           showmedians=True, showextrema=False)
        for body, s in zip(vp["bodies"], sig):
            body.set_facecolor(DIFF_COLOR if s else "lightgrey")
            body.set_alpha(0.6)
            body.set_edgecolor("none")
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(1.5)

        ax.axhline(0, color="k", linewidth=0.9, linestyle="--")

        # Match x-axis limits and ticks exactly to top row
        ax.set_xlim(top_ax.get_xlim())
        yr_labels = _deduplicate(x_yr, [fmt_years(y) for y in x_yr])
        ax.set_xticks(x_yr)
        ax.set_xticklabels(yr_labels, rotation=35, ha="right", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel("CRPS − VS skill" if lead == leads[0] else "", fontsize=9)
        ax.set_xlabel("Equivalent training record length", fontsize=9)

        sig_patch   = mpatches.Patch(color=DIFF_COLOR, alpha=0.6, label=f"Sig. ≠ 0 ({CI_PCT}%)")
        nosig_patch = mpatches.Patch(color="lightgrey", alpha=0.6, label="Not sig.")
        ax.legend(handles=[sig_patch, nosig_patch], fontsize=8, framealpha=0.9)

    ci_label = f"  |  {CI_PCT}% CI" if has_errors else ""
    fig.suptitle(f"L96 DDPM  |  N=40, F=8  |  $n_{{ens}}$=20{ci_label}", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    png_path = OUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")
    print(f"Saved → {png_path}")


if __name__ == "__main__":
    plot()
