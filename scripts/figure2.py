"""
Figure 2 — Shuffle test: how much of each joint score's skill comes from the
ensemble's dependence between gauges.

(a) Schematic: each location's values are permuted independently across ensemble
    members. Every marginal distribution is unchanged; the dependence between
    locations is destroyed. Δ = skill(original) − skill(shuffled), applied to the
    DDPM and to a same-month climatological (GLORYS) draw.
Left column skill vs lead, right column shuffle gaps; top row ES, bottom row VS:
(b) ES skill vs lead for the DDPM, the shuffled DDPM, and the shuffled GLORYS draw
(c) VS skill vs lead, same three ensembles
(d) ES shuffle gaps (mean over leads): Δ_DDPM, Δ_GLORYS, Δ_DDPM − Δ_GLORYS, 95 % CIs
(e) VS shuffle gaps, as (d)

Skills are ratio of means against the unshuffled GLORYS draw (figure1.summarise).
Scores come from figure1's cache (results/figure3_scores_{split}.npz).

Run:
    PYTHONPATH=.:scripts python scripts/figure2.py [--split test|val]
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

import figure1
from figure1 import (FS_LABEL, FS_LEG, FS_TICK, FS_TITLE, OUT_DIR, _BAR_COLORS,
                     _BAR_COLS, _BAR_LABELS, _CLIM_COLOR, _DDPM_COLOR, style_lead_ax)

_GAP_LABELS = [r"$\Delta_\mathrm{DDPM}$", r"$\Delta_\mathrm{GLORYS}$",
               r"$\Delta_\mathrm{DDPM}-\Delta_\mathrm{GLORYS}$"]
_GAP_COLORS = [_DDPM_COLOR, _CLIM_COLOR, "#b2182b"]

# Figure-3-local font sizes, matching figure 2
F_TICK, F_LABEL, F_TITLE, F_LEG = 8, 9, 9, 7
_LINE_LABELS = ["DDPM", "DDPM shuffled", "GLORYS shuffled"]


# ── (a) schematic ─────────────────────────────────────────────────────────────

def _ensemble_matrices(n_mem: int = 4, n_loc: int = 6, seed: int = 3):
    """A spatially coherent toy ensemble and its column-shuffled version.

    Each member is a smooth along-coast pattern (shared shape, member-specific
    amplitude and offset), so rows are internally coherent. The shuffle permutes
    each column independently across members (no column left in place)."""
    x = np.linspace(-1, 1, n_loc)
    amp = np.linspace(-1.0, 1.0, n_mem)
    X = amp[:, None] * (0.8 + 0.2 * x[None, :]) + 0.15 * x[None, :]
    rng = np.random.default_rng(seed)
    S = np.empty_like(X)
    for k in range(n_loc):
        while True:
            p = rng.permutation(n_mem)
            if not np.array_equal(p, np.arange(n_mem)):
                break
        S[:, k] = X[p, k]
    return X, S


def _draw_matrix(ax, M, x0, y0, cw, ch, cmap, norm, edge):
    n_mem, n_loc = M.shape
    for i in range(n_mem):
        for j in range(n_loc):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (n_mem - 1 - i) * ch), cw, ch,
                                   facecolor=cmap(norm(M[i, j])), edgecolor="white", lw=0.6))
    ax.add_patch(Rectangle((x0, y0), n_loc * cw, n_mem * ch, fill=False, edgecolor=edge, lw=1.0))


def draw_schematic(ax) -> None:
    X, S = _ensemble_matrices()
    n_mem, n_loc = X.shape
    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(-1.2, 1.2)
    cw, ch = 0.040, 0.15
    y0 = 0.14
    xo, xs = 0.17, 0.59                       # left edges of original / shuffled
    w = n_loc * cw

    _draw_matrix(ax, X, xo, y0, cw, ch, cmap, norm, "0.2")
    _draw_matrix(ax, S, xs, y0, cw, ch, cmap, norm, "0.2")

    top = y0 + n_mem * ch
    ax.text(xo + w / 2, top + 0.04, "Original ensemble", ha="center", va="bottom",
            fontsize=F_LABEL, fontweight="bold")
    ax.text(xs + w / 2, top + 0.04, "Shuffled ensemble", ha="center", va="bottom",
            fontsize=F_LABEL, fontweight="bold")
    for x_left in (xo, xs):
        ax.text(x_left + w / 2, y0 - 0.04, "locations (gauges)", ha="center", va="top",
                fontsize=F_TICK)
    ax.text(xo - 0.015, y0 + n_mem * ch / 2, "members", ha="right", va="center",
            rotation=90, fontsize=F_TICK)

    ax.add_patch(FancyArrowPatch((xo + w + 0.03, y0 + n_mem * ch / 2),
                                 (xs - 0.03, y0 + n_mem * ch / 2),
                                 arrowstyle="-|>", mutation_scale=9, lw=1.0, color="0.25"))
    ax.text((xo + w + xs) / 2, y0 + n_mem * ch / 2 - 0.06,
            "permute each\nlocation across\nmembers",
            ha="center", va="top", fontsize=F_TICK - 0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) Shuffle test", fontweight="bold", fontsize=F_TITLE, loc="left")


# ── (b)–(e) results ───────────────────────────────────────────────────────────

def _skill_lines(ax, sm, leads, title, ylabel, outline_shuffled=False):
    ld = np.array(leads)
    # Shaded CIs at low opacity, with faint edges so overlapping bands stay visible
    for col, label, color, ls, mk in zip(_BAR_COLS, _LINE_LABELS, _BAR_COLORS,
                                         ["-", "--", ":"], ["o", "s", "^"]):
        v, lo, hi = sm["skill"][:, col], sm["skill_lo"][:, col], sm["skill_hi"][:, col]
        ax.plot(ld, v, ls + mk, color=color, lw=1.2, ms=3, label=label, zorder=3)
        if outline_shuffled and col != _BAR_COLS[0]:
            # overlapping bands (ES): shuffled CIs as unfilled bounds
            for edge in (lo, hi):
                ax.plot(ld, edge, color=color, lw=0.9, alpha=0.9, ls=ls, zorder=2)
        elif outline_shuffled:
            ax.fill_between(ld, lo, hi, color=color, alpha=0.15, lw=0, zorder=1)
        else:
            ax.fill_between(ld, lo, hi, color=color, alpha=0.07, lw=0, zorder=1)
            for edge in (lo, hi):
                ax.plot(ld, edge, color=color, lw=0.5, alpha=0.45, ls=ls, zorder=2)
    style_lead_ax(ax, leads, ylabel, title)
    ax.set_xticklabels([str(l) for l in leads])
    ax.set_xlabel("Lead time (weeks)")
    ax.tick_params(labelsize=F_TICK)
    for t in ax.get_xticklabels():
        t.set_fontsize(F_TICK)
    ax.xaxis.label.set_fontsize(F_LABEL)
    ax.yaxis.label.set_fontsize(F_LABEL)
    _top_title(ax, title)


def _top_title(ax, title):
    """Title whose first line sits at a fixed height above the axes, so one- and
    two-line titles across the row line up at the top."""
    ax.set_title(title, fontsize=F_TITLE, fontweight="bold")


def _gap_bars(ax, sm, title, ylabel):
    x = np.arange(3)
    v, lo, hi = sm["delta"], sm["delta_lo"], sm["delta_hi"]
    bars = ax.bar(x, v, 0.7, color=_GAP_COLORS, alpha=0.85,
                  yerr=[np.maximum(0, v - lo), np.maximum(0, hi - v)],
                  error_kw=dict(elinewidth=0.6, capsize=2, ecolor="0.25"))
    # Diff. is a difference of gaps, not a gap: hatched outline + divider
    bars[2].set_facecolor("white")
    bars[2].set_edgecolor(_GAP_COLORS[2])
    bars[2].set_hatch("////")
    bars[2].set_linewidth(0.8)
    ax.axvline(1.5, color="0.6", lw=0.6, ls="-")
    top = max(hi.max(), 0)
    for xi, l, h in zip(x, lo, hi):
        if l > 0 or h < 0:
            ax.text(xi, h + 0.03 * top, "*", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["DDPM", "GLORYS", "Diff."], fontsize=F_TICK)
    ax.set_ylabel(ylabel, fontsize=F_LABEL)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, steps=[1, 2, 5, 10]))
    ax.set_ylim(min(lo.min(), 0) * 1.3, top * 1.18)
    _top_title(ax, title)
    ax.tick_params(labelsize=F_TICK)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)


def make_figure(summ: dict) -> None:
    leads = summ["leads"]
    fig = plt.figure(figsize=(7.0, 6.6))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.36, 1.0], hspace=0.18)
    draw_schematic(fig.add_subplot(outer[0]))
    # rows = score (ES, VS); columns = skill vs lead, shuffle gap
    grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1],
                                            width_ratios=[1, 0.6], wspace=0.32, hspace=0.32)
    # left column: skill vs lead (b ES, c VS); right column: shuffle gaps (d ES, e VS)
    ax_b = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[1, 0], sharex=ax_b)
    ax_d = fig.add_subplot(grid[0, 1])
    ax_e = fig.add_subplot(grid[1, 1], sharex=ax_d)

    _skill_lines(ax_b, summ["es"], leads, "(b) Energy Score", "", outline_shuffled=True)
    _skill_lines(ax_c, summ["vs"], leads, "(c) Variogram Score", "")
    _gap_bars(ax_d, summ["es"], r"(d) $\Delta$ Energy Score", "")
    _gap_bars(ax_e, summ["vs"], r"(e) $\Delta$ Variogram Score", "")
    # shared x: labels only on the bottom row
    for ax in (ax_b, ax_d):
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    # one y-label per column, centred on the pair, left of the wider tick labels
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for top_ax, bot_ax, text in ((ax_b, ax_c, "Skill Score"),
                                 (ax_d, ax_e, r"$\Delta$ skill (mean over leads)")):
        x_left = min(a.yaxis.get_tightbbox(rend).x0 for a in (top_ax, bot_ax))
        x_fig = fig.transFigure.inverted().transform((x_left, 0))[0] - 0.008
        y_mid = (top_ax.get_position().y1 + bot_ax.get_position().y0) / 2
        fig.text(x_fig, y_mid, text, rotation=90, ha="right", va="center", fontsize=F_LABEL)

    h, l = ax_b.get_legend_handles_labels()
    h.append(Line2D([], [], color="k", lw=0.8, ls="--"))
    l.append("GLORYS unshuffled (skill = 0)")
    pb, pe = ax_c.get_position(), ax_e.get_position()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=((pb.x0 + pe.x1) / 2, 0.035),
               ncol=4, fontsize=F_LEG, frameon=True)

    fig.savefig(OUT_DIR / "figure2.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "figure2.png", dpi=300, bbox_inches="tight")
    print("Saved figures/main/figure2.pdf and figures/main/figure2.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", help="val or test")
    args = parser.parse_args()
    make_figure(figure1.summarise(figure1.load_scores(args.split)))


if __name__ == "__main__":
    main()
