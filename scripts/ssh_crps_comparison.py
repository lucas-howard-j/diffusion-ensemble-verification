"""
SSH emulators: DDPM vs CRPS- and ES-trained generators, marginal and joint skill,
with a calibrated-dependence reference for the shuffle decomposition.

Why a reference: the raw shuffle gap Δ = skill(ens) − skill(ens shuffled)
scales with (ensemble spread / climatological SD)², so it cannot compare the
dependence of two ensembles with different spread.  For every model we build
a reference ensemble with the SAME per-case mean and per-gauge spread but
calibrated dependence, and report

    Δ / Δ_ref,   Δ_ref = skill(ref) − skill(ref shuffled)

≈ 1  → the model's inter-gauge dependence carries as much skill as calibrated
       dependence would at its own mean and spread;
< 1  → too little (0 = no better than independent members);
> 1  → more than calibrated (possible if the reference correlation is mis-set).

Reference construction (per init, lead): member deviations E = X − mean are
standardised per gauge, whitened by their own sample correlation, and
re-coloured to R_lead, the correlation of the model's ensemble-mean errors
(truth − mean) at that lead.  For a reliable ensemble, member deviations and
ensemble-mean errors share this correlation.  R_lead is estimated
leave-one-year-out so the case being scored never informs its own reference.
Per-gauge sample SDs and the ensemble mean are preserved exactly.

Skill reference (denominator) and shuffles follow scripts/figure1.py:
same-calendar-month GLORYS training-period draws, 500 members.  All skills are
ratio-of-means, 1 − mean(score)/mean(clim score), as in figure1.py.

Run:
    PYTHONPATH=. python scripts/ssh_crps_comparison.py [--split test] [--rerun]

Scores are cached per model NAME; models added to MODELS since the last run are
scored and appended without rescoring the others.  A retrained checkpoint is
not detected: use --rerun (regenerates ensembles and all scores) after retraining.

Outputs:
    results/ssh_members_{model}_{split}.npz     cached ensembles (physical units)
    results/ssh_crps_comparison_{split}.npz     per-init scores
    figures/supplement/figure_ssh_crps.{pdf,png}   CRPS and VS skill (test split, as Fig 1);
                                                   Δ/Δ_ref and spread/RMSE are printed only
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CHECKPOINTS, CRPS_SEEDS, LEAD_WEEKS, TRAIN_YEARS
from src.evaluation.metrics import ratio_of_means_skill
from src.evaluation.scores import crps, energy_score, variogram_score
from src.model.dataset import SSHDiffusionDataset
from src.model.sample import load_checkpoint

N_MEMBERS  = 500
BLOCK_SIZE = 3
N_BOOT     = 1000
CI_ALPHA   = 0.05
CACHE_DIR  = Path("results")
OUT_DIR    = Path("figures/supplement")

MODELS = {"ddpm": CHECKPOINTS / "ddpm_best_build2025.pt"}   # trained on the 2025 data build
MODELS.update({f"crps_s{s}": CHECKPOINTS / f"crps_gen_seed{s}.pt" for s in CRPS_SEEDS})
# M=8 members per training sample, DDPM-width trunk, lr 1e-3, no input noise
MODELS.update({f"crps8_s{s}": CHECKPOINTS / f"crps_gen_m8_seed{s}.pt" for s in CRPS_SEEDS})
# Same generator and settings as crps8, trained on the fair energy score
MODELS.update({f"es8_s{s}": CHECKPOINTS / f"es_gen_m8_seed{s}.pt" for s in CRPS_SEEDS})

# score-array columns
COLS = ["model", "model_shuf", "clim", "clim_shuf", "ref", "ref_shuf"]


# ── Ensembles ────────────────────────────────────────────────────────────────

def _cases(ds: SSHDiffusionDataset):
    """Init times with every lead available, and their dataset indices."""
    t_to_lead_idx: dict[int, dict[int, int]] = defaultdict(dict)
    for idx, (t, lead) in enumerate(ds.pairs):
        t_to_lead_idx[t][lead] = idx
    leads = sorted(LEAD_WEEKS)
    ts = sorted(t for t, ld in t_to_lead_idx.items() if all(l in ld for l in leads))
    return ts, leads, t_to_lead_idx


def generate_members(ckpt_path: Path, split: str) -> dict:
    model, schedule, ckpt, device = load_checkpoint(ckpt_path)
    gauge_mu, gauge_std = ckpt["gauge_mu"], ckpt["gauge_std"]
    target_dim = int(ckpt["target_dim"])
    ds = SSHDiffusionDataset(split)
    ts, leads, t_to_lead_idx = _cases(ds)
    print(f"{ckpt_path.name}: {len(ts)} inits on {device}")

    torch.manual_seed(0)
    members, truth = [], []
    with torch.no_grad():
        for i, t in enumerate(ts):
            if i % 20 == 0:
                print(f"  {i}/{len(ts)}", flush=True)
            items = [ds[t_to_lead_idx[t][lead]] for lead in leads]
            cond = torch.stack([c for c, _ in items]).to(device)        # (n_leads, cond_dim)
            if schedule is None:
                x = model.sample(cond, N_MEMBERS)                       # (n_leads, M, d)
            else:
                x = schedule.sample(model, cond.repeat_interleave(N_MEMBERS, 0), target_dim)
                x = x.view(len(leads), N_MEMBERS, target_dim)
            members.append(x.cpu().numpy() * gauge_std + gauge_mu)
            truth.append(np.stack([y.numpy() for _, y in items]) * gauge_std + gauge_mu)
    return dict(
        members=np.stack(members).astype(np.float32),   # (N, n_leads, M, d)
        truth=np.stack(truth),                          # (N, n_leads, d)
        t=np.array(ts), leads=np.array(leads),
        year=np.array([ds.time[t].year for t in ts]),
        month=np.array([[ds.time[t + l].month for l in leads] for t in ts]),
    )


def load_members(name: str, split: str, rerun: bool) -> dict:
    path = CACHE_DIR / f"ssh_members_{name}_{split}.npz"
    if path.exists() and not rerun:
        return dict(np.load(path))
    out = generate_members(MODELS[name], split)
    np.savez_compressed(path, **out)
    return out


def glorys_pool(split: str) -> dict[int, np.ndarray]:
    """Same-month GLORYS training-period states (physical units), as in figure1.py."""
    ds = SSHDiffusionDataset(split)
    train = ds.time.year.isin(TRAIN_YEARS)
    phys = ds.gauge_ssh[train] * ds.gauge_std + ds.gauge_mu
    months = ds.time[train].month
    return {m: phys[months == m] for m in range(1, 13)}


# ── Calibrated-dependence reference ──────────────────────────────────────────

def recolour(members: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Keep the ensemble mean and per-gauge sample SD; set sample correlation to R."""
    mu = members.mean(0)
    E = members - mu
    s = E.std(0, ddof=1)
    Zs = E / s
    L_e = np.linalg.cholesky(np.corrcoef(Zs, rowvar=False))
    W = np.linalg.solve(L_e, Zs.T).T            # sample correlation = I
    return mu + (W @ np.linalg.cholesky(R).T) * s


def error_correlations(d: dict) -> np.ndarray:
    """Leave-one-year-out correlation of ensemble-mean errors.  → (N, n_leads, d, d)."""
    err = d["truth"] - d["members"].mean(2)     # (N, n_leads, d)
    R = np.empty(err.shape[:2] + (err.shape[2],) * 2)
    for yr in np.unique(d["year"]):
        out = d["year"] == yr
        for j in range(err.shape[1]):
            R[out, j] = np.corrcoef(err[~out, j], rowvar=False)
    return R


def _shuffle(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.stack([arr[rng.permutation(len(arr)), k] for k in range(arr.shape[1])], 1)


def score_model(d: dict, pool: dict[int, np.ndarray]) -> dict:
    N, n_leads, M, n_st = d["members"].shape
    R = error_correlations(d)
    out = dict(crps=np.empty((N, n_leads, n_st)), crps_clim=np.empty((N, n_leads, n_st)),
               es=np.empty((N, n_leads, len(COLS))), vs=np.empty((N, n_leads, len(COLS))),
               spread=np.empty((N, n_leads, n_st)), sqerr=np.empty((N, n_leads, n_st)))
    for i in range(N):
        if i % 20 == 0:
            print(f"  scoring {i}/{N}", flush=True)
        for j in range(n_leads):
            t, lead = int(d["t"][i]), int(d["leads"][j])
            x, y = d["members"][i, j].astype(float), d["truth"][i, j]
            p = pool[int(d["month"][i, j])]
            rng_ref = np.random.default_rng(hash((t, lead)) % (2 ** 31))  # same draws as figure1.py
            clim = p[rng_ref.choice(len(p), M, replace=True)]
            rng = np.random.default_rng([t, lead, 7])
            ref = recolour(x, R[i, j])
            ens = [x, _shuffle(x, rng), clim, _shuffle(clim, rng), ref, _shuffle(ref, rng)]
            out["crps"][i, j] = crps(x, y)
            out["crps_clim"][i, j] = crps(clim, y)
            out["es"][i, j] = [energy_score(e, y) for e in ens]
            out["vs"][i, j] = [variogram_score(e, y) for e in ens]
            out["spread"][i, j] = x.var(0, ddof=1)
            out["sqerr"][i, j] = (x.mean(0) - y) ** 2
    return out


# ── Summaries ────────────────────────────────────────────────────────────────

def _block_idx(N: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(0, N - BLOCK_SIZE) + 1, size=int(np.ceil(N / BLOCK_SIZE)))
    return np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in starts])[:N]


def summarise(sc: dict, rng: np.random.Generator) -> dict:
    """Point estimates and block-bootstrap CIs.  Skills per lead; Δ ratios pooled over leads."""
    def stats(idx):
        crps_skill = (1 - sc["crps"][idx].mean(0) / sc["crps_clim"][idx].mean(0)).mean(-1)
        r = {"crps_skill": crps_skill,
             "spread_skill": np.sqrt(sc["spread"][idx].mean((0, 2)) / sc["sqerr"][idx].mean((0, 2)))}
        for name in ["es", "vs"]:
            m = ratio_of_means_skill(sc[name][idx])     # (n_leads, n_cols); clim = column 2
            r[f"{name}_skill"] = m[:, 0]
            r[f"{name}_skill_shuf"] = m[:, 1]
            r[f"{name}_skill_ref"] = m[:, 4]
            d_mod, d_ref = m[:, 0] - m[:, 1], m[:, 4] - m[:, 5]
            r[f"{name}_delta"] = d_mod
            r[f"{name}_delta_ref"] = d_ref
            r[f"{name}_ratio_lead"] = d_mod / d_ref
            r[f"{name}_ratio"] = np.array(d_mod.mean() / d_ref.mean())
        return r

    N = len(sc["crps"])
    point = stats(np.arange(N))
    boot = [stats(_block_idx(N, rng)) for _ in range(N_BOOT)]
    ci = {k: np.percentile([b[k] for b in boot], [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)], 0)
          for k in point}
    return dict(point=point, ci=ci)


# ── Figure ───────────────────────────────────────────────────────────────────

_COLORS = {"ddpm": "#2166ac"}
# CRPS-trained groups plotted: name prefix → (label, colour, marker).
# The M=2 generators (crps_s*) are still scored and printed but left off the
# figure; they belong with the ablation in the SI.
_GROUPS = {"crps8_s": ("CRPS-trained, M=8", "#b2182b", "^"),
           "es8_s":   ("ES-trained, M=8",   "#1b7837", "D")}
FS_TICK, FS_LABEL, FS_TITLE, FS_LEG = 6, 7, 7, 5.5


def make_figure(summ: dict, leads: list[int]) -> None:
    ld = np.array(leads)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.4))

    def lines(ax, key, ls="-", label_suffix="", band=True):
        p, ci = summ["ddpm"]["point"][key], summ["ddpm"]["ci"][key]
        ax.plot(ld, p, ls + "o", color=_COLORS["ddpm"], lw=1.2, ms=3,
                label="DDPM" + label_suffix + (" (95% CI)" if band else ""))
        if band:
            ax.fill_between(ld, ci[0], ci[1], color=_COLORS["ddpm"], alpha=0.15)
        for prefix, (label, color, marker) in _GROUPS.items():
            names = [k for k in summ if k.startswith(prefix)]
            if not names:
                continue
            vals = np.array([summ[n]["point"][key] for n in names])
            ax.plot(ld, vals.mean(0), ls + marker, color=color, lw=1.2, ms=3,
                    label=label + label_suffix + (f" ({len(names)}-seed mean, range)" if band else ""))
            if band:
                ax.fill_between(ld, vals.min(0), vals.max(0), color=color, alpha=0.15)

    def style(ax, ylabel, title, zero=True):
        if zero:
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(ld)
        ax.set_xticklabels([f"{l}wk" for l in leads], fontsize=FS_TICK)
        ax.set_xlabel("Lead time", fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontweight="bold", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.2, lw=0.5)

    lines(ax_a, "crps_skill")
    style(ax_a, "CRPS skill", "(a) Marginal CRPS skill")

    lines(ax_b, "vs_skill")
    lines(ax_b, "vs_skill_shuf", ls="--", label_suffix=" shuffled", band=False)
    style(ax_b, "VS skill", "(b) Variogram score skill")
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=FS_LEG, frameon=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "figure_ssh_crps.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "figure_ssh_crps.png", bbox_inches="tight", dpi=300)
    print(f"Saved {OUT_DIR / 'figure_ssh_crps.pdf'}")


def print_table(summ: dict, leads: list[int]) -> None:
    def fmt(n, k):
        p, ci = summ[n]["point"][k], summ[n]["ci"][k]
        if np.ndim(p) == 0:
            return f"{float(p):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
        return "  ".join(f"{v:+.3f}" for v in p)
    keys = ["crps_skill", "es_skill", "es_skill_ref", "vs_skill", "vs_skill_ref", "spread_skill",
            "es_delta", "es_delta_ref", "vs_delta", "vs_delta_ref",
            "es_ratio_lead", "vs_ratio_lead", "es_ratio", "vs_ratio"]
    print(f"\nPer-lead values at leads {leads}; ratios pooled over leads with 95% block-bootstrap CI")
    for k in keys:
        print(f"\n{k}")
        for n in summ:
            print(f"  {n:9s} {fmt(n, k)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", help="val or test")
    parser.add_argument("--rerun", action="store_true", help="regenerate ensembles and scores")
    args = parser.parse_args()

    score_path = CACHE_DIR / f"ssh_crps_comparison_{args.split}.npz"
    scores: dict[str, dict] = defaultdict(dict)
    leads = None
    if score_path.exists() and not args.rerun:
        flat = dict(np.load(score_path))
        for k, v in flat.items():
            if "/" in k:
                n, f = k.split("/")
                scores[n][f] = v
        leads = list(flat["leads"])
    missing = [n for n in MODELS if n not in scores]
    if missing:
        pool = glorys_pool(args.split)
        for name in missing:
            d = load_members(name, args.split, args.rerun)
            if leads is not None and list(d["leads"]) != leads:
                raise ValueError(f"{name} has leads {list(d['leads'])}, cached scores have {leads}; "
                                 "rerun with --rerun")
            print(f"Scoring {name}")
            scores[name] = score_model(d, pool)
            leads = list(d["leads"])
        np.savez(score_path, leads=np.array(leads),
                 **{f"{n}/{f}": v for n, s in scores.items() for f, v in s.items()})
    scores = {n: scores[n] for n in MODELS}     # MODELS order; drop stale cache entries

    rng = np.random.default_rng(42)
    summ = {n: summarise(s, rng) for n, s in scores.items()}
    print_table(summ, leads)
    if args.split == "test":
        make_figure(summ, leads)


if __name__ == "__main__":
    main()
