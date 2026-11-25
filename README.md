# Do AI Forecast Ensembles Sample the Correct Conditional Distribution?

Code for the paper:

> **Do AI Forecast Ensembles Sample the Correct Conditional Distribution?**  
> Lucas J. Howard, Elizabeth A. Barnes — Faculty of Computing and Data Sciences, Boston University

Ensemble forecasting aims to sample the conditional distribution of outcomes; whether AI forecast ensembles do this correctly in a joint sense remains largely untested. We train a diffusion model for probabilistic subseasonal coastal sea level forecasts at eight US East Coast tide gauge stations, with sea level derived from reanalysis, and find that marginal and joint forecast quality decouple: positive skill at every station and lead time marginally, while joint spatial structure is worse than climatological draws. A shuffle-based permutation decomposition reveals this failure is invisible to the energy score but detected by the variogram score. Lorenz-96 experiments show that the gap persists when the model is conditioned on the complete system state and does not close with increased training data volume up to roughly twice the length of the ERA5 record. A diffusion model, stochastic CRPS and ES-trained models, and a deterministic emulator all show a substantially larger gap than a perfect-model dynamical ensemble, indicating that the failure arises from data-driven emulation rather than from limited training data, model architecture, or training loss.

---

## Installation

```bash
conda env create -f envs/ml_env.yml
conda activate ssh_ml
```

The `ssh_ml` environment includes the Copernicus Marine and CDS clients used by the download
scripts.

All commands below are run from the repository root with `PYTHONPATH=.`.

---

## Data

**Included in this repository** — everything needed to train and evaluate the SSH models and
to regenerate every figure:

- `data/raw/tide_gauges/` — 8 NOAA tide gauge CSV files (Eastport ME → Key West FL)
- `data/processed/eof_ssh_pcs.npz`, `eof_sst_pcs.npz` — SSH and SST EOF principal components
  (PC-only copies of the full EOF files, without the spatial patterns)
- `data/processed/eof_wind.npz` — 10 m wind EOFs
- `data/processed/gauge_ssh_interp.npy` — GLORYS SSH anomaly interpolated to the 8 gauges
- `data/processed/nao_weekly.csv`, `enso_nino34_weekly.csv`, `nao_eof1.npz` — climate indices
- `checkpoints/` — trained SSH models; `lorenz96/sweep/checkpoints/` and
  `lorenz96/checkpoints/{det,crps,es}_emulator.pt` — trained L96 models
- `results/`, `lorenz96/ensemble_results/`, `lorenz96/sweep/*.npz` — cached scores behind the figures

**Data build.** All processed files cover 1993-01-04 → 2026-01-05 (1723 weeks). Seasonal
climatologies, linear trends and EOFs are fit on the training years (1993–2018) only. The
`Submission1` release used an earlier build (ending 2025-01-06) whose climatologies and trends
were fit over 1993–2020; its model is kept as `checkpoints/ddpm_best_gauge_init.pt`.

**Not included** (large; regenerate from the raw reanalyses): the weekly anomaly fields
(`data/processed/*_anom_weekly.nc`) and the full EOF files with spatial patterns
(`eof_ssh.npz`, `eof_sst.npz`). These are only needed to rebuild the inputs from scratch or to
project new data onto the EOFs; the code falls back to the PC-only copies otherwise.

**Rebuilding the inputs from the raw data:**
```bash
# GLORYS12 ocean reanalysis (requires a Copernicus Marine account): East Coast domain,
# Niño-3.4 box, and tide gauges
python scripts/download_glorys_bulk.py --end-year 2025
python scripts/download_glorys_bulk.py --end-year 2025 --region nino34
python scripts/download_tide_gauges.py

# ERA5 sea level pressure and 10 m winds (requires a CDS API key)
python scripts/download_era5_slp.py --end-year 2025

# Anomalies, EOFs and indices (training-period fits, 1993–2018)
python -m src.preprocessing.anomalies                  # GLORYS SST/SSH weekly anomalies
                                                       # (also clears the gauge SSH cache)
python -m src.eof.compute_eofs --variable ssh --n_eofs 200 --n_modes 200
python -m src.eof.compute_eofs --variable sst
python -m src.eof.compute_wind_eofs                    # wind anomalies + EOFs from raw ERA5
python -m src.indices.nao
python -m src.indices.enso
```
GLORYS12 and ERA5 are periodically revised by their providers, so a rebuild will be close to,
but not bit-identical with, the included files.

Raw data: GLORYS12v1 (E.U. Copernicus Marine Service) and ERA5 (Copernicus Climate Change
Service). Generated using Copernicus Marine Service and Copernicus Climate Change Service
information.

---

## Reproducing the results

### SSH diffusion model

```bash
# Primary model (seed 0) and three more training seeds
python -m src.model.train --out ddpm_best_build2025.pt --seed 0
for s in 1 2 3; do python -m src.model.train --out ddpm_best_build2025_s$s.pt --seed $s; done
PYTHONPATH=.:scripts python scripts/ddpm_seed_spread.py --split test

# One-at-a-time ablation, 8 variants x 3 seeds (Figure S1b)
bash scripts/run_ablation.sh 8
```

CRPS- and ES-trained SSH generators (not in the paper; same inputs and targets as the DDPM):
```bash
for s in 0 1 2; do python -m src.model.train_crps --seed $s --n-members 8 --lr 1e-3 \
    --hidden 512 512 256 --cond-noise 0 --out checkpoints/crps_gen_m8_seed$s.pt; done
for s in 0 1 2; do python -m src.model.train_crps --seed $s --loss es --n-members 8 --lr 1e-3 \
    --hidden 512 512 256 --cond-noise 0 --out checkpoints/es_gen_m8_seed$s.pt; done
python scripts/ssh_crps_comparison.py --split test
```
Generator hyperparameters come from a seed-0 sweep on validation fair CRPS (`logs/crps_sweep*`).

### Lorenz-96

```bash
python -m lorenz96.generate_data                      # → lorenz96/l96_trajectory.npz

# Training-volume sweep (Figure 4): train 24 DDPMs, then score all of them in one pass
# (500 test ICs x 500 members, shared ICs and climatology per lead)
python -m lorenz96.sweep
python -m lorenz96.sweep_eval

# Emulators (Figures 5-6)
python -m lorenz96.train_deterministic                # → lorenz96/checkpoints/det_emulator.pt
python -m lorenz96.train_crps                         # → crps_emulator.pt
python -m lorenz96.train_crps --loss es               # → es_emulator.pt

# Model comparison (Figure 6): 2,000 test ICs x 500 members, the same ICs and
# 500-member climatology for every model (seed 42)
python -m lorenz96.ensemble_ddpm
python -m lorenz96.ensemble_crps
python -m lorenz96.ensemble_crps --ckpt lorenz96/checkpoints/es_emulator.pt \
    --out lorenz96/ensemble_results/es_results.npz
python -m lorenz96.ensemble_dynamical    --sigma-ic 0.1578 --out lorenz96/ensemble_results/dyn_results.npz
python -m lorenz96.ensemble_deterministic --sigma-ic 0.1578 --out lorenz96/ensemble_results/det_results.npz

# Perturbation sweep (Figure 5): σ_IC = 1-40 % of climatological SD, 2,000 ICs x 500 members
python -m lorenz96.dyn_sweep
python -m lorenz96.det_sweep
```
σ_IC = 0.1578 (15.8 % of the climatological SD) makes the dynamical ensemble's spread match the
DDPM's at lead 1 TU; `python -m lorenz96.compare_ensembles --match-spread` recomputes it.

### Figures

```bash
python scripts/figure1.py --split test                 # Fig 1: SSH CRPS / ES / VS skill vs lead
PYTHONPATH=.:scripts python scripts/figure2.py --split test   # Fig 2: shuffle decomposition
python scripts/plots/figure3_domain_map.py             # Fig 3: gauges, correlations, per-pair VS
# (per-pair VS scores are cached in results/vs_per_pair_test.npz; to recompute them:
#  python -m scripts.vs_per_pair --ckpt checkpoints/ddpm_best_build2025.pt --split test)
python scripts/figure4.py                              # Fig 4: L96 training-volume sweep
python scripts/figure5.py                              # Fig 5: L96 σ_IC sweep
python scripts/figure6.py                              # Fig 6: L96 model comparison
python scripts/figure_training.py                      # Fig S1: SSH training diagnostics
```
Figures 1–3 read cached scores from `results/`; rerunning inference (`figure1.py --rerun`,
`vs_per_pair`) is seeded but will differ slightly from the released caches, which were drawn
before seeding was added. Main figures are
written to `figures/main/`; the supplement (`figures/supplement/supplement.tex`, Figure S1 and
Table S1) builds with pdfLaTeX.

---

## Key results

**SSH (test set 2022–2025, 184 initializations, 500 members).** Skill relative to a
same-calendar-month climatological draw, as 1 − mean(score) / mean(climatology score):

| Lead | CRPS | ES | VS |
|---|---|---|---|
| 2 wk | +0.087 | +0.071 | −0.034 |
| 4 wk | +0.080 | +0.077 | −0.043 |
| 8 wk | +0.063 | +0.057 | −0.047 |
| 12 wk | +0.055 | +0.047 | −0.066 |

CRPS and ES skill are significant at every lead; the VS 95 % block-bootstrap CI excludes zero at
4–12 weeks. Under the shuffle decomposition the DDPM's joint contribution exceeds climatology's
by +0.002 (ES, significant) and +0.043 (VS, not significant).

**Lorenz-96 (lead 1 TU, 2,000 test ICs, 500 members; σ_IC = 15.8 % for the reference ensembles).**

| Model | CRPS skill | VS skill | VS / CRPS |
|---|---|---|---|
| DDPM | 0.435 | 0.334 | 0.77 |
| CRPS-trained emulator | 0.452 | 0.353 | 0.78 |
| ES-trained emulator | 0.440 | 0.343 | 0.78 |
| Deterministic emulator | 0.371 | 0.216 | 0.58 |
| Dynamical (perfect model) | 0.629 | 0.570 | 0.91 |

Every emulator's VS/CRPS ratio is significantly below the dynamical ensemble's. The training-volume
sweep (Figure 4) shows a significant CRPS − VS gap at leads of 1 and 2 TU for every training size
from 1,000 to 249,000 samples (≈ 0.7–170 equivalent years).

---

## Repository structure

```
.
├── src/                        SSH model package
│   ├── config.py               Hyperparameters and paths
│   ├── model/                  DDPM (network, diffusion, dataset, train, sample); CRPS/ES generator
│   ├── eof/                    EOF computation, projection, PC-only copies
│   ├── indices/                NAO and Niño-3.4 indices
│   ├── preprocessing/          Anomalies, detrending, temporal aggregation
│   ├── evaluation/             Scores (CRPS, ES, VS), shuffle decomposition, bootstrap
│   └── data/                   Data loaders (GLORYS, ERA5, tide gauges)
│
├── scripts/                    Figure and analysis scripts
│   ├── figure1.py … figure6.py Main figures (figure1.py also holds the SSH scoring)
│   ├── figure_training.py      Figure S1
│   ├── plots/                  Figure 3 domain map and other plotting helpers
│   ├── vs_per_pair.py          Per-pair VS skill (Figure 3b)
│   ├── run_ablation.sh         SSH ablation (Figure S1b)
│   └── download_*.py           Data acquisition
│
├── lorenz96/                   Lorenz-96 (N = 40, F = 8) experiments
│   ├── generate_data.py        RK4 trajectory
│   ├── model.py                ConvScoreNet + DDPM, deterministic and generative conv nets
│   ├── train*.py               DDPM, deterministic (MSE) and CRPS/ES emulator training
│   ├── sweep.py, sweep_eval.py Training-volume sweep: training, single-pass evaluation
│   ├── ensemble_*.py           DDPM, CRPS/ES, deterministic and dynamical ensembles
│   ├── dyn_sweep.py, det_sweep.py  σ_IC sweeps
│   ├── eval.py                 CRPS, VS and bootstrap utilities
│   ├── sweep/                  Sweep checkpoints and results
│   └── ensemble_results/       Model-comparison results
│
├── lorenz63/                   Early Lorenz-63 testbed (archived; see lorenz63/README.md)
├── tests/                      Pytest suite
├── checkpoints/                SSH model weights (primary: ddpm_best_build2025.pt)
├── data/                       Tide gauges and processed predictors (see Data)
├── results/                    Cached SSH scores
├── logs/                       Training logs read by the figure scripts
├── figures/main/, figures/supplement/
└── envs/                       Conda environments
```
