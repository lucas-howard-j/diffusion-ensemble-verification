# Do AI Forecast Ensembles Sample the Correct Conditional Distribution?

Code for the paper:

> **Do AI Forecast Ensembles Sample the Correct Conditional Distribution?**  
> Lucas Howard, Elizabeth Barnes — Faculty of Computing and Data Sciences, Boston University

The fundamental problem ensemble forecasting attempts to solve is to generate samples from a conditional distribution that represents the true uncertainty in outcomes. For AI forecast models, the question of whether the ensemble samples from the correct joint distribution remains largely untested. We train a diffusion model to generate probabilistic subseasonal forecasts of coastal sea level at eight tide gauge stations along the US East Coast, with sea level derived from reanalysis, and find that marginal and joint forecast quality can decouple: the model achieves positive skill at every station and lead time when evaluated marginally, while its joint spatial structure is worse than drawing randomly from the climatological record. A shuffle-based permutation decomposition reveals this failure is invisible to the commonly used energy score metric but clearly detected by the variogram score, which is specifically sensitive to dependence structure. Lorenz-96 experiments across training volumes equivalent to 0.7–170 years of ERA5 demonstrate that the decoupling does not close with additional data and is reproduced by a linear baseline, indicating structural inadequacy of the learned distribution rather than insufficient training data. A further comparison in the L96 system between a dynamical model and a deterministic emulator reproduces the gap in the emulator but not the dynamical model, suggesting that it is a failure mode specific to learned emulators rather than of a specific architecture or method.

---

## Installation

```bash
conda env create -f envs/ml_env.yml
conda activate ssh_ml
```

For data download only (separate environment):
```bash
conda env create -f envs/glorys_dl_env.yml
conda activate copernicus
```

---

## Data

**Included in this repository:**
- `data/raw/tide_gauges/` — 8 NOAA tide gauge CSV files (Eastport ME → Key West FL)
- `data/processed/nao_weekly.csv`, `enso_nino34_weekly.csv` — precomputed climate indices
- `data/processed/eof_wind.npz` — wind EOF basis (9.8 MB)

**Auto-generated on first run (not tracked):**
- `data/processed/gauge_ssh_interp.npy` — SSH anomaly field interpolated to gauge locations; cached by `src/model/dataset.py` on first data load, requires `ssh_anom_weekly.nc`

**Available on Zenodo (link TBA at publication):**
- `data/processed/eof_ssh.npz`, `eof_sst.npz` — SSH/SST EOF bases (316 MB / 158 MB)
- `data/processed/ssh_anom_weekly.nc`, `sst_anom_weekly.nc`, `wind_anom_weekly.nc` — weekly anomaly fields
- `checkpoints/` — trained model weights (primary: `ddpm_best_gauge_init.pt`)

**Download raw data from scratch:**
```bash
# GLORYS ocean reanalysis (requires Copernicus Marine account)
conda activate copernicus
python scripts/download_glorys_bulk.py
python scripts/download_tide_gauges.py

# ERA5 atmospheric reanalysis (requires CDS API key)
python scripts/download_era5_slp.py
```

---

## Reproducing Results

All commands assume `conda activate ssh_ml` and `PYTHONPATH=.` set from the repo root.

### 1. Preprocess & compute EOFs

```bash
python -m src.eof.compute_eofs --variable ssh --n_eofs 200 --n_modes 200
python -m src.eof.compute_eofs --variable sst
python -m src.eof.compute_wind_eofs
```

### 2. Train the SSH DDPM

```bash
PYTHONPATH=. python -m src.model.train
# Best checkpoint saved to checkpoints/ddpm_best_gauge_init.pt
```

### 3. Evaluate the SSH model

```bash
CKPT=checkpoints/ddpm_best_gauge_init.pt

PYTHONPATH=. python scripts/rmse_vs_lead.py --ckpt $CKPT --split test
PYTHONPATH=. python scripts/ensemble_diagnostics.py --ckpt $CKPT
PYTHONPATH=. python scripts/gauge_correlation.py
PYTHONPATH=. python scripts/forecast_coherence_gauge.py
PYTHONPATH=. python scripts/rank_histograms.py
```

### 4. Lorenz-96 experiments

The L96 data and sweep checkpoints are gitignored (large binaries). Generate them with:

```bash
# Generate trajectory
PYTHONPATH=. python -m lorenz96.generate_data

# Training-volume sweep (produces lorenz96/sweep/checkpoints/ and sweep_results.npz)
PYTHONPATH=. python -m lorenz96.sweep

# Bootstrap CIs for the sweep (produces sweep_errors.npz)
PYTHONPATH=. python -m lorenz96.sweep.bootstrap_eval

# Train deterministic emulator (produces lorenz96/checkpoints/det_emulator.pt)
PYTHONPATH=. python -m lorenz96.train_deterministic

# Ensemble comparison: dynamical and det. emulator results are pre-computed
# and stored in lorenz96/ensemble_results/*.npz.  To regenerate:
PYTHONPATH=. python -m lorenz96.compare_ensembles \
    --ddpm-ckpt lorenz96/sweep/checkpoints/ddpm_lead20_249000.pt \
    --det-ckpt  lorenz96/checkpoints/det_emulator.pt
```

### 5. Generate figures

```bash
# Figure 1 — SSH central result (CRPS/ES/VS skill, decomposition)
PYTHONPATH=. python scripts/figure1.py --split test

# Figure 2 — SSH domain map and gauge correlation
PYTHONPATH=. python scripts/plots/figure2_domain_map.py

# Figure 3 — L96 training-volume sweep
PYTHONPATH=. python scripts/figure3.py

# Figure 4 — Ensemble comparison: DDPM vs dynamical vs det. emulator
PYTHONPATH=. python scripts/figure4_ensemble_comparison.py
```

Publication figures are written to `figures/main/`.

---

## Repository Structure

```
.
├── src/                        SSH model package
│   ├── config.py               Hyperparameters and paths
│   ├── model/                  DDPM: network, diffusion, dataset, train, sample
│   ├── eof/                    EOF computation and projection
│   ├── indices/                NAO and ENSO index computation
│   ├── preprocessing/          Anomalies, detrending, temporal aggregation
│   └── data/                   Data loaders (GLORYS, ERA5, tide gauges)
│
├── scripts/                    Analysis and figure scripts
│   ├── figure1.py              Fig 1 — SSH ES/VS/CRPS skill (main result)
│   ├── figure1_inset.py        Fig 1 variant — with inset panel for joint contribution
│   ├── figure3.py              Fig 3 — L96 training-volume sweep
│   ├── figure4_ensemble_comparison.py  Fig 4 — DDPM vs dynamical vs det. emulator
│   ├── ensemble_diagnostics.py ES, VS, spread-error diagnostics
│   ├── rmse_vs_lead.py         CRPS and RMSE skill vs lead
│   ├── gauge_correlation.py    Marginal Pearson r vs NOAA gauges
│   ├── forecast_coherence_gauge.py  Inter-station coherence vs lead
│   ├── rank_histograms.py      Marginal and multivariate rank histograms
│   ├── plots/                  Supporting figure scripts (domain map, schematics)
│   └── download_*.py           Data acquisition scripts
│
├── lorenz96/                   Lorenz-96 (40D) controlled experiments
│   ├── model.py                ConvScoreNet, DDPM, ConvDeterministicNet
│   ├── eval.py                 crps_ensemble, variogram_score, vs_per_ic,
│   │                             bootstrap_skill_ci
│   ├── train.py                DDPM training
│   ├── train_deterministic.py  MSE deterministic emulator training
│   ├── ensemble_dynamical.py   Perfect-model ensemble (vectorised RK4)
│   ├── ensemble_deterministic.py  Det. emulator ensemble (IC perturbation)
│   ├── compare_ensembles.py    Orchestrator: runs all three methods and
│   │                             calibrates σ_ic (results pre-computed below)
│   ├── weight_sensitivity.py   CRPS/VS gradient w.r.t. model weights (L96)
│   ├── sweep/                  Training-data-volume sweep
│   │   ├── checkpoints/        DDPM weights (lead × n_train grid)
│   │   ├── sweep_results.npz   Point estimates
│   │   ├── sweep_errors.npz    Bootstrap CI distributions (1000 draws)
│   │   └── *.py                Sweep, bootstrap eval, MVN baseline, plotting
│   └── ensemble_results/       Pre-computed ensemble comparison results
│       ├── dyn_results.npz         Dynamical, σ_ic=0.158 (spread-matched)
│       ├── dyn_results_sig0p01.npz Dynamical, σ_ic=0.01 (linear regime)
│       ├── det_results.npz         Det. emulator, σ_ic=0.158
│       └── det_results_sig0p01.npz Det. emulator, σ_ic=0.01
│
├── lorenz63/                   Lorenz-63 testbed (archived — see lorenz63/README.md)
│
├── tests/                      Pytest suite (142 tests)
│   ├── test_l96_eval.py        lorenz96 score functions
│   ├── test_l96_model.py       DetResBlock, ConvDeterministicNet
│   ├── test_scores.py          src score functions (CRPS, ES, VS)
│   ├── test_metrics.py         Bootstrap CI, correlation utilities
│   ├── test_network.py         SSH denoiser architecture
│   ├── test_diffusion.py       DDPM forward/reverse process
│   └── test_model_smoke.py     End-to-end ensemble output sanity
│
├── notebooks/                  Exploratory Jupyter notebooks
│
├── checkpoints/                SSH model weights (.pt files; large files on Zenodo)
│   └── ddpm_best_gauge_init.pt Primary model (val loss 0.1588)
│
├── data/
│   ├── raw/tide_gauges/        NOAA tide gauge observations (8 stations)
│   └── processed/              Precomputed indices and EOFs (large files on Zenodo)
│
├── results/                    Cached evaluation scores for figure reproducibility
│   ├── figure3_scores_test.npz — SSH skill scores, test split
│   └── figure3_scores_val.npz  — SSH skill scores, validation split
│
├── figures/main/               Publication-ready figures (PDF + PNG)
│   ├── figure1.pdf/png         SSH central result (paper Fig 1)
│   ├── figure1_inset.pdf/png   SSH central result with inset variant
│   ├── figure2.pdf/png         SSH domain map and arc diagram (paper Fig 2)
│   ├── figure3.pdf/png         L96 training-volume sweep (paper Fig 3)
│   └── figure4.pdf/png         Ensemble comparison (paper Fig 4)
│
├── envs/ml_env.yml             Conda environment (model training)
└── envs/glorys_dl_env.yml      Conda environment (data download)
```

---

## Model

The DDPM uses an MLP denoiser (`src/model/network.py`) with:
- **Architecture**: hidden dims (512, 512, 256), ~528k parameters, BetaSiLU activations
- **Conditioning**: 131-dim vector — SSH PCs (15 modes × 4 lags), SST PCs (15 × 3), wind PCs (10), NAO (3 lags), Niño3.4 (4 lags), gauge SSH at init time (8), lead (1)
- **Target**: SSH anomaly at 8 gauge locations (z-scored)
- **Training**: TRAIN 1993–2018, VAL 2019–2021, TEST 2022–2024

Test set CRPS skill vs GLORYS seasonal climatology: mean **+0.064** across all stations and leads (Key West strongest at +0.178 at 2 wk).

---

## Key Results (Lorenz-96, lead = 1.0 TU)

| Method | CRPS skill | VS skill | VS/CRPS |
|---|---|---|---|
| DDPM (one-shot) | +0.424 | +0.319 | 0.75 |
| Dynamical, σ_ic = 0.01 (linear) | +0.975 | +0.995 | 1.02 ≈ 1 |
| Dynamical, σ_ic = 0.158 (spread-matched) | +0.629 | +0.570 | 0.91 |
| Det. emulator, σ_ic = 0.158 | +0.371 | +0.216 | 0.58 |

The dynamical ensemble in the linear regime (VS/CRPS ≈ 1) shows that chaos alone does not cause joint-skill deficit. The gap is present in the DDPM and deterministic emulator but not in the dynamical model.

