# SSH Diffusion: Probabilistic S2S Coastal Sea Level Forecasting

Code for the paper (in preparation):

> **Decoupled Marginal and Joint Skill in a Diffusion Ensemble for Subseasonal Coastal Sea Level**  
> Barnes Lab — paper in preparation

A denoising diffusion probabilistic model (DDPM) trained to predict SSH anomalies at 8 NOAA tide gauges along the US East Coast at 2–12 week lead times. The paper introduces a permutation-based decomposition using energy score (ES) and variogram score (VS) to expose a failure mode in ML ensemble forecasts: positive marginal CRPS skill coexisting with near-zero or negative joint-structure skill. Controlled Lorenz-96 experiments show this decoupling does not close with increasing training data volume.

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
- `data/processed/trend_*.nc` — linear trend fields for detrending

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
# Compute SSH EOFs (200 modes; ~10 min)
python -m src.eof.compute_eofs --variable ssh --n_eofs 200 --n_modes 200
# Compute SST and wind EOFs similarly
python -m src.eof.compute_eofs --variable sst
python -m src.eof.compute_wind_eofs
```

### 2. Train

```bash
PYTHONPATH=. python -m src.model.train
# Best checkpoint saved to checkpoints/ddpm_best_gauge_init.pt
```

### 3. Evaluate

```bash
CKPT=checkpoints/ddpm_best_gauge_init.pt

PYTHONPATH=. python scripts/rmse_vs_lead.py --ckpt $CKPT --split test
PYTHONPATH=. python scripts/ensemble_diagnostics.py --ckpt $CKPT
PYTHONPATH=. python scripts/gauge_correlation.py
PYTHONPATH=. python scripts/forecast_coherence_gauge.py
PYTHONPATH=. python scripts/rank_histograms.py
```

### 4. Generate figures

```bash
# Figure 3 — SSH central result (reads from results/figure3_scores_test.npz)
PYTHONPATH=. python scripts/figure3.py --split test

# Figure 4 — L96 sweep (reads from lorenz96/sweep/sweep_results.npz)
PYTHONPATH=. python scripts/figure4.py

# Figure 1 — Verification protocol schematic
PYTHONPATH=. python scripts/plots/figure1_protocol.py

# Figure 2 — Domain map + per-pair VS diagnostic
PYTHONPATH=. python scripts/plots/figure2_domain_map.py
```

Publication figures are written to `figures/main/`.

---

## Repository Structure

```
phase1/
├── src/                        Core model package
│   ├── config.py               All hyperparameters and paths
│   ├── model/                  DDPM: network, diffusion, dataset, train, sample
│   ├── eof/                    EOF computation and projection
│   ├── indices/                NAO and ENSO index computation
│   ├── preprocessing/          Anomalies, detrending, temporal aggregation
│   └── data/                   Data loaders (GLORYS, ERA5, tide gauges)
│
├── scripts/                    Analysis and figure scripts
│   ├── figure3.py              Fig 3 — SSH ES/VS/CRPS skill (main result)
│   ├── figure4.py              Fig 4 — L96 sweep (main result)
│   ├── ensemble_diagnostics.py ES, VS, spread-error diagnostics
│   ├── rmse_vs_lead.py         CRPS and RMSE skill vs lead
│   ├── gauge_correlation.py    Marginal Pearson r vs NOAA gauges
│   ├── forecast_coherence_gauge.py  Inter-station coherence vs lead
│   ├── rank_histograms.py      Marginal and multivariate rank histograms
│   ├── plots/                  Publication figure scripts
│   └── download_*.py           Data acquisition scripts
│
├── lorenz96/                   Lorenz-96 (40D) controlled experiments
│   ├── sweep/                  Training-data-volume sweep (figures/data for Fig 4)
│   └── *.py                    Generate, train, evaluate L96 DDPM
│
├── lorenz63/                   Lorenz-63 testbed (archived — see lorenz63/README.md)
│
├── notebooks/                  Exploratory Jupyter notebooks
│
├── checkpoints/                Trained model weights (.pt files)
│   └── ddpm_best_gauge_init.pt Primary model (val loss 0.1588)
│
├── data/
│   ├── raw/tide_gauges/        NOAA tide gauge observations (8 stations)
│   └── processed/              Precomputed indices and EOFs (large files on Zenodo)
│
├── results/                    Cached evaluation scores for figure reproducibility
│   ├── figure3_scores_test.npz
│   └── figure3_scores_val.npz
│
├── figures/main/               Publication-ready figures (PDF + PNG)
│   ├── figure1.pdf/png         Verification protocol schematic
│   ├── figure2.pdf/png         SSH domain map
│   ├── figure3.pdf/png         SSH central result
│   └── figure4.pdf/png         L96 sweep
│
├── logs/beta_history.csv       BetaSiLU β convergence history
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

## Citation

Paper in preparation. Citation will be added upon publication.
