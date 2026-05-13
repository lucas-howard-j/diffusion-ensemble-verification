from pathlib import Path

ROOT = Path(__file__).parent.parent

# Raw data
RAW = ROOT / "data" / "raw"
GLORYS_DIR       = RAW / "glorys"
GLORYS_NINO34_DIR = RAW / "glorys_nino34"
ERA5_DIR         = RAW / "era5"
TIDE_GAUGE_DIR   = RAW / "tide_gauges"

# Processed data
PROC = ROOT / "data" / "processed"
SST_ANOM_WEEKLY  = PROC / "sst_anom_weekly.nc"
SSH_ANOM_WEEKLY  = PROC / "ssh_anom_weekly.nc"
WIND_ANOM_WEEKLY = PROC / "wind_anom_weekly.nc"
EOF_SST  = PROC / "eof_sst.npz"
EOF_SSH  = PROC / "eof_ssh.npz"
EOF_WIND = PROC / "eof_wind.npz"
NAO_WEEKLY  = PROC / "nao_weekly.csv"
ENSO_WEEKLY = PROC / "enso_nino34_weekly.csv"

# Checkpoints
CHECKPOINTS = ROOT / "checkpoints"

# ── Domain ────────────────────────────────────────────────────────────────────
DOMAIN = dict(lat_min=5.0, lat_max=50.0, lon_min=-98.0, lon_max=-50.0)
NINO34 = dict(lat_min=-5.0, lat_max=5.0, lon_min=-170.0, lon_max=-120.0)
ERA5_DOMAIN = dict(lat_min=20.0, lat_max=80.0, lon_min=-90.0, lon_max=40.0)

# ── Time periods ──────────────────────────────────────────────────────────────
TRAIN_YEARS = range(1993, 2019)   # 1993–2018  (26 years)
VAL_YEARS   = range(2019, 2022)   # 2019–2021  (3 years, ~131 inits)
TEST_YEARS  = range(2022, 2025)   # 2022–2024  (3 years, ~131 inits)

# ── EOF settings ──────────────────────────────────────────────────────────────
N_EOFS_MAX          = 200
VARIANCE_THRESHOLD  = 0.75   # retain SST/SSH modes to this cumulative fraction
WIND_VARIANCE_THRESHOLD = 0.80   # retain wind EOF modes to this cumulative fraction

# ── Predictor lag structure ───────────────────────────────────────────────────
# All lags are in weeks (positive = look back from current time t).
# SSH is lagged by at least 1 to avoid target information leaking into condition.
SSH_LAGS  = [0, 1, 2, 4]    # t, t-1, t-2, t-4  (init state + persistence + medium memory)
SST_LAGS  = [0, 1, 2]       # t,   t-1, t-2  (SST changes slowly; memory)
ENSO_LAGS = [0, 4, 8, 13]   # t, ~1mo, ~2mo, ~3mo  (ENSO teleconnection lag)
NAO_LAGS  = [0, 1, 2]       # extend NAO memory to 2 weeks back

MAX_LAG = max(SSH_LAGS + SST_LAGS + ENSO_LAGS + NAO_LAGS)   # = 13

# ── Mode counts (refined after running EOF analysis) ─────────────────────────
N_SSH_MODES  = 15
N_SST_MODES  = 15
N_WIND_MODES = 10   # joint u10/v10 EOF modes to include as predictors

# ── Diffusion model ───────────────────────────────────────────────────────────
N_DIFFUSION_STEPS = 1000
BETA_START   = 1e-4
BETA_END     = 0.02
BETA_SCHEDULE = "linear"   # "linear" or "cosine" (Nichol & Dhariwal 2021)
LEAD_WEEKS  = [2, 4, 8, 12]

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
LR           = 3e-5
WEIGHT_DECAY = 1e-3
MAX_EPOCHS   = 1000
PATIENCE     = 150
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9999
HIDDEN_DIMS  = (512, 512, 256)
DROPOUT      = 0.10

# ── Tide gauge stations ───────────────────────────────────────────────────────
TIDE_GAUGES = [
    {"id": "8410140", "name": "Eastport_ME",      "lat": 44.90, "lon": -67.00},
    {"id": "8443970", "name": "Boston_MA",         "lat": 42.36, "lon": -71.05},
    {"id": "8534720", "name": "Atlantic_City_NJ",  "lat": 39.36, "lon": -74.42},
    {"id": "8518750", "name": "Battery_NY",        "lat": 40.70, "lon": -74.01},
    {"id": "8638610", "name": "Sewells_Point_VA",  "lat": 36.95, "lon": -76.33},
    {"id": "8665530", "name": "Charleston_SC",     "lat": 32.78, "lon": -79.92},
    {"id": "8720218", "name": "Mayport_FL",        "lat": 30.40, "lon": -81.43},
    {"id": "8724580", "name": "Key_West_FL",       "lat": 24.56, "lon": -81.81},
]
