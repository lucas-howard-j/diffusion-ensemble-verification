"""
Lorenz-96 parameters and integration settings.

N=40, F=8 is the standard chaotic regime.
Leading Lyapunov exponent λ_1 ≈ 1.7 TU⁻¹  →  Lyapunov time ≈ 0.6 TU.
At dt=0.05:  1 TU = 20 steps,  Lyapunov time ≈ 12 steps.
"""

# L96 model parameters
N = 40
F = 8.0

# Numerical integration
DT = 0.05          # standard for L96 F=8 (stability + accuracy)

# Warmup before saving
WARMUP_TIME = 100.0   # TU  (~167 Lyapunov times)

# Trajectory length
TRAJ_LEN = 650_000    # steps = 32,500 TU  (>> pool + val + test requirements)

# Fixed IC: small perturbation on one node to break symmetry
IC_SEED = 42

# Default prediction lead
LEAD_STEPS = 20       # steps = 1 TU ≈ 1.7 Lyapunov times

# Train / val / test split (steps into saved trajectory)
# 100-step (5 TU ≫ Lyapunov time) gaps at each boundary
TRAIN_END  = 250_000
GAP        = 100
VAL_START  = 250_100
VAL_END    = 350_100
TEST_START = 350_200
# TEST_END = 650_000  (end of trajectory; ~299,800 steps = 14,990 TU)

# DDPM
T_DIFF     = 1000
BETA_START = 1e-4
BETA_END   = 0.02

# ConvScoreNet
HIDDEN_CH  = 32
N_RES      = 4
EMB_DIM    = 32
KERNEL     = 3

# Training
BATCH_SIZE = 1024
LR         = 3e-4
MAX_EPOCHS = 200
PATIENCE   = 20
