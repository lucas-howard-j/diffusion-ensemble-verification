"""
Standard Lorenz (1963) parameters and integration settings.

The Lyapunov time (1/lambda_1) is ~0.9 model time units at these parameters,
so WARMUP_TIME=200 TU is ~220 Lyapunov times — well past any IC memory.
"""

# Lorenz 1963 canonical parameters
SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0

# Numerical integration
DT = 0.01            # time step (model time units)

# Warmup before saving trajectories
WARMUP_TIME = 200.0  # time units to discard (200 / 0.9 ≈ 220 Lyapunov times)

# Default trajectory generation — single long trajectory for data-volume sweeps
TRAJ_LEN = 1_000_000  # steps to keep (10,000 TU ≈ 11,000 Lyapunov times)

# Fixed IC (on the attractor)
IC = [0.0, 1.0, 1.05]

# Prediction task
LEAD_STEPS = 300        # steps ahead to predict (300 steps = 3 TU ≈ 9 days)

# Train / val / test split indices (into the saved trajectory)
# 10-TU (1000-step) gaps at each boundary to avoid autocorrelation leakage
TRAIN_END  = 8_000_000
GAP        = 1_000
VAL_START  = 8_001_000
VAL_END    = 8_501_000
TEST_START = 8_502_000
# TEST_END = 10_000_000 (end of trajectory)

# DDPM
T_DIFF      = 100      # diffusion timesteps
BETA_START  = 1e-4
BETA_END    = 0.02
HIDDEN_DIM  = 64
N_LAYERS    = 2
EMB_DIM     = 32        # sinusoidal timestep embedding dimension

# Training
BATCH_SIZE  = 1024
LR          = 3e-4
MAX_EPOCHS  = 500
PATIENCE    = 50
