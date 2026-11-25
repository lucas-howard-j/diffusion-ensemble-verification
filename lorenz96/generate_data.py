"""
Generate a single long Lorenz-96 trajectory for the joint-calibration testbed.

L96 dynamics (Lorenz 1996):
    dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F     (k cyclic mod N)

Output (lorenz96/l96_trajectory.npz):
    states : (traj_len, N)   — all N grid points
    t      : (traj_len,)     — time axis in model TU
    config : JSON-encoded parameter dict

Usage:
    python -m lorenz96.generate_data [--traj_len L]
"""

import argparse
import json
import time as walltime

import numpy as np

from lorenz96.config import N, F, DT, WARMUP_TIME, TRAJ_LEN, IC_SEED


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

def _l96_deriv(x: np.ndarray) -> np.ndarray:
    """L96 tendency; x: (N,), returns (N,)."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def _rk4_step(x: np.ndarray, dt: float) -> np.ndarray:
    k1 = _l96_deriv(x)
    k2 = _l96_deriv(x + 0.5 * dt * k1)
    k3 = _l96_deriv(x + 0.5 * dt * k2)
    k4 = _l96_deriv(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(x0: np.ndarray, n_steps: int, dt: float = DT) -> np.ndarray:
    """Integrate L96 for n_steps; returns (n_steps+1, N) including x0."""
    out = np.empty((n_steps + 1, N), dtype=np.float64)
    out[0] = x0
    for i in range(n_steps):
        out[i + 1] = _rk4_step(out[i], dt)
    return out


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_trajectory(
    traj_len: int = TRAJ_LEN,
    dt: float = DT,
    warmup_time: float = WARMUP_TIME,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate L96 from a random IC, discard warmup, return traj_len steps.

    Returns
    -------
    states : (traj_len, N)
    t      : (traj_len,)  time axis in TU
    """
    rng = np.random.default_rng(IC_SEED)
    x0 = F * np.ones(N) + rng.standard_normal(N) * 0.01

    warmup_steps = int(round(warmup_time / dt))
    print(f"Warmup: {warmup_steps:,} steps ({warmup_time} TU) ...", end=" ", flush=True)
    t0 = walltime.perf_counter()
    warmup_traj = integrate(x0, warmup_steps, dt)
    print(f"{walltime.perf_counter()-t0:.1f}s")

    print(f"Saving: {traj_len:,} steps ({traj_len*dt:,.0f} TU) ...", end=" ", flush=True)
    t0 = walltime.perf_counter()
    states = integrate(warmup_traj[-1], traj_len - 1, dt)
    print(f"{walltime.perf_counter()-t0:.1f}s")

    return states, np.arange(traj_len) * dt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Lorenz-96 trajectory")
    parser.add_argument("--traj_len", type=int,   default=TRAJ_LEN)
    parser.add_argument("--dt",       type=float, default=DT)
    parser.add_argument("--warmup",   type=float, default=WARMUP_TIME)
    parser.add_argument("--out",      type=str,   default="lorenz96/l96_trajectory.npz")
    args = parser.parse_args()

    print(f"Lorenz-96  N={N}  F={F}")
    print(f"dt={args.dt}  warmup={args.warmup} TU  traj_len={args.traj_len:,}\n")

    states, t = generate_trajectory(args.traj_len, args.dt, args.warmup)

    meta = json.dumps({
        "N": N, "F": F, "dt": args.dt,
        "warmup_time": args.warmup,
        "warmup_steps": int(args.warmup / args.dt),
        "traj_len": args.traj_len,
        "ic_seed": IC_SEED,
    })

    np.savez_compressed(args.out, states=states, t=t, config=np.array(meta))

    print(f"\nSaved → {args.out}")
    print(f"  states : {states.shape}  {states.dtype}")
    print(f"  X range: [{states.min():.2f}, {states.max():.2f}]")
    print(f"  X mean : {states.mean():.4f}  std: {states.std():.4f}")


if __name__ == "__main__":
    main()
