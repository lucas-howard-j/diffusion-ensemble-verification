"""
Generate a single long Lorenz-63 trajectory for the joint-calibration testbed.

A fixed IC is integrated with a long warmup, then TRAJ_LEN steps are saved.
Downstream code samples (state_t, state_{t+lead}) pairs directly from this
trajectory; varying N of those pairs is how training data volume is controlled.

Output (lorenz63/data/l63_trajectory.npz):
    states : (traj_len, 3)  — x, y, z in model units
    t      : (traj_len,)    — time axis starting at 0
    config : dict encoded as JSON in a 0-d string array

Usage:
    python -m lorenz63.generate_data [--traj_len L] [--seed S]
"""

import argparse
import json
import time as walltime

import numpy as np

from lorenz63.config import SIGMA, RHO, BETA, DT, WARMUP_TIME, TRAJ_LEN, IC


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

def _l63_deriv(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def _rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = _l63_deriv(state)
    k2 = _l63_deriv(state + 0.5 * dt * k1)
    k3 = _l63_deriv(state + 0.5 * dt * k2)
    k4 = _l63_deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(state0: np.ndarray, n_steps: int, dt: float = DT) -> np.ndarray:
    """Integrate L63 for n_steps; returns (n_steps+1, 3) including state0."""
    out = np.empty((n_steps + 1, 3), dtype=np.float64)
    out[0] = state0
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
    Integrate L63 from IC, discard warmup, return traj_len saved steps.

    Returns
    -------
    states : (traj_len, 3)
    t      : (traj_len,)  time axis in model units
    """
    warmup_steps = int(round(warmup_time / dt))

    print(f"Warmup: {warmup_steps:,} steps ({warmup_time} TU) ...", end=" ", flush=True)
    t0 = walltime.perf_counter()
    warmup_traj = integrate(np.array(IC, dtype=float), warmup_steps, dt)
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
    parser = argparse.ArgumentParser(description="Generate Lorenz-63 trajectory")
    parser.add_argument("--traj_len", type=int,   default=TRAJ_LEN,    help="Steps to keep (after warmup)")
    parser.add_argument("--dt",       type=float, default=DT,          help="Integration time step")
    parser.add_argument("--warmup",   type=float, default=WARMUP_TIME, help="Warmup in model time units")
    parser.add_argument("--out",      type=str,   default="lorenz63/data/l63_trajectory.npz")
    args = parser.parse_args()

    print(f"Lorenz-63  σ={SIGMA}  ρ={RHO}  β={BETA:.4f}")
    print(f"IC: {IC}")
    print(f"dt={args.dt}  warmup={args.warmup} TU  traj_len={args.traj_len:,}\n")

    states, t = generate_trajectory(
        traj_len=args.traj_len,
        dt=args.dt,
        warmup_time=args.warmup,
    )

    meta = json.dumps({
        "sigma": SIGMA, "rho": RHO, "beta": BETA,
        "dt": args.dt, "warmup_time": args.warmup,
        "warmup_steps": int(args.warmup / args.dt),
        "traj_len": args.traj_len,
        "ic": IC,
    })

    np.savez_compressed(args.out, states=states, t=t, config=np.array(meta))

    print(f"\nSaved → {args.out}")
    print(f"  states : {states.shape}  {states.dtype}")
    print(f"  x range: [{states[:,0].min():.2f}, {states[:,0].max():.2f}]")
    print(f"  y range: [{states[:,1].min():.2f}, {states[:,1].max():.2f}]")
    print(f"  z range: [{states[:,2].min():.2f}, {states[:,2].max():.2f}]")


if __name__ == "__main__":
    main()
