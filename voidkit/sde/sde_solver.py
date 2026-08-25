"""Stochastic differential-equation solvers."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def sde_solver(
    drift_func: Callable[[np.ndarray], np.ndarray],
    diffusion_func: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve an Itô SDE with Euler-Maruyama on a bounded time grid.

    ``diffusion_func`` may return a scalar, a vector with one independent noise
    channel per state variable, or a matrix of shape ``(n_state, n_noise)``.
    When ``dt`` does not divide the interval exactly, the final step is shortened
    so returned times and state increments describe the same grid.
    """
    state0 = np.asarray(initial_state, dtype=float)
    if state0.ndim != 1 or state0.size == 0 or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must be a non-empty finite 1-D array.")
    if len(t_span) != 2:
        raise ValueError("t_span must contain exactly (t_start, t_end).")
    t_start, t_end = map(float, t_span)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("t_span must be finite with t_end > t_start.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value.")

    span = t_end - t_start
    full_steps = int(np.floor(span / dt))
    times = t_start + np.arange(full_steps + 1, dtype=float) * dt
    tolerance = np.finfo(float).eps * max(1.0, abs(t_end)) * 8.0
    if t_end - times[-1] > tolerance:
        times = np.append(times, t_end)
    else:
        times[-1] = t_end

    states = np.empty((times.size, state0.size), dtype=float)
    states[0] = state0
    generator = rng if rng is not None else np.random.default_rng()

    for i, step in enumerate(np.diff(times)):
        current = states[i]
        drift = np.asarray(drift_func(current.copy()), dtype=float)
        if drift.shape != current.shape or not np.all(np.isfinite(drift)):
            raise ValueError("drift_func must return a finite vector matching the state shape.")

        diffusion = np.asarray(diffusion_func(current.copy()), dtype=float)
        if diffusion.ndim == 0:
            d_w = generator.normal(0.0, np.sqrt(step), size=current.size)
            stochastic = float(diffusion) * d_w
        elif diffusion.shape == current.shape:
            if not np.all(np.isfinite(diffusion)):
                raise ValueError("diffusion_func returned non-finite values.")
            d_w = generator.normal(0.0, np.sqrt(step), size=current.size)
            stochastic = diffusion * d_w
        elif diffusion.ndim == 2 and diffusion.shape[0] == current.size:
            if not np.all(np.isfinite(diffusion)):
                raise ValueError("diffusion_func returned non-finite values.")
            d_w = generator.normal(0.0, np.sqrt(step), size=diffusion.shape[1])
            stochastic = diffusion @ d_w
        else:
            raise ValueError(
                "diffusion_func must return a scalar, state-shaped vector, or "
                "(n_state, n_noise) matrix."
            )

        states[i + 1] = current + drift * step + stochastic

    return times, states
