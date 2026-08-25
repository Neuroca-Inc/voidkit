"""Fractal-inspired stochastic spike-train generation."""

from __future__ import annotations

from typing import Optional

import numpy as np


def generate_fractal_spike_train(
    fractal_dimension: float,
    k: float,
    tau_f: float,
    duration: float,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate spike times from an exponentially decaying rate law.

    ``spike_rate(t) = k * fractal_dimension * exp(-t / tau_f)``.

    ``dt`` and ``duration`` are interpreted in milliseconds and the rate is
    interpreted in spikes/second, matching the original API's unit convention.
    Each bin uses the exact Poisson probability of one-or-more events rather than
    the small-rate approximation ``rate * dt``.
    """
    for name, value in {
        "fractal_dimension": fractal_dimension,
        "k": k,
        "tau_f": tau_f,
        "duration": duration,
        "dt": dt,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if fractal_dimension < 0.0 or k < 0.0:
        raise ValueError("fractal_dimension and k must be non-negative.")
    if tau_f <= 0.0 or duration < 0.0 or dt <= 0.0:
        raise ValueError("tau_f and dt must be positive; duration must be non-negative.")

    time = np.arange(0.0, duration, dt, dtype=float)
    if time.size == 0:
        return np.array([], dtype=float)

    spike_rate = k * fractal_dimension * np.exp(-time / tau_f)
    event_probability = -np.expm1(-spike_rate * dt / 1000.0)
    generator = rng if rng is not None else np.random.default_rng()
    spikes = generator.random(time.size) < event_probability
    return time[spikes]
