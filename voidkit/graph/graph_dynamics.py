"""Graph-dynamics utilities."""

from __future__ import annotations

import numpy as np


def simulate_sparsity_evolution(
    initial_sparsity: float,
    kappa: float,
    mu: float,
    spike_events: np.ndarray,
    dt: float,
    t_max: float,
) -> np.ndarray:
    """Integrate the legacy sparsity evolution law with explicit Euler steps."""
    events = np.asarray(spike_events, dtype=float)
    if events.ndim != 1 or not np.all(np.isfinite(events)):
        raise ValueError("spike_events must be a finite one-dimensional array.")
    if not 0.0 <= initial_sparsity <= 1.0:
        raise ValueError("initial_sparsity must satisfy 0 <= s <= 1.")
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(t_max) or t_max < 0.0:
        raise ValueError("dt must be positive and t_max non-negative; both must be finite.")
    if not np.isfinite(kappa) or not np.isfinite(mu):
        raise ValueError("kappa and mu must be finite.")

    n_steps = int(np.floor(t_max / dt))
    if events.size < n_steps:
        raise ValueError("spike_events must contain at least one value per integration step.")
    sparsity = np.empty(n_steps + 1, dtype=float)
    sparsity[0] = initial_sparsity
    for i in range(n_steps):
        ds_dt = -kappa * sparsity[i] * (1.0 - sparsity[i]) + mu * events[i]
        sparsity[i + 1] = sparsity[i] + ds_dt * dt
    return sparsity


def calculate_path_score(
    weights: np.ndarray,
    spike_times: np.ndarray,
    distances: np.ndarray,
    lambda_reg: float,
) -> float:
    """Evaluate ``sum(w * spike_time * exp(-distance/lambda_reg))``."""
    w = np.asarray(weights, dtype=float)
    spikes = np.asarray(spike_times, dtype=float)
    dist = np.asarray(distances, dtype=float)
    if w.ndim != 1 or spikes.ndim != 1 or dist.ndim != 1 or not (w.size == spikes.size == dist.size):
        raise ValueError("weights, spike_times, and distances must be equal-length 1-D arrays.")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(spikes)) or not np.all(np.isfinite(dist)):
        raise ValueError("Path inputs must be finite.")
    if np.any(dist < 0.0):
        raise ValueError("distances must be non-negative.")
    if not np.isfinite(lambda_reg) or lambda_reg <= 0.0:
        raise ValueError("lambda_reg must be positive and finite.")
    return float(np.sum(w * spikes * np.exp(-dist / lambda_reg)))
