"""Temporal-kernel spectral clustering."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering


def spectral_clustering_with_temporal_kernel(
    spike_rates: np.ndarray,
    spike_times: np.ndarray,
    sigma: float = 1.0,
    tau: float = 1.0,
    max_clusters: int = 10,
    random_state: Optional[int] = None,
) -> Tuple[int, np.ndarray]:
    """Cluster observations using a joint rate/time affinity kernel.

    ``W_ij = exp(-(rate_i-rate_j)^2/sigma^2 - |time_i-time_j|/tau)``.
    The cluster count is selected from the eigengap of the normalized graph
    Laplacian, rather than from eigenvalue gaps of the raw affinity matrix.
    """
    rates = np.asarray(spike_rates, dtype=float)
    times = np.asarray(spike_times, dtype=float)
    if rates.ndim != 1 or times.ndim != 1 or rates.size != times.size or rates.size == 0:
        raise ValueError("spike_rates and spike_times must be equal-length non-empty 1-D arrays.")
    if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(times)):
        raise ValueError("spike_rates and spike_times must be finite.")
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("sigma and tau must be positive finite values.")
    if max_clusters < 1:
        raise ValueError("max_clusters must be at least 1.")

    n_items = rates.size
    if n_items == 1:
        return 1, np.array([0], dtype=int)

    rate_diff = rates[:, None] - rates[None, :]
    time_diff = times[:, None] - times[None, :]
    affinity = np.exp(-(rate_diff**2) / (sigma**2) - np.abs(time_diff) / tau)

    normalized_laplacian = laplacian(affinity, normed=True)
    eigenvalues = np.linalg.eigvalsh(normalized_laplacian)
    max_candidate = min(max_clusters, n_items - 1)
    gaps = np.diff(eigenvalues[: max_candidate + 1])
    optimal_k = int(np.argmax(gaps) + 1)

    if optimal_k == 1:
        return 1, np.zeros(n_items, dtype=int)

    model = SpectralClustering(
        n_clusters=optimal_k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    return optimal_k, model.fit_predict(affinity)
