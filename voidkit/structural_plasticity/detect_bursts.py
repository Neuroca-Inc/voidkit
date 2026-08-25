"""Spike-burst detection."""

from __future__ import annotations

import numpy as np


def detect_bursts(
    spike_times: np.ndarray,
    max_interspike_interval: float = 10.0,
    min_spikes_in_burst: int = 3,
) -> np.ndarray:
    """Detect maximal spike runs whose consecutive gaps stay below a threshold."""
    spikes = np.asarray(spike_times, dtype=float)
    if spikes.ndim != 1:
        raise ValueError("spike_times must be one-dimensional.")
    if not np.all(np.isfinite(spikes)):
        raise ValueError("spike_times must be finite.")
    if np.any(np.diff(spikes) < 0.0):
        raise ValueError("spike_times must be sorted in non-decreasing order.")
    if not np.isfinite(max_interspike_interval) or max_interspike_interval < 0.0:
        raise ValueError("max_interspike_interval must be non-negative and finite.")
    if min_spikes_in_burst < 2:
        raise ValueError("min_spikes_in_burst must be at least 2.")
    if spikes.size < min_spikes_in_burst:
        return np.empty((0, 2), dtype=float)

    linked = np.diff(spikes) <= max_interspike_interval
    bursts = []
    start = 0
    for gap_index, stays_linked in enumerate(linked):
        if stays_linked:
            continue
        end = gap_index
        if end - start + 1 >= min_spikes_in_burst:
            bursts.append((spikes[start], spikes[end]))
        start = gap_index + 1

    end = spikes.size - 1
    if end - start + 1 >= min_spikes_in_burst:
        bursts.append((spikes[start], spikes[end]))
    return np.asarray(bursts, dtype=float).reshape(-1, 2)
