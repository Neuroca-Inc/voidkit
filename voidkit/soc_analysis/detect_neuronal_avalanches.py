"""Self-organized-criticality event analysis."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def detect_neuronal_avalanches(
    spike_times: np.ndarray,
    bin_width: float = 1.0,
) -> Dict[str, List[int]]:
    """Detect contiguous non-empty time-bin avalanches in a spike-time sequence."""
    spikes = np.asarray(spike_times, dtype=float)
    if spikes.ndim != 1:
        raise ValueError("spike_times must be one-dimensional.")
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("bin_width must be a positive finite value.")
    if spikes.size == 0:
        return {"sizes": [], "durations": []}
    if not np.all(np.isfinite(spikes)) or np.any(spikes < 0.0):
        raise ValueError("spike_times must be finite and non-negative.")

    max_time = float(np.max(spikes))
    n_bins = int(np.floor(max_time / bin_width)) + 1
    edges = np.arange(n_bins + 1, dtype=float) * bin_width
    binned_spikes, _ = np.histogram(spikes, bins=edges)

    avalanches: Dict[str, List[int]] = {"sizes": [], "durations": []}
    size = 0
    duration = 0
    for n_spikes in binned_spikes:
        if n_spikes > 0:
            size += int(n_spikes)
            duration += 1
        elif duration:
            avalanches["sizes"].append(size)
            avalanches["durations"].append(duration)
            size = 0
            duration = 0

    if duration:
        avalanches["sizes"].append(size)
        avalanches["durations"].append(duration)
    return avalanches
