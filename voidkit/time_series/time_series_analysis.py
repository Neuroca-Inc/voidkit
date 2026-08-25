"""Small, normalized time-series adapters."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.fft import fft, fftfreq


def calculate_fft(signal: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return the non-negative-frequency half of the FFT of a real 1-D signal."""
    values = np.asarray(signal)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value.")
    n = values.size
    yf = fft(values)
    xf = fftfreq(n, dt)[: n // 2]
    return xf, yf[: n // 2]


def calculate_autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Return normalized non-negative-lag autocorrelation.

    A constant signal has zero centered variance, so its centered normalized
    autocorrelation is returned as zeros rather than NaNs.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    centered = values - np.mean(values)
    variance = float(np.var(values))
    if variance == 0.0:
        return np.zeros(values.size, dtype=float)
    autocorr = np.correlate(centered, centered, mode="full")
    return autocorr[autocorr.size // 2 :] / (values.size * variance)


def calculate_cross_correlation(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """Return normalized non-negative-lag cross-correlation for equal-length signals."""
    first = np.asarray(signal1, dtype=float)
    second = np.asarray(signal2, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.size == 0 or second.size == 0:
        raise ValueError("Signals must be non-empty one-dimensional arrays.")
    if first.size != second.size:
        raise ValueError("Signals must have the same length.")

    mean1, mean2 = np.mean(first), np.mean(second)
    std1, std2 = float(np.std(first)), float(np.std(second))
    if std1 == 0.0 or std2 == 0.0:
        return np.zeros(first.size, dtype=float)

    cross_corr = np.correlate(first - mean1, second - mean2, mode="full")
    return cross_corr[cross_corr.size // 2 :] / (first.size * std1 * std2)
