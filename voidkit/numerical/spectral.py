"""Periodic one-dimensional Fourier spectral operators."""
from __future__ import annotations

import numpy as np


def angular_wavenumbers(size: int, dx: float) -> np.ndarray:
    """Angular wavenumbers for a periodic 1-D grid."""
    if size <= 0:
        raise ValueError("size must be positive")
    if dx <= 0:
        raise ValueError("dx must be positive")
    return 2.0 * np.pi * np.fft.fftfreq(size, d=dx)


def spectral_laplacian(values: np.ndarray, dx: float) -> np.ndarray:
    """Periodic Fourier-spectral second derivative."""
    u = np.asarray(values)
    if u.ndim != 1:
        raise ValueError("values must be one-dimensional")
    omega = angular_wavenumbers(u.size, dx)
    transformed = np.fft.fft(u)
    return np.fft.ifft(-(omega * omega) * transformed).real


def spectral_gradient(values: np.ndarray, dx: float) -> np.ndarray:
    """Periodic Fourier-spectral first derivative."""
    u = np.asarray(values)
    if u.ndim != 1:
        raise ValueError("values must be one-dimensional")
    omega = angular_wavenumbers(u.size, dx)
    transformed = np.fft.fft(u)
    return np.fft.ifft(1j * omega * transformed).real


# Historical source name.
spectral_grad = spectral_gradient
