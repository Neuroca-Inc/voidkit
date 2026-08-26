"""Exact periodic advection by spectral phase rotation.

Extracted from the VDM metriplectic J-step without the repository sys.path bootstrap.
The numerical function body is preserved.
"""
from __future__ import annotations
import numpy as np

def j_step_spectral_periodic(W: np.ndarray, dt: float, dx: float, c: float) -> np.ndarray:
    """Exact periodic advection by distance c*dt using spectral phase shift.

    Parameters:
    - W: state array (shape: (N,))
    - dt: time step (float)
    - dx: grid spacing (float)
    - c: advection speed (float)

    Returns: W at time t+dt under W_t + c W_x = 0 with periodic BC.
    """
    if dt == 0.0 or c == 0.0:
        return W.copy()
    N = W.size
    # Physical wavenumbers (rad/unit length)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    phase = np.exp(-1j * k * (c * dt))
    W_hat = np.fft.fft(W)
    Wn1 = np.fft.ifft(W_hat * phase).real
    return Wn1

exact_periodic_advection = j_step_spectral_periodic
__all__=["j_step_spectral_periodic","exact_periodic_advection"]
