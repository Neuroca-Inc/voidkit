#!/usr/bin/env python3
"""
CEG Instrumentation — Reference Spectral Operators

These are the spectral gradient and Laplacian used in the VDM Klein-Gordon
reference adapter.  If your model uses a different discretization (finite
difference, finite element, graph Laplacian, etc.), you do NOT need this file —
just implement the energy_norm_delta method in your adapter.

Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
"""
from __future__ import annotations

import numpy as np


def _omega(N: int, dx: float) -> np.ndarray:
    """Spectral wavenumbers for a periodic 1D grid."""
    k_cyc = np.fft.fftfreq(N, d=dx)
    return 2.0 * np.pi * k_cyc


def spectral_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """Periodic spectral Laplacian: Δ_h u."""
    N = u.size
    om = _omega(N, dx)
    U = np.fft.fft(u)
    return np.fft.ifft(-(om * om) * U).real


def spectral_grad(u: np.ndarray, dx: float) -> np.ndarray:
    """Periodic spectral gradient: ∂_x u."""
    N = u.size
    om = _omega(N, dx)
    U = np.fft.fft(u)
    return np.fft.ifft(1j * om * U).real


def kg_verlet_step(
    phi: np.ndarray, pi: np.ndarray,
    dt: float, dx: float,
    c: float, m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Störmer-Verlet step for the linear Klein-Gordon equation.

    ∂²φ/∂t² = c² Δφ − m² φ

    This is the reference J-limb (symplectic, conservative) integrator.
    """
    lap_phi = spectral_laplacian(phi, dx)
    pi_half = pi + 0.5 * dt * ((c * c) * lap_phi - (m * m) * phi)
    phi_new = phi + dt * pi_half
    lap_new = spectral_laplacian(phi_new, dx)
    pi_new = pi_half + 0.5 * dt * ((c * c) * lap_new - (m * m) * phi_new)
    return phi_new, pi_new


def stiffness(phi: np.ndarray, dx: float, c: float, m: float) -> np.ndarray:
    """Stiffness operator: K φ = −c² Δ_h φ + m² φ."""
    return -(c * c) * spectral_laplacian(phi, dx) + (m * m) * phi


def h_energy_norm_delta(
    phi_a: np.ndarray, pi_a: np.ndarray,
    phi_b: np.ndarray, pi_b: np.ndarray,
    dx: float, c: float, m: float,
) -> float:
    """Energy norm of the difference (φ_a − φ_b, π_a − π_b).

    ||δz||_H = sqrt( ∫ [δπ² + c²(∂_x δφ)² + m² δφ²] dx )

    This is the discrete Hamiltonian-density norm used as the echo error
    metric in VDM.  Your model should define its own analogous norm.
    """
    dphi = phi_a - phi_b
    dpi = pi_a - pi_b
    g = spectral_grad(dphi, dx)
    e2 = float(np.sum(dpi * dpi + (c * c) * (g * g) + (m * m) * (dphi * dphi)) * dx)
    return float(np.sqrt(max(e2, 0.0)))
