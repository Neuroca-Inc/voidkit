"""Discrete Noether-oriented Klein-Gordon primitives.
The source experiment runner and artifact gates are excluded; the invariant formulas and Verlet step are retained.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from voidkit.numerical.spectral import spectral_gradient as spectral_grad, spectral_laplacian

def stiffness(phi: np.ndarray, dx: float, c: float, m: float) -> np.ndarray:
    """K phi = -c^2 Δ_h phi + m^2 phi (periodic spectral)"""
    return -(c * c) * spectral_laplacian(phi, dx) + (m * m) * phi


def verlet_step_with_half(phi: np.ndarray, pi: np.ndarray, dt: float, dx: float, c: float, m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one Störmer-Verlet step returning (phi_new, pi_half, pi_new)."""
    lap_phi = spectral_laplacian(phi, dx)
    pi_half = pi + 0.5 * dt * ((c * c) * lap_phi - (m * m) * phi)
    phi_new = phi + dt * pi_half
    lap_phi_new = spectral_laplacian(phi_new, dx)
    pi_new = pi_half + 0.5 * dt * ((c * c) * lap_phi_new - (m * m) * phi_new)
    return phi_new, pi_half, pi_new


def discrete_energy(phi_n: np.ndarray, phi_np1: np.ndarray, pi_half: np.ndarray, dx: float, c: float, m: float) -> float:
    """Leapfrog/Verlet discrete energy exactly conserved for linear KG:

    E_d = 1/2 ||pi_{n+1/2}||^2 + 1/2 <phi_{n+1}, K phi_n>.
    """
    Kphi_n = stiffness(phi_n, dx, c, m)
    term_k = 0.5 * float(np.sum(pi_half * pi_half) * dx)
    term_p = 0.5 * float(np.sum(phi_np1 * Kphi_n) * dx)
    return term_k + term_p


def discrete_momentum(phi_n: np.ndarray, phi_np1: np.ndarray, pi_half: np.ndarray, dx: float) -> float:
    """Translation Noether momentum (discrete midpoint variant):

    P_d = < pi_{n+1/2}, ∇_h ( (phi_{n+1}+phi_n)/2 ) >.
    """
    phi_mid = 0.5 * (phi_np1 + phi_n)
    grad_mid = spectral_grad(phi_mid, dx)
    return float(np.sum(pi_half * grad_mid) * dx)

__all__=["stiffness","verlet_step_with_half","discrete_energy","discrete_momentum"]
