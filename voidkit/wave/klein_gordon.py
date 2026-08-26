"""Linear Klein-Gordon helpers using periodic spectral spatial operators."""
from __future__ import annotations

import numpy as np

from voidkit.numerical.spectral import spectral_gradient, spectral_laplacian



def energy(
    phi: np.ndarray,
    momentum: np.ndarray,
    dx: float,
    wave_speed: float,
    mass: float,
) -> float:
    """Continuous-semidscrete Klein-Gordon energy used by the source KG operator family."""
    gradient = spectral_gradient(phi, dx)
    kinetic = 0.5 * float(np.sum(momentum * momentum) * dx)
    gradient_energy = 0.5 * (wave_speed * wave_speed) * float(np.sum(gradient * gradient) * dx)
    potential = 0.5 * (mass * mass) * float(np.sum(phi * phi) * dx)
    return kinetic + gradient_energy + potential

def verlet_step(
    phi: np.ndarray,
    momentum: np.ndarray,
    dt: float,
    dx: float,
    wave_speed: float,
    mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Störmer-Verlet step for ``phi_tt = c^2 Δphi - m^2 phi``."""
    lap_phi = spectral_laplacian(phi, dx)
    half = momentum + 0.5 * dt * (
        (wave_speed * wave_speed) * lap_phi - (mass * mass) * phi
    )
    phi_new = phi + dt * half
    lap_new = spectral_laplacian(phi_new, dx)
    momentum_new = half + 0.5 * dt * (
        (wave_speed * wave_speed) * lap_new - (mass * mass) * phi_new
    )
    return phi_new, momentum_new


def stiffness(phi: np.ndarray, dx: float, wave_speed: float, mass: float) -> np.ndarray:
    """Return ``K phi = -c^2 Δphi + m^2 phi``."""
    return -(wave_speed * wave_speed) * spectral_laplacian(phi, dx) + (mass * mass) * phi


def energy_norm_delta(
    phi_a: np.ndarray,
    momentum_a: np.ndarray,
    phi_b: np.ndarray,
    momentum_b: np.ndarray,
    dx: float,
    wave_speed: float,
    mass: float,
) -> float:
    """Hamiltonian-density norm of the difference between two KG states."""
    dphi = phi_a - phi_b
    dpi = momentum_a - momentum_b
    gradient = spectral_gradient(dphi, dx)
    e2 = float(
        np.sum(
            dpi * dpi
            + (wave_speed * wave_speed) * (gradient * gradient)
            + (mass * mass) * (dphi * dphi)
        )
        * dx
    )
    return float(np.sqrt(max(e2, 0.0)))


# Historical source names for easy migration.
kg_energy = energy
kg_verlet_step = verlet_step
h_energy_norm_delta = energy_norm_delta
