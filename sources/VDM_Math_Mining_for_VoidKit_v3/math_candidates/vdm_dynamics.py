from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pc_vdm_lifted_descent_solver.domain.phase_lift import ExtendedLiftedState


@dataclass(frozen=True)
class TelegraphConfig:
    dt: float = 0.20
    damping: float = 1.50
    ridge: float = 1e-12
    stiffness: float = 0.0
    debt_rate: float = 0.015
    debt_decay: float = 0.985
    thermal_decay: float = 0.97
    thermal_rate: float = 0.03
    walker_threshold: float = 1e-8
    inner_steps: int = 1


def ring_laplacian(size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros((size, size), dtype=float)
    lap = np.zeros((size, size), dtype=float)
    for i in range(size):
        lap[i, i] = 2.0
        lap[i, (i - 1) % size] = -1.0
        lap[i, (i + 1) % size] = -1.0
    return lap


def residual(design: np.ndarray, y: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return design @ phi - y


def energy(
    design: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    laplacian: np.ndarray,
    cfg: TelegraphConfig,
) -> float:
    res = residual(design, y, phi)
    mse = 0.5 * float(np.mean(res * res))
    kinetic = 0.5 * float(np.dot(psi, psi))
    ridge = 0.5 * cfg.ridge * float(np.dot(phi, phi))
    elastic = 0.5 * cfg.stiffness * float(phi @ laplacian @ phi)
    return mse + kinetic + ridge + elastic


def gradient(
    design: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    laplacian: np.ndarray,
    cfg: TelegraphConfig,
) -> np.ndarray:
    res = residual(design, y, phi)
    return design.T @ res / float(design.shape[0]) + cfg.ridge * phi + cfg.stiffness * (laplacian @ phi)


def telegraph_step(
    state: ExtendedLiftedState,
    *,
    design: np.ndarray,
    y: np.ndarray,
    laplacian: np.ndarray,
    cfg: TelegraphConfig,
) -> ExtendedLiftedState:
    """One VDM metriplectic telegraph update of retained fields."""

    phi = state.phi.copy()
    psi = state.psi.copy()
    debt = state.debt.copy()
    kT = float(state.kT)
    dt = cfg.dt

    for _ in range(cfg.inner_steps):
        grad = gradient(design, y, phi, laplacian, cfg)
        debt = cfg.debt_decay * debt + cfg.debt_rate * np.abs(grad)
        local_damping = cfg.damping + 0.05 * debt
        psi = (1.0 - local_damping * dt) * psi - dt * grad
        phi = phi + dt * psi
        res = residual(design, y, phi)
        kT = cfg.thermal_decay * kT + cfg.thermal_rate * float(np.mean(res * res))

    walkers = int(np.count_nonzero(np.abs(psi) > cfg.walker_threshold))
    return state.copy_with(phi=phi, psi=psi, debt=debt, kT=kT, walkers=walkers)
