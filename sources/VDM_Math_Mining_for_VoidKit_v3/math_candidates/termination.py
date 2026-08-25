from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pc_vdm_lifted_descent_solver.domain.phase_lift import ExtendedLiftedState


@dataclass(frozen=True)
class TerminationConfig:
    residual_variance_tol: float = 1e-16
    stationary_energy_tol: float = 1e-12
    stationarity_window: int = 12
    min_macro_steps: int = 8


@dataclass(frozen=True)
class TerminationReport:
    terminated: bool
    residual_variance: float
    stationary_span: float
    zero_walkers: bool
    reason: str


def evaluate_termination(
    state: ExtendedLiftedState,
    *,
    residual_values: np.ndarray,
    energy_history: list[float],
    cfg: TerminationConfig,
) -> TerminationReport:
    residual_variance = float(np.mean(residual_values * residual_values))
    if len(energy_history) >= cfg.stationarity_window:
        tail = energy_history[-cfg.stationarity_window :]
        stationary_span = float(max(tail) - min(tail))
    else:
        stationary_span = float("inf")
    low_variance = residual_variance <= cfg.residual_variance_tol
    zero_walkers = state.walkers == 0
    stationary = stationary_span <= cfg.stationary_energy_tol
    old_enough = state.macro_step >= cfg.min_macro_steps
    terminated = bool(low_variance and zero_walkers and stationary and old_enough)
    if terminated:
        reason = "low_field_variance_zero_walkers_stationary_energy"
    else:
        reason = "active_lifted_descent"
    return TerminationReport(terminated, residual_variance, stationary_span, zero_walkers, reason)
