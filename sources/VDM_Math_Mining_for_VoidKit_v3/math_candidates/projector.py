from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pc_vdm_lifted_descent_solver.domain.basis import RetainedBasisFrame
from pc_vdm_lifted_descent_solver.domain.phase_lift import ExtendedLiftedState


@dataclass(frozen=True)
class FinalProjection:
    coefficients: np.ndarray
    expression: str
    predictions: np.ndarray
    rmse: float


def open_final_projection(
    state: ExtendedLiftedState,
    *,
    basis_frame: RetainedBasisFrame,
    y: np.ndarray,
    threshold: float = 1e-8,
) -> tuple[ExtendedLiftedState, FinalProjection]:
    """Open the visible projection exactly once at the final boundary."""

    if state.projection_opened:
        raise RuntimeError("final projection has already been opened")
    coefficients = basis_frame.raw_coefficients(state.phi)
    predictions = basis_frame.raw_design @ coefficients
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))
    projection = FinalProjection(
        coefficients=coefficients,
        expression=basis_frame.feature_map.expression(coefficients, threshold=threshold),
        predictions=predictions,
        rmse=rmse,
    )
    return state.copy_with(projection_opened=True), projection
