from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from pc_vdm_lifted_descent_solver.application.projector import FinalProjection, open_final_projection
from pc_vdm_lifted_descent_solver.domain.basis import FeatureMap, RetainedBasisFrame
from pc_vdm_lifted_descent_solver.domain.dataset import RegressionDataset
from pc_vdm_lifted_descent_solver.domain.operators import apply_discrete_transport
from pc_vdm_lifted_descent_solver.domain.phase_lift import ExtendedLiftedState
from pc_vdm_lifted_descent_solver.domain.termination import TerminationConfig, TerminationReport, evaluate_termination
from pc_vdm_lifted_descent_solver.domain.vdm_dynamics import TelegraphConfig, energy, residual, ring_laplacian, telegraph_step


@dataclass(frozen=True)
class SolverConfig:
    width: int = 64
    floor_den: int = 4096
    max_macro_steps: int = 512
    telegraph: TelegraphConfig = field(default_factory=TelegraphConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    projection_threshold: float = 1e-7


@dataclass(frozen=True)
class StepLog:
    macro_step: int
    operator: str
    A: int
    u: int
    v: int
    theta_tick: int
    kappa: int
    energy: float
    residual_variance: float
    walkers: int
    kT: float


@dataclass(frozen=True)
class SolveResult:
    dataset_name: str
    terminated: bool
    termination_reason: str
    macro_steps: int
    projection_open_count: int
    final_projection: FinalProjection
    history: List[StepLog]
    final_state: ExtendedLiftedState

    def to_jsonable(self) -> Dict[str, Any]:
        coeffs = [float(v) for v in self.final_projection.coefficients]
        return {
            "dataset": self.dataset_name,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "macro_steps": self.macro_steps,
            "projection_open_count": self.projection_open_count,
            "rmse": self.final_projection.rmse,
            "expression": self.final_projection.expression,
            "coefficients": coeffs,
            "history_tail": [log.__dict__ for log in self.history[-8:]],
        }


class LiftedDescentSolver:
    """Self-terminating VDM-style descent entirely inside retained lifted state."""

    def __init__(self, config: SolverConfig | None = None, feature_map: FeatureMap | None = None) -> None:
        self.config = config or SolverConfig()
        self.feature_map = feature_map or FeatureMap.default()

    def fit(self, dataset: RegressionDataset) -> SolveResult:
        basis_frame = RetainedBasisFrame.from_dataset(self.feature_map, dataset.x)
        design = basis_frame.orthogonal_design
        y = dataset.y
        lap = ring_laplacian(self.feature_map.size)
        state = ExtendedLiftedState.zero(self.feature_map.size)
        history: List[StepLog] = []
        energy_history: list[float] = []
        term = TerminationReport(False, float("inf"), float("inf"), False, "active_lifted_descent")

        for macro in range(1, self.config.max_macro_steps + 1):
            state, op = apply_discrete_transport(
                state,
                width=self.config.width,
                floor_den=self.config.floor_den,
            )
            state = state.copy_with(macro_step=macro)
            state = telegraph_step(state, design=design, y=y, laplacian=lap, cfg=self.config.telegraph)
            e = energy(design, y, state.phi, state.psi, lap, self.config.telegraph)
            energy_history.append(e)
            res = residual(design, y, state.phi)
            residual_variance = float(np.mean(res * res))
            history.append(
                StepLog(
                    macro_step=macro,
                    operator=op.value,
                    A=state.phase.A,
                    u=state.phase.q.u,
                    v=state.phase.q.v,
                    theta_tick=state.phase.theta_tick,
                    kappa=state.phase.kappa,
                    energy=e,
                    residual_variance=residual_variance,
                    walkers=state.walkers,
                    kT=state.kT,
                )
            )
            term = evaluate_termination(state, residual_values=res, energy_history=energy_history, cfg=self.config.termination)
            if term.terminated:
                break

        projected_state, projection = open_final_projection(
            state,
            basis_frame=basis_frame,
            y=y,
            threshold=self.config.projection_threshold,
        )
        return SolveResult(
            dataset_name=dataset.name,
            terminated=term.terminated,
            termination_reason=term.reason,
            macro_steps=state.macro_step,
            projection_open_count=1 if projected_state.projection_opened else 0,
            final_projection=projection,
            history=history,
            final_state=projected_state,
        )
