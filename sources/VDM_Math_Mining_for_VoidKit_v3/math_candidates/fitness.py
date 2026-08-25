from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pcsr.application.projection import ProjectedModel, fit_projected_model
from pcsr.domain.candidate import LiftedCandidate
from pcsr.domain.dataset import RegressionDataset


@dataclass(frozen=True, slots=True)
class FitnessResult:
    candidate: LiftedCandidate
    model: ProjectedModel
    rmse: float
    mae: float
    r2: float
    complexity: float
    score: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "complexity": self.complexity,
            "candidate": self.candidate.to_jsonable(),
            "projected_model": self.model.to_jsonable(),
        }


def evaluate_candidate(
    candidate: LiftedCandidate,
    dataset: RegressionDataset,
    ridge: float = 1e-10,
    max_degree: int = 12,
    complexity_penalty: float = 1e-4,
) -> FitnessResult:
    model = fit_projected_model(candidate, dataset.x, dataset.y, ridge=ridge, max_degree=max_degree)
    y_hat = model.predict(dataset.x)
    residual = y_hat - dataset.y
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((dataset.y - np.mean(dataset.y)) ** 2))
    r2 = 1.0 if denom == 0 else 1.0 - float(np.sum(residual**2)) / denom
    complexity = float(model.nonzero_terms() + 0.05 * candidate.depth)
    score = rmse + complexity_penalty * complexity
    return FitnessResult(
        candidate=candidate,
        model=model,
        rmse=rmse,
        mae=mae,
        r2=r2,
        complexity=complexity,
        score=score,
    )
