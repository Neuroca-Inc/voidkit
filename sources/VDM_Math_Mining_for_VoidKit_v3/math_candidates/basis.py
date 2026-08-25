from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


Evaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BasisAtom:
    name: str
    expression: str
    evaluator: Evaluator


@dataclass(frozen=True)
class FeatureMap:
    atoms: tuple[BasisAtom, ...]

    @classmethod
    def default(cls) -> "FeatureMap":
        return cls(
            atoms=(
                BasisAtom("one", "1", lambda x: np.ones_like(x)),
                BasisAtom("x", "x", lambda x: x),
                BasisAtom("x2", "x^2", lambda x: x * x),
                BasisAtom("x3", "x^3", lambda x: x * x * x),
                BasisAtom("sin", "sin(x)", np.sin),
                BasisAtom("cos", "cos(x)", np.cos),
            )
        )

    @property
    def size(self) -> int:
        return len(self.atoms)

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        columns: List[np.ndarray] = [atom.evaluator(x) for atom in self.atoms]
        return np.column_stack(columns).astype(float)

    def expression(self, coefficients: np.ndarray, threshold: float = 1e-8) -> str:
        terms: list[str] = []
        for coeff, atom in zip(coefficients, self.atoms):
            value = float(coeff)
            if abs(value) <= threshold:
                continue
            terms.append(f"({value:.12g})*{atom.expression}")
        return " + ".join(terms) if terms else "0"


@dataclass(frozen=True)
class RetainedBasisFrame:
    """Basis data retained inside the lifted solver, including QR re-articulation."""

    feature_map: FeatureMap
    raw_design: np.ndarray
    orthogonal_design: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_dataset(cls, feature_map: FeatureMap, x: np.ndarray) -> "RetainedBasisFrame":
        raw = feature_map.design_matrix(x)
        q, r = np.linalg.qr(raw, mode="reduced")
        scale = float(np.sqrt(raw.shape[0]))
        return cls(
            feature_map=feature_map,
            raw_design=raw,
            orthogonal_design=q * scale,
            upper=r / scale,
        )

    def raw_coefficients(self, retained_phi: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.upper, retained_phi)
