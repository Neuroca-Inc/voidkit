from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pcsr.domain.candidate import LiftedCandidate
from pcsr.domain.lifted_state import XiHat


@dataclass(frozen=True, slots=True)
class AtomKey:
    family: str
    degree: int
    phase: int
    host_mod: int

    def label(self) -> str:
        if self.family == "constant":
            return "1"
        if self.family == "poly":
            return "z" if self.degree == 1 else f"z^{self.degree}"
        if self.family == "sin":
            return f"sin({self.degree}*z)"
        if self.family == "cos":
            return f"cos({self.degree}*z)"
        if self.family == "rational":
            return f"1/(1+|z|^{self.degree})"
        raise ValueError(f"Unknown atom family: {self.family}")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "degree": self.degree,
            "phase": self.phase,
            "host_mod": self.host_mod,
            "label": self.label(),
        }


@dataclass(frozen=True, slots=True)
class ProjectedModel:
    atoms: tuple[AtomKey, ...]
    coefficients: tuple[float, ...]
    ridge: float
    max_degree: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = design_from_atoms(self.atoms, np.asarray(x, dtype=float))
        return matrix @ np.asarray(self.coefficients, dtype=float)

    def nonzero_terms(self, tol: float = 1e-10) -> int:
        return sum(abs(c) > tol for c in self.coefficients)

    def formula(self, tol: float = 1e-10) -> str:
        terms: list[str] = []
        for coeff, atom in zip(self.coefficients, self.atoms):
            if abs(coeff) <= tol:
                continue
            terms.append(f"({coeff:.12g})*{atom.label()}")
        return "0" if not terms else " + ".join(terms)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "atoms": [atom.to_jsonable() for atom in self.atoms],
            "coefficients": list(self.coefficients),
            "ridge": self.ridge,
            "max_degree": self.max_degree,
            "formula_in_scaled_variable_z": self.formula(),
        }


def atom_from_state(state: XiHat, max_degree: int) -> AtomKey:
    phase = state.phase_mod4
    host_mod = state.A % 4
    if phase == 0:
        return AtomKey("constant", 0, phase, host_mod)
    raw = (state.q.u, state.q.v, state.q.u + state.q.v)[phase - 1] + state.A
    degree = 1 + ((raw - 1) % max_degree)
    family = ("poly", "sin", "cos", "rational")[host_mod]
    return AtomKey(family, degree, phase, host_mod)


def atoms_from_candidate(candidate: LiftedCandidate, max_degree: int) -> tuple[AtomKey, ...]:
    atoms: list[AtomKey] = []
    seen: set[AtomKey] = set()
    for state in candidate.states():
        atom = atom_from_state(state, max_degree)
        if atom not in seen:
            atoms.append(atom)
            seen.add(atom)
    if not atoms or atoms[0].family != "constant":
        const = AtomKey("constant", 0, 0, 0)
        atoms.insert(0, const)
    return tuple(atoms)


def design_from_atoms(atoms: tuple[AtomKey, ...], z: np.ndarray) -> np.ndarray:
    cols: list[np.ndarray] = []
    z = np.asarray(z, dtype=float)
    clipped = np.clip(z, -50.0, 50.0)
    for atom in atoms:
        if atom.family == "constant":
            col = np.ones_like(clipped)
        elif atom.family == "poly":
            col = np.power(clipped, atom.degree)
        elif atom.family == "sin":
            col = np.sin(atom.degree * clipped)
        elif atom.family == "cos":
            col = np.cos(atom.degree * clipped)
        elif atom.family == "rational":
            col = 1.0 / (1.0 + np.power(np.abs(clipped), atom.degree))
        else:
            raise ValueError(f"Unknown atom family: {atom.family}")
        cols.append(col)
    return np.column_stack(cols)


def fit_projected_model(
    candidate: LiftedCandidate,
    x: np.ndarray,
    y: np.ndarray,
    ridge: float = 1e-10,
    max_degree: int = 12,
) -> ProjectedModel:
    atoms = atoms_from_candidate(candidate, max_degree=max_degree)
    matrix = design_from_atoms(atoms, np.asarray(x, dtype=float))
    target = np.asarray(y, dtype=float)
    gram = matrix.T @ matrix
    penalty = ridge * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0
    rhs = matrix.T @ target
    try:
        coeffs = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(gram + penalty) @ rhs
    return ProjectedModel(
        atoms=atoms,
        coefficients=tuple(float(c) for c in coeffs),
        ridge=float(ridge),
        max_degree=int(max_degree),
    )
