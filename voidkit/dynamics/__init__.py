"""General dynamical-system primitives extracted from research."""

from .logistic import (
    exact_step,
    invariant_q,
    logistic_invariant_Q,
    reaction_exact_step,
)

__all__ = [
    "exact_step",
    "invariant_q",
    "logistic_invariant_Q",
    "reaction_exact_step",
]
