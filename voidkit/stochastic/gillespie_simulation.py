"""Stochastic simulation algorithms."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def gillespie_simulation(
    initial_state: np.ndarray,
    propensity_func: Callable[[np.ndarray], np.ndarray],
    stoichiometry: np.ndarray,
    t_max: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a reaction network with the Gillespie direct SSA.

    Reactions sampled beyond ``t_max`` are not executed. Propensities must be
    finite and non-negative, and state/stoichiometry dimensions are validated
    before simulation.
    """
    state = np.asarray(initial_state).copy()
    stoich = np.asarray(stoichiometry)

    if state.ndim != 1 or state.size == 0:
        raise ValueError("initial_state must be a non-empty one-dimensional array.")
    if stoich.ndim != 2 or stoich.shape[1] != state.size:
        raise ValueError("stoichiometry must have shape (n_reactions, n_species).")
    if not np.isfinite(t_max) or t_max < 0.0:
        raise ValueError("t_max must be a non-negative finite value.")

    generator = rng if rng is not None else np.random.default_rng()
    times = [0.0]
    states = [state.copy()]
    t = 0.0

    while t < t_max:
        propensities = np.asarray(propensity_func(state.copy()), dtype=float)
        if propensities.ndim != 1 or propensities.size != stoich.shape[0]:
            raise ValueError("propensity_func must return one propensity per reaction.")
        if not np.all(np.isfinite(propensities)):
            raise ValueError("Propensities must be finite.")
        if np.any(propensities < 0.0):
            raise ValueError("Propensities must be non-negative.")

        total_propensity = float(np.sum(propensities))
        if total_propensity <= 0.0:
            break

        dt = float(generator.exponential(1.0 / total_propensity))
        next_time = t + dt
        if next_time > t_max:
            break

        reaction_index = int(generator.choice(propensities.size, p=propensities / total_propensity))
        state = state + stoich[reaction_index]
        t = next_time
        times.append(t)
        states.append(state.copy())

    return np.asarray(times, dtype=float), np.asarray(states)
