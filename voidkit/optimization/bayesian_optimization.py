"""Bayesian-optimization adapter with an optional scikit-optimize dependency."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np


def bayesian_optimization(
    objective_func: Callable[[List[Any]], float],
    param_space: List[Dict[str, Any]],
    n_calls: int = 50,
    n_initial_points: int = 10,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Minimize an objective with Gaussian-process Bayesian optimization."""
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical, Integer, Real
    except ImportError as exc:
        raise ImportError(
            "bayesian_optimization requires the optional 'scikit-optimize' dependency; "
            "install VoidKit with the 'optimization' extra."
        ) from exc

    if not callable(objective_func):
        raise TypeError("objective_func must be callable.")
    if not isinstance(param_space, list) or not param_space:
        raise ValueError("param_space must be a non-empty list of parameter specifications.")
    if n_calls < 1 or n_initial_points < 1 or n_initial_points > n_calls:
        raise ValueError("Require 1 <= n_initial_points <= n_calls.")

    space = []
    names = []
    seen = set()
    for spec in param_space:
        if not isinstance(spec, dict):
            raise TypeError("Each parameter specification must be a dictionary.")
        missing = {"type", "name", "range"} - set(spec)
        if missing:
            raise ValueError(f"Parameter specification missing keys: {sorted(missing)}")
        name = spec["name"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Parameter names must be non-empty and unique.")
        seen.add(name)
        names.append(name)

        kind = spec["type"]
        domain = spec["range"]
        if kind == "real":
            if len(domain) != 2 or not np.isfinite(domain[0]) or not np.isfinite(domain[1]) or domain[0] >= domain[1]:
                raise ValueError(f"Real parameter {name!r} requires finite increasing bounds.")
            space.append(Real(domain[0], domain[1], name=name))
        elif kind == "integer":
            if len(domain) != 2 or domain[0] > domain[1]:
                raise ValueError(f"Integer parameter {name!r} requires ordered bounds.")
            space.append(Integer(domain[0], domain[1], name=name))
        elif kind == "categorical":
            if not domain:
                raise ValueError(f"Categorical parameter {name!r} requires at least one category.")
            space.append(Categorical(domain, name=name))
        else:
            raise ValueError(f"Unsupported parameter type: {kind!r}")

    result = gp_minimize(
        objective_func,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
    )
    return {
        "best_params": dict(zip(names, result.x)),
        "best_value": float(result.fun),
        "result_object": result,
    }
