"""Normalized SciPy initial-value ODE adapter."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

_VALID_METHODS = {"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"}


def numerical_ode_solver(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: Union[List[float], np.ndarray],
    t_eval: Optional[Union[List[float], np.ndarray]] = None,
    args: Tuple = (),
    method: str = "RK45",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: Optional[float] = None,
) -> Any:
    """Solve ``dy/dt = fun(t, y, *args)`` using SciPy ``solve_ivp``.

    Both forward and backward integration are supported. A solver result with
    ``success=False`` is promoted to ``RuntimeError`` so callers cannot silently
    consume a failed numerical trajectory.
    """
    if not callable(fun):
        raise TypeError("fun must be callable.")
    if not isinstance(t_span, tuple) or len(t_span) != 2:
        raise ValueError("t_span must be a tuple (t0, tf).")
    t0, tf = map(float, t_span)
    if not np.isfinite(t0) or not np.isfinite(tf) or t0 == tf:
        raise ValueError("t_span endpoints must be finite and distinct.")

    initial = np.asarray(y0, dtype=float)
    if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
        raise ValueError("y0 must be a non-empty finite one-dimensional array.")
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple.")
    if method not in _VALID_METHODS:
        raise ValueError(f"Invalid method {method!r}; choose one of {sorted(_VALID_METHODS)}.")
    if not np.isfinite(rtol) or rtol <= 0.0 or not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("rtol and atol must be positive finite values.")
    if max_step is not None and (not np.isfinite(max_step) or max_step <= 0.0):
        raise ValueError("max_step must be a positive finite value when provided.")

    evaluation = None if t_eval is None else np.asarray(t_eval, dtype=float)
    if evaluation is not None:
        if evaluation.ndim != 1 or not np.all(np.isfinite(evaluation)):
            raise ValueError("t_eval must be a finite one-dimensional array.")

    initial_derivative = np.asarray(fun(t0, initial.copy(), *args), dtype=float)
    if initial_derivative.shape != initial.shape or not np.all(np.isfinite(initial_derivative)):
        raise ValueError("fun must return a finite array matching y0's shape.")

    kwargs = dict(
        fun=fun,
        t_span=(t0, tf),
        y0=initial,
        t_eval=evaluation,
        args=args,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if max_step is not None:
        kwargs["max_step"] = max_step

    solution = solve_ivp(**kwargs)
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")
    return solution
