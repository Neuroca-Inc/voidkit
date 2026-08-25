"""Normalized adaptive quadrature adapter."""

from __future__ import annotations

from numbers import Real
from typing import Callable, Tuple

from scipy import integrate


def numerical_integrate(
    func: Callable[[float], float],
    a: float,
    b: float,
    args: Tuple = (),
    *,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 50,
) -> Tuple[float, float]:
    """Integrate ``func`` from ``a`` to ``b`` using SciPy/QUADPACK.

    VoidKit preserves QUADPACK's useful behavior for equal or reversed bounds
    while normalizing validation and the tolerance surface.
    """
    if not callable(func):
        raise TypeError("func must be callable.")
    if not isinstance(a, Real) or not isinstance(b, Real):
        raise TypeError("a and b must be real numbers.")
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple.")
    if epsabs < 0.0 or epsrel < 0.0:
        raise ValueError("epsabs and epsrel must be non-negative.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    result, error = integrate.quad(
        func,
        float(a),
        float(b),
        args=args,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return float(result), float(error)
