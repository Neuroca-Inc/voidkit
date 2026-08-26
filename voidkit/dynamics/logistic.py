"""Exact finite-time flow and invariant for the autonomous logistic law.

For ``dW/dt = r W - u W^2`` the finite-time update is evaluated in closed form.
The implementation preserves the numerical stabilization used by the originating
VDM reaction-diffusion derivation (notably ``expm1`` and denominator guarding).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def exact_step(
    W: ArrayLike,
    r: ArrayLike,
    u: ArrayLike,
    dt: ArrayLike,
    clip_eps: float = 1e-12,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Compute the exact logistic reaction update over ``dt``."""
    x = np.array(W, dtype=dtype if dtype is not None else np.float64)
    r_arr = np.array(r, dtype=x.dtype)
    u_arr = np.array(u, dtype=x.dtype)
    dt_arr = np.array(dt, dtype=x.dtype)
    s = np.expm1(r_arr * dt_arr)
    e = s + 1.0
    u_zero = np.isclose(u_arr, 0.0)
    denom = u_arr * x * s + r_arr
    if np.isscalar(denom):
        if abs(denom) < clip_eps:
            denom = np.sign(denom) * clip_eps if denom != 0 else clip_eps
    else:
        zero_mask = np.isclose(denom, 0.0, atol=clip_eps, rtol=0.0)
        denom = np.where(
            zero_mask,
            np.where(denom > 0, clip_eps, -clip_eps),
            denom,
        )
    num = r_arr * x * e
    next_value = num / denom
    if np.any(u_zero):
        linear = x * e
        mask = u_zero
        if not np.array(mask).shape == x.shape:
            mask = np.broadcast_to(mask, x.shape)
        next_value = np.where(mask, linear, next_value)
    return next_value


def invariant_q(W: ArrayLike, r: ArrayLike, u: ArrayLike, t: ArrayLike) -> np.ndarray:
    """Logarithmic first integral ``Q=ln(W/(r-uW))-rt`` on its real domain."""
    x = np.array(W, dtype=np.float64)
    r_arr = np.array(r, dtype=x.dtype)
    u_arr = np.array(u, dtype=x.dtype)
    t_arr = np.array(t, dtype=x.dtype)
    eps = 1e-15
    denom = r_arr - u_arr * x
    zero_mask = np.isclose(denom, 0.0, atol=eps, rtol=0.0)
    denom = np.where(
        zero_mask,
        np.where(denom > 0, eps, -eps),
        denom,
    )
    return np.log(x / denom) - r_arr * t_arr


# Provenance-compatible names.
reaction_exact_step = exact_step
logistic_invariant_Q = invariant_q
