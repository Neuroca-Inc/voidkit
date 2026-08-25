"""Information Bottleneck objective utilities."""

from __future__ import annotations

import numpy as np

from .information_theory import calculate_mutual_information


def information_bottleneck(
    p_xy: np.ndarray,
    p_xt: np.ndarray,
    beta: float,
) -> float:
    """Evaluate the Information Bottleneck objective ``I(X;T) - beta I(T;Y)``.

    ``p_xy`` and ``p_xt`` must share the same ``X`` marginal. Under the standard
    Information Bottleneck Markov structure ``T <- X -> Y``, ``P(T,Y)`` is
    reconstructed from ``P(X,Y)`` and ``P(X,T)`` and used to evaluate the
    relevance term ``I(T;Y)``.
    """
    xy = np.asarray(p_xy, dtype=float)
    xt = np.asarray(p_xt, dtype=float)
    if xy.ndim != 2 or xt.ndim != 2:
        raise ValueError("p_xy and p_xt must be two-dimensional joint distributions.")
    if xy.shape[0] != xt.shape[0]:
        raise ValueError("p_xy and p_xt must have the same X dimension.")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be a non-negative finite value.")
    if np.any(xy < 0.0) or np.any(xt < 0.0):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(xy.sum(), 1.0) or not np.isclose(xt.sum(), 1.0):
        raise ValueError("Each joint distribution must sum to 1.")

    p_x_xy = xy.sum(axis=1)
    p_x_xt = xt.sum(axis=1)
    if not np.allclose(p_x_xy, p_x_xt, atol=1e-10, rtol=1e-8):
        raise ValueError("p_xy and p_xt must induce the same P(X) marginal.")

    p_t_given_x = np.divide(
        xt,
        p_x_xt[:, None],
        out=np.zeros_like(xt),
        where=p_x_xt[:, None] > 0.0,
    )
    p_ty = np.einsum("xy,xt->ty", xy, p_t_given_x)

    i_xt = calculate_mutual_information(xt)
    i_ty = calculate_mutual_information(p_ty)
    return float(i_xt - beta * i_ty)
