"""Legacy biologically inspired growth-trigger utilities."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def calculate_advanced_growth_trigger(
    avg_reward: float,
    burst_score: float,
    bdnf_proxy: float,
    kappa: float = 2.0,
    nu: float = 0.8,
    rho: float = 0.5,
) -> float:
    """Evaluate the legacy logistic growth-trigger formula without overflow."""
    values = np.asarray([avg_reward, burst_score, bdnf_proxy, kappa, nu, rho], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("All trigger parameters must be finite.")
    argument = kappa * (avg_reward - 0.5) + nu * burst_score + rho * bdnf_proxy
    return float(expit(argument))
