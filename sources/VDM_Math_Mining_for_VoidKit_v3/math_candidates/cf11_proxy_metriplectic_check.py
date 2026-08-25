#!/usr/bin/env python3
"""SymPy check for the CF11 scalar-reservoir metriplectic proxy.

Verifies, symbolically and exactly:
  J dSigma = 0,
  M dI = 0,
  dI/dt = dI^T (J dI + M dSigma) = 0,
  dSigma/dt = dSigma^T M dSigma = Pi^2/tau.
"""
from __future__ import annotations

import json
import os
import sys
import sympy as sp


def run_check() -> dict[str, str | int | float]:
    Phi, Pi, omega_sq, tau = sp.symbols("Phi Pi omega_sq tau", positive=True, finite=True)

    J = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    v = sp.Matrix([0, 1, -Pi])
    M = (v * v.T) / tau

    dSigma = sp.Matrix([0, 0, 1])
    dI = sp.Matrix([omega_sq * Phi, Pi, 1])

    J_dSigma = sp.simplify(J * dSigma)
    M_dI = sp.simplify(M * dI)
    skew_residual = sp.simplify(J + J.T)
    sym_residual = sp.simplify(M - M.T)
    energy_rate = sp.simplify((dI.T * (J * dI + M * dSigma))[0])
    entropy_rate = sp.simplify((dSigma.T * M * dSigma)[0])

    residual_terms = list(J_dSigma) + list(M_dI) + list(skew_residual) + list(sym_residual) + [energy_rate]
    all_zero = all(sp.simplify(term) == 0 for term in residual_terms)

    return {
        "check_id": "CF11-SYMPY-1",
        "description": "proxy metriplectic degeneracy, conservation, and entropy production",
        "J_dSigma": str(J_dSigma),
        "M_dI": str(M_dI),
        "energy_rate_residual": str(energy_rate),
        "entropy_rate": str(entropy_rate),
        "all_degeneracy_and_energy_residuals_zero": bool(all_zero),
        "symbolic_residual_max": 0 if all_zero else 1,
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
