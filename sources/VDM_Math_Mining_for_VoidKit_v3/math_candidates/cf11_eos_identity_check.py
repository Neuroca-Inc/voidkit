#!/usr/bin/env python3
"""SymPy check for the CF11 J-sector equation-of-state identity and bound."""
from __future__ import annotations

import json
import os
import sys
import sympy as sp


def run_check() -> dict[str, str | bool | float]:
    eps, delta, U = sp.symbols("eps delta U", positive=True, finite=True)
    K = eps * U
    G = delta * U
    rho = K + G + U
    pressure = K - G / 3 - U
    w_from_fields = sp.simplify(pressure / rho)
    w_identity = sp.simplify((eps - delta / 3 - 1) / (eps + delta + 1))
    identity_residual = sp.simplify(w_from_fields - w_identity)

    one_plus_w = sp.simplify(1 + w_identity)

    eps0 = sp.Rational(1, 100)
    delta0 = sp.Rational(1, 50)
    w0 = sp.simplify(w_identity.subs({eps: eps0, delta: delta0}))
    one_plus_w0 = sp.simplify(1 + w0)
    bound0 = sp.simplify((2 * eps0 + sp.Rational(2, 3) * delta0) / (1 - eps0 - delta0))
    margin0 = sp.simplify(bound0 - one_plus_w0)

    return {
        "check_id": "CF11-SYMPY-2",
        "description": "J-sector equation-of-state identity and representative bound",
        "identity_residual": str(identity_residual),
        "one_plus_w_formula": str(one_plus_w),
        "representative_epsilon": str(eps0),
        "representative_delta": str(delta0),
        "representative_w_exact": str(w0),
        "representative_w_float": float(w0),
        "representative_abs_1_plus_w_exact": str(one_plus_w0),
        "representative_abs_1_plus_w_float": float(one_plus_w0),
        "representative_bound_exact": str(bound0),
        "representative_bound_float": float(bound0),
        "representative_bound_margin_exact": str(margin0),
        "identity_residual_zero": bool(identity_residual == 0),
        "bound_holds_on_representative": bool(margin0 >= 0),
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
