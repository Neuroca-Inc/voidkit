#!/usr/bin/env python3
"""SymPy check for the CF11 hierarchy-throttled sound-speed relation."""
from __future__ import annotations

import json
import os
import sympy as sp


def run_check() -> dict[str, object]:
    cs0_sq, beta, D = sp.symbols("cs0_sq beta D", positive=True, finite=True)
    cs_eff_sq = cs0_sq * sp.exp(-beta * D)
    derivative = sp.simplify(sp.diff(cs_eff_sq, D))
    derivative_ratio = sp.simplify(derivative / cs_eff_sq)

    cs0 = sp.Rational(3, 625)  # 0.0048 from sigma_v=0.12
    beta0 = sp.Rational(2, 5)  # 0.4
    D0 = sp.Integer(3)
    value = sp.N(cs_eff_sq.subs({cs0_sq: cs0, beta: beta0, D: D0}), 18)

    return {
        "check_id": "CF11-SYMPY-7",
        "description": "hierarchy-throttled sound speed c_s_eff^2 = c_s0^2 exp(-beta D_void)",
        "cs_eff_sq_formula": str(cs_eff_sq),
        "d_dD_formula": str(derivative),
        "d_log_cs_eff_dD": str(derivative_ratio),
        "representative_cs0_sq": str(cs0),
        "representative_beta": str(beta0),
        "representative_D_void": str(D0),
        "representative_cs_eff_sq_float": float(value),
        "monotone_decreasing_for_positive_beta": bool(derivative_ratio == -beta),
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
