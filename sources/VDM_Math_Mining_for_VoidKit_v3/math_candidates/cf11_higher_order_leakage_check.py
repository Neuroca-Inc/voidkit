#!/usr/bin/env python3
"""SymPy check for the CF11 cubic higher-order leakage term."""
from __future__ import annotations

import json
import os
import sympy as sp


def run_check() -> dict[str, object]:
    lam, xJ, xM = sp.symbols("lam xJ xM", finite=True)
    E_full = sp.expand(lam * (xJ + xM)**3 / 6)
    E_J = sp.expand(lam * xJ**3 / 6)
    E_M = sp.expand(lam * xM**3 / 6)
    E_cross = sp.simplify(E_full - E_J - E_M)
    E_cross_formula = sp.simplify(lam / 2 * (xJ**2 * xM + xJ * xM**2))
    residual = sp.simplify(E_cross - E_cross_formula)

    vals = {lam: sp.Rational(3, 2), xJ: sp.Rational(1, 5), xM: sp.Rational(1, 10)}
    representative = sp.simplify(E_cross.subs(vals))

    return {
        "check_id": "CF11-SYMPY-8",
        "description": "cubic higher-order sector leakage E_cross^(3)",
        "E_full": str(E_full),
        "E_cross_formula": str(E_cross_formula),
        "formula_residual": str(residual),
        "representative_lambda": str(vals[lam]),
        "representative_xJ": str(vals[xJ]),
        "representative_xM": str(vals[xM]),
        "representative_E_cross_exact": str(representative),
        "representative_E_cross_float": float(representative),
        "formula_residual_zero": bool(residual == 0),
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
