#!/usr/bin/env python3
"""SymPy check for the CF11 Phase Calculus quotient algebra.

Verifies a finite exact descendant relation Pi E = G Pi, then compares
against a hidden-channel leak that intentionally breaks the quotient.
"""
from __future__ import annotations

import json
import os
import sympy as sp


def _max_abs_matrix_entry(M: sp.Matrix) -> int:
    vals = [abs(sp.simplify(x)) for x in list(M)]
    return int(max(vals) if vals else 0)


def run_check() -> dict[str, object]:
    # Retained state coordinates: (x_J, x_M, r_cross, hidden).
    a, b, c, h = sp.symbols("a b c h")
    E = sp.Matrix([
        [a, 0, 0, 0],
        [0, b, 0, 0],
        [0, 0, c, 0],
        [sp.Rational(1, 7), 0, 0, 1],
    ])
    Pi = sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    G = sp.diag(a, b, c)
    residual = sp.simplify(Pi * E - G * Pi)

    # Bad control: hidden state leaks into the visible J-channel.
    E_bad = sp.Matrix(E)
    E_bad[0, 3] = h
    bad_residual = sp.simplify(Pi * E_bad - G * Pi)

    return {
        "check_id": "CF11-SYMPY-5",
        "description": "finite Phase Calculus quotient relation Pi E = G Pi for the dark-sector readout proxy",
        "quotient_residual_matrix": str(residual),
        "quotient_residual_max": _max_abs_matrix_entry(residual),
        "bad_control_residual_matrix": str(bad_residual),
        "bad_control_hidden_leak_term": str(bad_residual[0, 3]),
        "exact_quotient_residual_zero": bool(residual == sp.zeros(3, 4)),
        "bad_control_detects_hidden_leak": bool(bad_residual[0, 3] == h),
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
