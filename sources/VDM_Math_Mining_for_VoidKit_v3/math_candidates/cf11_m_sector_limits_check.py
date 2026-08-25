#!/usr/bin/env python3
"""SymPy check for the two CF11 M-sector dust-like limits."""
from __future__ import annotations

import json
import os
import sys
import sympy as sp


def run_check() -> dict[str, str | bool | float]:
    A, m = sp.symbols("A m", positive=True, finite=True)
    # Harmonic oscillator averages use <sin^2>=<cos^2>=1/2 over one period.
    avg_K = sp.Rational(1, 4) * A**2 * m**2
    avg_U = sp.Rational(1, 4) * A**2 * m**2
    avg_w = sp.simplify((avg_K - avg_U) / (avg_K + avg_U))

    sigma_v = sp.Rational(3, 25)  # 0.12
    defect_w = sp.simplify(sigma_v**2 / 3)

    return {
        "check_id": "CF11-SYMPY-4",
        "description": "M-sector defect and harmonic-oscillator dust limits",
        "average_K": str(avg_K),
        "average_U": str(avg_U),
        "oscillator_average_w": str(avg_w),
        "oscillator_average_w_zero": bool(avg_w == 0),
        "representative_sigma_v": str(sigma_v),
        "representative_defect_w_exact": str(defect_w),
        "representative_defect_w_float": float(defect_w),
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
