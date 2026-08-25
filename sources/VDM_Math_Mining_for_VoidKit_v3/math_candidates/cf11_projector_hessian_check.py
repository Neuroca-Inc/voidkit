#!/usr/bin/env python3
"""SymPy check for complementary projectors and Hessian-sector energy bookkeeping."""
from __future__ import annotations

import json
import os
import sys
import sympy as sp


def _max_abs_matrix_entry(M: sp.Matrix) -> int:
    vals = [abs(sp.simplify(x)) for x in list(M)]
    return int(max(vals) if vals else 0)


def run_check() -> dict[str, str | int | bool]:
    H = sp.diag(2, 3, 5, 7)
    PJ = sp.diag(1, 1, 0, 0)
    PM = sp.diag(0, 0, 1, 1)
    dq = sp.Matrix([1, 2, 3, 5])

    dqJ = PJ * dq
    dqM = PM * dq
    EJ = sp.simplify(sp.Rational(1, 2) * (dqJ.T * H * dqJ)[0])
    EM = sp.simplify(sp.Rational(1, 2) * (dqM.T * H * dqM)[0])
    Ex = sp.simplify((dqJ.T * H * dqM)[0])
    Etot = sp.simplify(sp.Rational(1, 2) * (dq.T * H * dq)[0])
    partition_residual = sp.simplify(Etot - EJ - EM - Ex)

    algebra_residuals = {
        "PJ_idempotence": _max_abs_matrix_entry(PJ * PJ - PJ),
        "PM_idempotence": _max_abs_matrix_entry(PM * PM - PM),
        "PJPM_zero": _max_abs_matrix_entry(PJ * PM),
        "PMPJ_zero": _max_abs_matrix_entry(PM * PJ),
        "sum_identity": _max_abs_matrix_entry(PJ + PM - sp.eye(4)),
        "H_cross_orthogonality": _max_abs_matrix_entry(PJ.T * H * PM),
    }

    all_zero = all(v == 0 for v in algebra_residuals.values()) and Ex == 0 and partition_residual == 0

    return {
        "check_id": "CF11-SYMPY-3",
        "description": "spectral-projector algebra proxy and Hessian energy partition",
        "EJ_exact": str(EJ),
        "EM_exact": str(EM),
        "E_cross_exact": str(Ex),
        "E_total_exact": str(Etot),
        "partition_residual": str(partition_residual),
        "algebra_residuals": algebra_residuals,
        "all_projector_and_energy_residuals_zero": bool(all_zero),
    }


if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
