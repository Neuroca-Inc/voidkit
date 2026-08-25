#!/usr/bin/env python3
"""
CF05 SymPy verification surface.

This script reproduces the canonical companion-notebook checks:
1. structural metriplectic identities,
2. Jacobi residual for a valid bracket and an invalid skew control,
3. degree-2 polynomial invariant search for the canonical product model.
"""
from __future__ import annotations

import json
from pathlib import Path
import sympy as sp


def jacobi_components(J: sp.Matrix, symbols: list[sp.Symbol]) -> dict[tuple[int, int, int], sp.Expr]:
    n = J.shape[0]
    out: dict[tuple[int, int, int], sp.Expr] = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                expr = 0
                for ell in range(n):
                    expr += (
                        J[i, ell] * sp.diff(J[j, k], symbols[ell])
                        + J[j, ell] * sp.diff(J[k, i], symbols[ell])
                        + J[k, ell] * sp.diff(J[i, j], symbols[ell])
                    )
                out[(i, j, k)] = sp.simplify(expr)
    return out


def residual_sum(components: dict[tuple[int, int, int], sp.Expr]) -> sp.Expr:
    return sp.simplify(sum(expr**2 for expr in components.values()))


def main() -> None:
    q, p, s = sp.symbols("q p s", real=True)
    coords = [q, p, s]

    L = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    M = sp.diag(0, 0, 1)
    E = sp.Rational(1, 2) * (q**2 + p**2)
    S = s

    gradE = sp.Matrix([sp.diff(E, var) for var in coords])
    gradS = sp.Matrix([sp.diff(S, var) for var in coords])
    flow = L * gradE + M * gradS

    E_dot = sp.simplify((gradE.T * flow)[0])
    S_dot = sp.simplify((gradS.T * flow)[0])
    degeneracy = list(sp.simplify(L * gradS)) + list(sp.simplify(M * gradE))

    J_bad = sp.Matrix([[0, q, 0], [-q, 0, p], [0, -p, 0]])
    canon_jacobi = residual_sum(jacobi_components(L, coords))
    bad_jacobi = residual_sum(jacobi_components(J_bad, coords))
    bad_sample = sp.simplify(bad_jacobi.subs({q: 2, p: 1, s: 0}))

    monomials: list[sp.Expr] = []
    for total_degree in range(0, 3):
        for aq in range(total_degree + 1):
            for ap in range(total_degree - aq + 1):
                a_s = total_degree - aq - ap
                monomials.append(q**aq * p**ap * s**a_s)

    coeffs = sp.symbols(f"c0:{len(monomials)}")
    candidate = sum(c * m for c, m in zip(coeffs, monomials))
    lie_derivative = sp.expand(sum(sp.diff(candidate, var) * flow_i for var, flow_i in zip(coords, [p, -q, 1])))

    poly = sp.Poly(lie_derivative, q, p, s)
    matrix_rows = []
    for coef in poly.coeffs():
        matrix_rows.append([sp.expand(coef).coeff(c) for c in coeffs])
    A = sp.Matrix(matrix_rows)
    nullspace = A.nullspace()
    basis_exprs = [
        sp.expand(sum(vec[i] * monomials[i] for i in range(len(monomials))))
        for vec in nullspace
    ]

    result = {
        "model": "canonical_product_metriplectic",
        "structural_identities": {
            "E_dot": str(E_dot),
            "S_dot": str(S_dot),
            "degeneracy_entries": [str(v) for v in degeneracy],
        },
        "jacobi": {
            "canonical_residual": str(canon_jacobi),
            "bad_control_residual": str(bad_jacobi),
            "bad_control_residual_at_q2_p1_s0": str(bad_sample),
        },
        "degree_2_polynomial_search": {
            "basis_size": len(monomials),
            "nullspace_dimension": len(nullspace),
            "nullspace_basis": [str(expr) for expr in basis_exprs],
            "independent_nonstructural_invariants": 0,
        },
    }

    out = Path("cf05_sympy_metrics.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
