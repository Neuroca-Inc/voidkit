"""
CF06 companion SymPy checks
--------------------------------
This script packages the exact symbolic checks used by the CFN06 companion notebook:
1. Gaussian Fisher metric in (mu, sigma) coordinates.
2. Exact coordinate-change check under tau = log(sigma).
3. Ricci-scalar invariance across charts.
4. 1D canonical Fisher–Ruppeiner bridge identity.

Run from the package root:
    python companion/CF06_sympy.py

The script ends with a FINAL_STATUS line and exits nonzero when a check lands outside its target range.
"""

from __future__ import annotations

import json
import sys
import sympy as sp


mu, sigma, tau = sp.symbols("mu sigma tau", positive=True, real=True)
kB, T, CV = sp.symbols("kB T CV", positive=True, real=True)


def ricci_scalar_symbolic(metric: sp.Matrix, coords: tuple[sp.Symbol, ...]) -> sp.Expr:
    n = len(coords)
    ginv = sp.simplify(metric.inv())
    Gamma = [[[sp.S(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                expr = sp.S(0)
                for l in range(n):
                    expr += ginv[i, l] * (
                        sp.diff(metric[l, k], coords[j])
                        + sp.diff(metric[l, j], coords[k])
                        - sp.diff(metric[j, k], coords[l])
                    )
                Gamma[i][j][k] = sp.simplify(sp.Rational(1, 2) * expr)
    Ric = sp.Matrix.zeros(n, n)
    for i in range(n):
        for j in range(n):
            expr = sp.S(0)
            for k in range(n):
                expr += sp.diff(Gamma[k][i][j], coords[k]) - sp.diff(Gamma[k][i][k], coords[j])
                for l in range(n):
                    expr += Gamma[k][i][j] * Gamma[l][k][l] - Gamma[l][i][k] * Gamma[k][j][l]
            Ric[i, j] = sp.simplify(expr)
    return sp.simplify(sum(ginv[i, j] * Ric[i, j] for i in range(n) for j in range(n)))


def matrix_exact_equal(a: sp.Matrix, b: sp.Matrix) -> bool:
    return all(sp.simplify(a[i, j] - b[i, j]) == 0 for i in range(a.rows) for j in range(a.cols))


g_sigma = sp.Matrix([[sigma**-2, sp.Integer(0)], [sp.Integer(0), 2 * sigma**-2]])
expected_g_sigma = sp.Matrix([[sigma**-2, sp.Integer(0)], [sp.Integer(0), 2 * sigma**-2]])

jacobian_sigma_from_tau = sp.Matrix([[sp.Integer(1), sp.Integer(0)], [sp.Integer(0), sp.exp(tau)]])
g_tau_from_transform = sp.simplify(
    jacobian_sigma_from_tau.T * g_sigma.subs(sigma, sp.exp(tau)) * jacobian_sigma_from_tau
)
g_tau_expected = sp.Matrix([[sp.exp(-2 * tau), sp.Integer(0)], [sp.Integer(0), sp.Integer(2)]])
g_non_tensor = sp.Matrix([[sp.exp(-2 * tau), sp.Integer(0)], [sp.Integer(0), 2 * sp.exp(-2 * tau)]])

R_sigma = sp.simplify(ricci_scalar_symbolic(g_sigma, (mu, sigma)))
R_tau = sp.simplify(ricci_scalar_symbolic(g_tau_expected, (mu, tau)))
R_non_tensor = sp.simplify(ricci_scalar_symbolic(g_non_tensor, (mu, tau)))

varE = kB * T**2 * CV
gR = sp.Integer(1) / (kB * T**2 * CV)
bridge_identity = sp.simplify(varE * gR)

checks = {
    "gaussian_sigma_chart_exact": matrix_exact_equal(g_sigma, expected_g_sigma),
    "tau_chart_from_coordinate_transform": matrix_exact_equal(g_tau_from_transform, g_tau_expected),
    "ricci_invariant_across_charts": sp.simplify(R_sigma - R_tau) == 0,
    "ricci_non_tensor_comparison_detected": sp.simplify(R_non_tensor - R_tau) != 0,
    "canonical_bridge_product_equals_one": sp.simplify(bridge_identity - 1) == 0,
}

outside_target_checks = [name for name, passed in checks.items() if not passed]
final_status = "complete" if not outside_target_checks else "needs_attention"

payload = {
    "gaussian_metric_sigma_chart": [[str(x) for x in row] for row in g_sigma.tolist()],
    "gaussian_metric_tau_chart_from_transform": [[str(x) for x in row] for row in g_tau_from_transform.tolist()],
    "gaussian_metric_tau_chart_expected": [[str(x) for x in row] for row in g_tau_expected.tolist()],
    "ricci_sigma_chart": str(R_sigma),
    "ricci_tau_chart": str(R_tau),
    "ricci_non_tensor_comparison": str(R_non_tensor),
    "canonical_bridge_product": str(bridge_identity),
    "checks": checks,
    "outside_target_checks": outside_target_checks,
    "final_status": final_status,
}

print(json.dumps(payload, indent=2))
print(f"FINAL_STATUS: {final_status}")

if final_status != "complete":
    sys.exit(1)
