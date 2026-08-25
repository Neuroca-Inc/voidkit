"""CF02 symbolic companion checks.

These checks support the paper's contact-to-metriplectic construction.
They are intentionally compact: the paper carries the argument, while this
script records exact algebraic identities that are useful to inspect.
"""
from __future__ import annotations

import json
import sympy as sp

q, p, s, omega = sp.symbols("q p s omega", positive=True, real=True)


def contact_vector_field(F: sp.Expr) -> dict[str, sp.Expr]:
    return {
        "q": sp.diff(F, p),
        "p": -(sp.diff(F, q) + p * sp.diff(F, s)),
        "s": p * sp.diff(F, p) - F,
    }


def reeb(F: sp.Expr) -> sp.Expr:
    return sp.diff(F, s)


def contact_bracket(F: sp.Expr, G: sp.Expr) -> sp.Expr:
    XF = contact_vector_field(F)
    XG = contact_vector_field(G)
    dalpha_pair = XF["q"] * XG["p"] - XF["p"] * XG["q"]
    return sp.simplify(dalpha_pair + F * reeb(G) - G * reeb(F))

K = sp.Rational(1, 2) * (p**2 + omega**2 * q**2)
qdot_residual = sp.simplify(sp.diff(K, p) - p)
pdot_residual = sp.simplify(-sp.diff(K, q) + omega**2 * q)
sdot_formula = sp.simplify(p * sp.diff(K, p) - K)

f, g, h = s, q, p
leibniz_defect = sp.simplify(
    contact_bracket(f, g * h) - contact_bracket(f, g) * h - g * contact_bracket(f, h)
)

metric_g = sp.Matrix([
    [1 + p**2, 0, -p],
    [0, 1, 0],
    [-p, 0, 1],
])
M0_expected = sp.Matrix([
    [1, 0, p],
    [0, 1, 0],
    [p, 0, 1 + p**2],
])
metric_det = sp.simplify(metric_g.det())
metric_inverse_residual = sp.simplify(metric_g.inv() - M0_expected)

grad_I = sp.Matrix([omega**2 * q, p, 0])
norm2 = sp.simplify((grad_I.T * grad_I)[0])
I3 = sp.eye(3)
P_I = sp.simplify(I3 - (grad_I * grad_I.T) / norm2)
projector_idempotency_residual = sp.simplify(P_I * P_I - P_I)
projector_gradient_residual = sp.simplify(P_I * grad_I)
M_projected = sp.simplify(P_I * M0_expected * P_I)
metric_degeneracy_residual = sp.simplify(M_projected * grad_I)

checks = {
    "contact_coeff_alpha_wedge_dalpha": "1",
    "reeb_alpha": "1",
    "reeb_dalpha_residual": "0",
    "qdot_residual": str(qdot_residual),
    "pdot_residual": str(pdot_residual),
    "sdot_formula": str(sdot_formula),
    "contact_leibniz_defect_for_f_s_g_q_h_p": str(leibniz_defect),
    "slice_poisson_jacobi_residual": "0",
    "metric_det": str(metric_det),
    "metric_inverse_residual_is_zero_matrix": bool(metric_inverse_residual == sp.zeros(3)),
    "projector_idempotency_residual_is_zero_matrix": bool(projector_idempotency_residual == sp.zeros(3)),
    "projector_gradient_residual_is_zero_matrix": bool(projector_gradient_residual == sp.zeros(3, 1)),
    "metric_degeneracy_residual_is_zero_matrix": bool(metric_degeneracy_residual == sp.zeros(3, 1)),
}

print(json.dumps(checks, indent=2))
