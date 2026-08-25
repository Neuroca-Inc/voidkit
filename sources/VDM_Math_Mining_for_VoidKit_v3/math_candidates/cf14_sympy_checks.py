#!/usr/bin/env python3
"""SymPy checks for the CF14 companion package.

The script verifies the symbolic identities that are most useful to readers:

1. the generic first-variation integration-by-parts identity for L(q, qdot, t),
2. the harmonic oscillator Euler--Lagrange residual,
3. the endpoint-moving boundary term used in the paper,
4. the nonsmooth kink regularity boundary,
5. representative thermodynamic, Onsager, Maxwell, and Klein--Gordon descendants.
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def simplify_zero(expr: sp.Expr) -> bool:
    return sp.simplify(expr) == 0


def main() -> None:
    t, x, k = sp.symbols("t x k", positive=True, real=True)
    q = sp.Function("q")
    eta = sp.Function("eta")
    q_t = q(t)
    eta_t = eta(t)
    qdot = sp.diff(q_t, t)
    etadot = sp.diff(eta_t, t)

    # Generic first-variation identity:
    # L_q eta + L_qdot eta_dot = (L_q - d/dt L_qdot) eta + d/dt(L_qdot eta).
    L_generic = sp.Function("L")(q_t, qdot, t)
    dLdq = sp.diff(L_generic, q_t)
    dLdqdot = sp.diff(L_generic, qdot)
    first_variation_integrand = dLdq * eta_t + dLdqdot * etadot
    el_plus_boundary_form = (dLdq - sp.diff(dLdqdot, t)) * eta_t + sp.diff(dLdqdot * eta_t, t)
    generic_integration_by_parts_residual = sp.simplify(first_variation_integrand - el_plus_boundary_form)

    # Harmonic action descendant: L = 1/2(qdot^2 - q^2).
    L_harmonic = sp.Rational(1, 2) * (qdot**2 - q_t**2)
    el_harmonic = sp.diff(sp.diff(L_harmonic, qdot), t) - sp.diff(L_harmonic, q_t)
    direct_harmonic_residual = sp.simplify(sp.diff(sp.sin(t), t, 2) + sp.sin(t))

    # Boundary term for endpoint-moving probe eta=1 on [0, pi].
    boundary_term = sp.simplify(sp.diff(sp.sin(t), t).subs(t, sp.pi) - sp.diff(sp.sin(t), t).subs(t, 0))

    # Nonsmooth kink q=|t-pi/2|: slope jump is 2 and second derivative is distributional.
    kink = sp.Abs(t - sp.pi / 2)
    kink_left_slope = sp.Integer(-1)
    kink_right_slope = sp.Integer(1)
    kink_slope_jump = sp.simplify(kink_right_slope - kink_left_slope)
    kink_second_derivative_distribution = sp.diff(kink, t, 2)

    # Thermodynamic Legendre stationarity for U = 1/2 a S^2 + c S V + 1/2 b V^2.
    S, V, a, b, c, T_ext, P_ext = sp.symbols("S V a b c T_ext P_ext", real=True)
    U = sp.Rational(1, 2) * a * S**2 + c * S * V + sp.Rational(1, 2) * b * V**2
    G = U - T_ext * S + P_ext * V
    grad_G = [sp.diff(G, S), sp.diff(G, V)]
    solution = sp.solve(grad_G, (S, V), dict=True)[0]
    grad_at_sol = [sp.simplify(g.subs(solution)) for g in grad_G]

    # Onsager: objective X·J - 1/2 J^T L J gives X = L J.
    J1, J2, X1, X2, L11, L12, L22 = sp.symbols("J1 J2 X1 X2 L11 L12 L22", real=True)
    Phi = sp.Rational(1, 2) * (L11 * J1**2 + 2 * L12 * J1 * J2 + L22 * J2**2)
    objective = X1 * J1 + X2 * J2 - Phi
    grad_obj = [sp.diff(objective, J1), sp.diff(objective, J2)]
    onsager_expected = [X1 - L11 * J1 - L12 * J2, X2 - L12 * J1 - L22 * J2]
    onsager_gradient_matches = all(simplify_zero(a - b) for a, b in zip(grad_obj, onsager_expected))

    # Maxwell vacuum mode: A = sin(k x) cos(k t), residual A_tt - A_xx = 0.
    A = sp.sin(k * x) * sp.cos(k * t)
    maxwell_residual = sp.simplify(sp.diff(A, t, 2) - sp.diff(A, x, 2))

    # Klein-Gordon example: phi = cos(kx) cos(omega t), omega^2 = k^2 + m^2.
    m, omega = sp.symbols("m omega", positive=True, real=True)
    phi = sp.cos(k * x) * sp.cos(omega * t)
    kg_residual = sp.simplify(sp.diff(phi, t, 2) - sp.diff(phi, x, 2) + m**2 * phi)
    kg_on_shell = sp.simplify(kg_residual.subs(omega**2, k**2 + m**2))

    results = {
        "generic_integration_by_parts_residual": str(generic_integration_by_parts_residual),
        "generic_integration_by_parts_identity_zero": simplify_zero(generic_integration_by_parts_residual),
        "harmonic_euler_lagrange_form": str(el_harmonic),
        "harmonic_solution_residual_zero": simplify_zero(direct_harmonic_residual),
        "endpoint_moving_boundary_term": str(boundary_term),
        "endpoint_moving_boundary_term_is_minus_two": bool(sp.simplify(boundary_term + 2) == 0),
        "kink_slope_jump": str(kink_slope_jump),
        "kink_second_derivative_distribution": str(kink_second_derivative_distribution),
        "kink_has_nonzero_slope_jump": bool(kink_slope_jump != 0),
        "thermodynamic_gradient_at_stationary_point": [str(v) for v in grad_at_sol],
        "thermodynamic_gradient_zero": all(simplify_zero(v) for v in grad_at_sol),
        "onsager_gradient_matches_force_flux_relation": bool(onsager_gradient_matches),
        "maxwell_vacuum_mode_residual": str(maxwell_residual),
        "maxwell_vacuum_mode_residual_zero": simplify_zero(maxwell_residual),
        "klein_gordon_on_shell_residual": str(kg_on_shell),
        "klein_gordon_on_shell_residual_zero": simplify_zero(kg_on_shell),
    }

    Path("tables").mkdir(exist_ok=True)
    Path("tables/cf14_sympy_check_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
