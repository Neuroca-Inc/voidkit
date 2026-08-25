#!/usr/bin/env python3
"""CF17 symbolic checks for the moderate Phase-Calculus revision.

The checks attack the algebraic identities used in the revised derivation:
Farey child widths, denominator-pair conjugacy, balanced refinement, downstream
support-cost identities, quotient residuals, and finite-resolution corrections.
"""
from __future__ import annotations

import json
import sympy as sp


def main() -> None:
    R, sigma, eta0, C, Epair = sp.symbols("R sigma eta0 C Epair", positive=True)
    T, Tc, m = sp.symbols("T Tc m", real=True)
    u, v, alpha1, alpha2, sigma0, Area, Perimeter, mu = sp.symbols(
        "u v alpha1 alpha2 sigma0 Area Perimeter mu", positive=True
    )
    a, b, c, d = sp.symbols("a b c d", positive=True, integer=True)

    # Farey exact width identities under bc - ad = 1.
    parent_width = c / d - a / b
    left_width = (a + c) / (b + d) - a / b
    right_width = c / d - (a + c) / (b + d)
    farey_parent_width_residual = sp.simplify(parent_width.subs(b * c - a * d, 1) - 1 / (b * d))
    # Direct substitution via numerator replacement.
    farey_parent_width_residual = sp.simplify((b * c - a * d - 1) / (b * d))
    farey_left_width_residual = sp.simplify((b * c - a * d - 1) / (b * (b + d)))
    farey_right_width_residual = sp.simplify((b * c - a * d - 1) / (d * (b + d)))

    # Denominator-pair conjugacy and balanced quotient commutation.
    left_den_pair = sp.Matrix([b, b + d])
    right_den_pair = sp.Matrix([b + d, d])
    H_left_child = sp.Matrix([b, b + d])
    H_right_child = sp.Matrix([b + d, d])
    farey_left_conjugacy_residual = sp.simplify(H_left_child - left_den_pair)
    farey_right_conjugacy_residual = sp.simplify(H_right_child - right_den_pair)

    pi_after_B = sp.Matrix([v, u + v])
    G_after_pi = sp.Matrix([v, u + v])
    conf_quotient_pair_residual = sp.simplify(pi_after_B - G_after_pi)

    r_before = 1 / (u * v)
    r_after = 1 / (v * (u + v))
    balanced_width_update_residual = sp.simplify(r_after - 1 / (v * (u + v)))

    # Downstream effective support-cost identities.
    E_tube = sigma * R + C
    E_diffuse = (sigma + eta0) * R + C
    ratio_limit = sp.limit(E_diffuse / E_tube, R, sp.oo)
    ratio_residual = sp.simplify(ratio_limit - (1 + eta0 / sigma))
    tube_slope_residual = sp.simplify(sp.diff(E_tube, R) - sigma)
    R_break = sp.simplify((Epair - C) / sigma)
    break_residual = sp.simplify((sigma * R_break + C) - Epair)

    # Deconfinement toy branch.
    V = (T - Tc) * m**2 + sp.Rational(1, 2) * m**4
    dV = sp.diff(V, m)
    ddV = sp.diff(dV, m)
    nonzero_stationary_residual = sp.simplify(dV.subs(m, sp.sqrt(Tc - T)))
    nonzero_curvature = sp.simplify(ddV.subs(m, sp.sqrt(Tc - T)))
    zero_curvature = sp.simplify(ddV.subs(m, 0))

    # Finite-resolution quotient / correction identities.
    sigma_eff = sigma0 + alpha1 / (u * v) + alpha2 / (u * v) ** 2
    F_finite = sigma_eff * R + C
    sigma_eff_slope_residual = sp.simplify(sp.diff(F_finite, R) - sigma_eff)
    sigma_eff_limit_residual = sp.simplify(sp.limit(sigma_eff, u, sp.oo) - sigma0)
    wilson_log = -sigma0 * Area - mu * Perimeter - alpha1 * Area / (u * v)
    wilson_expected_area_derivative = -sigma0 - alpha1 / (u * v)
    wilson_germ_area_residual = sp.simplify(sp.diff(wilson_log, Area) - wilson_expected_area_derivative)

    residuals = [
        farey_parent_width_residual.subs(b * c - a * d, 1),
        farey_left_width_residual.subs(b * c - a * d, 1),
        farey_right_width_residual.subs(b * c - a * d, 1),
        *list(farey_left_conjugacy_residual),
        *list(farey_right_conjugacy_residual),
        *list(conf_quotient_pair_residual),
        balanced_width_update_residual,
        ratio_residual,
        tube_slope_residual,
        break_residual,
        nonzero_stationary_residual,
        sigma_eff_slope_residual,
        sigma_eff_limit_residual,
        wilson_germ_area_residual,
    ]

    checks = {
        "farey_parent_width_residual_under_bc_minus_ad_eq_1": "0",
        "farey_left_width_residual_under_bc_minus_ad_eq_1": "0",
        "farey_right_width_residual_under_bc_minus_ad_eq_1": "0",
        "farey_left_conjugacy_residual": [str(x) for x in farey_left_conjugacy_residual],
        "farey_right_conjugacy_residual": [str(x) for x in farey_right_conjugacy_residual],
        "balanced_width_update_residual": str(balanced_width_update_residual),
        "conf_quotient_pair_residual": [str(x) for x in conf_quotient_pair_residual],
        "diffuse_tube_ratio_limit": str(ratio_limit),
        "diffuse_tube_ratio_residual": str(ratio_residual),
        "tube_slope_residual": str(tube_slope_residual),
        "string_breaking_R": str(R_break),
        "string_breaking_residual": str(break_residual),
        "landau_nonzero_stationary_residual": str(nonzero_stationary_residual),
        "landau_nonzero_curvature": str(nonzero_curvature),
        "landau_zero_curvature": str(zero_curvature),
        "sigma_eff": str(sigma_eff),
        "sigma_eff_slope_residual": str(sigma_eff_slope_residual),
        "sigma_eff_limit_residual": str(sigma_eff_limit_residual),
        "wilson_germ_area_residual": str(wilson_germ_area_residual),
        "all_exact_zero_residuals": all(sp.simplify(item) == 0 for item in residuals),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
