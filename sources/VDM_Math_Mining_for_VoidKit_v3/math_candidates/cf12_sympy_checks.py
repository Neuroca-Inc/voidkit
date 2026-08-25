#!/usr/bin/env python3
"""SymPy algebra checks for the CF12 Option-1 rewrite.

This script mirrors the exact symbolic section of the companion notebook and
writes a compact JSON record under results/cf12_sympy_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> None:
    theta = sp.symbols("theta", real=True)
    x, y, k = sp.symbols("x y k", real=True)
    r, rH, M, G, c, hbar, kB = sp.symbols("r rH M G c hbar kB", positive=True)
    a_coeff, b_coeff = sp.symbols("a b", nonzero=True)
    H11, xi, zeta, gradPhi, sigma_sat = sp.symbols("H11 xi zeta gradPhi sigma_sat")

    # Projection loss: visible phase periodicity.
    visible_periodic = sp.simplify(sp.exp(sp.I * (theta + 2 * sp.pi)) - sp.exp(sp.I * theta))

    # Native plaquette continuous shadow.
    Phi_quad = k * (x**2 + y**2) / 2
    symbolic_defect = sp.simplify(sp.diff(Phi_quad, x, 2) + sp.diff(Phi_quad, y, 2))

    # Horizon and thermodynamics.
    rH_expr = 2 * G * M / c**2
    chi_at_horizon = sp.simplify(2 * G * M / (c**2 * rH_expr))
    f_at_horizon = sp.simplify(1 - 2 * G * M / (c**2 * rH_expr))
    kappa_H = sp.simplify(c**4 / (4 * G * M))
    T_H = sp.simplify(hbar * c**3 / (8 * sp.pi * kB * G * M))
    A_H = 4 * sp.pi * rH_expr**2
    ell_star_sq = hbar * G / c**3
    S_H = sp.simplify(kB * A_H / (4 * ell_star_sq))
    first_law_ratio = sp.simplify(T_H * sp.diff(S_H, M) / c**2)
    entropy_exponent = sp.simplify(M * sp.diff(S_H, M) / S_H)
    temperature_exponent = sp.simplify(M * sp.diff(T_H, M) / T_H)

    # Smooth-shadow recognition coefficient.
    divergence_coefficient = sp.simplify(a_coeff / 2 + b_coeff)
    b_solution = sp.solve(sp.Eq(divergence_coefficient, 0), b_coeff)[0]
    b_over_a = sp.simplify(b_solution / a_coeff)

    # Cosmogenesis baseline recovery.
    Hcorr = H11 + xi * gradPhi + zeta * sigma_sat
    baseline_residual = sp.simplify(Hcorr.subs({xi: 0, zeta: 0}) - H11)

    results = {
        "visible_periodic_residual": str(visible_periodic),
        "symbolic_defect_quadratic_phi": str(symbolic_defect),
        "chi_at_horizon": str(chi_at_horizon),
        "f_at_horizon": str(f_at_horizon),
        "kappa_H": str(kappa_H),
        "hawking_temperature": str(T_H),
        "entropy": str(S_H),
        "first_law_ratio": str(first_law_ratio),
        "entropy_log_exponent": str(entropy_exponent),
        "temperature_log_exponent": str(temperature_exponent),
        "smooth_shadow_divergence_coefficient": str(divergence_coefficient),
        "smooth_shadow_b_solution": str(b_solution),
        "smooth_shadow_b_over_a": str(b_over_a),
        "baseline_residual_when_xi_zeta_zero": str(baseline_residual),
        "expected": {
            "visible_periodic_residual": "0",
            "symbolic_defect_quadratic_phi": "2*k",
            "chi_at_horizon": "1",
            "f_at_horizon": "0",
            "kappa_H": "c**4/(4*G*M)",
            "hawking_temperature": "c**3*hbar/(8*pi*G*M*kB)",
            "first_law_ratio": "1",
            "entropy_log_exponent": "2",
            "temperature_log_exponent": "-1",
            "smooth_shadow_divergence_coefficient": "a/2 + b",
            "smooth_shadow_b_over_a": "-1/2",
            "baseline_residual_when_xi_zeta_zero": "0",
        },
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "cf12_sympy_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
