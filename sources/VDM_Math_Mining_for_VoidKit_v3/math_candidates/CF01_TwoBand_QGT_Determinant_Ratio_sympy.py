#!/usr/bin/env python3
"""
CF01 two-band QGT determinant-ratio symbolic audit.

This script attacks the exact symbolic identity behind the CF01 exact rational invariant.

Model:
    H(k, delta; m) = d_x sigma_x + d_y sigma_y + d_z sigma_z
    d_x = 1 + delta + (1 - delta) cos(k)
    d_y = (1 - delta) sin(k)
    d_z = m, m != 0

For a two-band Hamiltonian H=d.sigma:
    g_{mu nu} = 1/4 * partial_mu n . partial_nu n
    Omega_{mu nu}^2 = [d . (partial_mu d x partial_nu d)]^2 / (4 |d|^6)

The theorem identity is:
    det(g) / det(Omega_matrix) = det(g) / Omega_{k,delta}^2 = 1/4

The ungapped planar SSH negative control m=0 has Omega_{k,delta}=0 and det(g)=0,
so the ratio is undefined there rather than equal to 1/4.
"""

import sys
import sympy as sp


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    print("FINAL_RESULT: FAIL")
    sys.exit(1)


def main() -> None:
    k, delta, m = sp.symbols("k delta m", real=True)
    c = sp.cos(k)
    s = sp.sin(k)

    d = sp.Matrix([
        1 + delta + (1 - delta) * c,
        (1 - delta) * s,
        m,
    ])

    r2 = sp.simplify((d.T * d)[0])
    dk = d.diff(k)
    dd = d.diff(delta)

    def metric_component(a, b):
        da = d.diff(a)
        db = d.diff(b)
        # g_ab = 1/4 [(da.db)/r2 - (d.da)(d.db)/r2^2]
        return sp.factor(sp.simplify(
            sp.Rational(1, 4)
            * (((da.T * db)[0] * r2 - (d.T * da)[0] * (d.T * db)[0]) / r2**2)
        ))

    gkk = metric_component(k, k)
    gkd = metric_component(k, delta)
    gdd = metric_component(delta, delta)
    det_g = sp.factor(sp.simplify(gkk * gdd - gkd**2))

    triple = sp.factor(sp.simplify(d.dot(dk.cross(dd))))
    omega_sq = sp.factor(sp.simplify(triple**2 / (4 * r2**3)))

    expected_r2 = sp.factor(m**2 + 2 * (1 + sp.cos(k)) + 2 * delta**2 * (1 - sp.cos(k)))
    expected_triple = sp.factor(m * (1 - delta) * (1 - sp.cos(k)))
    expected_det_g = sp.factor(m**2 * (1 - delta)**2 * (1 - sp.cos(k))**2 / (16 * r2**3))
    expected_omega_sq = sp.factor(m**2 * (1 - delta)**2 * (1 - sp.cos(k))**2 / (4 * r2**3))

    checks = {
        "r2_formula": sp.simplify(r2 - expected_r2),
        "triple_formula": sp.simplify(triple - expected_triple),
        "det_g_formula": sp.simplify(det_g - expected_det_g),
        "omega_sq_formula": sp.simplify(omega_sq - expected_omega_sq),
        "ratio_identity_cross_multiplied": sp.simplify(det_g - sp.Rational(1, 4) * omega_sq),
        "pure_ssh_curvature_zero": sp.simplify(omega_sq.subs(m, 0)),
        "pure_ssh_detg_zero": sp.simplify(det_g.subs(m, 0)),
    }

    print("CF01 SSH/Rice-Mele symbolic audit")
    print("----------------------------------")
    print(f"r2 = {sp.factor(r2)}")
    print(f"d.(dk x ddelta) = {triple}")
    print(f"det(g) = {det_g}")
    print(f"Omega_kdelta^2 = {omega_sq}")
    print()

    for name, residual in checks.items():
        residual = sp.factor(sp.simplify(residual))
        print(f"{name}: residual = {residual}")
        if residual != 0:
            fail(f"{name} residual is nonzero")

    ratio = sp.factor(sp.simplify(det_g / omega_sq))
    print(f"ratio det(g)/Omega^2 = {ratio}")
    if sp.simplify(ratio - sp.Rational(1, 4)) != 0:
        fail("ratio is not exactly 1/4")

    print("All symbolic identities verified.")
    print("FINAL_RESULT: PASS")


if __name__ == "__main__":
    main()
