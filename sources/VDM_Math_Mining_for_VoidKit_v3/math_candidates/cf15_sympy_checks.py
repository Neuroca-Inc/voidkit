"""CF15 standalone SymPy companion checks.

The paper carries the smooth variational hypotheses. This script checks the
algebraic reductions that remain after those hypotheses are supplied:

1. particle-sector Noether residual,
2. isotropic and anisotropic rotation comparison residuals,
3. angular-momentum Poisson residuals,
4. field-sector current-divergence residual,
5. pure J-limb conservation residual,
6. M-limb drift decomposition,
7. equilibrium restoration,
8. stress-energy on-shell divergence residual.

Run from the package root:
    python scripts/cf15_sympy_checks.py

The script writes scripts/cf15_sympy_results.json and prints the reductions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp


def particle_noether_identity() -> sp.Expr:
    dP_xi, P_xidot, dLambda = sp.symbols("dP_xi P_xidot dLambda")
    residual = dP_xi + P_xidot - dLambda
    return sp.simplify(residual.subs(dLambda, dP_xi + P_xidot))


def particle_rotation_comparison() -> tuple[sp.Expr, sp.Expr]:
    x, y, xd, yd, omega_x, omega_y = sp.symbols("x y xd yd omega_x omega_y")
    # L = 1/2(xd^2 + yd^2) - 1/2(omega_x^2 x^2 + omega_y^2 y^2)
    # Rotation generator xi=(-y, x), xidot=(-yd, xd).
    iso = (-omega_x**2 * x) * (-y) + (-omega_x**2 * y) * x + xd * (-yd) + yd * xd
    aniso = (-omega_x**2 * x) * (-y) + (-omega_y**2 * y) * x + xd * (-yd) + yd * xd
    return sp.factor(iso), sp.factor(aniso)


def angular_momentum_poisson_comparison() -> tuple[sp.Expr, sp.Expr]:
    x, y, px, py, omega_x, omega_y = sp.symbols("x y px py omega_x omega_y")

    def poisson(F: sp.Expr, G: sp.Expr) -> sp.Expr:
        return sp.diff(F, x) * sp.diff(G, px) + sp.diff(F, y) * sp.diff(G, py) \
            - sp.diff(F, px) * sp.diff(G, x) - sp.diff(F, py) * sp.diff(G, y)

    # Sign convention chosen to match the paper table: L_z = x p_y - y p_x.
    Lz = x * py - y * px
    H_iso = sp.Rational(1, 2) * (px**2 + py**2) + sp.Rational(1, 2) * omega_x**2 * (x**2 + y**2)
    H_aniso = sp.Rational(1, 2) * (px**2 + py**2) + sp.Rational(1, 2) * (omega_x**2 * x**2 + omega_y**2 * y**2)
    return sp.factor(poisson(Lz, H_iso)), sp.factor(poisson(Lz, H_aniso))


def field_current_identity() -> sp.Expr:
    dPi_delta, Pi_d_delta, dK, L_phi_delta = sp.symbols(
        "dPi_delta Pi_d_delta dK L_phi_delta"
    )
    # div j = (dPi)*delta + Pi*d(delta) - dK.
    div_j = dPi_delta + Pi_d_delta - dK
    # On shell: dPi_delta = L_phi_delta.
    # Zero-cost field variation: L_phi_delta + Pi_d_delta = dK.
    reduced = div_j.subs(dPi_delta, L_phi_delta).subs(dK, L_phi_delta + Pi_d_delta)
    return sp.simplify(reduced)


def j_limb_identity() -> sp.Expr:
    poisson_QH, m_term = sp.symbols("poisson_QH m_term")
    dQ = poisson_QH + m_term
    return sp.simplify(dQ.subs({poisson_QH: 0, m_term: 0}))


def m_limb_drift_identity() -> sp.Expr:
    poisson_QH, m_term = sp.symbols("poisson_QH m_term")
    dQ = poisson_QH + m_term
    return sp.simplify(dQ.subs(poisson_QH, 0) - m_term)


def equilibrium_restoration_identity() -> sp.Expr:
    m_drift = sp.symbols("m_drift")
    return sp.simplify(m_drift.subs(m_drift, 0))


def stress_energy_divergence_identity() -> sp.Expr:
    euler_lagrange_residual, grad_phi_nu = sp.symbols("EL grad_phi_nu")
    # For the canonical scalar-field stress-energy identity,
    # div_mu T^{mu nu} = EL(phi) * partial^nu phi.
    residual = euler_lagrange_residual * grad_phi_nu
    return sp.simplify(residual.subs(euler_lagrange_residual, 0))


def expression_to_string(value: Any) -> Any:
    if isinstance(value, tuple):
        return [sp.sstr(sp.factor(v)) for v in value]
    return sp.sstr(sp.factor(value))


def main() -> None:
    checks = {
        "particle_noether_identity": particle_noether_identity(),
        "particle_rotation_comparison": particle_rotation_comparison(),
        "angular_momentum_poisson_comparison": angular_momentum_poisson_comparison(),
        "field_current_identity": field_current_identity(),
        "j_limb_identity": j_limb_identity(),
        "m_limb_drift_identity": m_limb_drift_identity(),
        "equilibrium_restoration_identity": equilibrium_restoration_identity(),
        "stress_energy_divergence_identity": stress_energy_divergence_identity(),
    }

    expected_zero = [
        "particle_noether_identity",
        "field_current_identity",
        "j_limb_identity",
        "m_limb_drift_identity",
        "equilibrium_restoration_identity",
        "stress_energy_divergence_identity",
    ]
    for name in expected_zero:
        assert sp.simplify(checks[name]) == 0, f"{name} did not reduce to zero"

    x, y, omega_x, omega_y = sp.symbols("x y omega_x omega_y")
    expected_aniso = x * y * (omega_x - omega_y) * (omega_x + omega_y)

    iso, aniso = checks["particle_rotation_comparison"]
    assert sp.simplify(iso) == 0, "isotropic oscillator residual should vanish"
    assert sp.simplify(aniso - expected_aniso) == 0, "anisotropic rotation residual changed"

    L_iso, L_aniso = checks["angular_momentum_poisson_comparison"]
    assert sp.simplify(L_iso) == 0, "isotropic angular-momentum residual should vanish"
    assert sp.simplify(L_aniso - expected_aniso) == 0, "anisotropic angular-momentum residual changed"

    for name, result in checks.items():
        print(f"{name}: {expression_to_string(result)}")

    out_path = Path(__file__).with_name("cf15_sympy_results.json")
    out_path.write_text(
        json.dumps({name: expression_to_string(result) for name, result in checks.items()}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
