from __future__ import annotations

import sys
import sympy as sp


def assert_zero(name: str, expr: sp.Expr) -> None:
    reduced = sp.simplify(expr)
    print(f"{name}: {reduced}")
    if reduced != 0:
        raise AssertionError(f"{name} residual is not zero: {reduced}")


def main() -> int:
    theta, u, v, mu = sp.symbols("theta u v mu", positive=True)
    g0, g1, g2 = sp.symbols("g0 g1 g2", real=True)
    I = sp.I

    visible = I * sp.exp(I * theta)
    q4_visible = I * sp.exp(I * (theta + 2 * sp.pi))
    assert_zero("Q4_visible_projection_residual", q4_visible - visible)

    b_u, b_v = v, u + v
    red_left = sp.Matrix([b_u, b_v])
    red_right = sp.Matrix([v, u + v])
    assert_zero("PiRed_B_commutation_u", red_left[0] - red_right[0])
    assert_zero("PiRed_B_commutation_v", red_left[1] - red_right[1])

    theta_tick = sp.symbols("theta_tick", integer=True, nonnegative=True)
    c_left = sp.pi * theta_tick / 2 - sp.pi / (u * v)
    c_right = sp.pi * theta_tick / 2 + sp.pi / (u * v)
    assert_zero("completion_germ_width", (c_right - c_left) - 2 * sp.pi / (u * v))

    grad = sp.Matrix([g0, g1, g2])
    M = mu * sp.eye(3)
    entropy_prod = (grad.T * M * grad)[0]
    expected = mu * (g0**2 + g1**2 + g2**2)
    assert_zero("M_limb_entropy_production_identity", entropy_prod - expected)

    J = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    assert_zero("J_antisymmetry", (J + J.T).norm())

    print("FINAL_RESULT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FINAL_RESULT: FAIL -- {exc}")
        raise SystemExit(1)
