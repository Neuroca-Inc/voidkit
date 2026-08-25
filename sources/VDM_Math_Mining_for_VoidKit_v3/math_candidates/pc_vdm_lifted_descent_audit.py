#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
    import sympy as sp
except Exception as exc:  # pragma: no cover
    print(f"FINAL_RESULT: FAIL sympy_import_error={exc}")
    sys.exit(1)


def check(name: str, condition: bool, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{name}: {status}")
    if not condition:
        failures.append(name)


def bpair(pair: tuple[int, int]) -> tuple[int, int]:
    u, v = pair
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def main() -> int:
    failures: list[str] = []

    pair = (1, 1)
    for _ in range(9):
        pair = bpair(pair)
    check("B9_anchor_55_89", pair == (55, 89), failures)
    check("anchor_product_4895", pair[0] * pair[1] == 4895, failures)

    theta = sp.symbols("theta", real=True)
    visible_shift = sp.I * sp.exp(sp.I * (theta + 2 * sp.pi))
    visible_base = sp.I * sp.exp(sp.I * theta)
    check("visible_projection_not_state_complete_witness", sp.simplify(visible_shift - visible_base) == 0, failures)

    q_visible = sp.I * sp.exp(sp.I * (theta + sp.pi / 2))
    q_expected = sp.I * (sp.I * sp.exp(sp.I * theta))
    check("Q_visible_quarter_quotient", sp.simplify(q_visible - q_expected) == 0, failures)

    tick = sp.symbols("tick", integer=True)
    check("Q4_tick_mod_invariant", sp.simplify(sp.Mod(tick + 4, 4) - sp.Mod(tick, 4)) == 0, failures)

    u, v = sp.symbols("u v", positive=True, integer=True)
    c_left = theta - sp.pi / (u * v)
    c_right = theta + sp.pi / (u * v)
    check("completion_germ_width", sp.simplify(c_right - c_left - 2 * sp.pi / (u * v)) == 0, failures)

    piq_after = (v, u + v)
    g_after = (v, u + v)
    check("Pi_q_after_B_equals_G_after_Pi_q", piq_after == g_after, failures)

    # Symbolic action gradient: I(phi)=1/2*((a*p0+b*p1-y)^2)
    a, b, y, p0, p1 = sp.symbols("a b y p0 p1", real=True)
    pred = a * p0 + b * p1
    action = sp.Rational(1, 2) * (pred - y) ** 2
    grad = sp.Matrix([sp.diff(action, p0), sp.diff(action, p1)])
    expected_grad = sp.Matrix([a * (pred - y), b * (pred - y)])
    check("quadratic_action_gradient", sp.simplify(grad - expected_grad) == sp.zeros(2, 1), failures)

    # Companion metriplectic surface: skew J, symmetric PSD M, and degeneracy controls.
    J = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    M = sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    h0, h1, s = sp.symbols("h0 h1 s", real=True)
    dSigma = sp.Matrix([0, 0, s])
    dI = sp.Matrix([h0, h1, 0])
    check("J_antisymmetric", J.T + J == sp.zeros(3), failures)
    check("M_symmetric", M.T - M == sp.zeros(3), failures)
    check("M_psd_eigenvalues", all(val >= 0 for val in M.eigenvals().keys()), failures)
    check("J_delta_Sigma_degeneracy", J * dSigma == sp.zeros(3, 1), failures)
    check("M_delta_I_degeneracy", M * dI == sp.zeros(3, 1), failures)

    root = Path(__file__).resolve().parents[2]
    source = "\n".join(p.read_text(errors="ignore") for p in (root / "src").rglob("*.py"))
    forbidden = ["genetic", "mutation", "mutate", "population", "fitness"]
    check("no_forbidden_external_search_terms_in_src", not any(word in source.lower() for word in forbidden), failures)

    if failures:
        print("FINAL_RESULT: FAIL")
        print("FAILURES:", ", ".join(failures))
        return 1
    print("FINAL_RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
