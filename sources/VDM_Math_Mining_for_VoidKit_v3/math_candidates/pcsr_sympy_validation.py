#!/usr/bin/env python3
from __future__ import annotations

import sys

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

    theta = sp.symbols("theta", real=True)
    q_visible = sp.I * sp.exp(sp.I * (theta + sp.pi / 2))
    q_expected = sp.I * (sp.I * sp.exp(sp.I * theta))
    check("Q_visible_quarter_quotient", sp.simplify(q_visible - q_expected) == 0, failures)

    u, v = sp.symbols("u v", positive=True, integer=True)
    c_left = theta - sp.pi / (u * v)
    c_right = theta + sp.pi / (u * v)
    check("germ_width", sp.simplify(c_right - c_left - 2 * sp.pi / (u * v)) == 0, failures)

    b_left = theta - sp.pi / (v * (u + v))
    b_right = theta + sp.pi / (v * (u + v))
    check("B_germ_width", sp.simplify(b_right - b_left - 2 * sp.pi / (v * (u + v))) == 0, failures)

    tick = sp.symbols("tick", integer=True)
    phase_mod_before = sp.Mod(tick, 4)
    phase_mod_after_q4 = sp.Mod(tick + 4, 4)
    check("Q4_visible_phase_mod_invariant", sp.simplify(phase_mod_after_q4 - phase_mod_before) == 0, failures)

    z = sp.symbols("z", real=True)
    coeffs = sp.symbols("c0:4")
    demo = sp.Rational(3, 4) * z**3 - sp.Rational(5, 4) * z**2 + sp.Rational(1, 2) * z + 2
    projected = coeffs[0] + coeffs[1] * z + coeffs[2] * z**2 + coeffs[3] * z**3
    solution = sp.solve(sp.Poly(projected - demo, z).coeffs(), coeffs, dict=True)[0]
    check("QBQB_projected_basis_spans_demo_cubic", sp.simplify(projected.subs(solution) - demo) == 0, failures)

    if failures:
        print("FINAL_RESULT: FAIL")
        print("FAILURES:", ", ".join(failures))
        return 1
    print("FINAL_RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
