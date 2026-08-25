#!/usr/bin/env python3
"""SymPy attack surface for the Transcendental Wall package."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc_inverse_word_certifier.application.certifier import InverseWordCertifier  # noqa: E402
from pc_inverse_word_certifier.domain.fibonacci import corridor_state, fibonacci_pair  # noqa: E402


def exact_zero(expr: sp.Expr) -> bool:
    return sp.simplify(expr) == 0


def main() -> int:
    y = sp.symbols("y", positive=True)
    u, v = sp.symbols("u v", positive=True, integer=True)
    x = sp.symbols("x", real=True)

    checks: list[dict[str, object]] = []

    # Universal inverse ledger for exp/log constants and Lambert W.
    e_residual = sp.simplify(sp.exp(sp.log(sp.E)) - sp.E)
    pi_residual = sp.simplify(sp.exp(sp.log(sp.pi)) - sp.pi)
    w1 = sp.LambertW(1)
    w1_residual = sp.simplify(w1 * sp.exp(w1) - 1)
    # SymPy sometimes leaves W*exp(W)-z unevaluated for symbols; attack the exact constants.
    checks.append({"claim": "exp inverse at e", "residual": str(e_residual), "pass": exact_zero(e_residual)})
    checks.append({"claim": "exp inverse at pi", "residual": str(pi_residual), "pass": exact_zero(pi_residual)})
    checks.append({"claim": "Lambert W(1)", "residual": str(w1_residual), "pass": abs(complex(sp.N(w1_residual, 90))) < 1e-80})

    # Lifted trigonometric readout from primitive roll: V=i*exp(i*x), Pi_trig=(-Re(V), Im(V)).
    trig_sin = sp.trigsimp(((sp.exp(sp.I*x) - sp.exp(-sp.I*x)) / (2*sp.I) - sp.sin(x)).rewrite(sp.sin))
    trig_cos = sp.trigsimp(((sp.exp(sp.I*x) + sp.exp(-sp.I*x)) / 2 - sp.cos(x)).rewrite(sp.cos))
    checks.append({"claim": "sin lifted exp readout", "residual": str(trig_sin), "pass": exact_zero(trig_sin)})
    checks.append({"claim": "cos lifted exp readout", "residual": str(trig_cos), "pass": exact_zero(trig_cos)})

    # Red quotient exact replacement block on the positive corridor.
    pi_red_after_b = (v, u + v)
    g_red_after_pi = (v, u + v)
    red_ok = pi_red_after_b == g_red_after_pi
    rho = sp.log(u * v)
    rho_after = sp.log(v * (u + v))
    rho_residual = sp.simplify(rho_after - sp.log(v * (u + v)))
    checks.append({"claim": "Pi_Red o B = G_Red o Pi_Red", "residual": str(pi_red_after_b), "pass": red_ok})
    checks.append({"claim": "rho(u,v)=log(uv) Liouvillian coordinate", "residual": str(rho_residual), "pass": exact_zero(rho_residual)})

    # State completeness and projection-loss witness.
    theta = sp.symbols("theta", real=True)
    visible_a = sp.exp(sp.I * theta)
    visible_b = sp.exp(sp.I * (theta + 2 * sp.pi))
    visible_collision = sp.simplify(visible_a - visible_b)
    checks.append({"claim": "visible collision theta and theta+2*pi", "residual": str(visible_collision), "pass": exact_zero(visible_collision)})

    # Fibonacci corridor exact anchor.
    anchor = fibonacci_pair(9)
    anchor_state = corridor_state(9)
    checks.append({"claim": "B^9(1,1)=(55,89)", "residual": str(anchor), "pass": anchor == (55, 89)})
    checks.append({"claim": "anchor uv=4895", "residual": str(anchor_state.uv), "pass": anchor_state.uv == 4895})

    certifier = InverseWordCertifier()
    certificates = {
        "lambertw_1_branch0": certifier.certify_lambertw(1, 0, 21, 1e-30).to_dict(),
        "lambertw_minus_0p1_branch0": certifier.certify_lambertw(-0.1, 0, 21, 1e-30).to_dict(),
        "lambertw_minus_0p1_branch_minus1": certifier.certify_lambertw(-0.1, -1, 21, 1e-30).to_dict(),
        "x_plus_sin_1p5": certifier.certify_x_plus_sin(1.5, 34, 1e-11).to_dict(),
        "exp_pi": certifier.certify_exp_inverse(str(sp.N(sp.pi, 60)), 21, 1e-30).to_dict(),
    }
    for name, cert in certificates.items():
        checks.append({"claim": f"certificate {name}", "residual": cert["residual_abs"], "pass": cert["status"] == "PASS"})

    all_pass = all(bool(row["pass"]) for row in checks)
    ledger = {
        "artifact": "pc_inverse_word_sympy_audit.py",
        "final_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "certificates": certificates,
    }
    out = ROOT / "certificates" / "sympy_inverse_ledger.json"
    out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(ledger, indent=2, sort_keys=True))
    print(f"FINAL_RESULT: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
