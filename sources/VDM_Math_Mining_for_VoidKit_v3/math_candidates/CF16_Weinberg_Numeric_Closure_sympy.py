#!/usr/bin/env python3
"""CF16 Weinberg-angle numeric closure symbolic/numeric audit.

This script attacks the exact rational closure:
    (u, v) = (55, 89)
    epsilon = 1/24 per edge
    chiral projected edge count = (u+v)/(2*24) = 3
    b^2 = v-u = 34
    a^2 + b^2 = u+v+3 = 147
    a^2 = 113
    sin^2(theta_EW) = 34/147

It exits nonzero if any symbolic identity or numerical comparison gate fails.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List

import sympy as sp


@dataclass
class Gate:
    gate_id: str
    name: str
    passed: bool
    details: str


def fail(msg: str) -> None:
    print(f"FINAL_RESULT: FAIL -- {msg}")
    sys.exit(1)


def require(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def main() -> int:
    gates: List[Gate] = []

    u = sp.Integer(55)
    v = sp.Integer(89)
    eps = sp.Rational(1, 24)

    S = u + v
    D = v - u
    C = sp.Rational(1, 2) * eps * S
    total = S + C
    b2 = D
    a2 = total - b2
    sin2 = sp.simplify(b2 / total)
    cos2 = sp.simplify(a2 / total)
    tan2 = sp.simplify(b2 / a2)

    gates.append(Gate("S1", "anchor arithmetic", (S == 144 and D == 34 and C == 3), f"S={S}, D={D}, C={C}"))
    gates.append(Gate("S2", "coefficient closure", (a2 == 113 and b2 == 34 and total == 147), f"a2={a2}, b2={b2}, total={total}"))
    gates.append(Gate("S3", "mixing identities", (sin2 == sp.Rational(34, 147) and cos2 == sp.Rational(113, 147) and tan2 == sp.Rational(34, 113)), f"sin2={sin2}, cos2={cos2}, tan2={tan2}"))
    gates.append(Gate("S4", "reduced rational", (sp.gcd(int(sp.numer(sin2)), int(sp.denom(sin2))) == 1), f"gcd={sp.gcd(int(sp.numer(sin2)), int(sp.denom(sin2)))}"))

    pdg_global = sp.Float("0.23129")
    pdg_global_sigma = sp.Float("0.00004")
    pdg_lhc = sp.Float("0.23122")
    pdg_lhc_sigma = sp.Float("0.00008")
    sin2_float = sp.N(sin2, 30)
    global_delta = abs(sin2_float - pdg_global)
    lhc_delta = abs(sin2_float - pdg_lhc)
    global_sigma_units = global_delta / pdg_global_sigma
    lhc_sigma_units = lhc_delta / pdg_lhc_sigma
    gates.append(Gate("N1", "PDG global-fit comparison", bool(global_sigma_units < 1), f"prediction={sin2_float}, delta={global_delta}, sigma_units={global_sigma_units}"))
    gates.append(Gate("N2", "PDG LHC-only comparison", bool(lhc_sigma_units < 1), f"prediction={sin2_float}, delta={lhc_delta}, sigma_units={lhc_sigma_units}"))

    no_edge = sp.Rational(34, 144)
    full_edge = sp.Rational(34, 150)
    wrong_gap = sp.Rational(55, 147)
    controls_far = all(abs(sp.N(x, 30) - pdg_global) > sp.Float("0.001") for x in [no_edge, full_edge, wrong_gap])
    gates.append(Gate("C1", "ablation controls are not accidentally close", bool(controls_far), f"no_edge={sp.N(no_edge,16)}, full_edge={sp.N(full_edge,16)}, wrong_gap={sp.N(wrong_gap,16)}"))

    print("CF16 Weinberg-angle numeric closure symbolic audit")
    print("-" * 78)
    for g in gates:
        print(f"{g.gate_id:>3} | {'PASS' if g.passed else 'FAIL':<4} | {g.name} | {g.details}")
    print("-" * 78)
    require(all(g.passed for g in gates), "one or more gates failed")
    print(f"PREDICTION sin^2(theta_EW) = {sin2} = {float(sin2):.16f}")
    print(f"COEFFICIENTS a^2:b^2 = {a2}:{b2}; tan^2(theta_EW) = {tan2}")
    print(f"MASS_RATIO mW/mZ = sqrt(113/147) = {math.sqrt(113/147):.16f}")
    print("FINAL_RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
