#!/usr/bin/env python3
"""Symbolic and integer audit for the PCVDM XiGraph carry hierarchy."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

MASK = 2**64 - 1


def lowbit(x: int) -> int:
    return x & -x


def xi_word_step(x: int) -> tuple[int, int]:
    lb = lowbit(x)
    raw = x + lb
    carry = 1 if raw > MASK else 0
    return ((raw & MASK) + carry) & MASK, carry


def audit_bit_carry() -> dict:
    rows = []
    for exponent in range(0, 16):
        x = 1 << exponent
        first_carry = None
        for step in range(1, 80):
            x, carry = xi_word_step(x)
            if carry:
                first_carry = step
                break
        expected = 64 - exponent
        assert first_carry == expected
        assert x == 1
        rows.append({"start_power": exponent, "first_carry_step": first_carry})
    return {"checked_start_powers": len(rows), "rows": rows[:4]}


def audit_balanced_corridor() -> dict:
    u, v = sp.Integer(1), sp.Integer(1)
    fib = [sp.Integer(0), sp.Integer(1)]
    for _ in range(40):
        fib.append(fib[-1] + fib[-2])
    products = []
    for n in range(0, 22):
        assert (u, v) == (fib[n + 1], fib[n + 2])
        products.append(u * v)
        u, v = v, u + v
    assert all(products[i] < products[i + 1] for i in range(len(products) - 1))
    anchor = (fib[10], fib[11])
    assert anchor == (55, 89)
    return {"depth_9_anchor": [int(anchor[0]), int(anchor[1])], "depth_21_uv": int(products[21])}


def audit_resolution_order() -> dict:
    debt, threshold, capacity = sp.symbols("debt threshold capacity", integer=True, nonnegative=True)
    residual = sp.Max(0, debt - capacity)
    node_birth_condition = sp.Gt(residual, threshold)
    edge_resolved_condition = sp.Le(residual, threshold)
    assert sp.simplify(sp.Not(sp.And(node_birth_condition, edge_resolved_condition)))
    return {
        "residual_formula": str(residual),
        "birth_condition": str(node_birth_condition),
        "edge_resolved_condition": str(edge_resolved_condition),
    }


def main() -> None:
    report = {
        "bit_carry_1d": audit_bit_carry(),
        "balanced_corridor": audit_balanced_corridor(),
        "hierarchical_resolution_order": audit_resolution_order(),
    }
    out = Path(__file__).with_name("xigraph_sympy_audit_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
