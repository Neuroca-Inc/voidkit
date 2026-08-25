from __future__ import annotations

import itertools
import json
from fractions import Fraction


def strict_order_edges(values: tuple[Fraction, ...]) -> set[tuple[Fraction, Fraction]]:
    return {(a, b) for a in values for b in values if a < b}


def is_lawful_pointed_order(
    values: set[Fraction],
    edges: set[tuple[Fraction, Fraction]],
    left: Fraction,
    right: Fraction,
) -> bool:
    if left not in values or right not in values or not left < right:
        return False
    expected = strict_order_edges(tuple(sorted(values)))
    return edges == expected


def chart_plus(values: set[Fraction]) -> dict[Fraction, Fraction]:
    return {x: x for x in values}


def chart_minus(values: set[Fraction], left: Fraction, right: Fraction) -> dict[Fraction, Fraction]:
    ordered = sorted(values)
    reversed_values = list(reversed(ordered))
    return {x: y for x, y in zip(ordered, reversed_values)}


def compose(f: dict, g: dict) -> dict:
    # f after g
    return {x: f[g[x]] for x in g}


def invert_map(f: dict) -> dict:
    return {v: k for k, v in f.items()}


def first_b_update(
    values: set[Fraction],
    edges: set[tuple[Fraction, Fraction]],
    left: Fraction,
    right: Fraction,
) -> dict:
    if not is_lawful_pointed_order(values, edges, left, right):
        return {"accepted": False, "reason": "invalid prior pointed order"}

    midpoint = Fraction(left.numerator + right.numerator, left.denominator + right.denominator)
    if midpoint in values or not (left < midpoint < right):
        return {"accepted": False, "reason": "B did not create a genuine interior determination"}

    next_values = set(values)
    next_values.add(midpoint)
    next_edges = set(edges)
    next_edges.add((left, midpoint))
    next_edges.add((midpoint, right))

    # The old relation is retained. The only additional old-to-new edge forced
    # at the first tick is already represented by the two inserted edges and
    # the retained old edge.
    if not is_lawful_pointed_order(next_values, next_edges, left, right):
        return {"accepted": False, "reason": "result is not the exact minimal pointed order"}

    plus = chart_plus(next_values)
    minus = chart_minus(next_values, left, right)
    t_pm = compose(minus, invert_map(plus))
    t_mp = compose(plus, invert_map(minus))

    return {
        "accepted": True,
        "midpoint": str(midpoint),
        "values": [str(x) for x in sorted(next_values)],
        "edges": [[str(a), str(b)] for a, b in sorted(next_edges)],
        "chart_plus": {str(k): str(v) for k, v in plus.items()},
        "chart_minus": {str(k): str(v) for k, v in minus.items()},
        "transfer_plus_to_minus": {str(k): str(v) for k, v in t_pm.items()},
        "transfer_minus_to_plus": {str(k): str(v) for k, v in t_mp.items()},
    }


def calculate() -> dict:
    zero = Fraction(0, 1)
    one = Fraction(1, 1)
    values0 = {zero, one}
    edges0 = {(zero, one)}

    plus0 = chart_plus(values0)
    minus0 = chart_minus(values0, zero, one)
    t0_pm = compose(minus0, invert_map(plus0))
    t0_mp = compose(plus0, invert_map(minus0))

    lawful = first_b_update(values0, edges0, zero, one)
    corrupted = first_b_update(values0, set(), zero, one)

    denominator_two = [
        Fraction(n, 2) for n in range(0, 3)
    ]
    interior = [x for x in denominator_two if zero < x < one]

    checks = {
        "P0_lawful": is_lawful_pointed_order(values0, edges0, zero, one),
        "plus_chart_unique": plus0 == {zero: zero, one: one},
        "minus_chart_unique": minus0 == {zero: one, one: zero},
        "initial_transfers_mutual_inverse": compose(t0_mp, t0_pm) == plus0,
        "unique_new_determination": interior == [Fraction(1, 2)],
        "first_B_accepted": lawful["accepted"],
        "corrupted_prior_rejected": not corrupted["accepted"],
        "updated_transfer_order_two": (
            lawful["transfer_plus_to_minus"] == lawful["transfer_minus_to_plus"]
            and lawful["transfer_plus_to_minus"]["1/2"] == "1/2"
        ),
    }

    return {
        "research_scope": "first-B pointed-order primary relation and derived atlas maps",
        "P0": {
            "values": ["0", "1"],
            "edges": [["0", "1"]],
            "chart_plus": {"0": "0", "1": "1"},
            "chart_minus": {"0": "1", "1": "0"},
            "transfer_plus_to_minus": {"0": "1", "1": "0"},
            "transfer_minus_to_plus": {"0": "1", "1": "0"},
        },
        "B": {
            "primitive_pair": ["1", "1"],
            "next_pair": ["1", "2"],
        },
        "P1": lawful,
        "negative_control": corrupted,
        "checks": checks,
        "computed_verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    result = calculate()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["computed_verdict"] != "PASS":
        raise SystemExit(1)
