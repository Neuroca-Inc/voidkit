from __future__ import annotations

import json
from typing import Dict, Iterable, Sequence, Tuple

import sympy as sp


I = sp.I
PI = sp.pi
THETA_0 = sp.Symbol("theta_0", real=True)
ZERO = sp.Integer(0)
HALF = sp.Rational(1, 2)
ONE = sp.Integer(1)


def canonical(expr):
    return sp.simplify(expr)


def expr_key(expr) -> str:
    return str(sp.simplify(expr))


def invert_map(mapping: Dict[sp.Expr, sp.Expr]) -> Dict[sp.Expr, sp.Expr]:
    inverse = {value: key for key, value in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("map is not bijective")
    return inverse


def compose(after: Dict[sp.Expr, sp.Expr], before: Dict[sp.Expr, sp.Expr]) -> Dict[sp.Expr, sp.Expr]:
    return {key: canonical(after[before[key]]) for key in before}


def pushforward_edges(edges: Iterable[Tuple[sp.Expr, sp.Expr]], fn):
    return {(canonical(fn(a)), canonical(fn(b))) for a, b in edges}


def mismatch_count(left: Dict[sp.Expr, sp.Expr], right: Dict[sp.Expr, sp.Expr]) -> int:
    keys = set(left) | set(right)
    return sum(int(left.get(key) != right.get(key)) for key in keys)


def calculate() -> dict:
    # Accepted prior state and exact scope.
    Xi1 = {
        "A": 0,
        "q": [1, 2],
        "theta": THETA_0,
        "k": 0,
        "j": 1,
        "W": "B",
    }
    prior_L_count = Xi1["W"].count("L")
    whole_active_sector = prior_L_count == 0

    # Primitive selection at Xi1.
    next_B_pair = [2, 3]
    current_capacity = 2
    next_B_product = next_B_pair[0] * next_B_pair[1]
    B_available = next_B_product <= current_capacity
    Q_available = Xi1["k"] + 1 < 2
    selected_primitive = "B" if B_available else ("Q" if Q_available else "L")

    Xi2 = {
        "A": Xi1["A"],
        "q": list(Xi1["q"]),
        "theta": sp.simplify(Xi1["theta"] + PI / 2),
        "k": Xi1["k"] + 1,
        "j": Xi1["j"] + 1,
        "W": Xi1["W"] + "Q",
    }

    # Accepted first-B relation.
    D1 = (ZERO, HALF, ONE)
    E1 = {(ZERO, HALF), (HALF, ONE), (ZERO, ONE)}
    plus1 = {ZERO: ZERO, HALF: HALF, ONE: ONE}
    minus1 = {ZERO: ONE, HALF: HALF, ONE: ZERO}
    t1_pm = compose(minus1, invert_map(plus1))
    t1_mp = compose(plus1, invert_map(minus1))

    # Exact first-Q law on the wholly active P1 sector.
    FQ = lambda z: canonical(I * z)
    FQ_inv = lambda z: canonical(-I * z)

    D2 = tuple(FQ(z) for z in D1)
    E2 = pushforward_edges(E1, FQ)

    # Uniquely forced by iota_2 o FQ = FQ o iota_1.
    plus2 = {FQ(x): FQ(plus1[x]) for x in D1}
    minus2 = {FQ(x): FQ(minus1[x]) for x in D1}

    chart_plus_commuting_mismatches = sum(
        int(canonical(plus2[FQ(x)] - FQ(plus1[x])) != 0)
        for x in D1
    )
    chart_minus_commuting_mismatches = sum(
        int(canonical(minus2[FQ(x)] - FQ(minus1[x])) != 0)
        for x in D1
    )

    t2_pm = compose(minus2, invert_map(plus2))
    t2_mp = compose(plus2, invert_map(minus2))

    transported_t1_pm = {FQ(x): FQ(t1_pm[x]) for x in D1}
    transported_t1_mp = {FQ(x): FQ(t1_mp[x]) for x in D1}
    naturality_pm_mismatches = mismatch_count(t2_pm, transported_t1_pm)
    naturality_mp_mismatches = mismatch_count(t2_mp, transported_t1_mp)

    # Corrupted prior: remove the old endpoint relation.
    corrupted_E1 = {(ZERO, HALF), (HALF, ONE)}
    corrupted_E2 = pushforward_edges(corrupted_E1, FQ)
    corrupted_difference = len(E2.symmetric_difference(corrupted_E2))

    # Same-primitive controls.
    plus_carrier = set(D2)
    stale_minus_carrier = set(D1)
    inverse_minus_carrier = {canonical(-I * z) for z in D1}
    stale_mismatch = len(plus_carrier.symmetric_difference(stale_minus_carrier))
    inverse_mismatch = len(plus_carrier.symmetric_difference(inverse_minus_carrier))

    # Primitive Q is not the transfer.
    q_image_D2 = {FQ(z) for z in D2}
    q_closure_failures = len(q_image_D2 - set(D2))
    transfer_square = compose(t2_mp, t2_pm)
    transfer_square_mismatches = sum(int(transfer_square[z] != z) for z in D2)

    expected_E2 = {(ZERO, I / 2), (I / 2, I), (ZERO, I)}
    expected_plus2 = {ZERO: ZERO, I / 2: I / 2, I: I}
    expected_minus2 = {ZERO: I, I / 2: I / 2, I: ZERO}

    checks = {
        "prior_prefix_exact": Xi1["W"] == "B",
        "no_prior_L": prior_L_count == 0,
        "P1_wholly_active": whole_active_sector,
        "Q_selected": selected_primitive == "Q",
        "Xi2_A_exact": Xi2["A"] == 0,
        "Xi2_q_exact": Xi2["q"] == [1, 2],
        "Xi2_theta_exact": sp.simplify(Xi2["theta"] - (THETA_0 + PI / 2)) == 0,
        "Xi2_k_exact": Xi2["k"] == 1,
        "Xi2_j_exact": Xi2["j"] == 2,
        "Xi2_word_exact": Xi2["W"] == "BQ",
        "P2_exact": E2 == expected_E2 and set(D2) == {ZERO, I / 2, I},
        "plus_chart_exact": plus2 == expected_plus2,
        "minus_chart_exact": minus2 == expected_minus2,
        "plus_chart_commuting": chart_plus_commuting_mismatches == 0,
        "minus_chart_commuting": chart_minus_commuting_mismatches == 0,
        "transfer_pm_exact": t2_pm == expected_minus2,
        "transfer_mp_exact": t2_mp == expected_minus2,
        "naturality_pm": naturality_pm_mismatches == 0,
        "naturality_mp": naturality_mp_mismatches == 0,
        "counts_retained": len(D1) == len(D2) == 3 and len(E1) == len(E2) == 3,
        "corrupted_prior_preserved": corrupted_difference == 1,
        "stale_minus_rejected": stale_mismatch > 0,
        "inverse_minus_rejected": inverse_mismatch > 0,
        "Q_not_transfer": q_closure_failures > 0 and transfer_square_mismatches == 0,
    }

    dependency_table = [
        {"component": "Xi2", "sources": ["Xi1", "selected Q"], "passed": all(checks[key] for key in [
            "Q_selected", "Xi2_A_exact", "Xi2_q_exact", "Xi2_theta_exact",
            "Xi2_k_exact", "Xi2_j_exact", "Xi2_word_exact"
        ])},
        {"component": "P2", "sources": ["actual P1", "FQ"], "passed": checks["P2_exact"]},
        {"component": "plus chart", "sources": ["plus1", "FQ", "commuting equation"], "passed": checks["plus_chart_exact"] and checks["plus_chart_commuting"]},
        {"component": "minus chart", "sources": ["minus1", "FQ", "commuting equation"], "passed": checks["minus_chart_exact"] and checks["minus_chart_commuting"]},
        {"component": "T plus-to-minus", "sources": ["updated charts"], "passed": checks["transfer_pm_exact"] and checks["naturality_pm"]},
        {"component": "T minus-to-plus", "sources": ["updated charts"], "passed": checks["transfer_mp_exact"] and checks["naturality_mp"]},
        {"component": "actual-prior dependence", "sources": ["corrupted P1 control"], "passed": checks["corrupted_prior_preserved"]},
        {"component": "same-primitive condition", "sources": ["stale/inverse controls"], "passed": checks["stale_minus_rejected"] and checks["inverse_minus_rejected"]},
        {"component": "Q/transfer distinction", "sources": ["closure/square control"], "passed": checks["Q_not_transfer"]},
    ]

    theorem_pass = all(row["passed"] for row in dependency_table)

    def edge_json(edges):
        return sorted([[expr_key(a), expr_key(b)] for a, b in edges])

    def map_json(mapping):
        return {expr_key(key): expr_key(value) for key, value in mapping.items()}

    result = {
        "scope": {
            "theorem": "complete first-Q transition only",
            "prior_prefix": Xi1["W"],
            "prior_L_count": prior_L_count,
            "P1_wholly_active": whole_active_sector,
            "post_L_whole_object_Q_formula": "OPEN",
        },
        "selection": {
            "next_B_pair": next_B_pair,
            "next_B_product": next_B_product,
            "current_capacity": current_capacity,
            "B_available": B_available,
            "Q_available": Q_available,
            "selected_primitive": selected_primitive,
        },
        "Xi1": {
            "A": Xi1["A"],
            "q": Xi1["q"],
            "theta": expr_key(Xi1["theta"]),
            "k": Xi1["k"],
            "j": Xi1["j"],
            "W": Xi1["W"],
        },
        "Xi2": {
            "A": Xi2["A"],
            "q": Xi2["q"],
            "theta": expr_key(Xi2["theta"]),
            "k": Xi2["k"],
            "j": Xi2["j"],
            "W": Xi2["W"],
        },
        "P1": {
            "determinations": [expr_key(z) for z in D1],
            "edges": edge_json(E1),
        },
        "P2": {
            "determinations": [expr_key(z) for z in D2],
            "edges": edge_json(E2),
        },
        "charts": {
            "plus1": map_json(plus1),
            "minus1": map_json(minus1),
            "plus2": map_json(plus2),
            "minus2": map_json(minus2),
            "commuting_mismatches": {
                "plus": chart_plus_commuting_mismatches,
                "minus": chart_minus_commuting_mismatches,
            },
        },
        "transfers": {
            "plus_to_minus_1": map_json(t1_pm),
            "minus_to_plus_1": map_json(t1_mp),
            "plus_to_minus_2": map_json(t2_pm),
            "minus_to_plus_2": map_json(t2_mp),
            "naturality_mismatches": {
                "plus_to_minus": naturality_pm_mismatches,
                "minus_to_plus": naturality_mp_mismatches,
            },
        },
        "controls": {
            "corrupted_prior_edge_difference": corrupted_difference,
            "stale_minus_carrier_difference": stale_mismatch,
            "inverse_minus_carrier_difference": inverse_mismatch,
            "Q_closure_failures_on_D2": q_closure_failures,
            "transfer_square_mismatches": transfer_square_mismatches,
        },
        "dependency_table": dependency_table,
        "checks": checks,
        "passed_checks": sum(int(value) for value in checks.values()),
        "total_checks": len(checks),
        "first_Q_theorem_pass": theorem_pass,
        "computed_verdict": "PASS" if all(checks.values()) and theorem_pass else "FAIL",
    }
    return result


if __name__ == "__main__":
    result = calculate()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["computed_verdict"] != "PASS":
        raise SystemExit(1)
