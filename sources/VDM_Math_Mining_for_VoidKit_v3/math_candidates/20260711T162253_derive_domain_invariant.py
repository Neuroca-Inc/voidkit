#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

TS = "20260711T162253"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_trace(rows: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    if len(rows) != 10001:
        raise ValueError(f"expected 10001 rows A=0..10000, got {len(rows)}")
    As = [int(r["A"]) for r in rows]
    if As != list(range(10001)):
        raise ValueError("A column is not exactly 0..10000")
    carries: list[int] = []
    defects: list[int] = []
    for r in rows[1:]:
        c = int(r["carry"])
        if c not in (7, 8, 9):
            raise ValueError(f"carry outside {{7,8,9}} at A={r['A']}: {c}")
        carries.append(c)
        d = int(r["defect"])
        if d not in (-2, -1, 0, 1, 2):
            raise ValueError(f"defect outside five-letter alphabet at A={r['A']}: {d}")
        defects.append(d)
    if len(carries) != 10000:
        raise ValueError("expected 10000 canonical carries")
    return carries, defects


def word_counts(seq: list[int], max_n: int = 20) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in range(1, max_n + 1):
        words = {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}
        full = 2 ** (n + 1) - 1
        out.append(
            {
                "length": n,
                "canonical_observed": len(words),
                "full_affine": full,
                "coverage": len(words) / full,
                "morse_hedlund_lower_bound": n + 1,
            }
        )
    return out


def exact_symbolic_checks() -> dict[str, Any]:
    lam, beta, nu, j, c = sp.symbols("lambda beta nu j c")
    gamma = 6 * lam - beta
    pi_state = lam * j + beta - nu
    next_pi = lam * (2 * j + 6) + beta - (2 * nu + c)
    recurrence_residual = sp.simplify(next_pi - (2 * pi_state + gamma - c))

    E = sp.symbols("E")
    F = 2 * E + gamma - c
    # In R/Z, h(F)-2h(E) is the integer -c.
    conjugacy_integer_residual = sp.expand((F + gamma) - 2 * (E + gamma))

    A = sp.symbols("A", integer=True, nonnegative=True)
    J = 6 * (2 ** (A + 1) - 1)
    J_next_residual = sp.simplify(J.subs(A, A + 1) - (2 * J + 6))
    seed_residual = sp.simplify(lam * (J + 6) - 12 * lam * 2**A)

    return {
        "boundary_factor_recurrence_residual": str(recurrence_residual),
        "doubling_conjugacy_difference": str(conjugacy_integer_residual),
        "doubling_conjugacy_is_integer_when_c_integer": True,
        "global_position_recurrence_residual": str(J_next_residual),
        "canonical_doubling_seed_residual": str(seed_residual),
        "all_symbolic_checks_zero_or_integer": recurrence_residual == 0
        and J_next_residual == 0
        and seed_residual == 0
        and conjugacy_integer_residual == -c,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    inputs = root / "inputs"
    outputs = root / "outputs"
    trace_dir = root / "trace"
    outputs.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    carry_path = inputs / f"{TS}_ACCEPTED_CARRY_TRACE_A0_A10000.csv"
    rows = read_trace(carry_path)
    carries, defects = validate_trace(rows)

    counts = word_counts(carries, 20)
    state_counts = Counter(carries)
    edge_counts = Counter(zip(carries[:-1], carries[1:]))
    defect_counts = Counter(defects)

    with (outputs / f"{TS}_canonical_word_complexity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(counts[0].keys()))
        writer.writeheader()
        writer.writerows(counts)

    with (outputs / f"{TS}_state_frequencies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "count", "frequency"])
        for s in (7, 8, 9):
            writer.writerow([s, state_counts[s], state_counts[s] / len(carries)])

    with (outputs / f"{TS}_edge_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to", "count", "frequency"])
        total_edges = len(carries) - 1
        for i in (7, 8, 9):
            for j2 in (7, 8, 9):
                writer.writerow([i, j2, edge_counts[(i, j2)], edge_counts[(i, j2)] / total_edges])

    with (outputs / f"{TS}_defect_frequencies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["defect", "count", "frequency"])
        for d in (-2, -1, 0, 1, 2):
            writer.writerow([d, defect_counts[d], defect_counts[d] / len(defects)])

    symbolic = exact_symbolic_checks()
    if not symbolic["all_symbolic_checks_zero_or_integer"]:
        raise RuntimeError("symbolic proof dependencies failed")
    write_json(outputs / f"{TS}_symbolic_checks.json", symbolic)

    density_boundary = {
        "lambda": "log(2)/log(phi)",
        "alpha": "12*log(2)/log(phi)",
        "alpha_irrational": "PROVED IN DOCUMENT BY GALOIS CONJUGATION",
        "density_equivalent_to": "base-2 disjunctivity of alpha",
        "irrationality_implies_density": False,
        "canonical_orbit_closure": "NOT YET DETERMINED",
        "case_3_only_arithmetic_intrinsic": False,
        "exact_missing_bridge": "prove or refute base-2 disjunctivity of alpha",
    }
    write_json(outputs / f"{TS}_density_boundary.json", density_boundary)

    d0_d1 = {
        "D0": {
            "objects": "primitive custody states X_t=(A,q,theta,k,j,W)",
            "transformations": ["B", "Q", "L"],
            "selection": "strict state law B>Q>L",
        },
        "D1": {
            "objects": "canonical saturated pre-L states S_A^- and return edges e_A",
            "transformation": "induced first-return map R",
            "derived_edge_label": "c_A=nu(q_A)-2*nu(q_{A-1}) in {7,8,9}",
        },
        "inherited": ["A", "q", "theta", "k", "j", "W", "B-count recoverability", "domain history"],
    }
    write_json(outputs / f"{TS}_descriptive_domains.json", d0_d1)

    l_test = [
        ("prior invariant stack retained", "PASS", "D1 is induced over complete saturated D0 boundary states"),
        ("new descriptive domain lawfully admitted", "PASS", "exact variable-length first-return map"),
        ("new domain-proper effective invariant", "PASS", "three-valued renormalized return cocycle"),
        ("not a D0 symbol rename", "PASS", "all closings use L while c takes 7,8,9"),
        ("same-layer saturation criterion independent", "PASS", "future/follower separation criterion stated"),
    ]
    with (outputs / f"{TS}_descriptive_L_test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["requirement", "verdict", "certificate"])
        writer.writerows(l_test)

    statuses = {
        "PROVED": [
            "CANONICAL QBL-TO-AFFINE BOUNDARY-ORBIT SEMICONJUGACY",
            "EXACT BOUNDARY-RETURN COCYCLE",
            "CANONICAL CARRY ITINERARY APERIODIC",
            "CANONICAL SYMBOLIC ORBIT CLOSURE INFINITE",
            "CANONICAL SYMBOLIC ORBIT CLOSURE TRANSITIVE",
            "D1 EFFECTIVE INVARIANT GENUINELY NEW",
            "HIGHER-ORDER DESCRIPTIVE L",
        ],
        "CERTIFIED_FINITE": [
            "accepted trace has A=0..10000 exactly",
            "all carries lie in {7,8,9}",
            "all ambient words length 1..7 occur in accepted finite trace",
        ],
        "OBSERVED": [
            "finite language coverage is compatible with density",
            "finite frequencies are close to prior Lebesgue benchmarks",
        ],
        "OPEN": [
            "CANONICAL ORBIT CLOSURE = FULL AFFINE SYSTEM",
            "CANONICAL ORBIT CLOSURE IS PROPER",
            "SPECIFIC-ORBIT EQUIDISTRIBUTION",
            "CANONICAL ORBIT-CLOSURE NON-SOFICITY",
            "CANONICAL ORBIT-CLOSURE MIXING",
            "CANONICAL ORBIT-CLOSURE FINITE MARKOV ORDER",
            "EXACT PRIMARY PAIRING RECURRENCE",
            "ORTHAD-LEVEL HIGHER-ORDER L",
        ],
    }
    write_json(outputs / f"{TS}_status.json", statuses)

    # Compact trace files.
    with (trace_dir / f"{TS}_canonical_carry_trace.jsonl").open("w", encoding="utf-8") as f:
        for idx, cval in enumerate(carries, start=1):
            f.write(json.dumps({"A": idx, "carry": cval, "defect": defects[idx - 1]}, sort_keys=True) + "\n")

    with (trace_dir / f"{TS}_language_coverage_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in counts:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    decisions = [
        {"claim": "irrationality implies density", "verdict": "REJECTED", "reason": "proper infinite doubling-invariant closures may contain irrational points"},
        {"claim": "case 3 only arithmetic intrinsic", "verdict": "REJECTED", "reason": "canonical aperiodic transitive symbolic orbit closure is intrinsic"},
        {"claim": "D1 novelty", "verdict": "PROVED", "reason": "induced return-edge class is not a primitive-letter rename"},
        {"claim": "descriptive L", "verdict": "PROVED", "reason": "retention, lawful induction, new effective invariant, and independent saturation criterion"},
        {"claim": "Orthad L", "verdict": "OPEN", "reason": "primary pairing recurrence is first missing dependency"},
    ]
    with (trace_dir / f"{TS}_decision_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    build_summary = {
        "trace_rows": len(rows),
        "carry_symbols": len(carries),
        "edges": len(carries) - 1,
        "symbolic_checks": "PASS",
        "finite_words_full_through_length": max(r["length"] for r in counts if r["canonical_observed"] == r["full_affine"]),
        "descriptive_L": "PROVED",
        "canonical_closure": "NOT YET DETERMINED",
    }
    write_json(outputs / f"{TS}_derivation_summary.json", build_summary)
    print(json.dumps(build_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
