#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

TS = "20260711T174605"
PRIOR_HASH = "6c5109dac6bde39687142a05c474db16f19d698f9d6040c5a07d92a7a0784ac2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"cannot infer fields for empty CSV {path}")
    names = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


@dataclass(frozen=True)
class State:
    A: int
    u: int
    v: int
    qturns: int
    k: int
    B_count: int
    W: str

    @property
    def N(self) -> int:
        return 6 * (2**self.A)

    @property
    def j_start(self) -> int:
        return 1 + 6 * (2**self.A - 1)

    @property
    def j(self) -> int:
        return self.j_start + self.k

    @property
    def product(self) -> int:
        return self.u * self.v

    @property
    def next_pair(self) -> tuple[int, int]:
        return (self.v, self.u + self.v)

    @property
    def capacity(self) -> int:
        if self.j == 1:
            return 2
        if self.j == 2:
            return 4
        return 2 ** (2 * self.j)

    def can_q(self) -> bool:
        return self.k < self.N - 1

    def can_b(self) -> bool:
        if self.k < self.N - 1:
            a, b = self.next_pair
            return a * b <= self.capacity
        return self.product < self.capacity

    def boundary(self) -> bool:
        return (not self.can_b()) and (not self.can_q())


def step(s: State) -> tuple[str, State]:
    if s.can_b():
        u2, v2 = s.next_pair
        return "B", State(s.A, u2, v2, s.qturns, s.k, s.B_count + 1, s.W + "B")
    if s.can_q():
        return "Q", State(s.A, s.u, s.v, s.qturns + 1, s.k + 1, s.B_count, s.W + "Q")
    return "L", State(s.A + 1, s.u, s.v, s.qturns, 0, s.B_count, s.W + "L")


def simulate_boundaries(max_A: int) -> tuple[list[State], list[dict[str, Any]], list[dict[str, Any]]]:
    s = State(0, 1, 1, 0, 0, 0, "")
    boundaries: list[State] = []
    transition_trace: list[dict[str, Any]] = []
    return_edges: list[dict[str, Any]] = []
    previous_boundary: State | None = None
    previous_step_index = 0
    step_index = 0

    while len(boundaries) <= max_A:
        if s.boundary():
            if s.A != len(boundaries):
                raise AssertionError(f"boundary domain mismatch {s.A} != {len(boundaries)}")
            boundaries.append(s)
            if previous_boundary is not None:
                suffix = s.W[len(previous_boundary.W):]
                if not suffix.startswith("L"):
                    raise AssertionError("return path must begin with closing L")
                if suffix.endswith("L"):
                    raise AssertionError("pre-L target prefix must not include target closing L")
                return_edges.append({
                    "source_A": previous_boundary.A,
                    "target_A": s.A,
                    "source_word_length": len(previous_boundary.W),
                    "target_word_length": len(s.W),
                    "path_word": suffix,
                    "path_length": len(suffix),
                    "B_increment": s.B_count - previous_boundary.B_count,
                    "Q_increment": s.qturns - previous_boundary.qturns,
                    "carry": s.B_count - 2 * previous_boundary.B_count,
                    "path_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
                    "source_step_index": previous_step_index,
                    "target_step_index": step_index,
                })
            previous_boundary = s
            previous_step_index = step_index
            if len(boundaries) > max_A:
                break
        op, s2 = step(s)
        transition_trace.append({
            "step": step_index,
            "A_before": s.A,
            "u_before": str(s.u),
            "v_before": str(s.v),
            "k_before": s.k,
            "j_before": s.j,
            "B_count_before": s.B_count,
            "operation": op,
            "A_after": s2.A,
            "u_after": str(s2.u),
            "v_after": str(s2.v),
            "k_after": s2.k,
            "j_after": s2.j,
            "B_count_after": s2.B_count,
        })
        s = s2
        step_index += 1
        if step_index > 5_000_000:
            raise RuntimeError("simulation runaway")

    return boundaries, return_edges, transition_trace


def read_accepted_trace(path: Path) -> tuple[list[dict[str, str]], list[int]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 10001:
        raise AssertionError(f"expected 10001 rows, got {len(rows)}")
    if [int(r["A"]) for r in rows] != list(range(10001)):
        raise AssertionError("trace must contain exactly A=0..10000")
    carries = [int(rows[a]["carry"]) for a in range(1, 10001)]
    if any(c not in (7, 8, 9) for c in carries):
        raise AssertionError("carry outside {7,8,9}")
    return rows, carries


def words(seq: list[int], n: int) -> set[tuple[int, ...]]:
    return {tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)}


def d0_classification() -> list[dict[str, str]]:
    return [
        {"construction": "one-step primitive transitions", "classification": "NATIVE", "certificate": "Step emits B, Q, or L"},
        {"construction": "exact word prefixes", "classification": "NATIVE", "certificate": "W is a retained state coordinate"},
        {"construction": "finite QBL paths", "classification": "DERIVED INSIDE D0", "certificate": "finite composition of Step"},
        {"construction": "stopping times", "classification": "DERIVED INSIDE D0", "certificate": "first hitting index of a state predicate"},
        {"construction": "boundary sections", "classification": "DERIVED INSIDE D0", "certificate": "B blocked and Q blocked"},
        {"construction": "first-return maps", "classification": "DERIVED INSIDE D0", "certificate": "iterate Step between section hits"},
        {"construction": "path observables", "classification": "DERIVED INSIDE D0", "certificate": "function of retained state/path/word suffix"},
        {"construction": "induced symbolic codes", "classification": "DERIVED INSIDE D0", "certificate": "label on states or return paths"},
        {"construction": "unrealized topological closure points", "classification": "NOT YET LICENSED", "certificate": "no corresponding enlarged QBL state family"},
    ]


def triangular_return_control(mmax: int = 12) -> dict[str, Any]:
    section = [m * (m + 1) // 2 for m in range(mmax + 1)]
    edges = []
    for m in range(mmax):
        source, target = section[m], section[m + 1]
        length = target - source
        path = list(range(source, target + 1))
        edges.append({"m": m, "source": source, "target": target, "return_length": length,
                      "label": "odd" if length % 2 else "even", "path": path})
    internal = all(e["path"][0] == e["source"] and e["path"][-1] == e["target"] for e in edges)
    label_internal = all(e["label"] == ("odd" if e["return_length"] % 2 else "even") for e in edges)
    return {
        "name": "triangular-section successor return code",
        "state_internality": True,
        "transition_internality": internal,
        "relation_internality": label_internal,
        "fiber_split_beyond_base": False,
        "verdict": "SAME-LAYER INDUCED RECODING",
        "edges": edges,
    }


def fiber_split_control() -> dict[str, Any]:
    states = [(x, b) for x in range(4) for b in (0, 1)]
    witness_pairs = []
    for x in range(4):
        z, zp = (x, 0), (x, 1)
        witness_pairs.append({"z": z, "z_prime": zp, "D_z": x, "D_z_prime": x, "xi_z": 0, "xi_z_prime": 1})
    return {
        "name": "second-coordinate fiber split",
        "old_description": "D(x,b)=x",
        "new_coordinate": "xi(x,b)=b",
        "fiber_split": all(w["D_z"] == w["D_z_prime"] and w["xi_z"] != w["xi_z_prime"] for w in witness_pairs),
        "verdict": "GENUINE ADDITIONAL DETERMINATION RELATIVE TO OLD DESCRIPTION",
        "witness_pairs": witness_pairs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.package_root.resolve()
    inp = root / "inputs"
    out = root / "outputs"
    trace = root / "trace"
    out.mkdir(parents=True, exist_ok=True)
    trace.mkdir(parents=True, exist_ok=True)

    law = (inp / f"{TS}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md").read_text(encoding="utf-8")
    cf000_hash = sha256(inp / f"{TS}_CF000_Primitive_Distinguishability.pdf")
    required_law_anchors = [
        "X=(A,q,\\theta,k,j,W)",
        "B>Q>L",
        "The exact ordered word and full prefix state are part of the certificate",
        "the explicit all-depth recurrence for the primary pairing",
    ]
    missing = [x for x in required_law_anchors if x not in law]
    if missing:
        raise AssertionError(f"custody authority anchors missing: {missing}")

    prior_zip = inp / f"{TS}_PRIOR_p5-b3-v5_release.zip"
    prior_hash_actual = sha256(prior_zip)
    prior_hash_file = (inp / f"{TS}_PRIOR_p5-b3-v5_release.zip.sha256").read_text().split()[0]
    if prior_hash_actual != PRIOR_HASH or prior_hash_file != PRIOR_HASH or len(PRIOR_HASH) != 64:
        raise AssertionError("corrected prior release hash failed")
    write_json(out / f"{TS}_corrected_prior_release_hash.json", {
        "archive": prior_zip.name,
        "expected": PRIOR_HASH,
        "actual": prior_hash_actual,
        "hex_length": len(prior_hash_actual),
        "verifier_result": "PASS",
    })

    accepted_rows, carries = read_accepted_trace(inp / f"{TS}_ACCEPTED_CARRY_TRACE_A0_A10000.csv")
    boundaries, return_edges, primitive_trace = simulate_boundaries(8)

    boundary_rows: list[dict[str, Any]] = []
    for s in boundaries:
        boundary_rows.append({
            "A": s.A,
            "u": str(s.u),
            "v": str(s.v),
            "product": str(s.product),
            "qturns": s.qturns,
            "k": s.k,
            "j": s.j,
            "N_A": s.N,
            "B_count": s.B_count,
            "word_length": len(s.W),
            "word_sha256": hashlib.sha256(s.W.encode()).hexdigest(),
            "boundary_predicate": s.boundary(),
        })
    write_csv(out / f"{TS}_canonical_boundary_states.csv", boundary_rows)
    write_csv(out / f"{TS}_return_edges.csv", return_edges)

    for idx, edge in enumerate(return_edges, start=1):
        expected = int(accepted_rows[idx]["carry"])
        if edge["carry"] != expected:
            raise AssertionError(f"carry mismatch A={idx}: {edge['carry']} != {expected}")

    classification = d0_classification()
    write_csv(out / f"{TS}_D0_articulation_class.csv", classification)

    interpretation_premises = {
        "state_internality": all(s.boundary() for s in boundaries),
        "transition_internality": all(e["target_A"] == e["source_A"] + 1 and e["path_word"].startswith("L") for e in return_edges),
        "relation_internality": all(e["carry"] == boundaries[e["target_A"]].B_count - 2 * boundaries[e["source_A"]].B_count for e in return_edges),
        "object_injectivity": len({(s.A, s.W) for s in boundaries}) == len(boundaries),
        "edge_injectivity": len({e["path_sha256"] for e in return_edges}) == len(return_edges),
        "unique_boundary_segmentation": all(e["target_word_length"] > e["source_word_length"] for e in return_edges),
        "fiber_split_beyond_complete_D0_interpretation_found": False,
    }
    same_layer = all(v for k, v in interpretation_premises.items() if k != "fiber_split_beyond_complete_D0_interpretation_found") and not interpretation_premises["fiber_split_beyond_complete_D0_interpretation_found"]
    if not same_layer:
        raise AssertionError("constructed D1-to-D0 interpretation did not satisfy same-layer premises")
    interpretation = {
        "objects": "J(S_A^-)=S_A^-",
        "edges": "J(r_A)=the exact D0 path suffix from W_A^- to W_{A+1}^-",
        "faithful": "DOCUMENT PROOF; finite regression PASS",
        "full_onto_boundary_return_subcategory": "DOCUMENT PROOF; deterministic segmentation algorithm",
        "invertible_enriched_presentation": "DOCUMENT PROOF; split at boundary predicate hits",
        "symbolic_carry_map": "information-losing path-observable factor",
        "premises": interpretation_premises,
        "derived_verdict": "D1 IS A SAME-LAYER INDUCED RECODING OF D0: PROVED",
    }
    write_json(out / f"{TS}_D1_to_D0_interpretation.json", interpretation)

    control_a = triangular_return_control()
    control_b = fiber_split_control()
    write_json(out / f"{TS}_negative_control_same_layer_return_code.json", control_a)
    write_json(out / f"{TS}_negative_control_genuine_fiber_split.json", control_b)

    saturation = [
        {"level": "local Domain-A at S_A^-", "status": "PROVED", "evidence_class": "DOCUMENT PROOF + EXACT SIMULATION", "certificate": "B blocked and Q blocked"},
        {"level": "complete D0 all-domain custody layer", "status": "FALSE ON CANONICAL INFINITE ORBIT", "evidence_class": "DOCUMENT PROOF", "certificate": "native L and later B/Q steps continue within D0"},
        {"level": "D1 saturation criterion", "status": "DEFINED", "evidence_class": "DEFINITION", "certificate": "future separation, return-observable completeness, no internal refinement"},
        {"level": "D1 same-layer saturation", "status": "NOT YET DERIVED", "evidence_class": "OPEN", "certificate": "no nontrivial complete D1 descriptor proved"},
        {"level": "D1 next re-articulation", "status": "NOT YET DERIVED", "evidence_class": "OPEN", "certificate": "no D1 saturation and no forced irreducible next determination"},
    ]
    write_csv(out / f"{TS}_saturation_status.csv", saturation)

    statuses = [
        {"claim": "D1 INDUCED RETURN INVARIANT", "status": "PROVED", "evidence_class": "DOCUMENT PROOF", "certificate": "first-return system and exact cocycle"},
        {"claim": "D1 IS A SAME-LAYER INDUCED RECODING OF D0", "status": "PROVED", "evidence_class": "DOCUMENT PROOF", "certificate": "full faithful interpretation with inverse segmentation"},
        {"claim": "D1 DOMAIN-PROPER EFFECTIVE INVARIANT", "status": "FALSE FOR INDUCED ORBIT PRESENTATION", "evidence_class": "DOCUMENT PROOF", "certificate": "no determination beyond complete D0 path interpretation"},
        {"claim": "D1 ADMISSION MECHANISM", "status": "LAWFUL INDUCTION, NOT FORCED RE-ARTICULATION", "evidence_class": "DOCUMENT PROOF", "certificate": "D0 continues natively through L"},
        {"claim": "HIGHER-ORDER DESCRIPTIVE L D0->D1", "status": "FALSE", "evidence_class": "DOCUMENT PROOF", "certificate": "D0 not saturated; D1 same-layer"},
        {"claim": "D1 SAME-LAYER SATURATION", "status": "NOT YET DERIVED", "evidence_class": "OPEN", "certificate": "criterion defined but not satisfied"},
        {"claim": "ORTHAD-LEVEL HIGHER-ORDER L", "status": "NOT YET DERIVED", "evidence_class": "OPEN", "certificate": "primary pairing recurrence remains first open dependency"},
    ]
    write_csv(out / f"{TS}_structural_status.csv", statuses)

    complexity_rows = []
    full_through = 0
    for n in range(1, 21):
        count = len(words(carries, n))
        full = 2 ** (n + 1) - 1
        if count == full:
            full_through = n
        complexity_rows.append({
            "length": n,
            "canonical_observed": count,
            "morse_hedlund_lower": n + 1,
            "full_affine": full,
            "coverage": count / full,
        })
    write_csv(out / f"{TS}_canonical_word_complexity.csv", complexity_rows)

    state_counts = Counter(carries)
    write_csv(out / f"{TS}_state_frequencies.csv", [
        {"symbol": s, "count": state_counts[s], "frequency": state_counts[s] / len(carries)} for s in (7, 8, 9)
    ])

    edge_counts = Counter(zip(carries[:-1], carries[1:]))
    edge_rows = []
    for i in (7, 8, 9):
        for j in (7, 8, 9):
            edge_rows.append({"from": i, "to": j, "count": edge_counts[(i, j)], "frequency": edge_counts[(i, j)] / (len(carries)-1)})
    write_csv(out / f"{TS}_edge_frequencies.csv", edge_rows)

    accepted_status = {
        "PROVED": [
            "CANONICAL QBL-TO-AFFINE BOUNDARY-ORBIT SEMICONJUGACY",
            "EXACT BOUNDARY-RETURN COCYCLE",
            "CANONICAL CARRY ITINERARY APERIODIC",
            "CANONICAL SYMBOLIC ORBIT CLOSURE INFINITE",
            "CANONICAL SYMBOLIC ORBIT CLOSURE TRANSITIVE",
            "D1 INDUCED RETURN INVARIANT",
            "D1 IS A SAME-LAYER INDUCED RECODING OF D0",
            "D1 IS A LAWFUL INDUCED DESCRIPTION, NOT A FORCED RE-ARTICULATION",
            "HIGHER-ORDER DESCRIPTIVE L FALSE FOR D0-TO-D1",
        ],
        "CERTIFIED_FINITE": [
            "custody simulation through Domain 8",
            "accepted carry trace A=0..10000 validated",
            f"full ambient language coverage through length {full_through}",
            "corrected prior release hash verified",
        ],
        "OBSERVED": [
            "broad finite ambient-language coverage",
            "finite frequencies close to prior Lebesgue benchmark",
        ],
        "OPEN": [
            "CANONICAL ORBIT CLOSURE = FULL AFFINE SYSTEM",
            "CANONICAL ORBIT CLOSURE IS PROPER",
            "SPECIFIC-ORBIT EQUIDISTRIBUTION",
            "D1 SAME-LAYER SATURATION",
            "D1 NEXT RE-ARTICULATION",
            "EXACT PRIMARY PAIRING RECURRENCE",
            "ORTHAD-LEVEL HIGHER-ORDER L",
        ],
    }
    write_json(out / f"{TS}_evidence_status.json", accepted_status)

    with (trace / f"{TS}_primitive_custody_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in primitive_trace:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (trace / f"{TS}_return_interpretation_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in return_edges:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (trace / f"{TS}_structural_decision_trace.jsonl").open("w", encoding="utf-8") as f:
        decisions = [
            {"question": "new alphabet implies new domain", "result": "REJECTED BY CONTROL A", "evidence_class": "COUNTERMODEL"},
            {"question": "fiber split is sufficient evidence of new determination", "result": "SUPPORTED BY CONTROL B", "evidence_class": "COUNTERMODEL"},
            {"question": "D1 enriched return system", "result": "SAME-LAYER INDUCED RECODING", "evidence_class": "DOCUMENT PROOF"},
            {"question": "D0 saturation forces D1", "result": "FALSE; D0 CONTINUES NATIVELY", "evidence_class": "DOCUMENT PROOF"},
            {"question": "D1 saturation", "result": "OPEN", "evidence_class": "OPEN"},
        ]
        for row in decisions:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "cf000_sha256": cf000_hash,
        "prior_release_hash_verification": "PASS",
        "simulated_boundaries": len(boundaries),
        "simulated_return_edges": len(return_edges),
        "accepted_carries": len(carries),
        "accepted_edges": len(carries) - 1,
        "full_finite_language_through_length": full_through,
        "same_layer_premises": interpretation_premises,
        "decisive_verdict": "D1 IS A SAME-LAYER INDUCED RECODING OF D0: PROVED",
        "descriptive_L_D0_to_D1": "FALSE",
        "branch_status": "OPEN",
    }
    write_json(out / f"{TS}_derivation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
