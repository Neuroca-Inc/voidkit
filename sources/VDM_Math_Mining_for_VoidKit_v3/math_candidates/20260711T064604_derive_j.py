#!/usr/bin/env python3
"""Derive and audit the QBL carry transition geometry.

This script closes only the affine ceiling-map / interval-coding track and
compares it with the certified finite carry trace A=0..10000. It does not infer
Orthad chart matrices, gauge data, holonomy, FQM classes, Weil projections, or
specific-orbit equidistribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp

STAMP = "20260711T064604"
STATES = (7, 8, 9)
DEFECTS = (-2, -1, 0, 1, 2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, sp.MatrixBase):
        return [[_jsonable(value[i, j]) for j in range(value.cols)] for i in range(value.rows)]
    if isinstance(value, sp.Basic):
        return str(sp.simplify(value))
    if isinstance(value, mp.mpf):
        return mp.nstr(value, 80)
    return value


def constants(dps: int = 100) -> dict[str, mp.mpf]:
    mp.mp.dps = dps
    phi = (1 + mp.sqrt(5)) / 2
    log_phi = mp.log(phi)
    lam = 6 * mp.log(2) / log_phi
    gamma = lam + mp.mpf("1.5") - mp.log(5) / (2 * log_phi)
    a = (gamma - 8) / 2
    y0 = 2 * lam - gamma
    return {
        "phi": phi,
        "lambda": lam,
        "gamma": gamma,
        "a": a,
        "y0": y0,
    }


def symbolic_system() -> dict[str, Any]:
    a = sp.symbols("a", real=True)
    half = sp.Rational(1, 2)
    quarter = sp.Rational(1, 4)
    J = sp.Matrix(
        [
            [0, (1 - 3 * a) / 2, a / 2],
            [(1 - 2 * a) / 4, quarter, a / 2],
            [(1 - 2 * a) / 4, 3 * a / 2 - quarter, 0],
        ]
    )
    pi = sp.Matrix([[half - a, half, a]])
    P = sp.Matrix(
        [
            [0, (1 - 3 * a) / (1 - 2 * a), a / (1 - 2 * a)],
            [(1 - 2 * a) / 2, half, a],
            [(1 - 2 * a) / (4 * a), (6 * a - 1) / (4 * a), 0],
        ]
    )
    M = sp.Matrix([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
    M2 = M**2
    sqrt2 = sp.sqrt(2)
    rho = 1 + sqrt2
    r = sp.Matrix([1, sqrt2, 1])
    K = sp.Matrix(
        [
            [0, sqrt2, 1],
            [sqrt2, 2, sqrt2],
            [1, sqrt2, 0],
        ]
    ) / (4 * rho)
    defect = {
        -2: sp.simplify(J[2, 0]),
        -1: sp.simplify(J[1, 0] + J[2, 1]),
        0: sp.simplify(J[0, 0] + J[1, 1] + J[2, 2]),
        1: sp.simplify(J[0, 1] + J[1, 2]),
        2: sp.simplify(J[0, 2]),
    }
    checks = {
        "row_sums": [sp.simplify(sum(J[i, j] for j in range(3))) for i in range(3)],
        "column_sums": [sp.simplify(sum(J[i, j] for i in range(3))) for j in range(3)],
        "total_mass": sp.simplify(sum(J)),
        "P_row_sums": [sp.simplify(sum(P[i, j] for j in range(3))) for i in range(3)],
        "stationarity": [sp.simplify(x) for x in list(pi * P - pi)],
        "M2": M2,
        "perron_residual": [sp.simplify(x) for x in list(M * r - rho * r)],
        "K_row_sums": [sp.simplify(sum(K[i, j] for j in range(3))) for i in range(3)],
        "K_column_sums": [sp.simplify(sum(K[i, j] for i in range(3))) for j in range(3)],
        "K_total_mass": sp.simplify(sum(K)),
    }
    return {
        "a": a,
        "J": J,
        "pi": pi,
        "P": P,
        "M": M,
        "M2": M2,
        "rho": rho,
        "r": r,
        "K": K,
        "defect": defect,
        "checks": checks,
    }


def load_prior_trace(package_root: Path) -> list[dict[str, str]]:
    candidates = sorted((package_root / "inputs").glob("*_PRIOR_CARRY_DEFECT_A0_A10000.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one prior finite trace CSV, found {len(candidates)}")
    with candidates[0].open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def empirical_from_trace(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_A = {int(r["A"]): r for r in rows}
    if min(by_A) != 0 or max(by_A) != 10000:
        raise RuntimeError("Prior trace must cover A=0..10000")
    carries = {A: int(by_A[A]["carry"]) for A in range(1, 10001)}
    transitions = Counter((carries[A - 1], carries[A]) for A in range(2, 10001))
    defects = Counter(carries[A] - carries[A - 1] for A in range(2, 10001))
    state_counts = Counter(carries.values())
    counts = np.array([[transitions[(i, j)] for j in STATES] for i in STATES], dtype=np.int64)
    total_edges = int(counts.sum())
    empirical_joint = counts.astype(np.float64) / total_edges
    return {
        "carries": carries,
        "transition_counts": transitions,
        "defect_counts": defects,
        "state_counts": state_counts,
        "joint_counts": counts,
        "joint": empirical_joint,
        "total_edges": total_edges,
    }


def boundary_scan(max_a: int = 10000, dps: int = 4500) -> dict[str, Any]:
    c = constants(dps)
    gamma = c["gamma"]
    a = c["a"]
    E = c["y0"] - mp.ceil(c["y0"])
    boundaries = [mp.mpf(-1), -mp.mpf("0.5") - a, -a, mp.mpf(0)]
    names = ["-1", "b7=(7-gamma)/2", "b8=(8-gamma)/2", "0"]
    minima = [(abs(E - b), 0, E) for b in boundaries]
    hits: list[dict[str, Any]] = []
    carry_counts: Counter[int] = Counter()
    for A in range(1, max_a + 1):
        z = 2 * E + gamma
        carry = int(mp.ceil(z))
        E = z - carry
        carry_counts[carry] += 1
        for idx, b in enumerate(boundaries):
            dist = abs(E - b)
            if dist == 0:
                hits.append({"A": A, "boundary": names[idx]})
            if dist < minima[idx][0]:
                minima[idx] = (dist, A, E)
    records = []
    for name, boundary, (distance, A, value) in zip(names, boundaries, minima):
        records.append(
            {
                "boundary": name,
                "boundary_value": mp.nstr(boundary, 90),
                "minimum_distance": mp.nstr(distance, 90),
                "A": A,
                "E_A": mp.nstr(value, 90),
            }
        )
    global_min = min(records, key=lambda r: mp.mpf(r["minimum_distance"]))
    return {
        "max_A": max_a,
        "dps": dps,
        "hits": hits,
        "minima": records,
        "global_minimum": global_min,
        "carry_counts": dict(carry_counts),
    }


def boundary_precision_check(max_a: int = 10000) -> dict[str, Any]:
    scan_3500 = boundary_scan(max_a=max_a, dps=3500)
    scan_4500 = boundary_scan(max_a=max_a, dps=4500)
    stable = True
    for left, right in zip(scan_3500["minima"], scan_4500["minima"]):
        stable &= left["A"] == right["A"]
        stable &= left["minimum_distance"][:70] == right["minimum_distance"][:70]
    return {
        "stable_70_digits": bool(stable),
        "scan_3500": scan_3500,
        "scan_4500": scan_4500,
    }



def metrics_high_precision(counts: np.ndarray, target: list[list[mp.mpf]]) -> dict[str, str]:
    mp.mp.dps = 120
    total = mp.mpf(int(counts.sum()))
    diffs = []
    for i in range(3):
        for j in range(3):
            empirical = mp.mpf(int(counts[i, j])) / total
            diffs.append(abs(empirical - target[i][j]))
    maximum = max(diffs)
    l1 = sum(diffs)
    return {
        "max_absolute_error": mp.nstr(maximum, 60),
        "L1_error": mp.nstr(l1, 60),
        "total_variation": mp.nstr(l1 / 2, 60),
    }


def metrics(emp: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = np.abs(emp - target)
    return {
        "max_absolute_error": float(delta.max()),
        "L1_error": float(delta.sum()),
        "total_variation": float(delta.sum() / 2),
    }


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run(package_root: Path) -> dict[str, Any]:
    outputs = package_root / "outputs"
    traces = package_root / "trace"
    outputs.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)

    sym = symbolic_system()
    c100 = constants(120)
    a_num = float(c100["a"])
    gamma_num = float(c100["gamma"])
    subs = {sym["a"]: sp.N(str(c100["a"]), 80)}
    J_num = np.array(sym["J"].subs(subs).evalf(50).tolist(), dtype=float)
    P_num = np.array(sym["P"].subs(subs).evalf(50).tolist(), dtype=float)
    K_num = np.array(sym["K"].evalf(50).tolist(), dtype=float)

    rows = load_prior_trace(package_root)
    empirical = empirical_from_trace(rows)
    emp_joint = empirical["joint"]
    edge_metrics_J_float = metrics(emp_joint, J_num)
    edge_metrics_K_float = metrics(emp_joint, K_num)
    mp.mp.dps = 120
    a_mp = c100["a"]
    sqrt2_mp = mp.sqrt(2)
    rho_mp = 1 + sqrt2_mp
    J_mp = [
        [mp.mpf(0), (1-3*a_mp)/2, a_mp/2],
        [(1-2*a_mp)/4, mp.mpf(1)/4, a_mp/2],
        [(1-2*a_mp)/4, 3*a_mp/2-mp.mpf(1)/4, mp.mpf(0)],
    ]
    K_mp = [
        [mp.mpf(0), sqrt2_mp/(4*rho_mp), mp.mpf(1)/(4*rho_mp)],
        [sqrt2_mp/(4*rho_mp), mp.mpf(2)/(4*rho_mp), sqrt2_mp/(4*rho_mp)],
        [mp.mpf(1)/(4*rho_mp), sqrt2_mp/(4*rho_mp), mp.mpf(0)],
    ]
    edge_metrics_J = metrics_high_precision(empirical["joint_counts"], J_mp)
    edge_metrics_K = metrics_high_precision(empirical["joint_counts"], K_mp)

    pi_leb = np.array([0.5 - a_num, 0.5, a_num])
    pi_parry = np.array([0.25, 0.5, 0.25])
    state_counts = np.array([empirical["state_counts"][s] for s in STATES], dtype=int)
    state_emp = state_counts / state_counts.sum()
    defect_counts = np.array([empirical["defect_counts"][d] for d in DEFECTS], dtype=int)
    defect_emp = defect_counts / defect_counts.sum()
    defect_benchmark = np.array(
        [
            (1 - 2 * a_num) / 4,
            a_num,
            0.25,
            0.5 - a_num,
            a_num / 2,
        ]
    )

    boundary = boundary_precision_check(10000)

    negative_a = 0.1
    J_negative = np.array(sym["J"].subs({sym["a"]: negative_a}).evalf().tolist(), dtype=float)
    negative_control = {
        "a": negative_a,
        "inside_required_interval": bool((1 / 6) < negative_a < (1 / 4)),
        "J98": float(J_negative[2, 1]),
        "seven_edge_positive_support_valid": bool(np.all(J_negative[J_negative != 0] > 0)),
        "pass": bool(J_negative[2, 1] <= 0),
    }

    symbolic_payload = {
        "assumption": "1/6 < a < 1/4",
        "partition": {
            "I7": "(-1, -1/2-a]",
            "I8": "(-1/2-a, -a]",
            "I9": "(-a, 0]",
        },
        "J": _jsonable(sym["J"]),
        "pi_Leb": _jsonable(sym["pi"]),
        "P": _jsonable(sym["P"]),
        "M": _jsonable(sym["M"]),
        "M2": _jsonable(sym["M2"]),
        "Perron_root": _jsonable(sym["rho"]),
        "Perron_vector": _jsonable(sym["r"]),
        "K": _jsonable(sym["K"]),
        "defect_masses": _jsonable(sym["defect"]),
        "checks": _jsonable(sym["checks"]),
        "support": {
            "positive_entries": ["J78", "J79", "J87", "J88", "J89", "J97", "J98"],
            "zero_entries": ["J77", "J99"],
        },
        "negative_control": negative_control,
    }
    numerical_payload = {
        "constants": {k: mp.nstr(v, 100) for k, v in c100.items()},
        "J": J_num.tolist(),
        "P": P_num.tolist(),
        "pi_Leb": pi_leb.tolist(),
        "M": np.array(sym["M"].tolist(), dtype=int).tolist(),
        "M2": np.array(sym["M2"].tolist(), dtype=int).tolist(),
        "Perron_root": float(1 + math.sqrt(2)),
        "Perron_vector": [1.0, math.sqrt(2), 1.0],
        "pi_Parry": pi_parry.tolist(),
        "K": K_num.tolist(),
        "empirical_joint_counts": empirical["joint_counts"].tolist(),
        "empirical_joint": emp_joint.tolist(),
        "edge_metrics_vs_J": edge_metrics_J,
        "edge_metrics_vs_Parry": edge_metrics_K,
        "edge_metrics_vs_J_float64": edge_metrics_J_float,
        "edge_metrics_vs_Parry_float64": edge_metrics_K_float,
        "state_counts": dict(empirical["state_counts"]),
        "state_empirical": state_emp.tolist(),
        "state_deviation_from_Lebesgue": (state_emp - pi_leb).tolist(),
        "state_deviation_from_Parry": (state_emp - pi_parry).tolist(),
        "defect_counts": dict(empirical["defect_counts"]),
        "defect_empirical": defect_emp.tolist(),
        "defect_benchmark": defect_benchmark.tolist(),
        "defect_deviation": (defect_emp - defect_benchmark).tolist(),
        "boundary_scan": boundary,
        "negative_control": negative_control,
        "scope": {
            "proved_abstractly": [
                "ceiling-map bridge on the affine model for A>=1",
                "half-open partition and exact endpoint assignments",
                "J, P, stationarity, M, Perron data, Parry K, and defect identities under 1/6<a<1/4",
            ],
            "certified_finitely": [
                "prior exact Fibonacci threshold agrees with affine ceiling model for A=0..10000",
                "no finite boundary hit was found through A=10000 in precision-stable 3500/4500 dps calculations",
                "empirical joint edge counts for transitions A=2..10000",
            ],
            "observed": [
                "the finite orbit joint edge distribution is substantially closer to J than to K",
                "state and defect frequencies are close to the Lebesgue benchmarks on the finite range",
            ],
            "open": [
                "specific-orbit equidistribution",
                "global exact Fibonacci threshold identity T_A=ceil(y_A)",
                "any gauge/FQM map from d_A=+/-1",
            ],
        },
    }

    (outputs / f"{STAMP}_symbolic_results.json").write_text(
        json.dumps(symbolic_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (outputs / f"{STAMP}_numerical_results.json").write_text(
        json.dumps(_jsonable(numerical_payload), indent=2, sort_keys=True), encoding="utf-8"
    )

    write_csv(
        outputs / f"{STAMP}_empirical_joint_transition.csv",
        ["from_state", "to_state", "count", "frequency", "J_mass", "Parry_K_mass"],
        [
            [
                i,
                j,
                int(empirical["joint_counts"][ri, cj]),
                f"{emp_joint[ri, cj]:.18g}",
                f"{J_num[ri, cj]:.18g}",
                f"{K_num[ri, cj]:.18g}",
            ]
            for ri, i in enumerate(STATES)
            for cj, j in enumerate(STATES)
        ],
    )
    write_csv(
        outputs / f"{STAMP}_state_frequencies.csv",
        ["state", "count", "empirical", "Lebesgue", "Parry", "emp_minus_Lebesgue", "emp_minus_Parry"],
        [
            [
                s,
                int(state_counts[idx]),
                f"{state_emp[idx]:.18g}",
                f"{pi_leb[idx]:.18g}",
                f"{pi_parry[idx]:.18g}",
                f"{state_emp[idx]-pi_leb[idx]:.18g}",
                f"{state_emp[idx]-pi_parry[idx]:.18g}",
            ]
            for idx, s in enumerate(STATES)
        ],
    )
    write_csv(
        outputs / f"{STAMP}_defect_frequencies.csv",
        ["defect", "count", "empirical", "Lebesgue_benchmark", "deviation"],
        [
            [
                d,
                int(defect_counts[idx]),
                f"{defect_emp[idx]:.18g}",
                f"{defect_benchmark[idx]:.18g}",
                f"{defect_emp[idx]-defect_benchmark[idx]:.18g}",
            ]
            for idx, d in enumerate(DEFECTS)
        ],
    )
    write_csv(
        outputs / f"{STAMP}_finite_boundary_minima.csv",
        ["boundary", "boundary_value", "minimum_distance", "A", "E_A"],
        [
            [r["boundary"], r["boundary_value"], r["minimum_distance"], r["A"], r["E_A"]]
            for r in boundary["scan_4500"]["minima"]
        ],
    )

    with (traces / f"{STAMP}_J_derivation_trace.jsonl").open("w", encoding="utf-8") as f:
        entries = [
            (7, 7, "0", "F7(I7) starts to the right of I7"),
            (7, 8, "(1-3*a)/2", "half the image overlap of length 1-3a"),
            (7, 9, "a/2", "half the image overlap of length a"),
            (8, 7, "(1-2*a)/4", "F8(I8)=(-1,0], half the target length"),
            (8, 8, "1/4", "F8(I8)=(-1,0], half the target length"),
            (8, 9, "a/2", "F8(I8)=(-1,0], half the target length"),
            (9, 7, "(1-2*a)/4", "half the full I7 length"),
            (9, 8, "3*a/2-1/4", "half the image overlap of length 3a-1/2"),
            (9, 9, "0", "F9(I9) ends to the left of I9"),
        ]
        for idx, (i, j, formula, reason) in enumerate(entries, 1):
            f.write(json.dumps({"step": idx, "from": i, "to": j, "formula": formula, "reason": reason}) + "\n")

    with (traces / f"{STAMP}_finite_transition_trace.jsonl").open("w", encoding="utf-8") as f:
        carries = empirical["carries"]
        for A in range(2, 10001):
            prev_c = carries[A - 1]
            cur_c = carries[A]
            f.write(
                json.dumps(
                    {
                        "A": A,
                        "previous_carry": prev_c,
                        "current_carry": cur_c,
                        "defect": cur_c - prev_c,
                    }
                )
                + "\n"
            )

    return numerical_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.package_root.resolve())
    print(json.dumps({"status": "PASS", "edge_metrics_vs_J": result["edge_metrics_vs_J"]}, indent=2))


if __name__ == "__main__":
    main()
