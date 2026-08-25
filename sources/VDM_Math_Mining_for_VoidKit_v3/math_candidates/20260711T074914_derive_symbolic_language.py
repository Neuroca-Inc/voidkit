#!/usr/bin/env python3
"""Derive the affine QBL carry interval language and audit the finite trace.

Closed here:
- affine ceiling/error map;
- exact half-open partition;
- one-step Lebesgue law J and conditional table P;
- pairwise edge-shift envelope M and its Parry edge measure K;
- forbidden word 989 and all length-three cylinders at the current constant;
- exact affine-language word complexity and entropy log(2);
- finite trace validation, empirical edge metrics, and outward interval boundary certificate.

Not closed here:
- a finite-state/sofic presentation of the full affine carry language;
- specific-orbit equidistribution;
- the global exact-Fibonacci threshold bridge;
- any gauge/FQM map from count parity or primality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import mpmath as mp
import numpy as np
import sympy as sp

STAMP = "20260711T074914"
STATES = (7, 8, 9)
STATE_CHARS = ("7", "8", "9")
DEFECTS = (-2, -1, 0, 1, 2)
M_INT = ((0, 1, 1), (1, 1, 1), (1, 1, 0))


def constants(dps: int = 120) -> dict[str, mp.mpf]:
    mp.mp.dps = dps
    phi = (1 + mp.sqrt(5)) / 2
    log_phi = mp.log(phi)
    lam = 6 * mp.log(2) / log_phi
    gamma = lam + mp.mpf("1.5") - mp.log(5) / (2 * log_phi)
    a = (gamma - 8) / 2
    y0 = 2 * lam - gamma
    p = 2 * a
    return {"phi": phi, "lambda": lam, "gamma": gamma, "a": a, "p": p, "y0": y0}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, mp.mpf):
        return mp.nstr(x, 100)
    if isinstance(x, sp.MatrixBase):
        return [[str(sp.simplify(x[i, j])) for j in range(x.cols)] for i in range(x.rows)]
    if isinstance(x, sp.Basic):
        return str(sp.simplify(x))
    if isinstance(x, Fraction):
        return str(x)
    return x


def symbolic_system() -> dict[str, Any]:
    a = sp.symbols("a", real=True)
    J = sp.Matrix([
        [0, (1 - 3 * a) / 2, a / 2],
        [(1 - 2 * a) / 4, sp.Rational(1, 4), a / 2],
        [(1 - 2 * a) / 4, 3 * a / 2 - sp.Rational(1, 4), 0],
    ])
    pi_leb = sp.Matrix([[sp.Rational(1, 2) - a, sp.Rational(1, 2), a]])
    P = sp.Matrix([
        [0, (1 - 3 * a) / (1 - 2 * a), a / (1 - 2 * a)],
        [(1 - 2 * a) / 2, sp.Rational(1, 2), a],
        [(1 - 2 * a) / (4 * a), (6 * a - 1) / (4 * a), 0],
    ])
    M = sp.Matrix(M_INT)
    M2 = M**2
    rho = 1 + sp.sqrt(2)
    r = sp.Matrix([1, sp.sqrt(2), 1])
    K = sp.Matrix([
        [0, sp.sqrt(2), 1],
        [sp.sqrt(2), 2, sp.sqrt(2)],
        [1, sp.sqrt(2), 0],
    ]) / (4 * rho)
    defect = {
        -2: sp.simplify(J[2, 0]),
        -1: sp.simplify(J[1, 0] + J[2, 1]),
        0: sp.simplify(J[0, 0] + J[1, 1] + J[2, 2]),
        1: sp.simplify(J[0, 1] + J[1, 2]),
        2: sp.simplify(J[0, 2]),
    }
    checks = {
        "J_row_sums": [sp.simplify(sum(J[i, j] for j in range(3))) for i in range(3)],
        "J_col_sums": [sp.simplify(sum(J[i, j] for i in range(3))) for j in range(3)],
        "J_total": sp.simplify(sum(J)),
        "P_row_sums": [sp.simplify(sum(P[i, j] for j in range(3))) for i in range(3)],
        "stationarity": [sp.simplify(x) for x in list(pi_leb * P - pi_leb)],
        "M2": M2,
        "perron_residual": [sp.simplify(x) for x in list(M * r - rho * r)],
        "K_row_sums": [sp.simplify(sum(K[i, j] for j in range(3))) for i in range(3)],
        "K_col_sums": [sp.simplify(sum(K[i, j] for i in range(3))) for j in range(3)],
        "K_total": sp.simplify(sum(K)),
    }
    expected = {
        "J_row_sums": [sp.Rational(1, 2) - a, sp.Rational(1, 2), a],
        "J_col_sums": [sp.Rational(1, 2) - a, sp.Rational(1, 2), a],
        "J_total": 1,
        "P_row_sums": [1, 1, 1],
        "stationarity": [0, 0, 0],
        "M2": sp.Matrix([[2, 2, 1], [2, 3, 2], [1, 2, 2]]),
        "perron_residual": [0, 0, 0],
        "K_row_sums": [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)],
        "K_col_sums": [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)],
        "K_total": 1,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise AssertionError(f"symbolic check failed: {key}: {checks[key]} != {value}")
    positivity_proofs = {
        "J78": "(1-3a)/2 > (1-3/4)/2 > 0 from a<1/4",
        "J79": "a/2 > 0 from a>1/6",
        "J87": "(1-2a)/4 > (1-1/2)/4 > 0 from a<1/4",
        "J88": "1/4 > 0",
        "J89": "a/2 > 0 from a>1/6",
        "J97": "(1-2a)/4 > 0 from a<1/4",
        "J98": "3a/2-1/4 > 0 from a>1/6",
    }
    return {"a": a, "J": J, "pi_leb": pi_leb, "P": P, "M": M, "M2": M2,
            "rho": rho, "r": r, "K": K, "defect": defect, "checks": checks,
            "positivity_proofs": positivity_proofs}


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted((root / "inputs").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {pattern}, found {len(matches)}")
    return matches[0]


def load_trace(root: Path) -> list[dict[str, str]]:
    path = find_one(root, "*_PRIOR_CARRY_DEFECT_A0_A10000.csv")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 10001:
        raise RuntimeError(f"expected 10001 rows, found {len(rows)}")
    seen: set[int] = set()
    for row in rows:
        A = int(row["A"])
        if A in seen:
            raise RuntimeError(f"duplicate A={A}")
        seen.add(A)
    if seen != set(range(10001)):
        missing = sorted(set(range(10001)) - seen)[:10]
        extra = sorted(seen - set(range(10001)))[:10]
        raise RuntimeError(f"trace A coverage invalid; missing={missing}, extra={extra}")
    by_A = {int(r["A"]): r for r in rows}
    rows = [by_A[A] for A in range(10001)]
    if rows[0]["carry"].strip() != "":
        raise RuntimeError("A=0 carry must be blank")
    for A in range(1, 10001):
        c = int(rows[A]["carry"])
        if c not in STATES:
            raise RuntimeError(f"invalid carry at A={A}: {c}")
    return rows


def load_prior_transition_counts(root: Path) -> np.ndarray:
    path = find_one(root, "*_PRIOR_TRANSITION_COUNTS.csv")
    counts = {(int(i), int(j)): None for i in STATES for j in STATES}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 9:
        raise RuntimeError(f"prior transition count table must have 9 rows, found {len(rows)}")
    for row in rows:
        key = (int(row["from_state"]), int(row["to_state"]))
        if key not in counts or counts[key] is not None:
            raise RuntimeError(f"bad/duplicate prior transition key {key}")
        counts[key] = int(row["count"])
    if any(v is None for v in counts.values()):
        raise RuntimeError("prior transition table incomplete")
    return np.array([[counts[(i, j)] for j in STATES] for i in STATES], dtype=np.int64)


def empirical(rows: list[dict[str, str]], prior_counts: np.ndarray) -> dict[str, Any]:
    carries = {A: int(rows[A]["carry"]) for A in range(1, 10001)}
    transition_counter = Counter((carries[A - 1], carries[A]) for A in range(2, 10001))
    defect_counter = Counter(carries[A] - carries[A - 1] for A in range(2, 10001))
    state_counter = Counter(carries.values())
    counts = np.array([[transition_counter[(i, j)] for j in STATES] for i in STATES], dtype=np.int64)
    if int(counts.sum()) != 9999:
        raise RuntimeError(f"expected 9999 transitions, found {counts.sum()}")
    if not np.array_equal(counts, prior_counts):
        raise RuntimeError(f"derived transitions disagree with PRIOR_TRANSITION_COUNTS.csv\nderived={counts}\nprior={prior_counts}")
    defects = np.array([defect_counter[d] for d in DEFECTS], dtype=np.int64)
    states = np.array([state_counter[s] for s in STATES], dtype=np.int64)
    return {
        "carries": carries,
        "transition_counts": counts,
        "joint": counts / counts.sum(),
        "defect_counts": defects,
        "state_counts": states,
        "total_transitions": int(counts.sum()),
    }


@dataclass(frozen=True)
class Affine:
    qa: Fraction
    c: Fraction

    def __add__(self, other: "Affine") -> "Affine":
        return Affine(self.qa + other.qa, self.c + other.c)

    def __sub__(self, other: "Affine") -> "Affine":
        return Affine(self.qa - other.qa, self.c - other.c)

    def scale(self, k: Fraction | int) -> "Affine":
        return Affine(self.qa * k, self.c * k)

    def divide(self, k: Fraction | int) -> "Affine":
        return Affine(self.qa / k, self.c / k)

    def eval(self, aval: mp.mpf) -> mp.mpf:
        return mp.mpf(self.qa.numerator) / self.qa.denominator * aval + mp.mpf(self.c.numerator) / self.c.denominator

    def plain(self) -> str:
        terms: list[str] = []
        if self.qa:
            q = self.qa
            if q == 1:
                terms.append("a")
            elif q == -1:
                terms.append("-a")
            else:
                terms.append(f"{q}*a")
        if self.c:
            c = self.c
            if terms and c > 0:
                terms.append(f"+ {c}")
            else:
                terms.append(str(c))
        return " ".join(terms) if terms else "0"

    def latex(self) -> str:
        def frac(q: Fraction) -> str:
            if q.denominator == 1:
                return str(q.numerator)
            return rf"\frac{{{q.numerator}}}{{{q.denominator}}}"
        pieces: list[str] = []
        if self.qa:
            q = self.qa
            if q == 1:
                pieces.append("a")
            elif q == -1:
                pieces.append("-a")
            else:
                pieces.append(frac(q) + "a")
        if self.c:
            cs = frac(abs(self.c))
            if pieces:
                pieces.append(("+" if self.c > 0 else "-") + cs)
            else:
                pieces.append(frac(self.c))
        return "".join(pieces) if pieces else "0"


PARTITIONS_AFF = {
    "7": (Affine(Fraction(0), Fraction(-1)), Affine(Fraction(-1), Fraction(-1, 2))),
    "8": (Affine(Fraction(-1), Fraction(-1, 2)), Affine(Fraction(-1), Fraction(0))),
    "9": (Affine(Fraction(-1), Fraction(0)), Affine(Fraction(0), Fraction(0))),
}


def compare_affine(x: Affine, y: Affine, a_lo: mp.mpf, a_hi: mp.mpf) -> int:
    d = x - y
    q = mp.mpf(d.qa.numerator) / d.qa.denominator
    c = mp.mpf(d.c.numerator) / d.c.denominator
    vals = [q * a_lo + c, q * a_hi + c]
    low, high = min(vals), max(vals)
    if high < 0:
        return -1
    if low > 0:
        return 1
    if low == 0 and high == 0:
        return 0
    raise RuntimeError(f"affine order not certified for {x.plain()} vs {y.plain()}: [{low},{high}]")


def max_affine(x: Affine, y: Affine, a_lo: mp.mpf, a_hi: mp.mpf) -> Affine:
    return y if compare_affine(x, y, a_lo, a_hi) < 0 else x


def min_affine(x: Affine, y: Affine, a_lo: mp.mpf, a_hi: mp.mpf) -> Affine:
    return x if compare_affine(x, y, a_lo, a_hi) < 0 else y


def cylinder_initial(word: str, a_lo: mp.mpf, a_hi: mp.mpf) -> tuple[Affine, Affine, bool]:
    lo, hi = PARTITIONS_AFF[word[0]]
    slope = Fraction(1)
    off = Affine(Fraction(0), Fraction(0))
    for idx in range(1, len(word)):
        c = int(word[idx - 1])
        slope *= 2
        off = off.scale(2) + Affine(Fraction(2), Fraction(8 - c))
        target_lo, target_hi = PARTITIONS_AFF[word[idx]]
        pre_lo = (target_lo - off).divide(slope)
        pre_hi = (target_hi - off).divide(slope)
        lo = max_affine(lo, pre_lo, a_lo, a_hi)
        hi = min_affine(hi, pre_hi, a_lo, a_hi)
    cmp = compare_affine(lo, hi, a_lo, a_hi)
    return lo, hi, cmp < 0


def edge_envelope_word(word: str) -> bool:
    idx = {"7": 0, "8": 1, "9": 2}
    return all(M_INT[idx[x]][idx[y]] == 1 for x, y in zip(word, word[1:]))


def length3_table(a_lo: mp.mpf, a_hi: mp.mpf) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in STATE_CHARS:
        for j in STATE_CHARS:
            for k in STATE_CHARS:
                word = i + j + k
                lo, hi, nonempty = cylinder_initial(word, a_lo, a_hi)
                envelope = edge_envelope_word(word)
                if nonempty:
                    reason = "nonempty exact affine cylinder at the current parameter enclosure"
                    interval = f"({lo.plain()}, {hi.plain()}]"
                    latex = rf"({lo.latex()},{hi.latex()}]"
                else:
                    interval = ""
                    latex = ""
                    if "77" in word:
                        reason = "empty because 7->7 has zero one-step mass"
                    elif "99" in word:
                        reason = "empty because 9->9 has zero one-step mass"
                    elif word == "989":
                        reason = "empty: after 9->8, F8 image is (-1,-2+6a], which lies left of I9 for a<1/4"
                    elif word == "787":
                        reason = "empty at the current constant: after 7->8, F8 image starts at -2+6a; a>3/14 makes it miss I7"
                    else:
                        reason = "empty exact affine cylinder"
                rows.append({
                    "word": word,
                    "pairwise_edge_envelope": envelope,
                    "realizable_current_parameter": nonempty,
                    "cylinder_initial_E": interval,
                    "cylinder_latex": latex,
                    "reason": reason,
                })
    actual = [r["word"] for r in rows if r["realizable_current_parameter"]]
    envelope = [r["word"] for r in rows if r["pairwise_edge_envelope"]]
    if len(actual) != 15 or len(envelope) != 17:
        raise AssertionError(f"length-3 count mismatch actual={len(actual)} envelope={len(envelope)}")
    missing = sorted(set(envelope) - set(actual))
    if missing != ["787", "989"]:
        raise AssertionError(f"unexpected pairwise-envelope excess: {missing}")
    return rows


def direct_current_cylinders(max_depth: int, gamma: Decimal) -> list[dict[str, tuple[Decimal, Decimal]]]:
    a = (gamma - Decimal(8)) / Decimal(2)
    parts = {
        "7": (Decimal(-1), -Decimal("0.5") - a),
        "8": (-Decimal("0.5") - a, -a),
        "9": (-a, Decimal(0)),
    }
    def intersect(x: tuple[Decimal, Decimal], y: tuple[Decimal, Decimal]):
        lo, hi = max(x[0], y[0]), min(x[1], y[1])
        return (lo, hi) if hi > lo else None
    def image(iv: tuple[Decimal, Decimal], carry: str):
        off = gamma - Decimal(int(carry))
        return (2 * iv[0] + off, 2 * iv[1] + off)
    levels = [{s: parts[s] for s in STATE_CHARS}]
    for _ in range(2, max_depth + 1):
        cur: dict[str, tuple[Decimal, Decimal]] = {}
        for word, iv in levels[-1].items():
            img = image(iv, word[-1])
            for s in STATE_CHARS:
                clipped = intersect(img, parts[s])
                if clipped is not None:
                    cur[word + s] = clipped
        levels.append(cur)
    return levels


def edge_path_count(n: int) -> int:
    vec = [1, 1, 1]
    for _ in range(1, n):
        vec = [sum(vec[i] * M_INT[i][j] for i in range(3)) for j in range(3)]
    return sum(vec)


def markov_order_counterexamples(levels: list[dict[str, tuple[Decimal, Decimal]]], max_k: int = 10) -> list[dict[str, Any]]:
    results = []
    max_n = len(levels) - 1
    for k in range(1, max_k + 1):
        found = None
        for n in range(k, max_n + 1):
            current = levels[n - 1]
            next_level = levels[n]
            groups: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
            for w in current:
                ext = tuple(s for s in STATE_CHARS if w + s in next_level)
                groups[w[-k:]].append((w, ext))
            for suffix, values in groups.items():
                by_ext: dict[tuple[str, ...], str] = {}
                for w, ext in values:
                    by_ext.setdefault(ext, w)
                if len(by_ext) > 1:
                    pairs = list(by_ext.items())[:2]
                    found = {
                        "tested_markov_order": k,
                        "witness_length": n,
                        "shared_suffix": suffix,
                        "word_1": pairs[0][1],
                        "extensions_1": "".join(pairs[0][0]),
                        "word_2": pairs[1][1],
                        "extensions_2": "".join(pairs[1][0]),
                    }
                    break
            if found:
                break
        if not found:
            raise AssertionError(f"no counterexample found for Markov order {k} within direct depth")
        results.append(found)
    return results


def complexity(max_n: int, direct_depth: int, direct_levels: list[dict[str, tuple[Decimal, Decimal]]]) -> list[dict[str, Any]]:
    rows = []
    for n in range(1, max_n + 1):
        exact = 2 ** (n + 1) - 1
        direct = len(direct_levels[n - 1]) if n <= direct_depth else None
        if direct is not None and direct != exact:
            raise AssertionError(f"direct cylinder count mismatch at n={n}: {direct} != {exact}")
        envelope = edge_path_count(n)
        rows.append({
            "length": n,
            "actual_affine_cylinders_exact": exact,
            "direct_interval_enumeration": "" if direct is None else direct,
            "pairwise_edge_envelope_paths": envelope,
            "envelope_excess": envelope - exact,
            "exact_method": "boundary-preimage theorem",
        })
    return rows


def boundary_interval_certificate(rows: list[dict[str, str]], dps: int = 3300) -> dict[str, Any]:
    """Outward interval check of the affine orbit against all partition boundaries.

    mpmath.iv performs interval operations with outward rounding. The 3300-digit
    input enclosure is propagated with the imported finite carry word. Widths
    grow by roughly 2^10000 but remain far below the observed boundary margin.
    """
    mp.mp.dps = dps
    mp.iv.dps = dps
    phi = (mp.iv.mpf(1) + mp.iv.sqrt(5)) / 2
    log_phi = mp.iv.log(phi)
    lam = 6 * mp.iv.log(2) / log_phi
    gamma = lam + mp.iv.mpf("1.5") - mp.iv.log(5) / (2 * log_phi)
    a = (gamma - 8) / 2
    y0 = 2 * lam - gamma

    def lo(x):
        return mp.mpf(x.a)
    def hi(x):
        return mp.mpf(x.b)
    def interval_payload(x):
        return {"lower": mp.nstr(lo(x), 120), "upper": mp.nstr(hi(x), 120),
                "width": mp.nstr(hi(x) - lo(x), 120)}

    if not (lo(y0) > 8 and hi(y0) < 9):
        raise RuntimeError("y0 interval does not certify ceil(y0)=9")
    E = y0 - 9
    b7 = -mp.iv.mpf("0.5") - a
    b8 = -a
    boundary_items = [("-1", mp.iv.mpf(-1)), ("b7", b7), ("b8", b8), ("0", mp.iv.mpf(0))]
    min_margin = mp.inf
    min_record = None
    max_width = mp.mpf(0)
    carry_certified = 0
    for A in range(0, 10001):
        width = hi(E) - lo(E)
        if width > max_width:
            max_width = width
        for name, b in boundary_items:
            if hi(E) < lo(b):
                dist = lo(b) - hi(E)
            elif lo(E) > hi(b):
                dist = lo(E) - hi(b)
            else:
                raise RuntimeError(f"outward interval overlaps boundary {name} at A={A}")
            if dist < min_margin:
                min_margin = dist
                min_record = {
                    "A": A,
                    "boundary": name,
                    "distance_lower_bound": mp.nstr(dist, 120),
                    "E_interval": interval_payload(E),
                    "boundary_interval": interval_payload(b),
                }
        if A == 10000:
            break
        c = int(rows[A + 1]["carry"])
        if c == 7:
            partition_ok = lo(E) > -1 and hi(E) <= lo(b7)
        elif c == 8:
            partition_ok = lo(E) > hi(b7) and hi(E) <= lo(b8)
        elif c == 9:
            partition_ok = lo(E) > hi(b8) and hi(E) <= 0
        else:
            raise RuntimeError(f"invalid carry {c}")
        if not partition_ok:
            raise RuntimeError(f"partition membership not certified at source A={A}, carry c_{A+1}={c}")
        z = 2 * E + gamma
        if not (lo(z) > c - 1 and hi(z) <= c):
            raise RuntimeError(f"ceiling assignment not certified at step A={A+1}")
        E = z - c
        carry_certified += 1
    return {
        "method": "mpmath.iv outward-rounded interval arithmetic",
        "mpmath_version": mp.__version__,
        "decimal_digits": dps,
        "gamma_interval": interval_payload(gamma),
        "a_interval": interval_payload(a),
        "y0_interval": interval_payload(y0),
        "E0_interval": interval_payload(y0 - 9),
        "all_imported_carries_certified": carry_certified == 10000,
        "certified_carry_steps": carry_certified,
        "all_E_A_boundary_disjoint_for_A_0_10000": True,
        "minimum_boundary_distance_lower_bound": min_record,
        "maximum_orbit_interval_width": mp.nstr(max_width, 120),
        "final_E10000_interval": interval_payload(E),
        "scope": "finite affine orbit A=0..10000 only; global nonintegrality is not proved",
    }


def metrics_high_precision(counts: np.ndarray, target: list[list[mp.mpf]]) -> dict[str, str]:
    mp.mp.dps = 120
    total = mp.mpf(int(counts.sum()))
    diffs = []
    for i in range(3):
        for j in range(3):
            empirical_value = mp.mpf(int(counts[i, j])) / total
            diffs.append(abs(empirical_value - target[i][j]))
    l1 = sum(diffs)
    return {
        "max_absolute_error": mp.nstr(max(diffs), 80),
        "L1_error": mp.nstr(l1, 80),
        "total_variation": mp.nstr(l1 / 2, 80),
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path) -> dict[str, Any]:
    outputs = root / "outputs"
    trace_dir = root / "trace"
    outputs.mkdir(exist_ok=True)
    trace_dir.mkdir(exist_ok=True)

    sym = symbolic_system()
    c = constants(160)
    if not (mp.mpf(1) / 6 < c["a"] < mp.mpf(1) / 4):
        raise RuntimeError("current a is outside the assumed interval")
    if not (c["a"] > mp.mpf(3) / 14):
        raise RuntimeError("current a enclosure does not establish a>3/14 for forbidden 787")

    rows = load_trace(root)
    prior_counts = load_prior_transition_counts(root)
    emp = empirical(rows, prior_counts)

    # Certified parameter enclosure reused for exact affine ordering in length-3 cylinders.
    mp.mp.dps = 220
    aval = constants(220)["a"]
    eps = mp.mpf("1e-200")
    a_lo, a_hi = aval - eps, aval + eps
    l3 = length3_table(a_lo, a_hi)

    getcontext().prec = 100
    gamma_dec = Decimal(mp.nstr(c["gamma"], 95))
    direct_depth = 12
    levels = direct_current_cylinders(direct_depth + 1, gamma_dec)
    complexity_rows = complexity(20, direct_depth, levels)
    markov_witnesses = markov_order_counterexamples(levels, max_k=10)

    boundary = boundary_interval_certificate(rows, dps=3300)

    a_mp = c["a"]
    sqrt2 = mp.sqrt(2)
    rho = 1 + sqrt2
    J_mp = [
        [mp.mpf(0), (1 - 3 * a_mp) / 2, a_mp / 2],
        [(1 - 2 * a_mp) / 4, mp.mpf(1) / 4, a_mp / 2],
        [(1 - 2 * a_mp) / 4, 3 * a_mp / 2 - mp.mpf(1) / 4, mp.mpf(0)],
    ]
    K_mp = [
        [mp.mpf(0), sqrt2 / (4 * rho), mp.mpf(1) / (4 * rho)],
        [sqrt2 / (4 * rho), mp.mpf(2) / (4 * rho), sqrt2 / (4 * rho)],
        [mp.mpf(1) / (4 * rho), sqrt2 / (4 * rho), mp.mpf(0)],
    ]
    metrics_J = metrics_high_precision(emp["transition_counts"], J_mp)
    metrics_K = metrics_high_precision(emp["transition_counts"], K_mp)

    joint = emp["joint"]
    pi_leb = np.array([0.5 - float(a_mp), 0.5, float(a_mp)])
    pi_parry = np.array([0.25, 0.5, 0.25])
    state_freq = emp["state_counts"] / emp["state_counts"].sum()
    defect_freq = emp["defect_counts"] / emp["defect_counts"].sum()
    defect_bench = np.array([(1 - 2 * float(a_mp)) / 4, float(a_mp), 0.25,
                             0.5 - float(a_mp), float(a_mp) / 2])

    symbolic_payload = {
        "assumption": "1/6 < a < 1/4",
        "J": sym["J"], "P": sym["P"], "pi_Leb": sym["pi_leb"],
        "defect_masses": sym["defect"], "symbolic_checks": sym["checks"],
        "positivity_proofs": sym["positivity_proofs"],
        "pairwise_edge_envelope_M": sym["M"], "M2": sym["M2"],
        "edge_envelope_Perron_root": sym["rho"], "edge_envelope_Perron_vector": sym["r"],
        "edge_envelope_Parry_joint_K": sym["K"],
        "forbidden_989": {
            "prefix_98_current_interval": "(-1/2-a,-1+2a]",
            "image_under_F8": "(-1,-2+6a]",
            "comparison": "a<1/4 implies -2+6a<-a",
            "conclusion": "intersection with I9=(-a,0] is empty",
        },
        "forbidden_787_current_parameter": {
            "prefix_78_current_interval": "(-1+2a,-a]",
            "image_under_F8": "(-2+6a,0]",
            "condition_for_I7_intersection": "a<3/14",
            "current_parameter": "a>3/14, so 787 is empty",
        },
        "word_complexity_theorem": {
            "conjugacy": "z=E+1; y=z+2a mod 1; y_next=2y mod 1",
            "cut_point": "p=2a",
            "internal_partition_boundaries": "D^{-1}(p)={a,a+1/2}",
            "irrationality": "p is irrational because rational p would make log(4096/5)/log(phi) rational, forcing a rational power identity between 4096/5 and phi",
            "complexity": "p(n)=1+sum_{m=1}^n 2^m=2^(n+1)-1",
            "entropy": "log(2)",
        },
        "status_holds": {
            "actual_carry_language_presentation": "NOT YET DERIVED",
            "specific_orbit_equidistribution": "NOT PROVED",
            "global_threshold_bridge": "NOT YET PROVED",
            "gauge_fqm_map": "NOT YET DERIVED",
        },
    }
    (outputs / f"{STAMP}_symbolic_results.json").write_text(json.dumps(_jsonable(symbolic_payload), indent=2) + "\n", encoding="utf-8")

    write_csv(outputs / f"{STAMP}_realizable_length3_words.csv",
              ["word", "pairwise_edge_envelope", "realizable_current_parameter", "cylinder_initial_E", "cylinder_latex", "reason"], l3)
    write_csv(outputs / f"{STAMP}_word_complexity.csv",
              ["length", "actual_affine_cylinders_exact", "direct_interval_enumeration", "pairwise_edge_envelope_paths", "envelope_excess", "exact_method"], complexity_rows)
    write_csv(outputs / f"{STAMP}_empirical_joint_transition.csv",
              ["from_state", "to_state", "count", "frequency"],
              ({"from_state": i, "to_state": j, "count": int(emp["transition_counts"][ii, jj]),
                "frequency": f"{joint[ii,jj]:.17g}"}
               for ii, i in enumerate(STATES) for jj, j in enumerate(STATES)))
    write_csv(outputs / f"{STAMP}_state_frequencies.csv",
              ["state", "count", "empirical", "Lebesgue", "Parry_envelope", "emp_minus_Lebesgue", "emp_minus_Parry"],
              ({"state": s, "count": int(emp["state_counts"][i]), "empirical": f"{state_freq[i]:.17g}",
                "Lebesgue": f"{pi_leb[i]:.17g}", "Parry_envelope": f"{pi_parry[i]:.17g}",
                "emp_minus_Lebesgue": f"{state_freq[i]-pi_leb[i]:.17g}",
                "emp_minus_Parry": f"{state_freq[i]-pi_parry[i]:.17g}"}
               for i, s in enumerate(STATES)))
    write_csv(outputs / f"{STAMP}_defect_frequencies.csv",
              ["defect", "count", "empirical", "Lebesgue_benchmark", "deviation"],
              ({"defect": d, "count": int(emp["defect_counts"][i]), "empirical": f"{defect_freq[i]:.17g}",
                "Lebesgue_benchmark": f"{defect_bench[i]:.17g}", "deviation": f"{defect_freq[i]-defect_bench[i]:.17g}"}
               for i, d in enumerate(DEFECTS)))
    write_csv(outputs / f"{STAMP}_markov_order_counterexamples.csv",
              ["tested_markov_order", "witness_length", "shared_suffix", "word_1", "extensions_1", "word_2", "extensions_2"], markov_witnesses)

    comparison = {
        "labels": {
            "M": "pairwise-support edge-shift envelope matrix",
            "K": "Parry joint edge measure of the pairwise edge-shift envelope",
            "J": "one-step Lebesgue joint transition law of the affine interval coding",
        },
        "empirical_counts": emp["transition_counts"].tolist(),
        "empirical_joint": joint.tolist(),
        "J": [[float(x) for x in row] for row in J_mp],
        "K_edge_envelope": [[float(x) for x in row] for row in K_mp],
        "metrics_empirical_vs_J": metrics_J,
        "metrics_empirical_vs_K_edge_envelope": metrics_K,
        "edge_envelope_entropy": mp.nstr(mp.log(1 + mp.sqrt(2)), 80),
        "actual_affine_coding_entropy": mp.nstr(mp.log(2), 80),
        "actual_mixing_status": "NOT YET DERIVED",
    }
    (outputs / f"{STAMP}_edge_envelope_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (outputs / f"{STAMP}_finite_boundary_certificate.json").write_text(json.dumps(boundary, indent=2) + "\n", encoding="utf-8")

    numerical = {
        "constants": {k: mp.nstr(v, 100) for k, v in c.items()},
        "state_counts": emp["state_counts"].tolist(), "state_frequencies": state_freq.tolist(),
        "defect_counts": emp["defect_counts"].tolist(), "defect_frequencies": defect_freq.tolist(),
        "transition_counts": emp["transition_counts"].tolist(), "empirical_joint": joint.tolist(),
        "metrics_J": metrics_J, "metrics_K_edge_envelope": metrics_K,
        "length3_actual_count": sum(bool(r["realizable_current_parameter"]) for r in l3),
        "length3_edge_envelope_count": sum(bool(r["pairwise_edge_envelope"]) for r in l3),
        "word_complexity_1_20": complexity_rows,
        "finite_boundary_certificate": boundary,
        "finite_threshold_status": "IMPORTED PRIOR FINITE CERTIFICATE; this package validates its row/carry/transition integrity but does not rerun the Fibonacci-threshold verifier",
    }
    (outputs / f"{STAMP}_numerical_results.json").write_text(json.dumps(numerical, indent=2) + "\n", encoding="utf-8")

    # Traces: exact level summaries and all imported finite transitions.
    with (trace_dir / f"{STAMP}_symbolic_cylinder_trace.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "parameter", "a": mp.nstr(c["a"], 100), "gamma": mp.nstr(c["gamma"], 100),
                            "a_gt_3_over_14": True, "a_between_1_over_6_and_1_over_4": True}) + "\n")
        for row in l3:
            f.write(json.dumps({"event": "length3_cylinder", **row}) + "\n")
        for row in complexity_rows:
            f.write(json.dumps({"event": "complexity_level", **row}) + "\n")
        for row in markov_witnesses:
            f.write(json.dumps({"event": "finite_markov_order_counterexample", **row}) + "\n")
        f.write(json.dumps({"event": "entropy_conclusion", "edge_shift_envelope": "log(1+sqrt(2))",
                            "actual_affine_coding": "log(2)",
                            "finite_state_presentation": "NOT YET DERIVED"}) + "\n")
    with (trace_dir / f"{STAMP}_finite_transition_trace.jsonl").open("w", encoding="utf-8") as f:
        carries = emp["carries"]
        for A in range(2, 10001):
            f.write(json.dumps({"A": A, "from_carry": carries[A - 1], "to_carry": carries[A],
                                "defect": carries[A] - carries[A - 1]}) + "\n")

    summary = {
        "status": "PASS",
        "abstract": {
            "J_P_preserved": True,
            "pairwise_graph_incomplete": True,
            "forbidden_989": True,
            "actual_length3_words": 15,
            "edge_envelope_length3_paths": 17,
            "actual_affine_word_complexity": "2^(n+1)-1",
            "actual_affine_entropy": "log(2)",
            "edge_envelope_entropy": "log(1+sqrt(2))",
            "finite_state_presentation": "NOT YET DERIVED",
        },
        "finite": {
            "trace_rows_valid": True,
            "transitions": 9999,
            "prior_transition_counts_match": True,
            "boundary_certificate": boundary,
            "metrics_J": metrics_J,
            "metrics_K_edge_envelope": metrics_K,
            "no_markov_order_le_10": markov_witnesses,
        },
        "holds": symbolic_payload["status_holds"],
    }
    (outputs / f"{STAMP}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.package_root.resolve()
    result = run(root)
    print("PASS")
    print(json.dumps(result["abstract"], indent=2))


if __name__ == "__main__":
    main()
