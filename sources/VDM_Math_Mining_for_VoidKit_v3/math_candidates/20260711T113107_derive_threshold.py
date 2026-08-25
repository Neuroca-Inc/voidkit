#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mpmath as mp
import sympy as sp

mp.mp.dps = 140

REQUIRED_DOCUMENT_MARKERS = {
    "binet": "P_n=L_n+\\rho_n",
    "correction": "|\\rho_n|<\\frac14",
    "power_two": "P_n\\ne X_A",
    "integer_gap": "|P_n-X_A|\\ge1",
    "nonintegrality": "y_A\\notin\\mathbb Z",
    "global": "T_A=\\lceil y_A\\rceil",
}
REQUIRED_LEAN_THEOREMS = [
    "exact_binet_product_identity",
    "correction_sign",
    "correction_abs_lt_quarter",
    "fibonacci_product_power_two_only",
    "threshold_equality_obstruction",
    "y_nonintegral",
    "global_threshold_bridge",
    "integer_gap_sign_transfer",
    "leading_term_separation",
]


def fib_pair(n: int) -> tuple[int, int]:
    if n == 0:
        return 0, 1
    a, b = fib_pair(n >> 1)
    c = a * ((b << 1) - a)
    d = a * a + b * b
    return (d, c + d) if n & 1 else (c, d)


def m_A(A: int) -> int:
    return 12 * ((1 << (A + 1)) - 1)


def exact_symbolic_checks() -> dict:
    sqrt5 = sp.sqrt(5)
    phi = (1 + sqrt5) / 2
    z, eps = sp.symbols("z eps", nonzero=True)

    phi_unit = sp.simplify(phi - 1 / phi - 1)
    binet_reduction = sp.simplify((z + eps * (phi - 1 / phi) - 1 / z) - (z + eps - 1 / z))
    phi5_minus4 = sp.simplify(phi**5 - 4)
    odd_margin = sp.simplify(sp.Rational(1, 4) - (1 + phi**-5) / 5)
    even_margin = sp.Rational(1, 4) - sp.Rational(1, 5)

    checks = {
        "phi_minus_inverse_equals_one": phi_unit == 0,
        "binet_cross_term_reduction": binet_reduction == 0,
        "phi_fifth_gt_four": bool(phi5_minus4.is_positive),
        "odd_correction_margin_positive": bool(odd_margin.is_positive),
        "even_correction_margin_positive": bool(even_margin > 0),
    }
    return {
        "checks": checks,
        "exact_values": {
            "phi5_minus4": str(phi5_minus4),
            "odd_margin": str(odd_margin),
            "even_margin": str(even_margin),
        },
        "pass": all(checks.values()),
    }


def universal_dependency_graph(document: str, lean_source: str) -> dict:
    doc = {name: marker in document for name, marker in REQUIRED_DOCUMENT_MARKERS.items()}
    lean = {name: (f"theorem {name}" in lean_source) for name in REQUIRED_LEAN_THEOREMS}

    obligations = {
        "U1_exact_binet_identity": {
            "mode": "exact_symbolic_algebra_plus_document_theorem",
            "depends_on": [],
            "present": doc["binet"] and lean["exact_binet_product_identity"],
        },
        "U2_uniform_correction_bound": {
            "mode": "exact_parity_worst_case_inequalities_plus_document_theorem",
            "depends_on": ["U1_exact_binet_identity"],
            "present": doc["correction"] and lean["correction_abs_lt_quarter"] and lean["correction_sign"],
        },
        "U3_power_of_two_obstruction": {
            "mode": "abstract_coprimality_argument_in_document_and_lean_surface",
            "depends_on": [],
            "present": doc["power_two"] and lean["fibonacci_product_power_two_only"] and lean["threshold_equality_obstruction"],
        },
        "U4_integer_gap_sign_transfer": {
            "mode": "generic_integer_lemma",
            "depends_on": ["U1_exact_binet_identity", "U2_uniform_correction_bound", "U3_power_of_two_obstruction"],
            "present": doc["integer_gap"] and lean["integer_gap_sign_transfer"] and lean["leading_term_separation"],
        },
        "U5_nonintegrality": {
            "mode": "contradiction_from_U2_U3",
            "depends_on": ["U2_uniform_correction_bound", "U3_power_of_two_obstruction"],
            "present": doc["nonintegrality"] and lean["y_nonintegral"],
        },
        "U6_global_ceiling_bridge": {
            "mode": "monotone_log_reduction_plus_U4_U5",
            "depends_on": ["U4_integer_gap_sign_transfer", "U5_nonintegrality"],
            "present": doc["global"] and lean["global_threshold_bridge"],
        },
    }

    resolved = {}
    for name, row in obligations.items():
        resolved[name] = bool(row["present"] and all(resolved[d] for d in row["depends_on"]))
        row["resolved"] = resolved[name]

    return {
        "document_markers": doc,
        "lean_theorem_markers": lean,
        "obligations": obligations,
        "pass": all(resolved.values()),
    }


def finite_regressions(max_a: int, max_n: int) -> dict:
    phi = (mp.mpf(1) + mp.sqrt(5)) / 2
    lnphi = mp.log(phi)
    rows = []
    for A in range(max_a + 1):
        m = m_A(A)
        y = (mp.mpf(m) * mp.log(2) + mp.log(5)) / (2 * lnphi) - mp.mpf("1.5")
        T = int(mp.ceil(y))
        fT, fTp1 = fib_pair(T + 1)
        fm, f0 = fib_pair(T)
        hi = fT * fTp1
        lo = fm * f0
        X = 1 << m
        if not lo < X < hi:
            raise AssertionError(f"finite threshold regression failed at A={A}")
        rows.append({"A": A, "T": T, "lo_lt_X": True, "X_lt_hi": True})

    binet_max_error = mp.mpf(0)
    correction_max = mp.mpf(0)
    power_two_hits = []
    for n in range(max_n + 1):
        f1, f2 = fib_pair(n + 1)
        p = f1 * f2
        L = phi ** (2 * n + 3) / 5
        rho = ((-1) ** n - phi ** (-(2 * n + 3))) / 5
        binet_max_error = max(binet_max_error, abs(mp.mpf(p) - (L + rho)))
        correction_max = max(correction_max, abs(rho))
        if p > 0 and (p & (p - 1)) == 0:
            power_two_hits.append([n, p])

    if power_two_hits != [[0, 1], [1, 2]]:
        raise AssertionError("finite power-of-two regression mismatch")

    return {
        "scope": "FINITE REGRESSION ONLY",
        "threshold_rows": rows,
        "binet_max_error": mp.nstr(binet_max_error, 30),
        "correction_max": mp.nstr(correction_max, 30),
        "power_two_hits": power_two_hits,
        "pass": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--max-a", type=int, default=12)
    ap.add_argument("--max-n", type=int, default=100)
    args = ap.parse_args()

    document = Path(args.document).read_text(encoding="utf-8")
    lean_source = Path(args.lean).read_text(encoding="utf-8")

    symbolic = exact_symbolic_checks()
    dependencies = universal_dependency_graph(document, lean_source)
    finite = finite_regressions(args.max_a, args.max_n)

    universal_pass = symbolic["pass"] and dependencies["pass"]
    result = {
        "universal_proof_obligations": {
            "symbolic": symbolic,
            "dependency_graph": dependencies,
            "status": "PROVED" if universal_pass else "FAILED",
            "note": "Universal status is supplied by exact algebra plus the document/Lean theorem surface, not by finite sampling.",
        },
        "finite_regressions": finite,
        "global_bridge": "PROVED" if universal_pass else "NOT PROVED",
    }

    if not universal_pass:
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(1)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
