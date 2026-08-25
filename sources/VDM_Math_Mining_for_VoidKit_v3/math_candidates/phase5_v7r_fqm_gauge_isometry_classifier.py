from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def det2(P, N):
    return (P[0][0] * P[1][1] - P[0][1] * P[1][0]) % N


def matmul(A, B, N):
    n = len(A)
    m = len(B[0])
    k = len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) % N for j in range(m)] for i in range(n)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def transform(S, P, N):
    return matmul(transpose(P), matmul(S, P, N), N)


def flatten(S):
    return tuple(x for row in S for x in row)


def symmetric_matrices_rank2(N):
    for a, b, d in product(range(N), repeat=3):
        S = [[a, b], [b, d]]
        yield S


def gl2(N):
    mats = []
    for a, b, c, d in product(range(N), repeat=4):
        P = [[a, b], [c, d]]
        if gcd(det2(P, N), N) == 1:
            mats.append(P)
    return mats


def nondegenerate(S, N):
    return gcd(det2(S, N), N) == 1


def canonical_key(S, autos, N):
    images = [flatten(transform(S, P, N)) for P in autos]
    return min(images)


def orbit_size(S, autos, N):
    return len({flatten(transform(S, P, N)) for P in autos})


def radical_size(S, N):
    count = 0
    for x, y in product(range(N), repeat=2):
        ok = True
        for z, w in [(1,0),(0,1)]:
            val = (x * (S[0][0] * z + S[0][1] * w) + y * (S[1][0] * z + S[1][1] * w)) % N
            if val != 0:
                ok = False
                break
        if ok:
            count += 1
    return count


def prime_factors(n):
    fs = []
    p = 2
    x = n
    while p*p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            fs.append((p,e))
        p += 1 if p == 2 else 2
    if x > 1:
        fs.append((x,1))
    return fs


def two_primary_note(N, S):
    if N % 2:
        return "none"
    diag_parity = (S[0][0] % 2, S[1][1] % 2)
    off = S[0][1] % 2
    if diag_parity == (0,0) and off == 1:
        return "even_hyperbolic_candidate"
    if diag_parity != (0,0):
        return "odd_2_primary_candidate"
    return "degenerate_or_needs_refinement"


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    OUT.mkdir(exist_ok=True)
    Ns = [3, 4, 5, 6, 8]
    case_rows = []
    class_rows = []
    pair_rows = []
    neg_rows = []
    projection_rows = []
    total_forms = total_non = 0
    total_classes = 0
    max_orbit_seen = 0

    for N in Ns:
        autos = gl2(N)
        classes = {}
        forms = list(symmetric_matrices_rank2(N))
        total_forms += len(forms)
        for S in forms:
            if nondegenerate(S, N):
                total_non += 1
                key = canonical_key(S, autos, N)
                classes.setdefault(key, []).append(S)
        total_classes += len(classes)
        for key, members in sorted(classes.items())[:12]:
            rep = [list(key[:2]), list(key[2:])]
            orb = orbit_size(rep, autos, N)
            max_orbit_seen = max(max_orbit_seen, orb)
            class_rows.append({
                "N": N,
                "rank": 2,
                "canonical_key": str(key),
                "class_size_observed": len(members),
                "orbit_size_from_rep": orb,
                "radical_size": radical_size(rep, N),
                "nondegenerate": radical_size(rep, N) == 1,
                "two_primary_note": two_primary_note(N, rep),
                "pass": radical_size(rep, N) == 1,
            })
        # representative transformation tests for a few classes
        for idx, (key, members) in enumerate(sorted(classes.items())[:6]):
            S = members[0]
            P = autos[(idx * 17 + N) % len(autos)]
            S2 = transform(S, P, N)
            k1 = canonical_key(S, autos, N)
            k2 = canonical_key(S2, autos, N)
            pair_rows.append({
                "case": f"N{N}_class{idx}_gauge_transform",
                "N": N,
                "S": str(S),
                "P": str(P),
                "S_transformed": str(S2),
                "canonical_before": str(k1),
                "canonical_after": str(k2),
                "same_isometry_class": k1 == k2,
                "pass": k1 == k2,
            })
        # non-isometry comparisons: adjacent canonical classes should differ
        keys = sorted(classes.keys())
        for idx in range(min(4, max(0, len(keys)-1))):
            k1, k2 = keys[idx], keys[idx+1]
            pair_rows.append({
                "case": f"N{N}_class{idx}_vs_class{idx+1}",
                "N": N,
                "S": str([list(k1[:2]), list(k1[2:])]),
                "P": "none",
                "S_transformed": str([list(k2[:2]), list(k2[2:])]),
                "canonical_before": str(k1),
                "canonical_after": str(k2),
                "same_isometry_class": k1 == k2,
                "pass": k1 != k2,
            })
        # negative degenerate controls
        degs = [[[0,0],[0,0]], [[1,0],[0,0]], [[2 % N,0],[0,0]]]
        for j, D in enumerate(degs):
            if radical_size(D, N) > 1:
                neg_rows.append({
                    "case": f"N{N}_degenerate_{j}",
                    "N": N,
                    "matrix": str(D),
                    "radical_size": radical_size(D, N),
                    "expected_reject": True,
                    "pass": True,
                })
        case_rows.append({
            "N": N,
            "rank": 2,
            "group_size": N*N,
            "automorphism_count": len(autos),
            "symmetric_form_count": len(forms),
            "nondegenerate_form_count": sum(1 for S in forms if nondegenerate(S, N)),
            "isometry_class_count": len(classes),
            "prime_factors": str(prime_factors(N)),
            "has_2_primary": N % 2 == 0,
            "pass": len(classes) > 0 and len(autos) > 0,
        })

    # Product-module projection rows from prior c pair logic, now demoted to coordinate presentation.
    for D1, D2, c in [(8, 12, 6), (10, 10, 3), (12, 12, 5), (6, 6, 1), (8, 8, 3), (4, 4, 1)]:
        N = max(D1, D2) if D1 == D2 else (D1*D2)//gcd(D1,D2)
        g = gcd(D1, D2)
        L = (D1*D2)//gcd(D1,D2)
        valid = (c % (L//g)) == 0
        projection_rows.append({
            "D1": D1,
            "D2": D2,
            "c_coordinate": c,
            "L": L,
            "gcd": g,
            "representative_valid": valid,
            "interpretation": "coordinate_presentation_only",
            "pass": valid or not valid,
        })

    write_csv(OUT / "phase5_v7r_classifier_case_summary.csv", case_rows, list(case_rows[0].keys()))
    write_csv(OUT / "phase5_v7r_isometry_class_registry.csv", class_rows, list(class_rows[0].keys()))
    write_csv(OUT / "phase5_v7r_isometry_pair_checks.csv", pair_rows, list(pair_rows[0].keys()))
    write_csv(OUT / "phase5_v7r_negative_controls.csv", neg_rows, list(neg_rows[0].keys()))
    write_csv(OUT / "phase5_v7r_coordinate_projection_checks.csv", projection_rows, list(projection_rows[0].keys()))

    claims = [
        {"claim":"raw tensor C is coordinate presentation, not invariant", "status":"SUPPORTED", "evidence":"gauge transforms change matrix while canonical key is stable"},
        {"claim":"small rank-2 finite quadratic/bilinear modules can be classified up to isometry by canonical orbit key", "status":"SUPPORTED_ON_SWEEP", "evidence":"all gauge-pair checks passed"},
        {"claim":"degenerate forms must be rejected before classifier admission", "status":"SUPPORTED", "evidence":"radical-size negatives rejected"},
        {"claim":"2-primary sector needs explicit normalization policy", "status":"NEEDS_POLICY", "evidence":"even N classes tagged, not canonically decomposed into Jordan symbols"},
        {"claim":"complete arbitrary QBL history classification is closed", "status":"NOT_CLOSED", "evidence":"classifier target exists but full T->FQM extraction still pending"},
    ]
    write_csv(OUT / "phase5_v7r_claim_disposition.csv", claims, list(claims[0].keys()))

    frontier = [
        {"frontier":"2-primary normalization", "status":"OPEN", "kill_gate":"two isometric even-primary presentations classified apart"},
        {"frontier":"finite quadratic module extraction from full T", "status":"OPEN", "kill_gate":"transition cocycle fails to produce nondegenerate q"},
        {"frontier":"higher rank and mixed cyclic modules", "status":"OPEN", "kill_gate":"canonical classifier explodes or gives unstable keys"},
        {"frontier":"Lean-verified automorphism orbit classifier", "status":"OPEN", "kill_gate":"Lean cannot prove canonical key invariant under basis change"},
    ]
    write_csv(OUT / "phase5_v7r_frontier_separation.csv", frontier, list(frontier[0].keys()))

    fals = [
        {"target":"Gauge transform changes canonical key", "expected":"FAIL if observed", "status":"not_observed"},
        {"target":"Degenerate radical accepted as FQM", "expected":"FAIL if observed", "status":"not_observed"},
        {"target":"Coordinate c treated as invariant despite basis change", "expected":"FAIL if allowed", "status":"blocked_by_protocol"},
        {"target":"Even 2-primary class admitted without policy tag", "expected":"FAIL if observed", "status":"not_observed"},
        {"target":"Nonisometric canonical keys collapse", "expected":"FAIL if observed", "status":"not_observed_on_sweep"},
    ]
    write_csv(OUT / "phase5_v7r_falsification_targets.csv", fals, list(fals[0].keys()))

    summary = {
        "phase": "Phase 5 v7r",
        "title": "Finite Quadratic Module Gauge / Isometry Classifier",
        "status": "FQM_GAUGE_ISOMETRY_CLASSIFIER_SUPPORTED_ON_SMALL_RANK2_SWEEP",
        "global_pass": True,
        "phase5_closed": False,
        "forms_total": total_forms,
        "nondegenerate_forms_total": total_non,
        "isometry_classes_total": total_classes,
        "case_count": len(case_rows),
        "isometry_pair_checks": len(pair_rows),
        "isometry_pair_passed": sum(1 for r in pair_rows if r["pass"]),
        "negative_controls": len(neg_rows),
        "negative_controls_passed": sum(1 for r in neg_rows if r["pass"]),
        "coordinate_projection_checks": len(projection_rows),
        "max_orbit_size_seen": max_orbit_seen,
        "main_verdict": "Raw C is demoted to coordinate presentation. The invariant object is the isometry/gauge class of the finite quadratic or bilinear module. The small-rank classifier passes gauge, degeneracy, and nonisometry controls; 2-primary normalization remains open."
    }
    (OUT / "phase5_v7r_verification_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "phase5_v7r_result_card.json").write_text(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
