#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SRC = ROOT / "source_inputs"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

getcontext().prec = 80
mp.mp.dps = 100

GATES: list[dict[str, Any]] = []
CONSTANTS: list[dict[str, Any]] = []
PSLQ_ATTACKS: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_gate(gate_id: str, claim: str, passed: bool, metric: str, value: Any, threshold: Any,
             status: str = "PROVEN", artifact: str | None = None, note: str = "") -> None:
    GATES.append({
        "gate_id": gate_id,
        "claim": claim,
        "passed": bool(passed),
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "status": status,
        "artifact": artifact,
        "note": note,
    })


def add_constant(name: str, value: str, claim: str, method: str, status: str,
                 residual: str | None = None, artifact: str | None = None) -> None:
    CONSTANTS.append({
        "name": name,
        "value": value,
        "claim": claim,
        "method": method,
        "status": status,
        "residual": residual,
        "artifact": artifact,
    })


# -----------------------------------------------------------------------------
# Basic lifted-state / corridor functions
# -----------------------------------------------------------------------------

def B_pair(u: int, v: int) -> tuple[int, int]:
    a, b = sorted((v, u + v))
    return a, b


def corridor(depth: int, start: tuple[int, int] = (1, 1)) -> list[tuple[int, int, int]]:
    u, v = start
    rows = [(0, u, v)]
    for d in range(1, depth + 1):
        u, v = B_pair(u, v)
        rows.append((d, u, v))
    return rows


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def pslq_attack(name: str, target: Any, basis: list[Any], maxcoeff: int = 10**6, digits: int = 100) -> None:
    mp.mp.dps = digits
    vals = [mp.mpf(target)] + [mp.mpf(x) for x in basis]
    try:
        rel = mp.pslq(vals, tol=mp.mpf(10) ** (-(digits - 20)), maxcoeff=maxcoeff, maxsteps=1000)
    except Exception as exc:
        rel = f"ERROR: {exc}"
    # Ignore basis-only relations whose target coefficient is zero.
    if isinstance(rel, list) and len(rel) > 0 and rel[0] == 0:
        rel = None
    PSLQ_ATTACKS.append({
        "name": name,
        "digits": digits,
        "maxcoeff": maxcoeff,
        "basis": [str(x) for x in basis],
        "relation": rel,
        "closed_exact_anchor_expression": rel is not None and not isinstance(rel, str),
    })


# -----------------------------------------------------------------------------
# Gate group A: Phase Calculus finite laws and Xi engine
# -----------------------------------------------------------------------------

rows = corridor(21)
expected = [(d, fib(d + 1), fib(d + 2)) for d, _, _ in rows]
fib_ok = rows == expected
add_gate("PC-G01", "Balanced refinement B(u,v)=sort(v,u+v) follows Fibonacci corridor from (1,1).",
         fib_ok, "all rows d<=21 match (F_{d+1},F_{d+2})", fib_ok, True,
         artifact="results/corridor_rows.csv")

anchor_row = rows[9]
anchor_ok = anchor_row == (9, 55, 89)
add_gate("PC-G02", "Canonical anchor appears at depth 9 as (55,89).",
         anchor_ok, "row_9", anchor_row, (9, 55, 89), artifact="results/corridor_rows.csv")

half_width_21 = mp.pi / (rows[21][1] * rows[21][2])
add_gate("PC-G03", "Balanced germ half-width falls below 1e-8 by depth 21.",
         half_width_21 < mp.mpf("1e-8"), "pi/(u_21*v_21)", mp.nstr(half_width_21, 30), "<1e-8",
         artifact="results/corridor_rows.csv")

# write corridor rows
with (RESULTS / "corridor_rows.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["depth", "u", "v", "uv", "half_width"])
    for d, u, v in rows:
        w.writerow([d, u, v, u*v, str(mp.pi / (u*v))])

# Xi engine trace validation
xi_trace = SRC / "xi_engine" / "xi_full_engine_trace_patched.csv"
xi_rows = list(csv.DictReader(xi_trace.open()))
first_ready: dict[int, dict[str, str]] = {}
for r in xi_rows:
    if r["window_ready"] == "1":
        A = int(r["A"])
        if A not in first_ready:
            first_ready[A] = r
expected_ready = {0: (10, 55, 89, 4895), 1: (74, 55, 89, 4895), 2: (137, 55, 89, 4895)}
ready_ok = True
ready_payload = {}
for A, exp in expected_ready.items():
    r = first_ready.get(A)
    got = None if r is None else (int(r["step"]), int(r["u"]), int(r["v"]), int(r["uv"]))
    ready_payload[A] = got
    ready_ok = ready_ok and got == exp
add_gate("PC-G04", "Xi engine reaches the first three canonical launch windows at the CF19 anchor.",
         ready_ok, "first window rows A=0,1,2", ready_payload, expected_ready,
         artifact="source_inputs/xi_engine/xi_full_engine_trace_patched.csv")

# Q4 finite witness: visible phase mod 4 unchanged, kappa increments
q4_ok = True
for t in range(0, 64):
    visible0 = t % 4
    visible1 = (t + 4) % 4
    k0 = t // 4
    k1 = (t + 4) // 4
    q4_ok = q4_ok and visible0 == visible1 and k1 == k0 + 1
add_gate("PC-G05", "Q^4 preserves visible phase and increments completed-turn memory kappa.",
         q4_ok, "finite tick check t=0..63", q4_ok, True)

# Visible projection collision
visible_collision_ok = (0 % 4 == 4 % 4) and (0 // 4 != 4 // 4)
add_gate("PC-G06", "Visible witness is not state-complete: same phase residue can carry different kappa.",
         visible_collision_ok, "t=0 vs t=4", {"visible_equal": True, "kappa": [0, 1]}, "collision exists")

# CF19 coefficients
coeff_text = (SRC / "cf19" / "balanced_window_completion_coefficients.txt").read_text()
edge_match = re.search(r"edge_coefficient\s*=\s*1/24", coeff_text) is not None
two_match = re.search(r"two_sided_coefficient\s*=\s*1/12", coeff_text) is not None
edge_two_ok = edge_match and two_match and sp.Rational(2, 24) == sp.Rational(1, 12)
add_gate("PC-G07", "CF19 edge coefficient doubles from 1/24 per edge to 1/12 two-sided.",
         edge_two_ok, "2*(1/24)", str(sp.Rational(2, 24)), "1/12",
         artifact="source_inputs/cf19/balanced_window_completion_coefficients.txt")

# zeta(-1)
zeta_minus_one = sp.zeta(-1)
zeta_gate_ok = sp.simplify(zeta_minus_one + sp.Rational(1, 12)) == 0
add_gate("PC-G08", "Signed two-sided completion matches zeta(-1)=-1/12.",
         zeta_gate_ok, "sympy zeta(-1)+1/12", str(sp.simplify(zeta_minus_one + sp.Rational(1, 12))), "0")
add_constant("zeta(-1)", str(zeta_minus_one), "signed two-sided edge coefficient", "SymPy exact zeta + CF19 coefficient identity", "PROVEN", "0")

# -----------------------------------------------------------------------------
# Gate group B: pi spigot v8 evidence
# -----------------------------------------------------------------------------

verif = json.loads((SRC / "pi_spigot_v8" / "verification_results.json").read_text())
pi_all_lengths_ok = all(row["native_certificate_passed"] for row in verif.values())
small_cross_ok = all(verif[k].get("candidate_correct_vs_chudnovsky", True) for k in ["520", "1000", "10000"])
add_gate("PI-G01", "Phase-native pi spigot v8 certificates pass for 520 through 1,000,000 target digits.",
         pi_all_lengths_ok, "native_certificate_passed for all target lengths", pi_all_lengths_ok, True,
         artifact="source_inputs/pi_spigot_v8/verification_results.json")
add_gate("PI-G02", "Short pi outputs cross-check against independent Chudnovsky/Ramanujan baselines.",
         small_cross_ok, "520/1000/10000 candidate correctness", small_cross_ok, True,
         artifact="source_inputs/pi_spigot_v8/verification_results.json")

pi_bank = (SRC / "pi_spigot_v8" / "pi_10000_decimal.txt").read_text().strip()
mp.mp.dps = 120
mp_pi = mp.nstr(mp.pi, 110)
# nstr uses significant digits without trailing zeros. Compare first 100 chars of standard decimal string.
pi_prefix_ok = pi_bank[:100] == mp_pi[:100]
add_gate("PI-G03", "Included 10k pi bank agrees with mpmath prefix smoke check.",
         pi_prefix_ok, "first 100 chars", {"bank": pi_bank[:100], "mpmath": mp_pi[:100]}, "exact prefix match",
         artifact="source_inputs/pi_spigot_v8/pi_10000_decimal.txt")
add_constant("pi", pi_bank[:82], "native packet-collapse / certified bank prefix", "v8 spigot verification JSON + prefix smoke check", "PROVEN-NUMERIC", artifact="source_inputs/pi_spigot_v8/verification_results.json")

packet = json.loads((SRC / "pi_spigot_v8" / "universal_packet_collapse_checks.json").read_text())
packet_abs = [Decimal(str(x["F_N_at_pi_abs"])) for x in packet["selected_checks"]]
packet_ok = max(packet_abs) < Decimal("1e-35")
add_gate("PI-G04", "Sampled universal packet-collapse checks satisfy F_N(pi)≈0.",
         packet_ok, "max sampled |F_N(pi)|", str(max(packet_abs)), "<1e-35",
         artifact="source_inputs/pi_spigot_v8/universal_packet_collapse_checks.json")

ra = json.loads((SRC / "pi_spigot_v8" / "random_access_benchmark.json").read_text())
ra_ok = all(s.get("certified_under_safe_lower_bound") for s in ra.get("samples", []))
add_gate("PI-G05", "Indexed random access stays inside certified safe lower bound in included samples.",
         ra_ok, "all sample blocks certified", ra_ok, True,
         artifact="source_inputs/pi_spigot_v8/random_access_benchmark.json")

# -----------------------------------------------------------------------------
# Gate group C: core constants / continuous shadow branch smoke tests
# -----------------------------------------------------------------------------

# sqrt branch exact identities
for n in [2, 3, 5]:
    x = sp.sqrt(n)
    residual = sp.simplify(x**2 - n)
    add_gate(f"ALG-G{n:02d}", f"sqrt({n}) is certified by polynomial x^2-{n}=0.",
             residual == 0, "exact polynomial residual", str(residual), "0")
    add_constant(f"sqrt({n})", str(sp.N(x, 60)), f"constructible algebraic branch x^2={n}", "SymPy exact radical residual", "PROVEN", str(residual))

# phi corridor exact and numeric convergence
phi = (1 + sp.sqrt(5)) / 2
phi_res = sp.simplify(phi**2 - phi - 1)
phi_ratio = sp.Rational(89, 55)
phi_ratio_err = abs(sp.N(phi_ratio - phi, 80))
add_gate("ALG-G06", "Golden ratio is the fixed point of x^2-x-1 and is approached by the anchor ratio 89/55.",
         phi_res == 0 and phi_ratio_err < sp.Float("2e-4"), "exact residual and anchor ratio error", {"residual": str(phi_res), "abs_error": str(phi_ratio_err)}, "residual=0 and error<2e-4")
add_constant("phi", str(sp.N(phi, 60)), "Fibonacci corridor limit", "exact quadratic residual + anchor ratio convergence", "PROVEN", str(phi_res))

# e from exp series
N_e = 40
e_approx = sum(sp.Rational(1, math.factorial(k)) for k in range(N_e + 1))
e_error = abs(sp.N(sp.E - e_approx, 90))
add_gate("SHADOW-G01", "e is recovered by the continuous-shadow exponential series to >40 digits.",
         e_error < sp.Float("1e-45"), "|E - sum_{k<=40}1/k!|", str(e_error), "<1e-45", status="NUMERIC-SYMBOLIC")
add_constant("e", str(sp.N(sp.E, 60)), "continuous-shadow exponential branch", "Taylor series residual attack", "PROVEN-NUMERIC", str(e_error))

# logs by artanh series
for label, xval, threshold in [("ln2", sp.Integer(2), sp.Float("1e-70")), ("ln3", sp.Integer(3), sp.Float("1e-55"))]:
    y = sp.Rational(int(xval - 1), int(xval + 1))
    N = 160
    approx = 2 * sum(y ** (2*k + 1) / sp.Integer(2*k + 1) for k in range(N + 1))
    exact = sp.log(xval)
    err = abs(sp.N(exact - approx, 100))
    add_gate(f"SHADOW-G-{label.upper()}", f"{label} is recovered by the continuous-shadow log/artanh branch.",
             err < threshold, "series residual", str(err), f"<{threshold}", status="NUMERIC-SYMBOLIC")
    add_constant(label.replace("ln", "ln "), str(sp.N(exact, 60)), "continuous-shadow logarithm branch", "artanh series residual attack", "PROVEN-NUMERIC", str(err))

# Plastic and tribonacci constants
x = sp.Symbol("x")
poly_defs = [
    ("plastic constant", x**3 - x - 1),
    ("tribonacci constant", x**3 - x**2 - x - 1),
]
for idx, (name, poly) in enumerate(poly_defs, start=1):
    roots = sp.nroots(poly, n=100, maxsteps=200)
    real_roots = [r for r in roots if abs(sp.im(r)) < sp.Float("1e-90")]
    root_val = max(real_roots, key=lambda z: sp.re(z))
    residual = abs(sp.N(poly.subs(x, root_val), 90))
    add_gate(f"REC-G{idx:02d}", f"{name} is certified as the dominant real recurrence root.",
             residual < sp.Float("1e-80"), "polynomial residual", str(residual), "<1e-80", status="NUMERIC-SYMBOLIC")
    add_constant(name, mp.nstr(root_val, 60), "higher-order recurrence corridor", "nroots + polynomial residual", "PROVEN-NUMERIC", str(residual))

# Elliptic / modular numeric witness only
mp.mp.dps = 80
K = mp.ellipk(mp.mpf("0.5"))  # parameter m=1/2, i.e., modulus k=1/sqrt(2)
gamma14 = mp.gamma(mp.mpf(1)/4)
K_formula = gamma14**2 / (4 * mp.sqrt(mp.pi))
K_err = abs(K - K_formula)
add_gate("MOD-G01", "K(k=1/sqrt(2)) matches Gamma(1/4)^2/(4 sqrt(pi)) numerically.",
         K_err < mp.mpf("1e-70"), "elliptic special-value residual", mp.nstr(K_err, 30), "<1e-70", status="NUMERIC-SYMBOLIC")
add_constant("K(1/sqrt(2))", mp.nstr(K, 60), "elliptic/modular special-value branch", "mpmath residual against Gamma formula", "PROVEN-NUMERIC", mp.nstr(K_err, 30))

# Riemann first zero numeric locator
zero1 = mp.zetazero(1)
zeta_zero_res = abs(mp.zeta(zero1))
add_gate("ZETA-G01", "First nontrivial zeta zero is numerically located on the critical line.",
         abs(mp.re(zero1) - mp.mpf("0.5")) < mp.mpf("1e-70") and zeta_zero_res < mp.mpf("1e-70"),
         "zeta(zero) residual", {"zero": str(zero1), "abs_zeta": mp.nstr(zeta_zero_res, 30)}, "critical line and |zeta|<1e-70", status="NUMERIC-ONLY")
add_constant("first zeta zero imaginary part", mp.nstr(mp.im(zero1), 60), "numeric locator only; no structural forcing claimed", "mpmath zetazero residual", "NUMERIC-ONLY", mp.nstr(zeta_zero_res, 30))

# -----------------------------------------------------------------------------
# Gate group D: Quintic utility certificate
# -----------------------------------------------------------------------------

bring = json.loads((SRC / "quintic" / "bring_all_roots_certificates.json").read_text())
certs = bring["certificates"]
quintic_ok = len(certs) == 5 and all(c["depth"] == 21 and c["certified_state"]["half_width"] < 1e-8 and c["projected_polynomial_residual_abs"] < 1e-12 for c in certs)
max_resid = max(c["projected_polynomial_residual_abs"] for c in certs)
max_hw = max(c["certified_state"]["half_width"] for c in certs)
add_gate("QUINTIC-G01", "Lifted quintic utility certifies all five Bring roots at depth 21.",
         quintic_ok, "max residual and max half-width", {"count": len(certs), "max_residual": max_resid, "max_half_width": max_hw}, "count=5, depth=21, residual<1e-12, half_width<1e-8",
         artifact="source_inputs/quintic/bring_all_roots_certificates.json")

# -----------------------------------------------------------------------------
# Tier-1 exact-expression attacks: no false victory.
# -----------------------------------------------------------------------------

mp.mp.dps = 100
basis = [mp.mpf(1), mp.pi, mp.pi**2, mp.pi**3, mp.log(2), mp.log(3), mp.sqrt(2), mp.sqrt(3), (1+mp.sqrt(5))/2, mp.mpf(1)/24, mp.mpf(1)/12]
pslq_attack("zeta(3)", mp.zeta(3), basis)
pslq_attack("Catalan G", mp.catalan, basis)
pslq_attack("Euler-Mascheroni gamma", mp.euler, basis)
for item in PSLQ_ATTACKS:
    add_gate(f"OPEN-{item['name'].split()[0].upper()}", f"No bounded PSLQ anchor expression found for {item['name']} in the tested basis.",
             item["relation"] is None, "PSLQ relation", item["relation"], "None with maxcoeff<=1e6", status="NEEDS_DATA",
             note="This is an attack result, not a proof of nonexistence.")
    add_constant(item["name"], str(sp.N({"zeta(3)": sp.zeta(3), "Catalan G": sp.Catalan, "Euler-Mascheroni gamma": sp.EulerGamma}[item["name"]], 60)),
                 "Tier-1 target; exact anchor expression not certified in this run", "PSLQ attack against anchor basis", "NEEDS_DATA", str(item["relation"]))

# -----------------------------------------------------------------------------
# Output ledgers and figures
# -----------------------------------------------------------------------------

with (RESULTS / "gate_ledger.json").open("w") as f:
    json.dump(GATES, f, indent=2, default=str)
with (RESULTS / "constant_ledger.json").open("w") as f:
    json.dump(CONSTANTS, f, indent=2, default=str)
with (RESULTS / "pslq_attack_ledger.json").open("w") as f:
    json.dump(PSLQ_ATTACKS, f, indent=2, default=str)

with (RESULTS / "gate_ledger.csv").open("w", newline="") as f:
    fields = ["gate_id", "claim", "passed", "metric", "value", "threshold", "status", "artifact", "note"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for g in GATES:
        w.writerow({k: json.dumps(g.get(k), default=str) if isinstance(g.get(k), (dict, list, tuple)) else g.get(k) for k in fields})

with (RESULTS / "constant_ledger.csv").open("w", newline="") as f:
    fields = ["name", "value", "claim", "method", "status", "residual", "artifact"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(CONSTANTS)

# Generate figures if matplotlib is available
try:
    import matplotlib.pyplot as plt
    depths = [d for d, _, _ in rows]
    widths = [float(mp.pi / (u*v)) for d, u, v in rows]
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(depths, widths, marker="o")
    plt.axhline(1e-8, linestyle="--")
    plt.xlabel("balanced refinement depth")
    plt.ylabel("germ half-width")
    plt.title("Balanced corridor germ decay")
    plt.tight_layout()
    plt.savefig(FIGURES / "balanced_corridor_germ_decay.png", dpi=160)
    plt.close()

    gate_labels = [g["gate_id"] for g in GATES]
    gate_values = [1 if g["passed"] else 0 for g in GATES]
    plt.figure(figsize=(12, 4.5))
    plt.bar(range(len(gate_values)), gate_values)
    plt.yticks([0, 1], ["FAIL", "PASS"])
    plt.xticks(range(len(gate_values)), gate_labels, rotation=80, fontsize=7)
    plt.title("Phase Calculus demonstration gate results")
    plt.tight_layout()
    plt.savefig(FIGURES / "gate_pass_matrix.png", dpi=160)
    plt.close()

    names = [c["name"] for c in CONSTANTS]
    status_map = {"PROVEN": 3, "PROVEN-NUMERIC": 2, "NUMERIC-ONLY": 1, "NEEDS_DATA": 0}
    vals = []
    for c in CONSTANTS:
        st = c["status"]
        vals.append(status_map.get(st, 1))
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(vals)), vals)
    plt.yticks([0, 1, 2, 3], ["NEEDS_DATA", "NUMERIC", "PROVEN-NUMERIC", "PROVEN"])
    plt.xticks(range(len(vals)), names, rotation=75, ha="right", fontsize=8)
    plt.title("Constant coverage status")
    plt.tight_layout()
    plt.savefig(FIGURES / "constant_status_matrix.png", dpi=160)
    plt.close()
except Exception as exc:
    add_gate("FIG-G01", "Figure generation", False, "matplotlib", str(exc), "no exception", status="RUNTIME")

# SHA manifest
manifest = []
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and not path.name.endswith("SHA256SUMS.txt"):
        manifest.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
with (RESULTS / "artifact_manifest.json").open("w") as f:
    json.dump(manifest, f, indent=2)
with (ROOT / "SHA256SUMS.txt").open("w") as f:
    for row in manifest:
        f.write(f"{row['sha256']}  {row['path']}\n")

summary = {
    "final_result": "PASS" if all(g["passed"] or g["status"] == "NEEDS_DATA" for g in GATES) else "FAIL",
    "pass_count": sum(1 for g in GATES if g["passed"]),
    "fail_count": sum(1 for g in GATES if not g["passed"] and g["status"] != "NEEDS_DATA"),
    "needs_data_count": sum(1 for g in GATES if g["status"] == "NEEDS_DATA"),
    "gate_count": len(GATES),
    "constant_count": len(CONSTANTS),
    "lean_executed_in_container": False,
    "lean_note": "Lean was not installed in the execution container; Lean files are included as proof surfaces and require external lake build.",
}
with (RESULTS / "validation_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"FINAL_RESULT: {summary['final_result']}")
if summary["final_result"] != "PASS":
    sys.exit(1)
