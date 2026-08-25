from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import sympy as sp

from pc_branch_certifier.application.compiler import BranchCoverCompiler
from pc_branch_certifier.domain.fibonacci import balanced_pair, depth_for_floor
from pc_branch_certifier.infrastructure.json_io import build_objects, load_spec

ROOT = Path(__file__).resolve().parents[1]


def load_report(name: str):
    spec = load_spec(ROOT / "data" / name)
    surface, presentation, generators, words, commutators, projection_words, floor = build_objects(spec)
    compiler = BranchCoverCompiler(surface, presentation, generators, floor)
    return compiler.compile(words, commutators, projection_words or None)


def assert_gate(name: str, cond: bool, details: dict[str, object], ledger: list[dict[str, object]]) -> None:
    ledger.append({"gate": name, "pass": bool(cond), "details": details})
    if not cond:
        raise AssertionError(f"{name} failed: {details}")


def packet_E(anchor_uv: int) -> mp.mpf:
    mp.mp.dps = 90
    t = 2 * mp.pi / mp.mpf(anchor_uv)
    q = mp.e ** (-t)
    log_poch = mp.nsum(lambda k: mp.log(1 - q**k), [1, mp.inf])
    return (log_poch + mp.pi**2 / (6 * t) - mp.mpf("0.5") * mp.log(2 * mp.pi / t)) / t


def main() -> int:
    ledger: list[dict[str, object]] = []

    # G1/G3: exact balanced corridor and logarithmic floor witness.
    assert_gate("G-FIB-ANCHOR", balanced_pair(9) == (55, 89), {"B^9(1,1)": balanced_pair(9)}, ledger)
    res = depth_for_floor(1e-8)
    assert_gate(
        "G-LOG-RESOLUTION",
        res.depth == 21 and res.u == 17711 and res.v == 28657 and res.half_width < 1e-8,
        {"depth": res.depth, "q": (res.u, res.v), "half_width": res.half_width},
        ledger,
    )

    # Symbolic Fibonacci identity for the first 24 depths.
    n = sp.symbols("n", integer=True, nonnegative=True)
    fib_pairs_ok = all(balanced_pair(i) == (int(sp.fibonacci(i + 1)), int(sp.fibonacci(i + 2))) for i in range(24))
    assert_gate("G-SYMPY-FIBONACCI-ROWS", fib_pairs_ok, {"checked_depths": 24, "formula": "B^n(1,1)=(F_{n+1},F_{n+2})"}, ledger)

    # G4/G10: arithmetic-completion packet collapse at anchor.
    E_anchor = packet_E(55 * 89)
    anchor_error = abs(E_anchor - mp.mpf(1) / 24)
    assert_gate("G-PACKET-ANCHOR-1-24", anchor_error < mp.mpf("1e-50"), {"E(2pi/4895)": str(E_anchor), "error": str(anchor_error)}, ledger)

    # G7: finite branch quotient compiler, S5 relators and commutator.
    q_report = load_report("quintic_s5_branch_surface.json")
    assert_gate("G-S5-COMPILE", q_report.all_gates_pass(), {"status": q_report.status, "resolution": q_report.resolution}, ledger)
    assert_gate("G-S5-RELATORS", all(r.pass_gate for r in q_report.relator_certificates), {"relator_count": len(q_report.relator_certificates)}, ledger)
    assert_gate("G-S5-COMMUTATOR", not q_report.commutator_certificates[0].is_identity, {"cycles": q_report.commutator_certificates[0].cycles}, ledger)
    assert_gate("G-S5-QUOTIENT-RESIDUAL", all(w.quotient_residual == 0 for w in q_report.word_registry), {"residuals": [w.quotient_residual for w in q_report.word_registry]}, ledger)

    # G7 countable analytic branch support: logarithm cover as integer shift action.
    log_report = load_report("log_countable_branch_surface.json")
    assert_gate("G-COUNTABLE-LOG-COMPILE", log_report.all_gates_pass(), {"status": log_report.status, "word_end_sheets": [w.end_sheet for w in log_report.word_registry]}, ledger)

    # G11: Bring quintic all-root certificate.
    root_report = json.loads((ROOT / "certificate_bring_quintic_roots.json").read_text(encoding="utf-8"))
    residuals = [r["residual_abs"] for r in root_report["roots"]]
    assert_gate("G-BRING-ROOTS", root_report["status"] == "PROVEN" and max(residuals) < 1.4e-15, {"max_residual": max(residuals), "root_count": len(residuals)}, ledger)

    out = ROOT / "sympy_audit_ledger.json"
    out.write_text(json.dumps({"status": "PASS", "gates": ledger}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "gate_count": len(ledger)}, indent=2))
    print("FINAL_RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
