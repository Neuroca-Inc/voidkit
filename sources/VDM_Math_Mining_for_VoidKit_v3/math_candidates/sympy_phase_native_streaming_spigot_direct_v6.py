#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

x = sp.symbols("x", positive=True, real=True)
tau = sp.I * x / (2 * sp.pi)
eta_tau = sp.symbols("eta_tau", nonzero=True)
eta_4tau = sp.symbols("eta_4tau", nonzero=True)

# Packet-law algebraic rewrite after substituting (q;q)_inf = q^(-1/24) eta(tau)
expr_raw = sp.Rational(1, 2) * sp.log(2) - (sp.log(eta_tau) + x / 24) - x / 8 + (sp.log(eta_4tau) + x / 6)
expr_reduced = sp.simplify(expr_raw)
expected = sp.Rational(1, 2) * sp.log(2) + sp.log(eta_4tau) - sp.log(eta_tau)

# Modular fixed packet equation 4*tau = -1/tau
fixed_eq_residual = sp.simplify(4 * (sp.I / 2) + 1 / (sp.I / 2))
fixed_x = sp.simplify(2 * sp.pi * sp.im(sp.I / 2))

# Legacy floor arithmetic from benchmark JSON
bench = json.loads((OUT / "benchmark_results.json").read_text())
row520 = bench["lengths"]["520"]
hot_520 = sp.nsimplify(row520["hot_path_seconds_mean"])
cert_520 = sp.nsimplify(row520["legacy_certificate_seconds_mean"])
both_520 = sp.nsimplify(row520["legacy_hot_plus_certificate_seconds_mean"])
target = sp.Rational(1, 5000)  # 0.0002

report = {
    "rewrite_matches_expected": sp.simplify(expr_reduced - expected) == 0,
    "rewrite_result": str(expr_reduced),
    "fixed_packet_equation_residual": str(fixed_eq_residual),
    "collapsed_x": str(fixed_x),
    "hot_520_lt_target": bool(hot_520 < target),
    "legacy_both_520_gt_target": bool(both_520 > target),
    "hot_520": str(hot_520),
    "cert_520": str(cert_520),
    "both_520": str(both_520),
    "target": str(target),
}

(ROOT / "sympy_output_v6.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
