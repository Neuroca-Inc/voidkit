#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "sympy_results_v7.json"

n, d, j = sp.symbols("n d j", integer=True, positive=True)
pi_sym = sp.symbols("pi", positive=True)
q = sp.exp(-pi_sym)

# symbolic packet-collapse / AGM-side coarse bound
exact_bound = sp.pi**2 * 2 ** (n + 4) * q ** (2 ** (n + 1))
coarse_bound = 2 ** (n + 8) * sp.exp(-3 * 2 ** (n + 1))

log10_coarse = sp.expand_log(sp.log(coarse_bound, 10), force=True)
carry_condition = sp.Lt(sp.log(coarse_bound, 10) + d, -j)

# proof-sketch algebra for the v7 certificate
B_scaled = sp.simplify(coarse_bound * 10**d)
margin_floor = sp.Symbol("margin_floor", positive=True)

results = {
    "exact_bound": sp.srepr(exact_bound),
    "coarse_bound": sp.srepr(coarse_bound),
    "log10_coarse_bound": sp.srepr(log10_coarse),
    "carry_condition": sp.srepr(carry_condition),
    "scaled_bound": sp.srepr(B_scaled),
    "notes": [
        "exact AGM-side bound: pi^2 * 2^(n+4) * exp(-pi * 2^(n+1))",
        "coarse native bound after q<e^-3 and pi^2<16: 2^(n+8) * exp(-3*2^(n+1))",
        "carry certificate condition: log10(B_n) + d < -j",
    ],
}

OUT.write_text(json.dumps(results, indent=2) + "\n")
print(OUT)
