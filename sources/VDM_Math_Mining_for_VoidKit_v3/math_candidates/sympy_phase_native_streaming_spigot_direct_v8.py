#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "sympy_v8_checks.json"

N, x = sp.symbols('N x', positive=True, real=True)
tau = sp.I * x / (sp.pi * N)

# Algebraic identities for the universal packet collapse derivation.
identities = {
    "tau_formula": sp.simplify(tau - sp.I * x / (sp.pi * N)) == 0,
    "tau_at_pi": sp.simplify(tau.subs(x, sp.pi) - sp.I / N) == 0,
    "modular_relation_at_pi": sp.simplify((N**2) * tau.subs(x, sp.pi) + 1 / tau.subs(x, sp.pi)) == 0,
}

# Numeric q-product checks of F_N(pi)=0 for sample balanced packets.
mp.mp.dps = 80
samples = []
for n in [2, 6, 15, 40, 104]:
    s1 = mp.mpf('0')
    s2 = mp.mpf('0')
    for m in range(1, 1801):
        s1 += mp.log(1 - mp.e ** (-(2 * mp.pi / n) * m))
        s2 += mp.log(1 - mp.e ** (-(2 * n * mp.pi) * m))
    F = mp.mpf('0.5') * mp.log(n) - s1 - ((n - 1 / mp.mpf(n)) / 12) * mp.pi + s2
    samples.append({"N": n, "abs_F_N_pi": str(abs(F))})

# q-side term-count estimate monotonicity check
D = sp.symbols('D', positive=True, real=True)
K_est = (1 + sp.sqrt(1 + 12 * N * D * sp.log(10) / sp.pi)) / 6
monotonic_dK_dN = sp.simplify(sp.diff(K_est, N))

payload = {
    "identities": identities,
    "sample_packet_checks": samples,
    "K_est": str(K_est),
    "dK_dN": str(monotonic_dK_dN),
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
