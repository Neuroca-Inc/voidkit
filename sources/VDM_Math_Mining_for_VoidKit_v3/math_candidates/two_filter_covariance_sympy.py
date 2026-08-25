
#!/usr/bin/env python3
"""
SymPy companion for the two-filter covariance package.

This script attacks the symbolic algebraic core:
1. The green branch pair lies on S^3: |h|^2 + |g|^2 = 1.
2. Quarter-turn transport acts by multiplication by i on (h,g).
3. The launched-family bridge delta = pi/(uv) = pi * Im(tau).
4. Product-only / tau-only filters cannot determine B uniquely.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

I = sp.I
theta = sp.symbols("theta", real=True)
u, v = sp.symbols("u v", positive=True, integer=True)
delta = sp.symbols("delta", positive=True, real=True)
pi = sp.pi

h = sp.exp(I * theta) * sp.cos(delta)
g = sp.exp(I * theta) * sp.sin(delta)

s3_identity = sp.simplify(sp.expand_complex(sp.Abs(h) ** 2 + sp.Abs(g) ** 2))
quarter_h = sp.simplify(sp.exp(I * (theta + sp.pi / 2)) * sp.cos(delta) - I * h)
quarter_g = sp.simplify(sp.exp(I * (theta + sp.pi / 2)) * sp.sin(delta) - I * g)

tau = theta / (2 * pi) + I / (u * v)
delta_bridge = sp.simplify(pi * sp.im(tau) - pi / (u * v))

# Product-only ambiguity witness:
pair_1 = (1, 6)
pair_2 = (2, 3)
same_product = pair_1[0] * pair_1[1] == pair_2[0] * pair_2[1]
B1 = (pair_1[1], pair_1[0] + pair_1[1])
B2 = (pair_2[1], pair_2[0] + pair_2[1])

summary = {
    "s3_identity": str(s3_identity),
    "quarter_turn_h_residual": str(sp.simplify(quarter_h)),
    "quarter_turn_g_residual": str(sp.simplify(quarter_g)),
    "delta_bridge_residual": str(delta_bridge),
    "same_product_example": {
        "pair_1": pair_1,
        "pair_2": pair_2,
        "same_product": same_product,
        "B_pair_1": B1,
        "B_pair_2": B2,
        "B_images_equal": B1 == B2,
    },
}

print(json.dumps(summary, indent=2))

out = Path(__file__).with_name("two_filter_sympy_output.json")
with out.open("w") as f:
    json.dump(summary, f, indent=2)
