#!/usr/bin/env python3
"""SymPy exactness checks for the three-filter covariance package."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


phi, delta = sp.symbols("phi delta", real=True)
I = sp.I

m = sp.exp(I * phi) * sp.cos(delta)
s = sp.exp(I * phi) * sp.sin(delta)

x1, x2, x3, x4 = sp.re(m), sp.im(m), sp.re(s), sp.im(s)

n1 = sp.simplify(2 * sp.re(m * sp.conjugate(s)))
n2 = sp.simplify(2 * sp.im(m * sp.conjugate(s)))
n3 = sp.simplify(sp.conjugate(m) * m - sp.conjugate(s) * s)

sheet_shift = sp.pi / 2
mQ = sp.simplify(m.subs(phi, phi + sheet_shift))
sQ = sp.simplify(s.subs(phi, phi + sheet_shift))

n1Q = sp.simplify(n1.subs(phi, phi + sheet_shift))
n2Q = sp.simplify(n2.subs(phi, phi + sheet_shift))
n3Q = sp.simplify(n3.subs(phi, phi + sheet_shift))

results = {
    "intrinsic_pair": {
        "m": sp.srepr(m),
        "s": sp.srepr(s),
    },
    "s3_unit_norm": sp.simplify(sp.conjugate(m) * m + sp.conjugate(s) * s - 1),
    "hopf_base": {
        "n1": sp.simplify(n1),
        "n2": sp.simplify(n2),
        "n3": sp.simplify(n3),
        "unit_sphere_residual": sp.simplify(n1**2 + n2**2 + n3**2 - 1),
    },
    "q_action": {
        "m_shift_minus_i_m": sp.simplify(mQ - I * m),
        "s_shift_minus_i_s": sp.simplify(sQ - I * s),
        "hopf_base_n1_invariant": sp.simplify(n1Q - n1),
        "hopf_base_n2_invariant": sp.simplify(n2Q - n2),
        "hopf_base_n3_invariant": sp.simplify(n3Q - n3),
    },
    "closed_forms": {
        "n1": sp.simplify(sp.trigsimp(n1.rewrite(sp.sin))),
        "n2": sp.simplify(sp.trigsimp(n2.rewrite(sp.sin))),
        "n3": sp.simplify(sp.trigsimp(n3.rewrite(sp.cos))),
    },
    "product_only_impossibility_witness": {
        "pair_1": [1, 6],
        "pair_2": [2, 3],
        "shared_product": 6,
        "B_pair_1": [6, 7],
        "B_pair_2": [3, 5],
        "B_images_equal": False,
    },
}


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    if isinstance(obj, sp.Basic):
        return sp.sstr(obj)
    return obj



def main(output_path: Path) -> None:
    output_path.write_text(json.dumps(serialize(results), indent=2))


if __name__ == "__main__":
    main(Path("phase_three_filter_sympy_output.json"))
