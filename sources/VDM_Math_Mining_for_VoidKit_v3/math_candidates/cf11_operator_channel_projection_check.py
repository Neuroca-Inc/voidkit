#!/usr/bin/env python3
"""SymPy check for the CF11 operator-core channel projection."""
from __future__ import annotations
import json, os
import sympy as sp

def run_check() -> dict[str, object]:
    raw_channel_projection = sp.Matrix([[1, 0, 0], [0, 1, 1]])  # rows C,R; columns Q,B,L
    macro_to_raw = sp.Matrix([
        [1, 1, 0],  # Q count in S,R,T
        [0, 1, 0],  # B count
        [0, 0, 1],  # L count
    ])
    expected_macro_projection = sp.Matrix([[1, 1, 0], [0, 1, 1]])
    macro_projection = sp.simplify(raw_channel_projection * macro_to_raw)
    residual = sp.simplify(macro_projection - expected_macro_projection)
    sample_counts = sp.simplify(macro_projection * sp.Matrix([1, 1, 1]))
    return {
        "check_id": "CF11-SYMPY-10",
        "description": "operator-core channel projection from Q,B,L to macro S,R,T",
        "raw_channel_projection": str(raw_channel_projection),
        "macro_to_raw": str(macro_to_raw),
        "macro_projection": str(macro_projection),
        "expected_macro_projection": str(expected_macro_projection),
        "projection_residual": str(residual),
        "projection_residual_zero": bool(residual == sp.zeros(2, 3)),
        "sample_word": "S R T",
        "sample_channel_counts_C_R": [str(sample_counts[0]), str(sample_counts[1])],
    }

if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
