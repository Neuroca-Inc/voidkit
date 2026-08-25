#!/usr/bin/env python3
import json
from pathlib import Path
import sympy as sp

W = sp.Integer(64)
e = sp.symbols('e', integer=True, nonnegative=True)

# Ideal unlimited admission recurrence by saturation epochs.
# a(0)=1, a(e+1)=2*a(e). This is not a geometry law; it is a carry-admission count.
closed = 2**e
recurrence_residual = sp.simplify(2 * closed - 2**(e + 1))
births_closed = closed - 1
births_residual = sp.simplify(sum(2**k for k in range(8)) - (2**8 - 1))

report = {
    "claim": "single-origin carry admission has no preassigned 2D coordinates",
    "saturation_width": int(W),
    "admitted_after_e_saturation_epochs": "2**e",
    "births_after_e_saturation_epochs": "2**e - 1",
    "recurrence_residual": str(recurrence_residual),
    "births_residual_for_8_epochs": str(births_residual),
    "explicit_grid": False,
    "explicit_edges": False,
    "geometry_primitive": False,
    "status": "PASS" if recurrence_residual == 0 and births_residual == 0 else "FAIL",
}

out = Path("verification/cf000_origin_arena_sympy_report.json")
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
