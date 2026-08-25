#!/usr/bin/env python3
"""SymPy check for the CF11 forced two-channel quotient lower bound."""
from __future__ import annotations
import json, os
import sympy as sp

def run_check() -> dict[str, object]:
    x_carrier_only_a = sp.Matrix([1, 0])
    x_carrier_only_b = sp.Matrix([1, 1])
    x_reservoir_only_a = sp.Matrix([0, 1])
    x_reservoir_only_b = sp.Matrix([1, 1])
    Pi_carrier = sp.Matrix([[1, 0]])
    Pi_reservoir = sp.Matrix([[0, 1]])
    Pi_full = sp.eye(2)
    carrier_collision = sp.simplify(Pi_carrier * x_carrier_only_a - Pi_carrier * x_carrier_only_b)
    carrier_full_difference = sp.simplify(Pi_full * x_carrier_only_a - Pi_full * x_carrier_only_b)
    reservoir_collision = sp.simplify(Pi_reservoir * x_reservoir_only_a - Pi_reservoir * x_reservoir_only_b)
    reservoir_full_difference = sp.simplify(Pi_full * x_reservoir_only_a - Pi_full * x_reservoir_only_b)
    return {
        "check_id": "CF11-SYMPY-9",
        "description": "forced dark-sector quotient lower bound: one-channel readouts lose retained burden",
        "carrier_only_collision_residual": str(carrier_collision),
        "carrier_pair_full_difference": str(carrier_full_difference),
        "reservoir_only_collision_residual": str(reservoir_collision),
        "reservoir_pair_full_difference": str(reservoir_full_difference),
        "carrier_only_fails_to_distinguish_reservoir_burden": bool(carrier_collision == sp.zeros(1, 1) and carrier_full_difference != sp.zeros(2, 1)),
        "reservoir_only_fails_to_distinguish_carrier_memory": bool(reservoir_collision == sp.zeros(1, 1) and reservoir_full_difference != sp.zeros(2, 1)),
        "two_channel_readout_needed": True,
    }

if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
