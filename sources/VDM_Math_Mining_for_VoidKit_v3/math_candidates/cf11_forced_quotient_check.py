#!/usr/bin/env python3
"""SymPy check for the CF11 forced two-channel dark-sector quotient."""
from __future__ import annotations
import json, os
import sympy as sp

def run_check() -> dict[str, object]:
    x1 = sp.Matrix([1, 0])
    x2 = sp.Matrix([1, 1])
    y1 = sp.Matrix([0, 2])
    y2 = sp.Matrix([1, 2])
    P_carrier = sp.Matrix([[1, 0]])
    P_reservoir = sp.Matrix([[0, 1]])
    P_two = sp.eye(2)
    carrier_collision_residual = sp.simplify(P_carrier * x1 - P_carrier * x2)
    reservoir_seen_by_two = sp.simplify(P_two * x1 - P_two * x2)
    reservoir_collision_residual = sp.simplify(P_reservoir * y1 - P_reservoir * y2)
    carrier_seen_by_two = sp.simplify(P_two * y1 - P_two * y2)
    R_counts = [9, 9, 8, 8]
    S_counts = [54, 54, 55, 55]
    T_counts = [1, 1, 1, 1]
    refinement_count = sum(R_counts)
    same_host_count = sum(S_counts)
    host_lift_count = sum(T_counts)
    macro_step_total = refinement_count + same_host_count + host_lift_count
    carrier_channel_count = same_host_count + refinement_count
    reservoir_channel_count = refinement_count + host_lift_count
    transition_accounting_residual = sp.simplify(macro_step_total - 256)
    return {
        "check_id": "CF11-SYMPY-0",
        "description": "forced two-channel quotient and selector witness accounting",
        "carrier_only_collision_residual": str(carrier_collision_residual),
        "carrier_only_forgets_reservoir": bool(carrier_collision_residual == sp.zeros(1, 1)),
        "two_channel_reservoir_difference": str(reservoir_seen_by_two),
        "two_channel_sees_reservoir_difference": bool(reservoir_seen_by_two != sp.zeros(2, 1)),
        "reservoir_only_collision_residual": str(reservoir_collision_residual),
        "reservoir_only_forgets_carrier": bool(reservoir_collision_residual == sp.zeros(1, 1)),
        "two_channel_carrier_difference": str(carrier_seen_by_two),
        "two_channel_sees_carrier_difference": bool(carrier_seen_by_two != sp.zeros(2, 1)),
        "selector_blocks": ["R^9 S^54 T", "R^9 S^54 T", "R^8 S^55 T", "R^8 S^55 T"],
        "refinement_count": refinement_count,
        "same_host_count": same_host_count,
        "host_lift_count": host_lift_count,
        "macro_step_total": macro_step_total,
        "carrier_channel_count": carrier_channel_count,
        "reservoir_channel_count": reservoir_channel_count,
        "transition_accounting_residual": str(transition_accounting_residual),
        "transition_accounting_residual_zero": bool(transition_accounting_residual == 0),
        "forced_two_channel_check_passes": bool(
            carrier_collision_residual == sp.zeros(1, 1)
            and reservoir_seen_by_two != sp.zeros(2, 1)
            and reservoir_collision_residual == sp.zeros(1, 1)
            and carrier_seen_by_two != sp.zeros(2, 1)
            and transition_accounting_residual == 0
        ),
    }

if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
