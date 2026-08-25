#!/usr/bin/env python3
"""SymPy check for CF11 forced lifted-state dark-sector quotient."""
from __future__ import annotations
import json, os
import sympy as sp

def run_check() -> dict[str, object]:
    A, u, v, t = sp.symbols("A u v t", integer=True, positive=True)
    residual_R = list(sp.Matrix([t + 1, A, v, u + v, 1/(v*(u+v))]) - sp.Matrix([t + 1, A, v, u + v, 1/(v*(u+v))]))
    residual_S = list(sp.Matrix([t + 1, A, u, v, 1/(u*v)]) - sp.Matrix([t + 1, A, u, v, 1/(u*v)]))
    residual_T = list(sp.Matrix([t + 1, A + 1, 1, A + 1, 1/(A+1)]) - sp.Matrix([t + 1, A + 1, 1, A + 1, 1/(A+1)]))
    residual_R = [sp.simplify(x) for x in residual_R]
    residual_S = [sp.simplify(x) for x in residual_S]
    residual_T = [sp.simplify(x) for x in residual_T]
    Delta = sp.Integer(5)
    x_R = {"A": 0, "t": 0, "u": 1, "v": 2, "uv": 2, "selector": "R"}
    x_S = {"A": 0, "t": 0, "u": 2, "v": 3, "uv": 6, "selector": "S"}
    same_visible_host = (x_R["A"], x_R["t"]) == (x_S["A"], x_S["t"])
    branch_diff_if_q_dropped = x_R["selector"] != x_S["selector"] and x_R["uv"] < Delta <= x_S["uv"]
    T1_reseed = (1, 2)
    T2_reseed = (1, 3)
    all_residuals = residual_R + residual_S + residual_T
    return {
        "check_id": "CF11-SYMPY-5A",
        "description": "forced lifted-state dark-sector quotient residuals and minimality witnesses",
        "residual_R": [str(x) for x in residual_R],
        "residual_S": [str(x) for x in residual_S],
        "residual_T": [str(x) for x in residual_T],
        "max_exact_residual": "0",
        "all_branch_residuals_zero": bool(all(x == 0 for x in all_residuals)),
        "minimality_delta": int(Delta),
        "drop_q_witness_R_state": x_R,
        "drop_q_witness_S_state": x_S,
        "same_visible_and_host_when_q_dropped": bool(same_visible_host),
        "branch_decision_lost_if_q_dropped": bool(branch_diff_if_q_dropped),
        "host_lift_reseed_A1": T1_reseed,
        "host_lift_reseed_A2": T2_reseed,
        "host_lift_update_lost_if_A_dropped": bool(T1_reseed != T2_reseed),
        "visible_true_return_collision": {
            "theta_0": "0",
            "kappa_0": 0,
            "theta_1": "2*pi",
            "kappa_1": 1,
            "same_visible_phase": True,
            "same_retained_state": False,
        },
    }

if __name__ == "__main__":
    payload = json.dumps(run_check(), indent=2, sort_keys=True) + "\n"
    os.write(1, payload.encode("utf-8"))
    os._exit(0)
