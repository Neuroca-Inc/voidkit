#!/usr/bin/env python3
import itertools, json
import numpy as np

def compute_contact(events):
    clock = {}
    C = {}
    Z = {}
    residuals = []
    for ev in events:
        pair = tuple(sorted(ev["pair"]))
        impulse = int(ev["impulse"])
        clock[pair] = clock.get(pair, 0) + 1
        p = clock[pair]
        dc = impulse
        dz = p * dc
        C[pair] = C.get(pair, 0) + dc
        Z[pair] = Z.get(pair, 0) + dz
        residuals.append(dz - p * dc)
    return C, Z, max([abs(x) for x in residuals] or [0])

if __name__ == "__main__":
    A = [{"pair": (0,1), "impulse": 1}, {"pair": (0,1), "impulse": 5}]
    B = [{"pair": (0,1), "impulse": 5}, {"pair": (0,1), "impulse": 1}]
    CA, ZA, rA = compute_contact(A)
    CB, ZB, rB = compute_contact(B)
    result = {
        "same_C": CA == CB,
        "same_Z": ZA == ZB,
        "A": {"C": str(CA), "Z": str(ZA), "alpha_residual": rA},
        "B": {"C": str(CB), "Z": str(ZB), "alpha_residual": rB},
        "pass": CA == CB and ZA != ZB and rA == 0 and rB == 0
    }
    print(json.dumps(result, indent=2))
