#!/usr/bin/env python3
import math, cmath, csv, json

def weil_F(N):
    return [[cmath.exp(-2j*math.pi*r*s/N)/(N**0.5) for s in range(N)] for r in range(N)]

def weil_G(N):
    return [[(cmath.exp(1j*math.pi*r*r/N) if r==s else 0) for s in range(N)] for r in range(N)]

# This script intentionally refuses to compare unless an explicit QBL word and Orthad projection are supplied.
def compare_readout(N, word, projection_fn, target):
    if word is None or projection_fn is None:
        return {"status":"NOT_RUN", "reason":"explicit QBL word and Orthad projection required", "residual":None}
    readout = projection_fn(word, N)
    M = weil_G(N) if target == "Weil_G_N" else weil_F(N)
    residual = max(abs(readout[i][j]-M[i][j]) for i in range(N) for j in range(N))
    return {"status":"RUN", "residual":residual, "match": residual < 1e-12}

if __name__ == "__main__":
    for N in [12,8]:
        print(json.dumps({
            "N": N,
            "Weil_G_N": compare_readout(N, None, None, "Weil_G_N"),
            "Weil_F_N": compare_readout(N, None, None, "Weil_F_N"),
        }, indent=2))
