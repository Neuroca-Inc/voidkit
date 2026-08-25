#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

delta, W, k = sp.symbols('delta W k', integer=True, nonnegative=True)

geom = sp.summation(3**k, (k, 0, delta - 1))
closed = sp.simplify(geom + 2 * 3**(W - 1))

print("Symbolic block-coordinate derivation")
print("----------------------------------")
print("sum_{k=0}^{delta-1} 3^k + 2*3^(W-1) =")
print(sp.simplify(closed))
print()

# Exact current-regime checks
W0 = 64
delta0 = 9
delta1 = 8

coord0 = sp.Integer((3**delta0 - 1)//2 + 2 * 3**(W0 - 1))
coord1 = sp.Integer((3**delta1 - 1)//2 + 2 * 3**(W0 - 1))

print(f"W={W0}, delta={delta0}")
print(coord0)
print()
print(f"W={W0}, delta={delta1}")
print(coord1)
print()

# Balanced ready-depth checks
def B(u: int, v: int) -> tuple[int, int]:
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)

def depth_to_floor(Delta: int, seed: tuple[int, int]) -> tuple[int, tuple[int, int], int]:
    u, v = seed
    n = 0
    while u * v < Delta:
        u, v = B(u, v)
        n += 1
    return n, (u, v), u * v

for seed in [(1,1), (1,2)]:
    d, pair, uv = depth_to_floor(4096, seed)
    print(f"seed={seed} -> depth={d}, pair={pair}, uv={uv}")
