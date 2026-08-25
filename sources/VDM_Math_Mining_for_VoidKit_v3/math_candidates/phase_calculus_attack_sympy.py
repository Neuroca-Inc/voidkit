from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import mpmath as mp
import sympy as sp


def B_pair(u: int, v: int) -> tuple[int, int]:
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def apply_B_pair(seed: tuple[int, int], n: int) -> tuple[int, int]:
    u, v = seed
    for _ in range(n):
        u, v = B_pair(u, v)
    return (u, v)


def dA(A: int, Delta: float) -> float:
    if Delta == math.inf:
        return math.inf
    u, v = 1, max(1, A)
    n = 0
    while u * v < Delta:
        u, v = B_pair(u, v)
        n += 1
        if n > 10000:
            raise RuntimeError("Depth exceeded search cap")
    return n


def qpoch_residual(uv: int, dps: int = 80) -> tuple[mp.mpf, mp.mpf]:
    mp.mp.dps = dps
    t = 2 * mp.pi / uv
    q = mp.e ** (-t)
    E = (mp.log(mp.qp(q, q, mp.inf)) + (mp.pi**2) / (6 * t) - mp.log(2 * mp.pi / t) / 2) / t
    return t, E


def shell_recursion_coeffs(max_n: int) -> tuple[List[int], Dict[tuple[int, int], int]]:
    Cdict: Dict[tuple[int, int], int] = {(0, 0): 1}
    for m in range(1, max_n + 1):
        Cdict[(0, m)] = 0
    max_r = int(math.isqrt(max_n)) + 3
    for r in range(0, max_r):
        step = r + 1
        for m in range(0, max_n + 1):
            total = 0
            for t in range(0, m // step + 1):
                total += ((-1) ** t) * (t + 1) * Cdict.get((r, m - t * step), 0)
            Cdict[(r + 1, m)] = total
    coeffs = []
    for n in range(0, max_n + 1):
        total = 0
        for r in range(0, int(math.isqrt(n)) + 3):
            if r * r <= n:
                total += Cdict.get((r, n - r * r), 0)
        coeffs.append(total)
    return coeffs, Cdict


def classical_mock_theta_coeffs(max_n: int) -> List[int]:
    q = sp.Symbol("q")
    expr = 0
    for r in range(0, max_n + 4):
        denom = sp.prod((1 + q**j) ** 2 for j in range(1, r + 1))
        expr += q ** (r * r) / denom
    poly = sp.series(expr, q, 0, max_n + 1).removeO().expand()
    return [int(poly.coeff(q, n)) for n in range(max_n + 1)]


def corrupted_mock_theta_coeffs(max_n: int) -> List[int]:
    q = sp.Symbol("q")
    expr = 0
    for r in range(0, max_n + 4):
        denom = sp.prod((1 + q**j) for j in range(1, r + 1))
        expr += q ** (r * r) / denom
    poly = sp.series(expr, q, 0, max_n + 1).removeO().expand()
    return [int(poly.coeff(q, n)) for n in range(max_n + 1)]


def main() -> None:
    anchor_from_A0 = apply_B_pair((1, 1), int(dA(0, 4096)))
    anchor_from_A2 = apply_B_pair((1, 2), int(dA(2, 4096)))
    _, E_4895 = qpoch_residual(4895, dps=80)

    coeffs, _ = shell_recursion_coeffs(12)
    classical = classical_mock_theta_coeffs(12)
    corrupted = corrupted_mock_theta_coeffs(12)

    result = {
        "anchor_from_A0": anchor_from_A0,
        "anchor_from_A2": anchor_from_A2,
        "anchor_uv": anchor_from_A0[0] * anchor_from_A0[1],
        "E_4895": str(E_4895),
        "E_minus_1_24": str(E_4895 - mp.mpf(1) / 24),
        "first_8_shell_coeffs": coeffs[:8],
        "first_8_classical_coeffs": classical[:8],
        "first_8_corrupted_coeffs": corrupted[:8],
        "coefficients_match_classical": coeffs == classical,
        "corrupted_matches_classical": corrupted == classical,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
