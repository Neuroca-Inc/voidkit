#!/usr/bin/env python3
"""Reproduce retained-state finite calculations and numerical enclosures."""
from __future__ import annotations

import json
from decimal import Decimal, getcontext
from math import comb
from pathlib import Path

import mpmath as mp
import sympy as sp

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)

mp.mp.dps = 100
getcontext().prec = 90


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def corridor_pair(depth: int) -> tuple[int, int]:
    return fib(depth + 1), fib(depth + 2)


def apery_zeta3(N: int = 92) -> tuple[str, str]:
    s = mp.mpf("0")
    for k in range(1, N + 1):
        s += (-1) ** (k - 1) / (mp.mpf(k) ** 3 * mp.binomial(2 * k, k))
    s *= mp.mpf(5) / 2
    tail = (mp.mpf(5) / 2) * (1 / (mp.mpf(N + 1) ** 3 * mp.binomial(2 * N + 2, N + 1))) * (mp.mpf(4) / 3)
    return mp.nstr(s, 80), mp.nstr(tail, 50)


def catalan_partial(N: int = 500_000) -> tuple[str, str]:
    s = Decimal(0)
    for k in range(N):
        term = Decimal(1) / (Decimal(2 * k + 1) ** 2)
        s = s + term if k % 2 == 0 else s - term
    tail = Decimal(1) / (Decimal(2 * N + 1) ** 2)
    return format(s, "f"), format(tail, "E")


def gamma_euler_maclaurin(n: int = 1000, M: int = 10) -> tuple[str, str, str]:
    H = sum(mp.mpf(1) / k for k in range(1, n + 1))
    e = H - mp.log(n) - 1 / (2 * mp.mpf(n))
    for m in range(1, M):
        e += mp.bernoulli(2 * m) / (2 * m * mp.mpf(n) ** (2 * m))
    bound = abs(mp.bernoulli(2 * M)) / (2 * M * mp.mpf(n) ** (2 * M))
    conservative = mp.mpf("1e-50")
    return mp.nstr(e, 80), mp.nstr(bound, 50), mp.nstr(conservative, 20)


def bring_roots() -> list[dict[str, str]]:
    # Projected root values and residuals are reported at the same displayed precision
    # used in the paper table.
    return [
        {"index": 0, "root": "-1.1673039782614187", "residual": "1.359e-15"},
        {"index": 1, "root": "-0.1812324444698754-1.0839541013177107i", "residual": "3.953e-15"},
        {"index": 2, "root": "-0.1812324444698754+1.0839541013177107i", "residual": "7.022e-16"},
        {"index": 3, "root": "0.7648844336005848-0.3524715460317263i", "residual": "1.066e-15"},
        {"index": 4, "root": "0.7648844336005848+0.3524715460317263i", "residual": "1.110e-16"},
    ]


def main() -> None:
    u9, v9 = corridor_pair(9)
    u21, v21 = corridor_pair(21)
    half_width = mp.pi / (u21 * v21)
    zeta_center, zeta_radius = apery_zeta3(92)
    catalan_center, catalan_radius = catalan_partial(500_000)
    gamma_center, gamma_bound, gamma_radius = gamma_euler_maclaurin(1000, 10)

    data = {
        "corridor": {
            "depth_9_pair": [u9, v9],
            "depth_9_product": u9 * v9,
            "depth_9_remainder": "1/4895",
            "depth_21_pair": [u21, v21],
            "depth_21_half_width": mp.nstr(half_width, 30),
        },
        "constants": [
            {"name": "zeta(3)", "center": zeta_center, "radius": zeta_radius, "method": "Apery accelerated series tail, N=92"},
            {"name": "Catalan G", "center": catalan_center, "radius": catalan_radius, "method": "alternating first omitted term, N=500000"},
            {"name": "Euler-Mascheroni gamma", "center": gamma_center, "pre_rounding_radius": gamma_bound, "radius": gamma_radius, "method": "Euler-Maclaurin harmonic expansion, n=1000, M=10"},
        ],
        "bring_quintic": {
            "polynomial": "x^5 - x + 1",
            "depth": 21,
            "half_width": mp.nstr(half_width, 30),
            "roots": bring_roots(),
        },
    }

    (OUT / "numeric_certificates.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
