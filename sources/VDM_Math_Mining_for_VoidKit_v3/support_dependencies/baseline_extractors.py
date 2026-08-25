#!/usr/bin/env python3
from __future__ import annotations

from decimal import Decimal, getcontext
from functools import lru_cache
from typing import Dict, List
import math
import time


HEX_ALPHABET = "0123456789ABCDEF"


def bbp_hex_digit(n: int) -> str:
    """
    n-th hexadecimal digit of pi after the point (0-indexed) via BBP.
    Pure Python reference baseline for extractor benchmarking.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")

    def S(j: int, n: int) -> float:
        s = 0.0
        for k in range(n + 1):
            r = 8 * k + j
            s += pow(16, n - k, r) / r
            s -= math.floor(s)
        t = 0.0
        k = n + 1
        while True:
            new = (16.0 ** (n - k)) / (8 * k + j)
            t += new
            if abs(new) < 1e-17:
                break
            k += 1
        return s + t

    x = 4.0 * S(1, n) - 2.0 * S(4, n) - S(5, n) - S(6, n)
    x = x - math.floor(x)
    digit = int(16 * x)
    return HEX_ALPHABET[digit]


def bbp_hex_digits(start: int, count: int) -> str:
    return "".join(bbp_hex_digit(start + i) for i in range(count))


def chudnovsky_pi_str(digits: int) -> str:
    """
    Decimal prefix baseline using the Chudnovsky series.
    Returns a plain string without decimal point, e.g. digits=20 -> '314159...'
    """
    if digits < 2:
        raise ValueError("digits must be at least 2")

    extra = 20
    getcontext().prec = digits + extra

    C = 426880 * Decimal(10005).sqrt()
    M = 1
    L = 13591409
    X = 1
    K = 6
    S = Decimal(L)

    terms = digits // 14 + 2
    for k in range(1, terms):
        M = M * (K**3 - 16 * K) // (k**3)
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / Decimal(X)
        K += 12

    pi = C / S
    s = format(pi, f".{digits}f").replace(".", "")
    return s[:digits + 1]  # include leading integer digit plus requested fractional digits


def pi_hex_plain(count_total: int) -> str:
    """
    Returns hexadecimal digits of pi without a point, starting with the leading integer '3'.
    count_total counts all digits including the leading 3.
    """
    if count_total < 1:
        raise ValueError("count_total must be positive")
    if count_total == 1:
        return "3"
    return "3" + bbp_hex_digits(0, count_total - 1)


def run_smoke_benchmarks() -> Dict[str, object]:
    out: Dict[str, object] = {}

    t0 = time.perf_counter()
    dec_1000 = chudnovsky_pi_str(1000)
    t1 = time.perf_counter()
    out["decimal_prefix_1000"] = {
        "seconds": t1 - t0,
        "prefix_32": dec_1000[:32],
        "total_digits_including_leading": len(dec_1000),
    }

    cases = []
    for start in [100, 1000]:
        t0 = time.perf_counter()
        hx = bbp_hex_digits(start, 16)
        t1 = time.perf_counter()
        cases.append({
            "start_after_point": start,
            "count": 16,
            "seconds": t1 - t0,
            "hex": hx,
        })
    out["hex_extractor_cases"] = cases
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_smoke_benchmarks(), indent=2))
