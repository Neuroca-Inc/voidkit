from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ResolutionCertificate:
    depth: int
    u: int
    v: int
    uv: int
    half_width: float
    exact_remainder: Fraction


def balanced_step(u: int, v: int) -> tuple[int, int]:
    a, b = sorted((v, u + v))
    return int(a), int(b)


def balanced_pair(depth: int, seed: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    u, v = seed
    if u <= 0 or v <= 0:
        raise ValueError("seed must be positive")
    u, v = sorted((u, v))
    for _ in range(depth):
        u, v = balanced_step(u, v)
    return u, v


def certificate_for_depth(depth: int) -> ResolutionCertificate:
    u, v = balanced_pair(depth)
    uv = u * v
    return ResolutionCertificate(depth, u, v, uv, math.pi / uv, Fraction(1, uv))


def depth_for_floor(delta: float) -> ResolutionCertificate:
    if delta <= 0:
        raise ValueError("resolution floor must be positive")
    depth = 0
    while True:
        cert = certificate_for_depth(depth)
        if cert.half_width <= delta:
            return cert
        depth += 1
