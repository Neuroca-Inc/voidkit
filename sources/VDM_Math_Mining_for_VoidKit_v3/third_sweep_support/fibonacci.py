"""Fibonacci corridor primitives for inverse-word certificates."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class CorridorState:
    depth: int
    u: int
    v: int
    uv: int
    half_width: float
    word: str


def fibonacci_pair(depth: int) -> tuple[int, int]:
    """Return the balanced-corridor pair B^depth(1,1)."""
    if depth < 0:
        raise ValueError("depth must be non-negative")
    u, v = 1, 1
    for _ in range(depth):
        u, v = v, u + v
        if u > v:
            u, v = v, u
    return u, v


def corridor_state(depth: int) -> CorridorState:
    u, v = fibonacci_pair(depth)
    uv = u * v
    return CorridorState(
        depth=depth,
        u=u,
        v=v,
        uv=uv,
        half_width=pi / uv,
        word=f"B^{depth}",
    )


def minimum_depth_for_half_width(target: float) -> int:
    if target <= 0:
        raise ValueError("target must be positive")
    depth = 0
    while corridor_state(depth).half_width > target:
        depth += 1
    return depth
