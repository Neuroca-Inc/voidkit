"""Deterministic real interval solvers used by the certifier."""
from __future__ import annotations

import math
from collections.abc import Callable


def bisect_real(
    func: Callable[[float], float],
    left: float,
    right: float,
    *,
    tolerance: float,
    max_steps: int = 400,
) -> tuple[float, float, float, int]:
    """Return midpoint, left, right, steps for a certified sign-changing bracket."""
    if not left < right:
        raise ValueError("left must be smaller than right")
    fl = func(left)
    fr = func(right)
    if fl == 0.0:
        return left, left, left, 0
    if fr == 0.0:
        return right, right, right, 0
    if fl * fr > 0:
        raise ValueError(f"bracket does not change sign: f(left)={fl}, f(right)={fr}")
    steps = 0
    mid = (left + right) / 2.0
    while steps < max_steps and (right - left) / 2.0 > tolerance:
        mid = (left + right) / 2.0
        fm = func(mid)
        if fm == 0.0:
            left = right = mid
            break
        if fl * fm <= 0:
            right = mid
            fr = fm
        else:
            left = mid
            fl = fm
        steps += 1
    mid = (left + right) / 2.0
    return mid, left, right, steps


def lambertw_real_bracket(y: float, branch: int) -> tuple[float, float]:
    """Return a real branch bracket for x*exp(x)=y when available."""
    lower_branch_point = -1.0 / math.e
    if branch == 0 and y >= 0:
        right = max(1.0, math.log1p(y) + 1.0, y + 1.0)
        return 0.0, right
    if lower_branch_point < y < 0 and branch == 0:
        return -1.0, 0.0
    if lower_branch_point < y < 0 and branch == -1:
        left = min(-2.0, -math.log(-y) - 2.0)
        while left * math.exp(left) - y <= 0:
            left *= 2.0
        return left, -1.0
    raise ValueError("real Lambert W branch requires branch=0 with y>=-1/e or branch=-1 with -1/e<y<0")


def x_plus_sin_bracket(y: float) -> tuple[float, float]:
    return y - 2.0, y + 2.0
