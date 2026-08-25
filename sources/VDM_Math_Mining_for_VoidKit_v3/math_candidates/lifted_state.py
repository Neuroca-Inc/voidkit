from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True, slots=True)
class BalancedPair:
    u: int = 1
    v: int = 1

    def __post_init__(self) -> None:
        if self.u < 1 or self.v < 1:
            raise ValueError("BalancedPair entries must be positive.")
        if self.u > self.v:
            raise ValueError("BalancedPair requires u <= v.")

    @property
    def product(self) -> int:
        return self.u * self.v

    @property
    def remainder(self) -> Fraction:
        return Fraction(1, self.product)

    def refine(self) -> "BalancedPair":
        a, b = self.v, self.u + self.v
        return BalancedPair(a, b) if a <= b else BalancedPair(b, a)

    def to_jsonable(self) -> dict[str, int]:
        return {"u": self.u, "v": self.v, "uv": self.product}


@dataclass(frozen=True, slots=True)
class XiHat:
    """Full balanced lifted object XiHat=(A,q,theta,kappa,c).

    theta is stored as exact quarter-turn ticks: theta = theta_tick*pi/2.
    The completion germ c is computed from (q, theta), never stored as
    mutable external memory.
    """

    A: int = 0
    q: BalancedPair = BalancedPair()
    theta_tick: int = 0

    def __post_init__(self) -> None:
        if self.A < 0:
            raise ValueError("Host class A must be non-negative.")

    @property
    def theta(self) -> float:
        return self.theta_tick * math.pi / 2.0

    @property
    def kappa(self) -> int:
        return self.theta_tick // 4

    @property
    def c(self) -> tuple[float, float]:
        half_width = math.pi / self.q.product
        return (self.theta - half_width, self.theta + half_width)

    @property
    def half_width(self) -> float:
        return math.pi / self.q.product

    @property
    def phase_mod4(self) -> int:
        return self.theta_tick % 4

    def Q(self) -> "XiHat":
        return XiHat(A=self.A, q=self.q, theta_tick=self.theta_tick + 1)

    def B(self) -> "XiHat":
        return XiHat(A=self.A, q=self.q.refine(), theta_tick=self.theta_tick)

    def L(self) -> "XiHat":
        return XiHat(A=self.A + 1, q=self.q, theta_tick=self.theta_tick + 1)

    def visible_witness(self) -> complex:
        return 1j * cmath.exp(1j * self.theta)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "A": self.A,
            "q": self.q.to_jsonable(),
            "theta_tick": self.theta_tick,
            "theta": self.theta,
            "kappa": self.kappa,
            "c": list(self.c),
            "half_width": self.half_width,
            "phase_mod4": self.phase_mod4,
        }


SEED_STATE = XiHat()
