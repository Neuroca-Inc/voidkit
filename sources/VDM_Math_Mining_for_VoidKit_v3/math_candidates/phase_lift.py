from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import pi
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BalancedPair:
    """Balanced denominator-pair coordinate q=(u,v)."""

    u: int = 1
    v: int = 1

    def __post_init__(self) -> None:
        if self.u < 1 or self.v < 1:
            raise ValueError("BalancedPair requires positive integers")
        if self.u > self.v:
            object.__setattr__(self, "u", self.v)
            object.__setattr__(self, "v", self.u)

    @property
    def product(self) -> int:
        return self.u * self.v

    def refine(self) -> "BalancedPair":
        a, b = self.v, self.u + self.v
        return BalancedPair(min(a, b), max(a, b))

    def as_tuple(self) -> Tuple[int, int]:
        return (self.u, self.v)


@dataclass(frozen=True)
class CompletionGerm:
    """Centered completion germ c=[theta-pi/(uv), theta+pi/(uv)]."""

    theta_tick: int
    denominator: int

    @property
    def theta_pi_units(self) -> Fraction:
        return Fraction(self.theta_tick, 2)

    @property
    def left_pi_units(self) -> Fraction:
        return self.theta_pi_units - Fraction(1, self.denominator)

    @property
    def right_pi_units(self) -> Fraction:
        return self.theta_pi_units + Fraction(1, self.denominator)

    @property
    def width_pi_units(self) -> Fraction:
        return Fraction(2, self.denominator)

    @property
    def half_width(self) -> float:
        return pi / float(self.denominator)

    def as_tuple_pi_units(self) -> Tuple[Fraction, Fraction]:
        return (self.left_pi_units, self.right_pi_units)


@dataclass(frozen=True)
class PhaseCoordinates:
    """Standard Phase Calculus lifted coordinate block."""

    A: int = 0
    q: BalancedPair = field(default_factory=BalancedPair)
    theta_tick: int = 0
    kappa: int = 0
    c: CompletionGerm = field(default_factory=lambda: CompletionGerm(0, 1))

    @classmethod
    def initial(cls) -> "PhaseCoordinates":
        return cls().recomputed()

    @property
    def theta(self) -> float:
        return 0.5 * pi * self.theta_tick

    def visible_witness(self) -> complex:
        return 1j * np.exp(1j * self.theta)

    def recomputed(self) -> "PhaseCoordinates":
        return replace(
            self,
            kappa=self.theta_tick // 4,
            c=CompletionGerm(self.theta_tick, self.q.product),
        )

    def with_values(
        self,
        *,
        A: int | None = None,
        q: BalancedPair | None = None,
        theta_tick: int | None = None,
    ) -> "PhaseCoordinates":
        return PhaseCoordinates(
            A=self.A if A is None else A,
            q=self.q if q is None else q,
            theta_tick=self.theta_tick if theta_tick is None else theta_tick,
        ).recomputed()


@dataclass
class ExtendedLiftedState:
    """Full retained state: Phase coordinates plus VDM dynamical fields."""

    phase: PhaseCoordinates
    phi: np.ndarray
    psi: np.ndarray
    debt: np.ndarray
    kT: float
    walkers: int
    macro_step: int = 0
    projection_opened: bool = False

    @classmethod
    def zero(cls, dimension: int) -> "ExtendedLiftedState":
        z = np.zeros(dimension, dtype=float)
        return cls(
            phase=PhaseCoordinates.initial(),
            phi=z.copy(),
            psi=z.copy(),
            debt=z.copy(),
            kT=0.0,
            walkers=dimension,
        )

    def copy_with(self, **kwargs: object) -> "ExtendedLiftedState":
        data = {
            "phase": self.phase,
            "phi": self.phi.copy(),
            "psi": self.psi.copy(),
            "debt": self.debt.copy(),
            "kT": self.kT,
            "walkers": self.walkers,
            "macro_step": self.macro_step,
            "projection_opened": self.projection_opened,
        }
        data.update(kwargs)
        return ExtendedLiftedState(**data)
