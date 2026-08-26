"""Discrete Heisenberg-group arithmetic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeisenbergState:
    """Discrete Heisenberg state ``(m, n, omega)``."""

    m: int
    n: int
    omega: int


# Historical alias retained for provenance/source parity.
HState = HeisenbergState


def multiply(left: HeisenbergState, right: HeisenbergState) -> HeisenbergState:
    """Product ``(m,n,r)(m',n',r')=(m+m',n+n',r+r'+m*n')``."""
    return HeisenbergState(
        left.m + right.m,
        left.n + right.n,
        left.omega + right.omega + left.m * right.n,
    )


def inverse(state: HeisenbergState) -> HeisenbergState:
    return HeisenbergState(-state.m, -state.n, -state.omega + state.m * state.n)


def commutator(first: HeisenbergState, second: HeisenbergState) -> HeisenbergState:
    return multiply(multiply(multiply(first, second), inverse(first)), inverse(second))


def visible(state: HeisenbergState) -> tuple[int, int]:
    return (state.m, state.n)


def order_charge(before: HeisenbergState, after: HeisenbergState) -> int:
    return after.omega - before.omega
