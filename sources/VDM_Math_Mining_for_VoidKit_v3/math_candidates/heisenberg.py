from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HState:
    """Discrete Heisenberg state (m, n, omega)."""

    m: int
    n: int
    omega: int


def multiply(left: HState, right: HState) -> HState:
    """Heisenberg product: (m,n,r)(m',n',r') = (m+m',n+n',r+r'+m*n')."""
    return HState(
        left.m + right.m,
        left.n + right.n,
        left.omega + right.omega + left.m * right.n,
    )


def inverse(state: HState) -> HState:
    """Group inverse for the discrete Heisenberg product."""
    return HState(-state.m, -state.n, -state.omega + state.m * state.n)


def commutator(first: HState, second: HState) -> HState:
    """Lifted commutator first * second * first^-1 * second^-1."""
    return multiply(multiply(multiply(first, second), inverse(first)), inverse(second))


def visible(state: HState) -> tuple[int, int]:
    """Visible projection that forgets the order register."""
    return (state.m, state.n)


def order_charge(before: HState, after: HState) -> int:
    """Change in hidden order register."""
    return after.omega - before.omega
