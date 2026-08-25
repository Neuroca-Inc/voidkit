#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple
import csv


SYMBOL_TO_DIGIT = {"S": 0, "R": 1, "T": 2}


def B_pair(u: int, v: int) -> Tuple[int, int]:
    """Balanced refinement law (u,v) -> sort(v,u+v)."""
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def seed_for_host(A: int) -> Tuple[int, int]:
    """
    Executable seed convention used in the current block-emission bundle:
    seed = (1, max(1,A)).
    """
    return (1, max(1, A))


def depth_to_floor(delta_floor: int, seed: Tuple[int, int]) -> Tuple[int, Tuple[int, int], int]:
    """Smallest n such that uv(B^n(seed)) >= delta_floor."""
    u, v = seed
    n = 0
    while u * v < delta_floor:
        u, v = B_pair(u, v)
        n += 1
    return n, (u, v), u * v


@dataclass(frozen=True)
class HostBlock:
    A: int
    W: int
    Delta: int
    seed_u: int
    seed_v: int
    depth_to_floor: int
    delta: int
    u: int
    v: int
    uv: int
    word: str
    coordinate: int

    def to_row(self) -> dict:
        return asdict(self)


def canonical_block_word(delta: int, W: int) -> str:
    """
    Canonical macro-word on the selector-closed executable branch:
        R^delta S^(W-1-delta) T
    """
    if W < 1:
        raise ValueError("W must be positive")
    if not (0 <= delta <= W - 1):
        raise ValueError(f"delta must lie in [0, W-1], got delta={delta}, W={W}")
    return "R" * delta + "S" * (W - 1 - delta) + "T"


def execution_coordinate(word: str) -> int:
    """
    Base-3 execution coordinate with least-significant symbol first:
        S -> 0, R -> 1, T -> 2.
    """
    total = 0
    for k, ch in enumerate(word):
        total += SYMBOL_TO_DIGIT[ch] * (3 ** k)
    return total


def execution_coordinate_closed(delta: int, W: int) -> int:
    """
    Exact closed form for E(R^delta S^(W-1-delta) T).
    """
    return (3 ** delta - 1) // 2 + 2 * (3 ** (W - 1))


def host_block(A: int, W: int, Delta: int) -> HostBlock:
    seed = seed_for_host(A)
    d, pair, uv = depth_to_floor(Delta, seed)
    delta = min(d, W - 1)
    word = canonical_block_word(delta, W)
    coord = execution_coordinate(word)
    closed = execution_coordinate_closed(delta, W)
    if coord != closed:
        raise AssertionError(f"coordinate mismatch: {coord} != {closed}")
    return HostBlock(
        A=A,
        W=W,
        Delta=Delta,
        seed_u=seed[0],
        seed_v=seed[1],
        depth_to_floor=d,
        delta=delta,
        u=pair[0],
        v=pair[1],
        uv=uv,
        word=word,
        coordinate=coord,
    )


def block_table(W: int, Delta: int, hosts: int) -> List[HostBlock]:
    return [host_block(A, W, Delta) for A in range(hosts)]


def write_block_table(path: str, rows: Iterable[HostBlock]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].to_row().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def selector_symbol(t: int, uv: int, W: int, Delta: int) -> str:
    """
    Macro selector from the v11 manuscript:
        T if t == W-1 mod W
        R if uv < Delta
        S if uv >= Delta
    """
    if t % W == W - 1:
        return "T"
    return "R" if uv < Delta else "S"


def block_schedule_trace(A: int, W: int, Delta: int) -> List[str]:
    """
    Explicit selector trace for a single canonical block, starting at the seed state for host A.
    This returns exactly the canonical word if the selector law is implemented correctly.
    """
    u, v = seed_for_host(A)
    out: List[str] = []
    for t in range(W):
        uv = u * v
        sym = selector_symbol(t=t, uv=uv, W=W, Delta=Delta)
        out.append(sym)
        if sym == "R":
            u, v = B_pair(u, v)
        elif sym == "S":
            # same-host quarter continuation only; arithmetic pair unchanged
            pass
        elif sym == "T":
            # host lift ends the block; arithmetic pair unchanged in the macro-view
            pass
        else:
            raise ValueError(sym)
    return out


if __name__ == "__main__":
    rows = block_table(W=64, Delta=4096, hosts=12)
    write_block_table("native_block_table.csv", rows)
    print("wrote native_block_table.csv")
    print("first 4 blocks:")
    for row in rows[:4]:
        print(row)
