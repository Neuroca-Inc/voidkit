#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Dict, Iterable, List, Tuple
import csv

SYMBOL_TO_DIGIT = {"S": 0, "R": 1, "T": 2}
DIGIT_ALPHABET = "0123456789ABCDEF"


def B_pair(u: int, v: int) -> Tuple[int, int]:
    """Balanced refinement law (u,v) -> sort(v,u+v)."""
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def seed_for_host(A: int) -> Tuple[int, int]:
    """Executable seed convention used on the selector-closed balanced-window branch."""
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

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def canonical_block_word(delta: int, W: int) -> str:
    """Canonical macro-word R^delta S^(W-1-delta) T."""
    if W < 1:
        raise ValueError("W must be positive")
    if not (0 <= delta <= W - 1):
        raise ValueError(f"delta must lie in [0, W-1], got delta={delta}, W={W}")
    return "R" * delta + "S" * (W - 1 - delta) + "T"


def execution_coordinate(word: str) -> int:
    """Exact base-3 execution coordinate E(U)=sum U_k 3^k, least-significant symbol first."""
    total = 0
    for k, ch in enumerate(word):
        total += SYMBOL_TO_DIGIT[ch] * (3 ** k)
    return total


def execution_coordinate_closed(delta: int, W: int) -> int:
    """Closed form for E(R^delta S^(W-1-delta) T)."""
    return (3 ** delta - 1) // 2 + 2 * (3 ** (W - 1))


def core_coordinate(delta: int) -> int:
    """Execution coordinate of R^delta S^... before the carry horizon symbol T."""
    return (3 ** delta - 1) // 2


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


def frac_part(x: Fraction) -> Fraction:
    return x - (x.numerator // x.denominator)


def to_base_digits_int(n: int, base: int) -> str:
    if not (2 <= base <= 16):
        raise ValueError("base must lie in [2,16]")
    if n == 0:
        return "0"
    s = ""
    x = abs(n)
    while x > 0:
        x, r = divmod(x, base)
        s = DIGIT_ALPHABET[r] + s
    return s


def frac_digits(frac: Fraction, base: int, m: int) -> str:
    """
    First m digits after the radix point of an exact rational observable.
    """
    if not (2 <= base <= 16):
        raise ValueError("base must lie in [2,16]")
    x = frac_part(Fraction(frac))
    out: List[str] = []
    for _ in range(m):
        x *= base
        d = x.numerator // x.denominator
        out.append(DIGIT_ALPHABET[d])
        x -= d
    return "".join(out)


def int_block(n: int, base: int, m: int, mode: str) -> str:
    """
    Standard-radix block read directly from an exact integer certificate.
    mode='msd' -> most-significant block
    mode='lsd' -> least-significant block
    """
    s = to_base_digits_int(n, base)
    if len(s) < m:
        s = s.rjust(m, "0")
    if mode == "msd":
        return s[:m]
    if mode == "lsd":
        return s[-m:]
    raise ValueError(f"unknown mode {mode!r}")


def observable_registry() -> Dict[str, callable]:
    """
    Exact local rational observables built only from the current host block.
    These are candidate local packet families for radix-block decoding.
    """
    return {
        "full_norm": lambda r: frac_part(Fraction(r.coordinate, 3 ** r.W)),
        "full_norm_plus_gap": lambda r: frac_part(Fraction(r.coordinate, 3 ** r.W) + Fraction(1, r.uv)),
        "full_norm_minus_gap": lambda r: frac_part(Fraction(r.coordinate, 3 ** r.W) - Fraction(1, r.uv)),
        "core_norm_W": lambda r: frac_part(Fraction(core_coordinate(r.delta), 3 ** (r.W - 1))),
        "core_norm_delta": lambda r: frac_part(Fraction(core_coordinate(r.delta), 3 ** max(1, r.delta))),
        "depth_norm": lambda r: frac_part(Fraction(r.delta, max(1, r.W - 1))),
        "inv_uv": lambda r: Fraction(1, r.uv),
        "u_over_v": lambda r: frac_part(Fraction(r.u, r.v)),
        "u_over_sum": lambda r: frac_part(Fraction(r.u, r.u + r.v)),
        "v_over_sum": lambda r: frac_part(Fraction(r.v, r.u + r.v)),
        "uv_over_Delta": lambda r: frac_part(Fraction(r.uv, r.Delta)),
        "Delta_over_uv": lambda r: frac_part(Fraction(r.Delta, r.uv)),
        "A_over_uv": lambda r: frac_part(Fraction(r.A + 1, r.uv)),
        "depth_plus_gap": lambda r: frac_part(Fraction(r.delta, max(1, r.W - 1)) + Fraction(1, r.uv)),
        "coord_mod_uv_norm": lambda r: frac_part(Fraction(r.coordinate % r.uv, r.uv)),
        "sum_over_uv": lambda r: frac_part(Fraction(r.u + r.v, r.uv)),
        "sum_over_Delta": lambda r: frac_part(Fraction(r.u + r.v, r.Delta)),
        "delta_over_uv": lambda r: frac_part(Fraction(r.delta, r.uv)),
    }


def integer_registry() -> Dict[str, callable]:
    return {
        "coord": lambda r: r.coordinate,
        "core_coord": lambda r: core_coordinate(r.delta),
        "uv": lambda r: r.uv,
        "sum": lambda r: r.u + r.v,
        "u": lambda r: r.u,
        "v": lambda r: r.v,
        "delta": lambda r: r.delta,
    }


def direct_fraction_block(row: HostBlock, family: str, base: int, m: int) -> str:
    obs = observable_registry()[family]
    return frac_digits(obs(row), base, m)


def stateful_fraction_blocks(rows: List[HostBlock], family: str, base: int, m: int, a: int) -> List[str]:
    """
    A minimal native stateful decoder family:
        y_{n+1} = frac(a * y_n + packet_n),
        emit the next m radix digits of y_{n+1}.
    This is intentionally exact and local; it does not claim to be the solved pi decoder.
    """
    if a < 0:
        raise ValueError("a must be nonnegative")
    obs = observable_registry()[family]
    y = Fraction(0, 1)
    out: List[str] = []
    for row in rows:
        y = frac_part(Fraction(a, 1) * y + obs(row))
        out.append(frac_digits(y, base, m))
    return out


def integer_block(row: HostBlock, family: str, base: int, m: int, mode: str) -> str:
    fn = integer_registry()[family]
    return int_block(fn(row), base, m, mode)


def write_rows(path: str, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    rows = block_table(W=64, Delta=4096, hosts=8)
    for row in rows[:4]:
        print(row)
        print("  full_norm dec8:", direct_fraction_block(row, "full_norm", 10, 8))
        print("  full_norm hex8:", direct_fraction_block(row, "full_norm", 16, 8))
        print("  coord lsd hex8:", integer_block(row, "coord", 16, 8, "lsd"))
