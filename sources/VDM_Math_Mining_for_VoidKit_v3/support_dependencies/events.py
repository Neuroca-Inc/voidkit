from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import FrozenSet, Literal, Tuple

Chart = Literal["A", "B", "C"]
Kind = Literal["Q", "B", "L", "P"]

CHARTS: tuple[Chart, ...] = ("A", "B", "C")
INDEX = {"A": 0, "B": 1, "C": 2}

@dataclass(frozen=True)
class Event:
    kind: Kind
    support: FrozenSet[Chart]
    sign: int = 1
    tag: str = ""

    def token(self) -> str:
        s = "".join(sorted(self.support))
        return f"{self.kind}{s}{'+' if self.sign >= 0 else '-'}{self.tag}"

    def depends_on(self, other: "Event") -> bool:
        if self.kind == "P" or other.kind == "P":
            return True
        return bool(self.support & other.support)


def Q(a: Chart, tag: str = "") -> Event:
    return Event("Q", frozenset([a]), +1, tag)


def B(a: Chart, tag: str = "") -> Event:
    return Event("B", frozenset([a]), +1, tag)


def L(a: Chart, b: Chart, sign: int = 1, tag: str = "") -> Event:
    return Event("L", frozenset([a, b]), 1 if sign >= 0 else -1, tag)


def P(tag: str = "") -> Event:
    return Event("P", frozenset(CHARTS), +1, tag)


def modulus_for(dims: dict[Chart, int], pair: Tuple[Chart, Chart]) -> int:
    return lcm(dims[pair[0]], dims[pair[1]])
