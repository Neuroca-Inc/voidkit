from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import lcm
from typing import Dict, Iterable, Tuple

from .events import CHARTS, INDEX, Chart, Event

Pair = Tuple[Chart, Chart]

def canon_pair(pair: Iterable[Chart]) -> Pair:
    a, b = sorted(pair)
    return (a, b)  # type: ignore

@dataclass(frozen=True)
class CouplingReport:
    C: tuple[tuple[int, ...], ...]
    pair_values: dict[str, int]
    holonomy: int
    cocycle_residual: int
    gauge_signature: str
    visible_projection: int
    admitted: bool


def extract_coupling(events: Iterable[Event], dims: Dict[Chart, int]) -> CouplingReport:
    """Extract a pairwise native coupling candidate from retained latch history.

    The extractor is deliberately close to the Phase 5 v7f shape:
      c_ij += sign_b * (q_i + 1) * (q_j + 1) mod lcm(D_i,D_j)

    Here q_i is carried by retained Q/B local history. L events are boundary/latch events.
    Projection is a lossy visible checksum and is intentionally not enough to recover C.
    """
    q = {a: 0 for a in CHARTS}
    pair_values: Dict[Pair, int] = {p: 0 for p in combinations(CHARTS, 2)}
    visible = 0
    for ev in events:
        if ev.kind in {"Q", "B"}:
            a = next(iter(ev.support))
            q[a] = (q[a] + (1 if ev.kind == "Q" else 2)) % dims[a]
        elif ev.kind == "L":
            a, b = canon_pair(ev.support)
            m = lcm(dims[a], dims[b])
            pair_values[(a, b)] = (pair_values[(a, b)] + ev.sign * (q[a] + 1) * (q[b] + 1)) % m
        elif ev.kind == "P":
            visible = (sum(q.values()) + sum(pair_values.values())) % lcm(*dims.values())
    holonomy = (pair_values[("A", "B")] + pair_values[("B", "C")] + pair_values[("A", "C")]) % lcm(*dims.values())
    cocycle_residual = holonomy
    admitted = cocycle_residual == 0
    n = len(CHARTS)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for (a, b), val in pair_values.items():
        i, j = INDEX[a], INDEX[b]
        C[i][j] = C[j][i] = val
    pv = {"".join(k): v for k, v in sorted(pair_values.items())}
    gauge_signature = "admitted:" + ";".join(f"{k}={v}" for k, v in pv.items()) if admitted else "rejected:cocycle_residual=" + str(cocycle_residual)
    return CouplingReport(tuple(tuple(r) for r in C), pv, holonomy, cocycle_residual, gauge_signature, visible, admitted)


def matrix_signature(C: tuple[tuple[int, ...], ...]) -> str:
    return ";".join(",".join(str(x) for x in row) for row in C)
