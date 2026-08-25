"""
Orthad state input adapter for VDM-RT listen-only probes.

Mandate:
The harness never scans, never reduces the graph, never asks the engine a question.
It only feeds structured Orthad state and listens to walker announcements.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, List, Sequence, Set


@dataclass(frozen=True)
class OrthadFrame:
    tick: int
    visible: tuple[str, ...]
    retained: tuple[str, ...]

    def symbols(self) -> tuple[str, ...]:
        return tuple(self.visible) + tuple(self.retained)


def _base_visible(tick: int) -> tuple[str, ...]:
    # Keep terminal projection intentionally small and shared across legal/illegal modes.
    return (
        "visible:charts:A-B-C",
        "visible:phase:terminal",
        "visible:projection:shared",
    )


def _retained_legal(tick: int) -> tuple[str, ...]:
    return (
        "retained:chart:A",
        "retained:chart:B",
        "retained:chart:C",
        "retained:overlap:AB:+",
        "retained:overlap:BC:+",
        "retained:overlap:CA:-",
        "retained:latch:L:stable",
        "retained:depth:stable",
    )


def _retained_legal_rewrite(tick: int) -> tuple[str, ...]:
    # Same retained content as legal, different order. The symbol-to-index sink is order-insensitive.
    base = list(_retained_legal(tick))
    return tuple(base[3:] + base[:3])


def _retained_flip(tick: int) -> tuple[str, ...]:
    # Same visible projection, retained CA orientation alternates.
    ca = "retained:overlap:CA:+" if (tick % 2 == 0) else "retained:overlap:CA:-"
    return (
        "retained:chart:A",
        "retained:chart:B",
        "retained:chart:C",
        "retained:overlap:AB:+",
        "retained:overlap:BC:+",
        ca,
        "retained:latch:L:stable",
        "retained:depth:stable",
    )


def _retained_cocycle_bad(tick: int) -> tuple[str, ...]:
    # No residual is supplied. The only difference is retained orientation on CA.
    return (
        "retained:chart:A",
        "retained:chart:B",
        "retained:chart:C",
        "retained:overlap:AB:+",
        "retained:overlap:BC:+",
        "retained:overlap:CA:+",
        "retained:latch:L:stable",
        "retained:depth:stable",
    )


def _retained_sparse_baseline(tick: int) -> tuple[str, ...]:
    return (
        "retained:chart:A",
        f"retained:latch:L:{tick % 3}",
    )


def make_frame(mode: str, tick: int) -> OrthadFrame:
    m = str(mode).strip().lower()
    visible = _base_visible(int(tick))
    if m == "legal":
        retained = _retained_legal(tick)
    elif m == "legal_rewrite":
        retained = _retained_legal_rewrite(tick)
    elif m == "illegal_flip":
        retained = _retained_flip(tick)
    elif m == "illegal_cocycle":
        retained = _retained_cocycle_bad(tick)
    elif m == "sparse_baseline":
        retained = _retained_sparse_baseline(tick)
    else:
        raise ValueError(f"unknown Orthad frame mode: {mode}")
    return OrthadFrame(tick=int(tick), visible=visible, retained=retained)


def make_sequence(mode: str, ticks: int) -> List[OrthadFrame]:
    return [make_frame(mode, t) for t in range(int(max(0, ticks)))]


def symbol_to_indices(symbol: str, N: int, group_size: int = 4, salt: str = "orthad-state-v1") -> List[int]:
    """Deterministic structured-state symbol mapping. No lexicon, decoder, or language model."""
    N = int(max(1, N))
    group_size = int(max(1, group_size))
    digest = blake2b((str(salt) + "|" + str(symbol)).encode("utf-8"), digest_size=16).digest()
    base = int.from_bytes(digest[:8], "little") % N
    stride = 1 + (int.from_bytes(digest[8:], "little") % max(1, N - 1)) if N > 1 else 1
    out = []
    cur = base
    for _ in range(group_size):
        out.append(int(cur % N))
        cur += stride
    return out


def frame_to_indices(frame: OrthadFrame, N: int, group_size: int = 4) -> List[int]:
    seen: Set[int] = set()
    for sym in frame.symbols():
        for idx in symbol_to_indices(sym, N=N, group_size=group_size):
            seen.add(int(idx))
    return sorted(seen)


__all__ = [
    "OrthadFrame",
    "make_frame",
    "make_sequence",
    "symbol_to_indices",
    "frame_to_indices",
]
