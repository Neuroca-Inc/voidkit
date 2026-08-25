from __future__ import annotations

from enum import Enum

from pcsr.domain.lifted_state import XiHat


class PCOp(str, Enum):
    Q = "Q"
    B = "B"
    L = "L"


PRIMITIVE_OPS: tuple[PCOp, PCOp, PCOp] = (PCOp.Q, PCOp.B, PCOp.L)


def apply_op(state: XiHat, op: PCOp) -> XiHat:
    if op is PCOp.Q:
        return state.Q()
    if op is PCOp.B:
        return state.B()
    if op is PCOp.L:
        return state.L()
    raise ValueError(f"Unknown Phase Calculus operator: {op!r}")


def parse_op(token: str) -> PCOp:
    try:
        return PCOp(token.upper())
    except ValueError as exc:
        raise ValueError("Operator word may contain only Q, B, and L.") from exc
