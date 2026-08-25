"""Domain models for retained inverse-fiber certificates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BranchMemory:
    sheet: str
    history: list[str]
    kappa: int = 0


@dataclass(frozen=True)
class InverseFiberState:
    equation: str
    y: str
    x: str
    residual_abs: float
    tolerance: float
    branch_memory: BranchMemory
    corridor: dict[str, Any]
    inverse_word: list[str]
    interval: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = "PASS" if self.residual_abs <= self.tolerance else "FAIL"
        data["fiber_law"] = "Fib_Pi_f(y) = {X : Pi_f(X) = y}"
        return data
