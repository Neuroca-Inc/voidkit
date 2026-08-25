from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Permutation:
    """Finite sheet permutation with images[i] = image of sheet i."""
    images: Tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.images)
        if sorted(self.images) != list(range(n)):
            raise ValueError(f"images must be a permutation of 0..{n - 1}: {self.images}")

    @classmethod
    def identity(cls, n: int) -> "Permutation":
        if n <= 0:
            raise ValueError("degree must be positive")
        return cls(tuple(range(n)))

    @classmethod
    def from_iterable(cls, values: Iterable[int]) -> "Permutation":
        return cls(tuple(int(v) for v in values))

    @property
    def degree(self) -> int:
        return len(self.images)

    def apply(self, sheet: int) -> int:
        if sheet < 0 or sheet >= self.degree:
            raise ValueError(f"sheet {sheet} outside 0..{self.degree - 1}")
        return self.images[sheet]

    def compose(self, other: "Permutation") -> "Permutation":
        self._check(other)
        return Permutation(tuple(self.images[other.images[i]] for i in range(self.degree)))

    def inverse(self) -> "Permutation":
        inv = [0] * self.degree
        for i, image in enumerate(self.images):
            inv[image] = i
        return Permutation(tuple(inv))

    def is_identity(self) -> bool:
        return self.images == tuple(range(self.degree))

    def cycles(self) -> Tuple[Tuple[int, ...], ...]:
        seen = [False] * self.degree
        cycles = []
        for i in range(self.degree):
            if seen[i]:
                continue
            cur = i
            cyc = []
            while not seen[cur]:
                seen[cur] = True
                cyc.append(cur)
                cur = self.apply(cur)
            if len(cyc) > 1:
                cycles.append(tuple(cyc))
        return tuple(cycles)

    def _check(self, other: "Permutation") -> None:
        if self.degree != other.degree:
            raise ValueError(f"degree mismatch: {self.degree} != {other.degree}")
