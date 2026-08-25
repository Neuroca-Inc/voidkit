from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Dict, List, Tuple

BASES: Tuple[str, ...] = ("b0", "b1", "b2", "b3")
RNA_NAMES: Dict[str, str] = {"b0": "u", "b1": "c", "b2": "a", "b3": "g"}
HOST_PARTITIONS: Dict[int, Tuple[Tuple[str, str], Tuple[str, str]]] = {
    0: (("b1", "b2"), ("b0", "b3")),
    1: (("b0", "b1"), ("b2", "b3")),
    2: (("b1", "b3"), ("b0", "b2")),
}
IUPAC_NAMES: Dict[int, Tuple[str, str]] = {
    0: ("m", "k"),
    1: ("y", "r"),
    2: ("s", "w"),
}

@dataclass(frozen=True)
class LiftedState:
    A: int
    u: int
    v: int
    theta: Fraction
    kappa: int

    @property
    def A_mod(self) -> int:
        return self.A % 3


def edge_residual() -> Fraction:
    return Fraction(1, 24)


def normalized_edge_bonus() -> Fraction:
    return 24 * edge_residual()


def quarter_weight(base: str) -> Fraction:
    if base not in BASES:
        raise KeyError(base)
    return Fraction(1, 1) + (normalized_edge_bonus() if base == "b1" else Fraction(0, 1))


def total_weight_budget() -> Fraction:
    return sum(quarter_weight(base) for base in BASES)


def triplet_words() -> List[Tuple[str, str, str]]:
    return list(product(BASES, repeat=3))


def occurrences_per_base() -> Dict[str, int]:
    counts = {base: 0 for base in BASES}
    for word in triplet_words():
        for symbol in word:
            counts[symbol] += 1
    return counts


def total_acceptors() -> Fraction:
    counts = occurrences_per_base()
    return sum(Fraction(counts[base], 1) * quarter_weight(base) for base in BASES)


def centered_count(base: str) -> Fraction:
    total = Fraction(0, 1)
    for left in BASES:
        for right in BASES:
            total += quarter_weight(left) + quarter_weight(base) + quarter_weight(right)
    return total


def named_centered_counts() -> Dict[str, Fraction]:
    return {RNA_NAMES[base]: centered_count(base) for base in BASES}


def host_partition_counts(A: int) -> Dict[str, Fraction]:
    pos, neg = HOST_PARTITIONS[A % 3]
    pos_name, neg_name = IUPAC_NAMES[A % 3]
    named = named_centered_counts()
    pos_total = sum(named[RNA_NAMES[b]] for b in pos)
    neg_total = sum(named[RNA_NAMES[b]] for b in neg)
    return {pos_name: pos_total, neg_name: neg_total}


def anchor_projection() -> Dict[str, object]:
    state = LiftedState(A=0, u=55, v=89, theta=Fraction(3, 10), kappa=0)
    pos, neg = HOST_PARTITIONS[state.A_mod]
    return {
        "A": state.A,
        "u": state.u,
        "v": state.v,
        "partition_named": list(IUPAC_NAMES[state.A_mod]),
        "partition_bases": [[RNA_NAMES[b] for b in pos], [RNA_NAMES[b] for b in neg]],
    }


def summary_payload() -> Dict[str, object]:
    return {
        "quarter_alphabet_size": len(BASES),
        "triplet_word_count": len(triplet_words()),
        "edge_residual": str(edge_residual()),
        "normalized_edge_bonus": str(normalized_edge_bonus()),
        "weights": {base: str(quarter_weight(base)) for base in BASES},
        "position_counts": occurrences_per_base(),
        "total_acceptors": str(total_acceptors()),
        "named_centered_counts": {k: str(v) for k, v in named_centered_counts().items()},
        "partition_counts": {str(A): {k: str(v) for k, v in host_partition_counts(A).items()} for A in range(3)},
        "anchor_projection": anchor_projection(),
    }

if __name__ == '__main__':
    import json
    print(json.dumps(summary_payload(), indent=2))
