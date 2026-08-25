#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple
import argparse
import csv

PI_PREFIX = "3141592653589793238462643383279502884197169399375105820974944592"

SYMBOL_TO_DIGIT = {"S": 0, "R": 1, "T": 2}


def B(u: int, v: int) -> Tuple[int, int]:
    """Balanced refinement law (u,v) -> sort(v,u+v)."""
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def depth_to_floor(delta_floor: int, seed: Tuple[int, int]) -> Tuple[int, Tuple[int, int], int]:
    """Smallest n such that uv(B^n(seed)) >= delta_floor."""
    u, v = seed
    n = 0
    while u * v < delta_floor:
        u, v = B(u, v)
        n += 1
    return n, (u, v), u * v


def seed_for_A(A: int) -> Tuple[int, int]:
    return (1, max(1, A))


def truncated_depth(W: int, Delta: int, A: int) -> Tuple[int, Tuple[int, int], int]:
    d, pair, uv = depth_to_floor(Delta, seed_for_A(A))
    return min(d, W - 1), pair, uv


def canonical_block_word(delta: int, W: int) -> str:
    if not (0 <= delta <= W - 1):
        raise ValueError(f"delta must lie in [0, W-1], got delta={delta}, W={W}")
    return "R" * delta + "S" * (W - 1 - delta) + "T"


def execution_coordinate(word: str) -> int:
    total = 0
    for k, ch in enumerate(word):
        total += SYMBOL_TO_DIGIT[ch] * (3 ** k)
    return total


def execution_coordinate_closed(delta: int, W: int) -> int:
    return (3 ** delta - 1) // 2 + 2 * (3 ** (W - 1))


@dataclass(frozen=True)
class BlockInfo:
    A: int
    W: int
    Delta: int
    seed_u: int
    seed_v: int
    depth: int
    trunc_depth: int
    u: int
    v: int
    uv: int
    word: str
    coordinate: int

    @property
    def depth_digit_candidate(self) -> int:
        return self.trunc_depth

    @property
    def coordinate_mod_10(self) -> int:
        return self.coordinate % 10


def block_info(A: int, W: int, Delta: int) -> BlockInfo:
    d, pair, uv = depth_to_floor(Delta, seed_for_A(A))
    delta = min(d, W - 1)
    word = canonical_block_word(delta, W)
    coord = execution_coordinate(word)
    closed = execution_coordinate_closed(delta, W)
    if coord != closed:
        raise AssertionError(f"coordinate mismatch: direct={coord}, closed={closed}")
    su, sv = seed_for_A(A)
    return BlockInfo(
        A=A, W=W, Delta=Delta, seed_u=su, seed_v=sv,
        depth=d, trunc_depth=delta, u=pair[0], v=pair[1], uv=uv,
        word=word, coordinate=coord
    )


def emit_blocks(W: int, Delta: int, n_blocks: int) -> List[BlockInfo]:
    return [block_info(A, W, Delta) for A in range(n_blocks)]


def compare_naive_depth_to_pi(blocks: Iterable[BlockInfo], pi_prefix: str = PI_PREFIX) -> List[Tuple[int, int, int]]:
    out = []
    for i, blk in enumerate(blocks):
        if i >= len(pi_prefix):
            break
        out.append((blk.A, blk.trunc_depth, int(pi_prefix[i])))
    return out


def write_blocks_csv(path: str, blocks: Iterable[BlockInfo]) -> None:
    rows = [asdict(b) for b in blocks]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact Phase Calculus native block emitter for the selector-closed executable layer.")
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--Delta", type=int, default=4096)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--csv", type=str, default="first_blocks.csv")
    args = ap.parse_args()

    blocks = emit_blocks(args.W, args.Delta, args.blocks)
    write_blocks_csv(args.csv, blocks)

    print(f"W={args.W}, Delta={args.Delta}, blocks={args.blocks}")
    print()
    for blk in blocks[:4]:
        print(f"A={blk.A:2d}  seed=({blk.seed_u},{blk.seed_v})  depth={blk.depth}  pair=({blk.u},{blk.v})  uv={blk.uv}")
        print(f"      word={blk.word[:min(20,len(blk.word))]}... len={len(blk.word)}")
        print(f"      coordinate={blk.coordinate}")
    print()
    print("Naive step-count-as-digit check against decimal pi prefix:")
    for A, depth_digit, pi_digit in compare_naive_depth_to_pi(blocks):
        print(f"A={A:2d}  emitted_depth={depth_digit}  pi_digit={pi_digit}")

    print()
    blk0 = blocks[0]
    print("Canonical exact block:")
    print(f"delta={blk0.trunc_depth}, word={blk0.word}, coordinate={blk0.coordinate}")


if __name__ == "__main__":
    main()
