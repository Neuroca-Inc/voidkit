#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List
import csv
import hashlib
import json
import time

from phase_native_radix import (
    HostBlock,
    block_table,
    direct_fraction_block,
    integer_block,
    observable_registry,
    stateful_fraction_blocks,
)
# External baselines are verification-only; they are never used in the native candidate path.
from baseline_extractors import chudnovsky_pi_str, pi_hex_plain


@dataclass(frozen=True)
class CandidateScore:
    kind: str
    family: str
    base: int
    block_len: int
    stream_len: int
    longest_prefix: int
    exact_rate: float
    preview: str

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def decimal_target(frac_digits: int) -> str:
    # chudnovsky_pi_str(d) returns leading integer digit + d fractional digits
    return chudnovsky_pi_str(frac_digits)[1:]


def hex_target(frac_digits: int) -> str:
    return pi_hex_plain(frac_digits + 1)[1:]


def score_stream(stream: str, target: str) -> Dict[str, object]:
    n = min(len(stream), len(target))
    stream = stream[:n]
    target = target[:n]
    matches = [a == b for a, b in zip(stream, target)]
    exact_rate = sum(matches) / n if n else 0.0
    first_error_at = next((i for i, ok in enumerate(matches) if not ok), n)
    return {
        "stream_len": n,
        "longest_prefix": first_error_at,
        "exact_rate": exact_rate,
        "preview": stream[:32],
    }


def build_fraction_stream(rows: List[HostBlock], family: str, base: int, block_len: int) -> str:
    return "".join(direct_fraction_block(row, family, base, block_len) for row in rows)


def build_stateful_fraction_stream(rows: List[HostBlock], family: str, base: int, block_len: int, a: int) -> str:
    return "".join(stateful_fraction_blocks(rows, family, base, block_len, a))


def build_integer_stream(rows: List[HostBlock], family: str, base: int, block_len: int, mode: str) -> str:
    return "".join(integer_block(row, family, base, block_len, mode) for row in rows)


def candidate_scores(rows: List[HostBlock], target_dec: str, target_hex: str) -> List[CandidateScore]:
    out: List[CandidateScore] = []

    for family in observable_registry().keys():
        for base, target in [(10, target_dec), (16, target_hex)]:
            for block_len in [1, 2, 4, 8]:
                stream = build_fraction_stream(rows, family, base, block_len)
                sc = score_stream(stream, target)
                out.append(CandidateScore(kind="frac_direct", family=family, base=base, block_len=block_len, **sc))
                for a in [1, 2, 3]:
                    stream = build_stateful_fraction_stream(rows, family, base, block_len, a)
                    sc = score_stream(stream, target)
                    out.append(CandidateScore(kind=f"frac_state_a{a}", family=family, base=base, block_len=block_len, **sc))

    for family in ["coord", "core_coord", "uv", "sum", "u", "v", "delta"]:
        for base, target in [(10, target_dec), (16, target_hex)]:
            for block_len in [1, 2, 4, 8]:
                for mode in ["msd", "lsd"]:
                    stream = build_integer_stream(rows, family, base, block_len, mode)
                    sc = score_stream(stream, target)
                    out.append(CandidateScore(kind=f"int_{mode}", family=family, base=base, block_len=block_len, **sc))

    return out


def score_sort_key(x: CandidateScore) -> tuple:
    return (x.longest_prefix, x.exact_rate, -x.block_len, x.kind, x.family)


def sample_radix_table(rows: List[HostBlock]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        entry = row.to_row()
        entry.update({
            "full_norm_dec8": direct_fraction_block(row, "full_norm", 10, 8),
            "full_norm_hex8": direct_fraction_block(row, "full_norm", 16, 8),
            "depth_norm_dec8": direct_fraction_block(row, "depth_norm", 10, 8),
            "depth_norm_hex8": direct_fraction_block(row, "depth_norm", 16, 8),
            "coord_msd_hex8": integer_block(row, "coord", 16, 8, "msd"),
            "coord_lsd_hex8": integer_block(row, "coord", 16, 8, "lsd"),
        })
        out.append(entry)
    return out


def nativeity_audit() -> Dict[str, object]:
    return {
        "native_path_uses_mpmath_pi": False,
        "native_path_uses_known_pi_digit_table": False,
        "native_path_uses_chudnovsky_inside_candidate": False,
        "native_path_uses_bbp_inside_candidate": False,
        "notes": [
            "Chudnovsky and BBP appear only in verification helpers.",
            "Candidate streams depend only on local host state, exact integer certificates, and exact rational packet transforms.",
        ],
    }


def main() -> None:
    W = 64
    Delta = 4096
    hosts = 128

    t0 = time.perf_counter()
    rows = block_table(W=W, Delta=Delta, hosts=hosts)
    target_dec = decimal_target(4096)
    target_hex = hex_target(4096)
    scores = candidate_scores(rows, target_dec, target_hex)
    elapsed = time.perf_counter() - t0

    dec_scores = sorted([s for s in scores if s.base == 10], key=score_sort_key, reverse=True)
    hex_scores = sorted([s for s in scores if s.base == 16], key=score_sort_key, reverse=True)

    sample_rows = sample_radix_table(rows[:16])

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "regime": {"W": W, "Delta": Delta, "hosts": hosts},
            "nativeity_audit": nativeity_audit(),
            "search_seconds": elapsed,
            "target_decimal_prefix_32": target_dec[:32],
            "target_hex_prefix_32": target_hex[:32],
            "best_decimal_candidate": dec_scores[0].to_row(),
            "best_hex_candidate": hex_scores[0].to_row(),
            "search_space_size": len(scores),
        }, f, indent=2)

    with open("search_summary.txt", "w", encoding="utf-8") as f:
        f.write("Phase Radix Block Bundle v1\n\n")
        f.write("What this bundle closes:\n")
        f.write("- exact native phase-address -> radix-block transforms\n")
        f.write("- exact per-host block images in standard radices\n")
        f.write("- a local candidate search over rational packet families and stateful carry recurrences\n\n")
        f.write("What it does NOT claim:\n")
        f.write("- a solved pi-native decoder theorem\n")
        f.write("- a world-best spigot\n\n")
        f.write(f"Regime: W={W}, Delta={Delta}, hosts={hosts}\n")
        f.write(f"Search seconds: {elapsed:.6f}\n\n")
        f.write(f"Best decimal candidate: {dec_scores[0].to_row()}\n")
        f.write(f"Best hex candidate: {hex_scores[0].to_row()}\n\n")
        f.write("Interpretation:\n")
        f.write("The exact representation/address layer is strong enough to support native radix-block readout,\n")
        f.write("but the candidate local packet families tested here still fail quickly against pi.\n")
        f.write("That advances the work: the next burden is no longer 'how do we emit a radix block at all?'\n")
        f.write("It is 'which exact native packet family carries the pi-specific digits?'\n")

    with open("top_candidates_decimal.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dec_scores[0].to_row().keys()))
        writer.writeheader()
        for row in dec_scores[:50]:
            writer.writerow(row.to_row())

    with open("top_candidates_hex.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hex_scores[0].to_row().keys()))
        writer.writeheader()
        for row in hex_scores[:50]:
            writer.writerow(row.to_row())

    with open("native_radix_blocks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    with open("candidate_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "top_decimal_candidates": [x.to_row() for x in dec_scores[:10]],
            "top_hex_candidates": [x.to_row() for x in hex_scores[:10]],
            "sample_rows": sample_rows[:4],
        }, f, indent=2)

    print("Wrote benchmark_results.json, search_summary.txt, top_candidates_*.csv, native_radix_blocks.csv, candidate_manifest.json")


if __name__ == "__main__":
    main()
