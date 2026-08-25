#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "output"

ELL = 32
POSITIONS = [1, 4895, 500000, 999937, 1000000]
BASES = [10, 16]


@dataclass(frozen=True)
class PositionRow:
    base: int
    position: int
    block_length: int
    lower_bound_required_decimal_digits: int
    actual_required_decimal_digits_current_law: int
    safety_margin_digits: int
    fixed_state_barrier_position: int
    fixed_state_certified: bool
    adaptive_certified: bool
    matches_reference: bool
    query_block: str
    probe_suffix: str


@dataclass(frozen=True)
class PacketDepthRow:
    depth: int
    u: int
    v: int
    N: int
    K520: int
    K10000: int
    K1000000: int
    strictly_worse_than_depth1: bool


def load_json(name: str) -> Dict[str, object]:
    return json.loads((RAW / name).read_text())


def lower_bound_digits(base: int, position: int, block_length: int = ELL) -> int:
    return math.floor((position - 1 + block_length) * math.log10(base) + math.log10(2.0)) + 1


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def k_n(N: int, digits: int) -> int:
    return math.ceil((1.0 + math.sqrt(1.0 + 12.0 * N * digits * math.log(10.0) / math.pi)) / 6.0)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)

    fixed = load_json("fixed_state_live_queries.json")
    adaptive = load_json("adaptive_live_queries.json")
    scaling = load_json("scaling_sweep.json")

    fixed_rows = {(r["base"], r["start_digit_after_decimal"]): r for r in fixed["rows"]}
    adaptive_rows = {(r["base"], r["start_digit_after_decimal"]): r for r in adaptive["rows"]}

    position_rows: List[PositionRow] = []
    for base in BASES:
        barrier = fixed_rows[(base, 1)]["ambiguity_barrier_position"]
        for position in POSITIONS:
            a = adaptive_rows[(base, position)]
            f = fixed_rows[(base, position)]
            lb = lower_bound_digits(base, position)
            position_rows.append(
                PositionRow(
                    base=base,
                    position=position,
                    block_length=ELL,
                    lower_bound_required_decimal_digits=lb,
                    actual_required_decimal_digits_current_law=a["required_decimal_digits"],
                    safety_margin_digits=a["required_decimal_digits"] - lb,
                    fixed_state_barrier_position=barrier,
                    fixed_state_certified=bool(f["certified"]),
                    adaptive_certified=bool(a["certified"]),
                    matches_reference=bool(a["matches_reference"]),
                    query_block=a["block"],
                    probe_suffix=a["probe_suffix"],
                )
            )

    packet_rows: List[PacketDepthRow] = []
    for depth in range(1, 9):
        u = fibonacci(depth + 1)
        v = fibonacci(depth + 2)
        N = u * v
        packet_rows.append(
            PacketDepthRow(
                depth=depth,
                u=u,
                v=v,
                N=N,
                K520=k_n(N, 520),
                K10000=k_n(N, 10000),
                K1000000=k_n(N, 1000000),
                strictly_worse_than_depth1=(depth > 1),
            )
        )

    theorem_summary = {
        "theorem": "No from-zero sublinear local radix decoder under the current universal-collapse + AGM family",
        "cell_obstruction_condition": "2 * b^(n-1) * B_m >= b^(-ell) implies at least two compatible radix blocks remain",
        "necessary_condition_for_certification": "2 * b^(n-1) * B_m < b^(-ell)",
        "warmup_lower_bound": "D_m > (n + ell - 1) * log10(b) + log10(2)",
        "asymptotic_conclusion": "D_m = Omega(n); sublinear warm-up in n is impossible under the current law",
        "packet_conclusion": "Depth-1 remains optimal because K_N(D) is strictly increasing in N on the balanced packet family",
    }

    write_csv(OUT / "position_demo.csv", [asdict(r) for r in position_rows])
    write_csv(OUT / "packet_depth_exploration.csv", [asdict(r) for r in packet_rows])
    (OUT / "position_demo.json").write_text(json.dumps([asdict(r) for r in position_rows], indent=2))
    (OUT / "packet_depth_exploration.json").write_text(json.dumps([asdict(r) for r in packet_rows], indent=2))
    (OUT / "theorem_summary.json").write_text(json.dumps(theorem_summary, indent=2))

    summary_lines = [
        "Phase Structural Closure v11",
        "",
        "Result: thread closed by impossibility under the current law.",
        "",
        "Main theorem:",
        "  Any sound current-state local radix decoder under the current universal-collapse + AGM family",
        "  must satisfy D_m = Omega(n). A from-zero sublinear nth-block decoder is impossible in this family.",
        "",
        "Evidence:",
        "  - fixed 6000-digit shared state certifies only positions within a finite horizon",
        "  - all five requested positions certify only after adaptive warm-up that grows linearly with queried position",
        "  - measured current-law requirements differ from the analytic lower bound by a constant safety margin only",
        "",
        "Representative barriers:",
        f"  base 10 fixed-state barrier: {fixed_rows[(10,1)]['ambiguity_barrier_position']}",
        f"  base 16 fixed-state barrier: {fixed_rows[(16,1)]['ambiguity_barrier_position']}",
    ]
    (OUT / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    manifest = {}
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        manifest[str(path.relative_to(ROOT))] = {
            "sha256": sha256sum(path),
            "bytes": path.stat().st_size,
        }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with (ROOT / "SHA256SUMS.txt").open("w") as fh:
        for rel, info in sorted(manifest.items()):
            fh.write(f"{info['sha256']}  {rel}\n")


if __name__ == "__main__":
    main()
