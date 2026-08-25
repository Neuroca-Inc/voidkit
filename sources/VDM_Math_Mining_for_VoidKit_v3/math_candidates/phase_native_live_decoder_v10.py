#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "native_phase_live_decoder_v10.so"
SRC_PATH = ROOT / "native_phase_live_decoder_v10.c"
MPFR_LIB = "/lib/x86_64-linux-gnu/libmpfr.so.6"

POSITIONS = [1, 4_895, 500_000, 999_937, 1_000_000]
BASES = [10, 16]
PROBE_DIGITS = 32
BLOCK_LENGTH = 32
FIXED_WARMUP_DIGITS = 6_000
SCALING_POSITIONS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
BENCH_REPS = 8


@dataclass(frozen=True)
class PrepareResult:
    required_decimal_digits: int
    probe_digits: int
    warmup_seconds: float
    agm_iterations: int
    coarse_bound_log10: float


@dataclass(frozen=True)
class QueryResult:
    base: int
    start_digit_after_decimal: int
    length: int
    block: str
    probe_suffix: str
    query_seconds: float
    first_nonmax_probe_position: int
    certified: bool
    safe_length_lower_bound: int
    reference_block: str
    matches_reference: bool


@dataclass(frozen=True)
class FixedStateRow:
    fixed_required_decimal_digits: int
    base: int
    start_digit_after_decimal: int
    block: str
    probe_suffix: str
    query_seconds: float
    first_nonmax_probe_position: int
    certified: bool
    safe_length_lower_bound: int
    reference_block: str
    matches_reference: bool
    ambiguity_barrier_position: int


@dataclass(frozen=True)
class AdaptiveRow:
    base: int
    start_digit_after_decimal: int
    required_decimal_digits: int
    warmup_seconds: float
    agm_iterations: int
    coarse_bound_log10: float
    block: str
    probe_suffix: str
    query_seconds: float
    first_nonmax_probe_position: int
    certified: bool
    safe_length_lower_bound: int
    reference_block: str
    matches_reference: bool


@dataclass(frozen=True)
class ScalingRow:
    base: int
    start_digit_after_decimal: int
    required_decimal_digits: int
    warmup_seconds: float
    query_seconds: float
    agm_iterations: int
    coarse_bound_log10: float


@dataclass(frozen=True)
class TrajectoryRow:
    step: int
    base: int
    start_digit_after_decimal: int
    previous_required_decimal_digits: int
    new_required_decimal_digits: int
    state_extended: bool
    warmup_seconds: float
    agm_iterations: int
    query_seconds: float
    certified: bool
    block: str


@dataclass(frozen=True)
class PacketRow:
    depth: int
    u: int
    v: int
    N: int
    estimated_q_terms_520: int
    estimated_q_terms_10000: int
    estimated_q_terms_1000000: int
    preferred_under_current_law: bool
    comment: str


def compile_native() -> None:
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SRC_PATH.stat().st_mtime:
        return
    cmd = [
        "gcc",
        "-Ofast",
        "-march=native",
        "-funroll-loops",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-shared",
        "-fPIC",
        str(SRC_PATH),
        MPFR_LIB,
        "-lgmp",
        "-o",
        str(LIB_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


class NativeLib:
    def __init__(self) -> None:
        compile_native()
        self.lib = ctypes.CDLL(str(LIB_PATH))
        self.lib.phase_local_prepare_v10.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.phase_local_prepare_v10.restype = ctypes.c_int
        self.lib.phase_local_query_v10.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.phase_local_query_v10.restype = ctypes.c_int
        self.lib.phase_reference_query_v10.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.phase_reference_query_v10.restype = ctypes.c_int
        self.lib.phase_local_reset_v10.argtypes = []
        self.lib.phase_local_reset_v10.restype = None

    def reset(self) -> None:
        self.lib.phase_local_reset_v10()

    def prepare(self, required_decimal_digits: int, probe_digits: int = PROBE_DIGITS) -> PrepareResult:
        seconds = ctypes.c_double()
        iters = ctypes.c_uint()
        bound_log10 = ctypes.c_double()
        rc = self.lib.phase_local_prepare_v10(
            required_decimal_digits,
            probe_digits,
            ctypes.byref(seconds),
            ctypes.byref(iters),
            ctypes.byref(bound_log10),
        )
        if rc != 0:
            raise RuntimeError(f"phase_local_prepare_v10 failed rc={rc}")
        return PrepareResult(
            required_decimal_digits=required_decimal_digits,
            probe_digits=probe_digits,
            warmup_seconds=seconds.value,
            agm_iterations=int(iters.value),
            coarse_bound_log10=bound_log10.value,
        )

    def query(self, base: int, start: int, length: int = BLOCK_LENGTH, probe_digits: int = PROBE_DIGITS) -> QueryResult:
        out = ctypes.create_string_buffer(length + 2)
        probe = ctypes.create_string_buffer(probe_digits + 2)
        seconds = ctypes.c_double()
        first_nonmax = ctypes.c_uint()
        cert_ok = ctypes.c_int()
        safe_len = ctypes.c_uint()
        rc = self.lib.phase_local_query_v10(
            base,
            start,
            length,
            probe_digits,
            out,
            len(out),
            probe,
            len(probe),
            ctypes.byref(seconds),
            ctypes.byref(first_nonmax),
            ctypes.byref(cert_ok),
            ctypes.byref(safe_len),
        )
        if rc != 0:
            raise RuntimeError(f"phase_local_query_v10 failed rc={rc}")
        ref = self.reference_query(base, start, length, probe_digits)
        block = out.value.decode()
        return QueryResult(
            base=base,
            start_digit_after_decimal=start,
            length=length,
            block=block,
            probe_suffix=probe.value.decode(),
            query_seconds=seconds.value,
            first_nonmax_probe_position=int(first_nonmax.value),
            certified=bool(cert_ok.value),
            safe_length_lower_bound=int(safe_len.value),
            reference_block=ref["block"],
            matches_reference=(block == ref["block"]),
        )

    def reference_query(self, base: int, start: int, length: int = BLOCK_LENGTH, probe_digits: int = PROBE_DIGITS) -> Dict[str, str]:
        out = ctypes.create_string_buffer(length + 2)
        probe = ctypes.create_string_buffer(probe_digits + 2)
        rc = self.lib.phase_reference_query_v10(
            base,
            start,
            length,
            probe_digits,
            out,
            len(out),
            probe,
            len(probe),
        )
        if rc != 0:
            raise RuntimeError(f"phase_reference_query_v10 failed rc={rc}")
        return {"block": out.value.decode(), "probe_suffix": probe.value.decode()}


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def required_decimal_digits_for_query(base: int, start: int, length: int = BLOCK_LENGTH, probe_digits: int = PROBE_DIGITS) -> int:
    return math.ceil((start - 1 + length + probe_digits + 12) * math.log10(base)) + 8


def ambiguity_barrier_position(bound_log10: float, base: int, length: int = BLOCK_LENGTH) -> int:
    log10b = math.log10(base)
    # Necessary condition for unique block from current state alone:
    # 2 * b^(n-1) * B < b^(-length)
    # where log10(B) = bound_log10.
    raw = (-math.log10(2.0) - bound_log10) / log10b - length + 1
    return max(0, math.floor(raw - 1e-12))


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def est_q_terms(N: int, digits: int) -> int:
    rhs = (N * digits * math.log(10.0)) / math.pi
    k = (1.0 + math.sqrt(1.0 + 12.0 * rhs)) / 6.0
    return max(1, math.ceil(k))


def packet_depth_rows(max_depth: int = 8) -> List[PacketRow]:
    rows: List[PacketRow] = []
    for depth in range(1, max_depth + 1):
        u = fibonacci(depth + 1)
        v = fibonacci(depth + 2)
        N = u * v
        rows.append(
            PacketRow(
                depth=depth,
                u=u,
                v=v,
                N=N,
                estimated_q_terms_520=est_q_terms(N, 520),
                estimated_q_terms_10000=est_q_terms(N, 10_000),
                estimated_q_terms_1000000=est_q_terms(N, 1_000_000),
                preferred_under_current_law=(depth == 1),
                comment="Under universal packet collapse, deeper packets increase the pre-collapse q-side burden and do not solve the local nth-block problem.",
            )
        )
    return rows


def run_fixed_state_demo(lib: NativeLib, fixed_digits: int = FIXED_WARMUP_DIGITS) -> Dict[str, object]:
    lib.reset()
    prep = lib.prepare(fixed_digits)
    rows: List[FixedStateRow] = []
    for base in BASES:
        barrier = ambiguity_barrier_position(prep.coarse_bound_log10, base, BLOCK_LENGTH)
        for start in POSITIONS:
            q = lib.query(base, start)
            rows.append(
                FixedStateRow(
                    fixed_required_decimal_digits=fixed_digits,
                    base=base,
                    start_digit_after_decimal=start,
                    block=q.block,
                    probe_suffix=q.probe_suffix,
                    query_seconds=q.query_seconds,
                    first_nonmax_probe_position=q.first_nonmax_probe_position,
                    certified=q.certified,
                    safe_length_lower_bound=q.safe_length_lower_bound,
                    reference_block=q.reference_block,
                    matches_reference=q.matches_reference,
                    ambiguity_barrier_position=barrier,
                )
            )
    return {"prepare": asdict(prep), "rows": [asdict(r) for r in rows]}


def run_adaptive_demo(lib: NativeLib) -> Dict[str, object]:
    rows: List[AdaptiveRow] = []
    for base in BASES:
        for start in POSITIONS:
            req = required_decimal_digits_for_query(base, start)
            lib.reset()
            prep = lib.prepare(req)
            q = lib.query(base, start)
            rows.append(
                AdaptiveRow(
                    base=base,
                    start_digit_after_decimal=start,
                    required_decimal_digits=req,
                    warmup_seconds=prep.warmup_seconds,
                    agm_iterations=prep.agm_iterations,
                    coarse_bound_log10=prep.coarse_bound_log10,
                    block=q.block,
                    probe_suffix=q.probe_suffix,
                    query_seconds=q.query_seconds,
                    first_nonmax_probe_position=q.first_nonmax_probe_position,
                    certified=q.certified,
                    safe_length_lower_bound=q.safe_length_lower_bound,
                    reference_block=q.reference_block,
                    matches_reference=q.matches_reference,
                )
            )
    return {"rows": [asdict(r) for r in rows]}


def run_scaling_sweep(lib: NativeLib) -> Dict[str, object]:
    rows: List[ScalingRow] = []
    for base in BASES:
        for start in SCALING_POSITIONS:
            req = required_decimal_digits_for_query(base, start)
            lib.reset()
            prep = lib.prepare(req)
            q = lib.query(base, start)
            rows.append(
                ScalingRow(
                    base=base,
                    start_digit_after_decimal=start,
                    required_decimal_digits=req,
                    warmup_seconds=prep.warmup_seconds,
                    query_seconds=q.query_seconds,
                    agm_iterations=prep.agm_iterations,
                    coarse_bound_log10=prep.coarse_bound_log10,
                )
            )
    return {"rows": [asdict(r) for r in rows]}


def run_live_trajectory(lib: NativeLib) -> Dict[str, object]:
    rows: List[TrajectoryRow] = []
    current_digits = 0
    step = 0
    for base in BASES:
        for start in POSITIONS:
            step += 1
            required = required_decimal_digits_for_query(base, start)
            extended = required > current_digits
            if extended:
                lib.reset()
                prep = lib.prepare(required)
                current_digits = required
            else:
                prep = PrepareResult(current_digits, PROBE_DIGITS, 0.0, 0, float("nan"))
            q = lib.query(base, start)
            rows.append(
                TrajectoryRow(
                    step=step,
                    base=base,
                    start_digit_after_decimal=start,
                    previous_required_decimal_digits=(0 if step == 1 else rows[-1].new_required_decimal_digits if rows else 0),
                    new_required_decimal_digits=current_digits,
                    state_extended=extended,
                    warmup_seconds=prep.warmup_seconds,
                    agm_iterations=prep.agm_iterations,
                    query_seconds=q.query_seconds,
                    certified=q.certified,
                    block=q.block,
                )
            )
    return {"rows": [asdict(r) for r in rows]}


def summarize(fixed: Dict[str, object], adaptive: Dict[str, object], scaling: Dict[str, object]) -> str:
    fixed_rows = fixed["rows"]
    adaptive_rows = adaptive["rows"]
    scaling_rows = scaling["rows"]

    fixed_ok = sum(1 for r in fixed_rows if r["certified"])
    adaptive_ok = sum(1 for r in adaptive_rows if r["certified"] and r["matches_reference"])
    worst_adaptive_prepare = max(r["warmup_seconds"] for r in adaptive_rows)
    worst_adaptive_query = max(r["query_seconds"] for r in adaptive_rows)

    dec_rows = [r for r in scaling_rows if r["base"] == 10]
    hex_rows = [r for r in scaling_rows if r["base"] == 16]

    lines = []
    lines.append("Phase Native Live Decoder v10 summary")
    lines.append("")
    lines.append(f"Fixed-state live demo: {fixed_ok}/{len(fixed_rows)} queries certified with one shared {FIXED_WARMUP_DIGITS}-digit warm-up state.")
    lines.append("The certified horizon is finite: near positions pass, far positions fail without state growth.")
    lines.append("")
    lines.append(f"Adaptive live demo: {adaptive_ok}/{len(adaptive_rows)} queries certified and matched the external reference.")
    lines.append(f"Worst adaptive warm-up time: {worst_adaptive_prepare:.6f} s")
    lines.append(f"Worst adaptive query time:  {worst_adaptive_query:.6f} s")
    lines.append("")
    lines.append("Scaling snapshot (required decimal digits):")
    for rows, label in [(dec_rows, "base 10"), (hex_rows, "base 16")]:
        parts = [f"{r['start_digit_after_decimal']}->{r['required_decimal_digits']}" for r in rows]
        lines.append(f"  {label}: " + ", ".join(parts))
    lines.append("")
    lines.append("Structural result:")
    lines.append("Under the current universal-collapse + AGM law, a fixed current state cannot certify arbitrarily far radix blocks.")
    lines.append("The oriented uncertainty grows like b^(n-1) * B_m, so the warm-up / precision burden must grow at least linearly with the queried position n.")
    lines.append("A true sublinear from-zero local nth-digit decoder therefore requires a new decoder law beyond the current collapsed AGM family.")
    return "\n".join(lines) + "\n"


def make_notebook(path: Path) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase Native Live Decoder v10\n",
                "\n",
                "This notebook records the v10 result: a genuinely live current-state orientation prototype, together with the exact obstruction that prevents a fixed warm-up state from certifying arbitrary far-away blocks under the current universal-collapse + AGM law.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "root = Path('.')\n",
                "for name in ['output/fixed_state_live_queries.json', 'output/adaptive_live_queries.json', 'output/scaling_sweep.json']:\n",
                "    p = root / name\n",
                "    print('---', p)\n",
                "    print(json.loads(p.read_text()) if p.exists() else 'missing')\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase native live decoder v10")
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    lib = NativeLib()
    fixed = run_fixed_state_demo(lib)
    adaptive = run_adaptive_demo(lib)
    scaling = run_scaling_sweep(lib)
    trajectory = run_live_trajectory(lib)
    packets = {"rows": [asdict(r) for r in packet_depth_rows()]}

    (out / "fixed_state_live_queries.json").write_text(json.dumps(fixed, indent=2))
    (out / "adaptive_live_queries.json").write_text(json.dumps(adaptive, indent=2))
    (out / "scaling_sweep.json").write_text(json.dumps(scaling, indent=2))
    (out / "live_state_trajectory.json").write_text(json.dumps(trajectory, indent=2))
    (out / "packet_depth_exploration.json").write_text(json.dumps(packets, indent=2))

    write_csv(out / "fixed_state_live_queries.csv", fixed["rows"])
    write_csv(out / "adaptive_live_queries.csv", adaptive["rows"])
    write_csv(out / "scaling_sweep.csv", scaling["rows"])
    write_csv(out / "live_state_trajectory.csv", trajectory["rows"])
    write_csv(out / "packet_depth_exploration.csv", packets["rows"])

    summary = summarize(fixed, adaptive, scaling)
    (out / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
