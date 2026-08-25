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
from typing import Dict, List, Tuple

import mpmath as mp

ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "native_phase_packet_v5.so"
SRC_PATH = ROOT / "native_phase_packet_v5.c"
MPFR_LIB = "/lib/x86_64-linux-gnu/libmpfr.so.6"


@dataclass(frozen=True)
class PacketState:
    depth: int
    u: int
    v: int
    N: int
    delta: int
    block_word: str
    execution_coordinate: int


@dataclass(frozen=True)
class EmissionRow:
    step: int
    decimal_from: int
    decimal_to: int
    newly_emitted: int
    digits_chunk: str


@dataclass(frozen=True)
class VerificationResult:
    target_digits: int
    candidate_correct_vs_chudnovsky: bool
    candidate_correct_vs_ramanujan_1914: bool
    first_mismatch_vs_chudnovsky: int | None
    first_mismatch_vs_ramanujan_1914: int | None


@dataclass(frozen=True)
class PacketDepthRow:
    depth: int
    u: int
    v: int
    N: int
    q_at_pi: float
    r_at_pi: float
    q_terms_tol_520: int
    r_terms_tol_520: int
    q_terms_tol_10000: int
    r_terms_tol_10000: int


class IterRow(ctypes.Structure):
    _fields_ = [
        ("stage_dps", ctypes.c_uint),
        ("q_terms", ctypes.c_uint),
        ("r_terms", ctypes.c_uint),
        ("safe_digits", ctypes.c_uint),
        ("stage_seconds", ctypes.c_double),
        ("abs_update", ctypes.c_char * 128),
        ("abs_residual", ctypes.c_char * 128),
        ("error_bound", ctypes.c_char * 128),
    ]


class Summary(ctypes.Structure):
    _fields_ = [
        ("target_digits", ctypes.c_uint),
        ("final_safe_digits", ctypes.c_uint),
        ("stages", ctypes.c_uint),
        ("q_terms_final", ctypes.c_uint),
        ("r_terms_final", ctypes.c_uint),
        ("total_seconds", ctypes.c_double),
    ]


def compile_native() -> None:
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SRC_PATH.stat().st_mtime:
        return
    cmd = [
        "gcc",
        "-Ofast",
        "-march=native",
        "-fPIC",
        "-shared",
        "-funroll-loops",
        "-fno-math-errno",
        "-fno-trapping-math",
        str(SRC_PATH),
        MPFR_LIB,
        "-lgmp",
        "-o",
        str(LIB_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def load_lib() -> ctypes.CDLL:
    compile_native()
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.phase_native_pi_fast_v5.argtypes = [
        ctypes.c_uint,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.phase_native_pi_fast_v5.restype = ctypes.c_int
    lib.phase_native_pi_trace_v5.argtypes = [
        ctypes.c_uint,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(IterRow),
        ctypes.c_uint,
        ctypes.POINTER(Summary),
    ]
    lib.phase_native_pi_trace_v5.restype = ctypes.c_int
    lib.phase_native_pi_benchmark_v5.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.phase_native_pi_benchmark_v5.restype = ctypes.c_int
    lib.phase_native_pi_pipeline_reset_v5.argtypes = []
    lib.phase_native_pi_pipeline_reset_v5.restype = None
    return lib


def candidate_fast(lib: ctypes.CDLL, target_digits: int) -> Tuple[str, float, int, int, int]:
    buf = ctypes.create_string_buffer(target_digits + 32)
    seconds = ctypes.c_double()
    safe = ctypes.c_uint()
    q_terms = ctypes.c_uint()
    r_terms = ctypes.c_uint()
    rc = lib.phase_native_pi_fast_v5(
        target_digits,
        buf,
        len(buf),
        ctypes.byref(seconds),
        ctypes.byref(safe),
        ctypes.byref(q_terms),
        ctypes.byref(r_terms),
    )
    if rc != 0:
        raise RuntimeError(f"phase_native_pi_fast_v5 failed with rc={rc}")
    return buf.value.decode(), seconds.value, safe.value, q_terms.value, r_terms.value


def candidate_trace(lib: ctypes.CDLL, target_digits: int) -> Tuple[str, Summary, List[IterRow]]:
    buf = ctypes.create_string_buffer(target_digits + 32)
    rows = (IterRow * 16)()
    summary = Summary()
    rc = lib.phase_native_pi_trace_v5(target_digits, buf, len(buf), rows, 16, ctypes.byref(summary))
    if rc != 0:
        raise RuntimeError(f"phase_native_pi_trace_v5 failed with rc={rc}")
    return buf.value.decode(), summary, list(rows[: summary.stages])


def candidate_benchmark(lib: ctypes.CDLL, target_digits: int, reps: int, warmup: bool = True) -> Tuple[float, float]:
    if warmup:
        _ = candidate_fast(lib, target_digits)
    best = ctypes.c_double()
    mean = ctypes.c_double()
    rc = lib.phase_native_pi_benchmark_v5(target_digits, reps, ctypes.byref(best), ctypes.byref(mean))
    if rc != 0:
        raise RuntimeError(f"phase_native_pi_benchmark_v5 failed with rc={rc}")
    return best.value, mean.value


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def packet_state(depth: int, W: int = 64) -> PacketState:
    u = fibonacci(depth + 1)
    v = fibonacci(depth + 2)
    delta = min(depth, W - 1)
    block_word = "R" * delta + "S" * (W - 1 - delta) + "T"
    execution_coordinate = (3**delta - 1) // 2 + 2 * 3 ** (W - 1)
    return PacketState(depth, u, v, u * v, delta, block_word, execution_coordinate)


def packet_family(max_depth: int = 10) -> List[PacketState]:
    return [packet_state(depth) for depth in range(max_depth + 1)]


def first_mismatch(a: str, b: str) -> int | None:
    fa = a.split(".", 1)[1]
    fb = b.split(".", 1)[1]
    for idx, (ca, cb) in enumerate(zip(fa, fb), start=1):
        if ca != cb:
            return idx
    return None


def int_to_base(n: int, base: int) -> str:
    alphabet = "0123456789ABCDEF"
    if n == 0:
        return "0"
    out = ""
    x = int(n)
    while x > 0:
        x, r = divmod(x, base)
        out = alphabet[r] + out
    return out


def expand_in_base(x: mp.mpf, digits: int, base: int = 10) -> str:
    alphabet = "0123456789ABCDEF"
    frac = mp.mpf(x)
    n = int(mp.floor(frac))
    frac -= n
    out: List[str] = []
    for _ in range(digits):
        frac *= base
        d = int(mp.floor(frac))
        out.append(alphabet[d])
        frac -= d
    return int_to_base(n, base) + "." + "".join(out)


def chudnovsky_pi(digits: int) -> mp.mpf:
    extra = 20
    mp.mp.dps = digits + extra
    C = 426880 * mp.sqrt(10005)
    M = mp.mpf(1)
    L = mp.mpf(13591409)
    X = mp.mpf(1)
    K = mp.mpf(6)
    S = L
    terms = digits // 14 + 2
    for n in range(1, terms):
        M = M * (K**3 - 16 * K) / (n**3)
        L += 545140134
        X *= -262537412640768000
        S += M * L / X
        K += 12
    pi = C / S
    mp.mp.dps = digits
    return +pi


def ramanujan_1914_pi(digits: int) -> mp.mpf:
    extra = 20
    mp.mp.dps = digits + extra
    S = mp.mpf(0)
    terms = digits // 8 + 2
    for n in range(terms):
        num = mp.factorial(4 * n) * (1103 + 26390 * n)
        den = (mp.factorial(n) ** 4) * mp.power(396, 4 * n)
        S += mp.mpf(num) / den
    pi = 9801 / (2 * mp.sqrt(2) * S)
    mp.mp.dps = digits
    return +pi


def benchmark_mean_seconds(fn, digits: int, reps: int, warmup: bool = True) -> Tuple[float, float]:
    if warmup:
        fn(digits)
    timings: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(digits)
        timings.append(time.perf_counter() - t0)
    return min(timings), sum(timings) / len(timings)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(bundle_dir: Path) -> None:
    files = []
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(bundle_dir)
        files.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"bundle": bundle_dir.name, "files": files}
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (bundle_dir / "SHA256SUMS.txt").open("w") as f:
        for entry in files:
            f.write(f"{entry['sha256']}  {entry['path']}\n")


def pentagonal_term_count(q: float, tol_digits: int) -> int:
    if q <= 0.0:
        return 0
    log_tol = -tol_digits * math.log(10.0)
    tminus = q
    q2 = q * q
    tplus = q2
    p1 = q2 * q
    p2 = q2
    k = 1
    def above(term: float) -> bool:
        return term != 0.0 and math.log(abs(term)) >= log_tol
    while above(tminus) or above(tplus):
        next_minus = tplus * p1
        next_plus = next_minus * p2
        tminus = next_minus
        tplus = next_plus
        p1 *= q2
        p2 *= q
        k += 1
        if k > 100000:
            break
    return k - 1


def packet_depth_exploration(max_depth: int = 4) -> List[PacketDepthRow]:
    rows: List[PacketDepthRow] = []
    for depth in range(1, max_depth + 1):
        p = packet_state(depth)
        q = math.exp(-2.0 * math.pi / p.N)
        r = math.exp(-2.0 * p.N * math.pi)
        rows.append(
            PacketDepthRow(
                depth=depth,
                u=p.u,
                v=p.v,
                N=p.N,
                q_at_pi=q,
                r_at_pi=r,
                q_terms_tol_520=pentagonal_term_count(q, 528),
                r_terms_tol_520=pentagonal_term_count(r, 528),
                q_terms_tol_10000=pentagonal_term_count(q, 10008),
                r_terms_tol_10000=pentagonal_term_count(r, 10008),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase native streaming spigot v5")
    parser.add_argument("--benchmark-520-reps", type=int, default=5)
    parser.add_argument("--benchmark-1000-reps", type=int, default=5)
    parser.add_argument("--benchmark-10000-reps", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    lib = load_lib()
    lib.phase_native_pi_pipeline_reset_v5()

    benchmark_plan = {520: args.benchmark_520_reps, 1000: args.benchmark_1000_reps, 10000: args.benchmark_10000_reps}
    outputs: Dict[int, Dict[str, object]] = {}
    verification_rows: Dict[int, Dict[str, object]] = {}

    # main artifacts / correctness
    for digits in (520, 1000, 10000):
        text, seconds_one, safe_digits, q_terms, r_terms = candidate_fast(lib, digits)
        outputs[digits] = {
            "text": text,
            "seconds_one": seconds_one,
            "safe_digits": safe_digits,
            "q_terms": q_terms,
            "r_terms": r_terms,
        }
        (outdir / f"pi_{digits}_decimal.txt").write_text(text + "\n")

        chud_text = expand_in_base(chudnovsky_pi(digits + 20), digits, base=10)
        ram_text = expand_in_base(ramanujan_1914_pi(digits + 20), digits, base=10)
        (outdir / f"baseline_chudnovsky_{digits}.txt").write_text(chud_text + "\n")
        (outdir / f"baseline_ramanujan_1914_{digits}.txt").write_text(ram_text + "\n")

        verification = VerificationResult(
            target_digits=digits,
            candidate_correct_vs_chudnovsky=(text == chud_text),
            candidate_correct_vs_ramanujan_1914=(text == ram_text),
            first_mismatch_vs_chudnovsky=first_mismatch(text, chud_text),
            first_mismatch_vs_ramanujan_1914=first_mismatch(text, ram_text),
        )
        verification_rows[digits] = asdict(verification)

    # detailed trace for 520 digits
    trace_text, summary, rows = candidate_trace(lib, 520)
    assert trace_text == outputs[520]["text"]
    frac = trace_text.split(".", 1)[1]
    emitted_through = 0
    emission_rows: List[Dict[str, object]] = []
    iter_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        safe = min(520, int(row.safe_digits))
        if safe > emitted_through:
            chunk = frac[emitted_through:safe]
            emission_rows.append(asdict(EmissionRow(idx, emitted_through + 1, safe, safe - emitted_through, chunk)))
            emitted_through = safe
        iter_rows.append(
            {
                "step": idx,
                "work_dps": int(row.stage_dps),
                "packet_depth": 1,
                "u": 1,
                "v": 2,
                "N": 2,
                "q_terms": int(row.q_terms),
                "r_terms": int(row.r_terms),
                "abs_update": row.abs_update.decode().rstrip("\x00"),
                "abs_residual": row.abs_residual.decode().rstrip("\x00"),
                "error_bound": row.error_bound.decode().rstrip("\x00"),
                "safe_decimal_digits": safe,
                "emitted_through": emitted_through,
                "stage_seconds": row.stage_seconds,
            }
        )
    write_csv(outdir / "iteration_trace_520.csv", iter_rows)
    write_csv(outdir / "emission_trace_520.csv", emission_rows)

    # packet family and deeper-packet exploration
    write_csv(outdir / "packet_family.csv", [asdict(x) for x in packet_family(10)])
    write_csv(outdir / "packet_depth_exploration.csv", [asdict(x) for x in packet_depth_exploration(4)])

    # benchmarks
    benchmark_results: Dict[str, object] = {"lengths": {}}
    for digits, reps in benchmark_plan.items():
        cand_min, cand_mean = candidate_benchmark(lib, digits, reps, warmup=True)
        chud_min, chud_mean = benchmark_mean_seconds(chudnovsky_pi, digits, reps, warmup=True)
        ram_min, ram_mean = benchmark_mean_seconds(ramanujan_1914_pi, digits, reps, warmup=True)
        benchmark_results["lengths"][str(digits)] = {
            "benchmark_repetitions": reps,
            "candidate_seconds_one_run": outputs[digits]["seconds_one"],
            "candidate_seconds_min": cand_min,
            "candidate_seconds_mean": cand_mean,
            "candidate_digits_per_second_mean": digits / cand_mean,
            "candidate_safe_digits": outputs[digits]["safe_digits"],
            "candidate_q_terms": outputs[digits]["q_terms"],
            "candidate_r_terms": outputs[digits]["r_terms"],
            "chudnovsky_seconds_min": chud_min,
            "chudnovsky_seconds_mean": chud_mean,
            "chudnovsky_digits_per_second_mean": digits / chud_mean,
            "ramanujan_1914_seconds_min": ram_min,
            "ramanujan_1914_seconds_mean": ram_mean,
            "ramanujan_1914_digits_per_second_mean": digits / ram_mean,
            "candidate_vs_chudnovsky_speed_ratio_mean": cand_mean / chud_mean,
            "candidate_vs_ramanujan_1914_speed_ratio_mean": cand_mean / ram_mean,
        }

    benchmark_results["headline"] = {
        "candidate_beats_chudnovsky_at_520": benchmark_results["lengths"]["520"]["candidate_seconds_mean"]
        <= benchmark_results["lengths"]["520"]["chudnovsky_seconds_mean"],
        "candidate_beats_ramanujan_1914_at_520": benchmark_results["lengths"]["520"]["candidate_seconds_mean"]
        <= benchmark_results["lengths"]["520"]["ramanujan_1914_seconds_mean"],
        "candidate_beats_chudnovsky_at_10000": benchmark_results["lengths"]["10000"]["candidate_seconds_mean"]
        <= benchmark_results["lengths"]["10000"]["chudnovsky_seconds_mean"],
        "candidate_beats_ramanujan_1914_at_10000": benchmark_results["lengths"]["10000"]["candidate_seconds_mean"]
        <= benchmark_results["lengths"]["10000"]["ramanujan_1914_seconds_mean"],
    }

    (outdir / "benchmark_results.json").write_text(json.dumps(benchmark_results, indent=2) + "\n")
    (outdir / "verification_results.json").write_text(json.dumps(verification_rows, indent=2) + "\n")

    candidate_summary = {
        "packet_depth": 1,
        "packet_u": 1,
        "packet_v": 2,
        "packet_N": 2,
        "packet_delta": 1,
        "packet_block_word": packet_state(1).block_word,
        "packet_execution_coordinate": str(packet_state(1).execution_coordinate),
        "trace_stages_520": int(summary.stages),
        "trace_safe_digits_520": int(summary.final_safe_digits),
        "trace_q_terms_final_520": int(summary.q_terms_final),
        "trace_r_terms_final_520": int(summary.r_terms_final),
        "final_prefix_80": outputs[520]["text"][:82],
    }
    (outdir / "candidate_summary.json").write_text(json.dumps(candidate_summary, indent=2) + "\n")

    summary_txt = [
        "phase_native_streaming_spigot_bundle_v5",
        f"520_mean={benchmark_results['lengths']['520']['candidate_seconds_mean']}",
        f"520_vs_chud={benchmark_results['lengths']['520']['candidate_vs_chudnovsky_speed_ratio_mean']}",
        f"520_vs_ram={benchmark_results['lengths']['520']['candidate_vs_ramanujan_1914_speed_ratio_mean']}",
        f"10000_mean={benchmark_results['lengths']['10000']['candidate_seconds_mean']}",
        f"10000_vs_chud={benchmark_results['lengths']['10000']['candidate_vs_chudnovsky_speed_ratio_mean']}",
        f"10000_vs_ram={benchmark_results['lengths']['10000']['candidate_vs_ramanujan_1914_speed_ratio_mean']}",
        f"verified_520={verification_rows[520]['candidate_correct_vs_chudnovsky'] and verification_rows[520]['candidate_correct_vs_ramanujan_1914']}",
        f"verified_1000={verification_rows[1000]['candidate_correct_vs_chudnovsky'] and verification_rows[1000]['candidate_correct_vs_ramanujan_1914']}",
        f"verified_10000={verification_rows[10000]['candidate_correct_vs_chudnovsky'] and verification_rows[10000]['candidate_correct_vs_ramanujan_1914']}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_txt) + "\n")

    build_manifest(ROOT)


if __name__ == "__main__":
    main()
