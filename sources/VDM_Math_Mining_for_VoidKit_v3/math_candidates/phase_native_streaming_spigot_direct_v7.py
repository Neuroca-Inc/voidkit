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
LIB_PATH = ROOT / "native_phase_packet_v7.so"
SRC_PATH = ROOT / "native_phase_packet_v7.c"
MPFR_LIB = "/lib/x86_64-linux-gnu/libmpfr.so.6"
PROBE_DIGITS = 16


@dataclass(frozen=True)
class EmissionRow:
    step: int
    decimal_from: int
    decimal_to: int
    newly_emitted: int
    digits_chunk: str


@dataclass(frozen=True)
class CandidateRow:
    target_digits: int
    iterations: int
    probe_digits: int
    probe_suffix: str
    first_non9_probe_position: int
    coarse_bound_log10: float
    coarse_bound_decimal_digits: float
    native_certificate_passed: bool
    safe_digits_lower_bound: int
    one_run_seconds: float
    prefix_80: str


@dataclass(frozen=True)
class VerificationRow:
    target_digits: int
    native_certificate_passed: bool
    candidate_correct_vs_chudnovsky: bool | None
    candidate_correct_vs_ramanujan_1914: bool | None
    first_mismatch_vs_chudnovsky: int | None
    first_mismatch_vs_ramanujan_1914: int | None
    probe_suffix: str
    first_non9_probe_position: int
    coarse_bound_decimal_digits: float
    safe_digits_lower_bound: int


@dataclass(frozen=True)
class PacketRow:
    depth: int
    u: int
    v: int
    N: int
    exact_packet_collapse_available: bool
    selected_by_collapsed_law: bool
    note: str


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
        self.lib.phase_native_pi_hot_v7.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.phase_native_pi_hot_v7.restype = ctypes.c_int
        self.lib.phase_native_pi_full_v7.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.phase_native_pi_full_v7.restype = ctypes.c_int
        self.lib.phase_native_pi_hot_benchmark_v7.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.phase_native_pi_hot_benchmark_v7.restype = ctypes.c_int
        self.lib.phase_native_pi_full_benchmark_v7.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.phase_native_pi_full_benchmark_v7.restype = ctypes.c_int
        self.lib.phase_native_reset_v7.argtypes = []
        self.lib.phase_native_reset_v7.restype = None

    def reset(self) -> None:
        self.lib.phase_native_reset_v7()

    def hot_run(self, digits: int, probe_digits: int = PROBE_DIGITS) -> Tuple[str, str, float, int, int]:
        buf = ctypes.create_string_buffer(digits + 32)
        probe = ctypes.create_string_buffer(probe_digits + 1)
        seconds = ctypes.c_double()
        iters = ctypes.c_uint()
        first_non9 = ctypes.c_uint()
        rc = self.lib.phase_native_pi_hot_v7(
            digits,
            probe_digits,
            buf,
            len(buf),
            probe,
            len(probe),
            ctypes.byref(seconds),
            ctypes.byref(iters),
            ctypes.byref(first_non9),
        )
        if rc != 0:
            raise RuntimeError(f"phase_native_pi_hot_v7 failed rc={rc}")
        return buf.value.decode(), probe.value.decode(), seconds.value, iters.value, first_non9.value

    def full_run(self, digits: int, probe_digits: int = PROBE_DIGITS) -> Tuple[str, str, float, int, bool, float, int, int]:
        buf = ctypes.create_string_buffer(digits + 32)
        probe = ctypes.create_string_buffer(probe_digits + 1)
        seconds = ctypes.c_double()
        iters = ctypes.c_uint()
        cert_ok = ctypes.c_int()
        bound_log10 = ctypes.c_double()
        safe_lb = ctypes.c_uint()
        first_non9 = ctypes.c_uint()
        rc = self.lib.phase_native_pi_full_v7(
            digits,
            probe_digits,
            buf,
            len(buf),
            probe,
            len(probe),
            ctypes.byref(seconds),
            ctypes.byref(iters),
            ctypes.byref(cert_ok),
            ctypes.byref(bound_log10),
            ctypes.byref(safe_lb),
            ctypes.byref(first_non9),
        )
        if rc != 0:
            raise RuntimeError(f"phase_native_pi_full_v7 failed rc={rc}")
        return (
            buf.value.decode(),
            probe.value.decode(),
            seconds.value,
            iters.value,
            bool(cert_ok.value),
            bound_log10.value,
            safe_lb.value,
            first_non9.value,
        )

    def benchmark(self, kind: str, digits: int, reps: int, probe_digits: int = PROBE_DIGITS) -> Tuple[float, float]:
        best = ctypes.c_double()
        mean = ctypes.c_double()
        fn = self.lib.phase_native_pi_hot_benchmark_v7 if kind == "hot" else self.lib.phase_native_pi_full_benchmark_v7
        rc = fn(digits, probe_digits, reps, ctypes.byref(best), ctypes.byref(mean))
        if rc != 0:
            raise RuntimeError(f"benchmark {kind} failed rc={rc}")
        return best.value, mean.value


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
    extra = 24
    mp.mp.dps = digits + extra
    C = 426880 * mp.sqrt(10005)
    M = mp.mpf(1)
    L = mp.mpf(13591409)
    X = mp.mpf(1)
    K = mp.mpf(6)
    S = L
    terms = digits // 14 + 3
    for n in range(1, terms):
        M = M * (K**3 - 16 * K) / (n**3)
        L += 545140134
        X *= -262537412640768000
        S += M * L / X
        K += 12
    mp.mp.dps = digits + 4
    return +(C / S)


def ramanujan_1914_pi(digits: int) -> mp.mpf:
    extra = 24
    mp.mp.dps = digits + extra
    S = mp.mpf(0)
    terms = digits // 8 + 3
    for n in range(terms):
        num = mp.factorial(4 * n) * (1103 + 26390 * n)
        den = (mp.factorial(n) ** 4) * mp.power(396, 4 * n)
        S += mp.mpf(num) / den
    mp.mp.dps = digits + 4
    return +(9801 / (2 * mp.sqrt(2) * S))


def benchmark_mean_seconds(fn, digits: int, reps: int) -> Tuple[float, float]:
    fn(digits)
    timings: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(digits)
        timings.append(time.perf_counter() - t0)
    return min(timings), sum(timings) / len(timings)


def first_mismatch(a: str, b: str) -> int | None:
    fa = a.split(".", 1)[1]
    fb = b.split(".", 1)[1]
    for idx, (ca, cb) in enumerate(zip(fa, fb), start=1):
        if ca != cb:
            return idx
    return None


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def packet_depth_exploration(max_depth: int = 4) -> List[PacketRow]:
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
                exact_packet_collapse_available=(N == 2),
                selected_by_collapsed_law=(N == 2),
                note=("singular fixed packet tau=i/2" if N == 2 else "no exact collapsed fixed packet closed in current law"),
            )
        )
    return rows


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
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file() and p.name not in {"manifest.json", "SHA256SUMS.txt"}):
        rel = path.relative_to(bundle_dir)
        files.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"bundle": bundle_dir.name, "files": files}
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (bundle_dir / "SHA256SUMS.txt").open("w") as f:
        for entry in files:
            f.write(f"{entry['sha256']}  {entry['path']}\n")
        f.write(f"{sha256_file(bundle_dir / 'manifest.json')}  manifest.json\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase native streaming spigot v7")
    parser.add_argument("--benchmark-520-reps", type=int, default=40)
    parser.add_argument("--benchmark-1000-reps", type=int, default=20)
    parser.add_argument("--benchmark-10000-reps", type=int, default=3)
    parser.add_argument("--benchmark-100000-reps", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    native = NativeLib()
    native.reset()

    plan = {520: args.benchmark_520_reps, 1000: args.benchmark_1000_reps, 10000: args.benchmark_10000_reps}
    optional_plan = {100000: args.benchmark_100000_reps}

    benchmark_results: Dict[str, object] = {"lengths": {}, "optional_lengths": {}}
    verification_results: Dict[str, Dict[str, object]] = {}
    candidate_rows: List[Dict[str, object]] = []

    for digits, reps in plan.items():
        text, probe, one_run, iters, cert_ok, bound_log10, safe_lb, first_non9 = native.full_run(digits)
        coarse_digits = -bound_log10
        candidate_rows.append(
            asdict(
                CandidateRow(
                    target_digits=digits,
                    iterations=iters,
                    probe_digits=PROBE_DIGITS,
                    probe_suffix=probe,
                    first_non9_probe_position=first_non9,
                    coarse_bound_log10=bound_log10,
                    coarse_bound_decimal_digits=coarse_digits,
                    native_certificate_passed=cert_ok,
                    safe_digits_lower_bound=safe_lb,
                    one_run_seconds=one_run,
                    prefix_80=text[:82],
                )
            )
        )

        (outdir / f"pi_{digits}_decimal.txt").write_text(text + "\n")
        write_csv(
            outdir / f"emission_trace_{digits}.csv",
            [asdict(EmissionRow(step=1, decimal_from=1, decimal_to=digits, newly_emitted=digits, digits_chunk=text.split('.', 1)[1]))],
        )

        hot_best, hot_mean = native.benchmark("hot", digits, reps)
        full_best, full_mean = native.benchmark("full", digits, reps)
        chud_best, chud_mean = benchmark_mean_seconds(chudnovsky_pi, digits, reps)
        ram_best, ram_mean = benchmark_mean_seconds(ramanujan_1914_pi, digits, reps)

        chud_text = expand_in_base(chudnovsky_pi(digits), digits, base=10)
        ram_text = expand_in_base(ramanujan_1914_pi(digits), digits, base=10)
        (outdir / f"baseline_chudnovsky_{digits}.txt").write_text(chud_text + "\n")
        (outdir / f"baseline_ramanujan_1914_{digits}.txt").write_text(ram_text + "\n")

        verification_results[str(digits)] = asdict(
            VerificationRow(
                target_digits=digits,
                native_certificate_passed=cert_ok,
                candidate_correct_vs_chudnovsky=(text == chud_text),
                candidate_correct_vs_ramanujan_1914=(text == ram_text),
                first_mismatch_vs_chudnovsky=first_mismatch(text, chud_text),
                first_mismatch_vs_ramanujan_1914=first_mismatch(text, ram_text),
                probe_suffix=probe,
                first_non9_probe_position=first_non9,
                coarse_bound_decimal_digits=coarse_digits,
                safe_digits_lower_bound=safe_lb,
            )
        )

        benchmark_results["lengths"][str(digits)] = {
            "benchmark_repetitions": reps,
            "probe_digits": PROBE_DIGITS,
            "iterations": iters,
            "candidate_hot_seconds_one_run": one_run,
            "candidate_hot_seconds_min": hot_best,
            "candidate_hot_seconds_mean": hot_mean,
            "candidate_full_seconds_min": full_best,
            "candidate_full_seconds_mean": full_mean,
            "candidate_full_digits_per_second_mean": digits / full_mean,
            "candidate_full_vs_target_0p0001": full_mean / 0.0001,
            "chudnovsky_seconds_min": chud_best,
            "chudnovsky_seconds_mean": chud_mean,
            "ramanujan_1914_seconds_min": ram_best,
            "ramanujan_1914_seconds_mean": ram_mean,
            "candidate_full_vs_chudnovsky_speed_ratio_mean": full_mean / chud_mean,
            "candidate_full_vs_ramanujan_1914_speed_ratio_mean": full_mean / ram_mean,
            "candidate_full_speedup_over_chudnovsky_mean": chud_mean / full_mean,
            "candidate_full_speedup_over_ramanujan_1914_mean": ram_mean / full_mean,
            "native_certificate_passed": cert_ok,
            "safe_digits_lower_bound": safe_lb,
            "coarse_bound_decimal_digits": coarse_digits,
            "first_non9_probe_position": first_non9,
            "probe_suffix": probe,
        }

    # optional 100k native scaling only
    for digits, reps in optional_plan.items():
        text, probe, one_run, iters, cert_ok, bound_log10, safe_lb, first_non9 = native.full_run(digits)
        coarse_digits = -bound_log10
        (outdir / f"pi_{digits}_decimal.txt").write_text(text + "\n")
        write_csv(
            outdir / f"emission_trace_{digits}.csv",
            [asdict(EmissionRow(step=1, decimal_from=1, decimal_to=digits, newly_emitted=digits, digits_chunk=text.split('.', 1)[1]))],
        )
        hot_best, hot_mean = native.benchmark("hot", digits, reps)
        full_best, full_mean = native.benchmark("full", digits, reps)
        benchmark_results["optional_lengths"][str(digits)] = {
            "benchmark_repetitions": reps,
            "probe_digits": PROBE_DIGITS,
            "iterations": iters,
            "candidate_hot_seconds_one_run": one_run,
            "candidate_hot_seconds_min": hot_best,
            "candidate_hot_seconds_mean": hot_mean,
            "candidate_full_seconds_min": full_best,
            "candidate_full_seconds_mean": full_mean,
            "candidate_full_digits_per_second_mean": digits / full_mean,
            "native_certificate_passed": cert_ok,
            "safe_digits_lower_bound": safe_lb,
            "coarse_bound_decimal_digits": coarse_digits,
            "first_non9_probe_position": first_non9,
            "probe_suffix": probe,
            "note": "optional native scaling run; classical baselines omitted at 100000 digits in this harness",
        }

    benchmark_results["headline"] = {
        "full_520_under_0p0001": benchmark_results["lengths"]["520"]["candidate_full_seconds_mean"] <= 0.0001,
        "native_certificate": "analytic coarse AGM bound + carry-margin probe",
        "claim_scope": "benchmarked on this machine and this harness only",
    }

    write_csv(outdir / "packet_depth_exploration.csv", [asdict(x) for x in packet_depth_exploration(4)])
    (outdir / "benchmark_results.json").write_text(json.dumps(benchmark_results, indent=2) + "\n")
    (outdir / "verification_results.json").write_text(json.dumps(verification_results, indent=2) + "\n")
    (outdir / "candidate_summary.json").write_text(json.dumps(candidate_rows, indent=2) + "\n")
    (outdir / "summary.txt").write_text(
        "\n".join(
            [
                "phase_native_streaming_spigot_bundle_v7",
                f"full_520_mean={benchmark_results['lengths']['520']['candidate_full_seconds_mean']}",
                f"full_520_vs_chud={benchmark_results['lengths']['520']['candidate_full_vs_chudnovsky_speed_ratio_mean']}",
                f"full_1000_mean={benchmark_results['lengths']['1000']['candidate_full_seconds_mean']}",
                f"full_10000_mean={benchmark_results['lengths']['10000']['candidate_full_seconds_mean']}",
                f"full_100000_mean={benchmark_results['optional_lengths']['100000']['candidate_full_seconds_mean']}",
                f"cert_520={verification_results['520']['native_certificate_passed']}",
                f"correct_520={verification_results['520']['candidate_correct_vs_chudnovsky'] and verification_results['520']['candidate_correct_vs_ramanujan_1914']}",
                f"correct_1000={verification_results['1000']['candidate_correct_vs_chudnovsky'] and verification_results['1000']['candidate_correct_vs_ramanujan_1914']}",
                f"correct_10000={verification_results['10000']['candidate_correct_vs_chudnovsky'] and verification_results['10000']['candidate_correct_vs_ramanujan_1914']}",
            ]
        )
        + "\n"
    )

    build_manifest(ROOT)


if __name__ == "__main__":
    main()
