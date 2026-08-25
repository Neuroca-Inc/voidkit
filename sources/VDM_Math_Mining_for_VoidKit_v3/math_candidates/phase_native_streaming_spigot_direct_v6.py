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
LIB_PATH = ROOT / "native_phase_packet_v6.so"
SRC_PATH = ROOT / "native_phase_packet_v6.c"
MPFR_LIB = "/lib/x86_64-linux-gnu/libmpfr.so.6"


@dataclass(frozen=True)
class PacketState:
    depth: int
    u: int
    v: int
    N: int
    q_at_pi: float
    r_at_pi: float
    q_terms_tol_520: int
    r_terms_tol_520: int
    q_terms_tol_1000: int
    r_terms_tol_1000: int
    q_terms_tol_10000: int
    r_terms_tol_10000: int


@dataclass(frozen=True)
class EmissionRow:
    step: int
    decimal_from: int
    decimal_to: int
    newly_emitted: int
    digits_chunk: str


@dataclass(frozen=True)
class VerificationRow:
    target_digits: int
    candidate_correct_vs_chudnovsky: bool
    candidate_correct_vs_ramanujan_1914: bool
    first_mismatch_vs_chudnovsky: int | None
    first_mismatch_vs_ramanujan_1914: int | None
    legacy_cert_safe_digits: int
    legacy_cert_seconds_one_run: float
    legacy_cert_q_terms: int
    legacy_cert_r_terms: int


@dataclass(frozen=True)
class HotSummary:
    target_digits: int
    one_run_seconds: float
    iterations: int
    prefix_80: str


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


def load_lib() -> ctypes.CDLL:
    compile_native()
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.phase_native_pi_hot_v6.argtypes = [
        ctypes.c_uint,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.phase_native_pi_hot_v6.restype = ctypes.c_int
    lib.phase_native_pi_hot_benchmark_v6.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.phase_native_pi_hot_benchmark_v6.restype = ctypes.c_int
    lib.phase_current_law_cert_v6.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.phase_current_law_cert_v6.restype = ctypes.c_int
    lib.phase_current_law_cert_benchmark_v6.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.phase_current_law_cert_benchmark_v6.restype = ctypes.c_int
    lib.phase_native_pi_hot_with_cert_benchmark_v6.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.phase_native_pi_hot_with_cert_benchmark_v6.restype = ctypes.c_int
    lib.phase_native_reset_v6.argtypes = []
    lib.phase_native_reset_v6.restype = None
    return lib


def hot_run(lib: ctypes.CDLL, target_digits: int) -> Tuple[str, float, int]:
    buf = ctypes.create_string_buffer(target_digits + 32)
    seconds = ctypes.c_double()
    iterations = ctypes.c_uint()
    rc = lib.phase_native_pi_hot_v6(target_digits, buf, len(buf), ctypes.byref(seconds), ctypes.byref(iterations))
    if rc != 0:
        raise RuntimeError(f"phase_native_pi_hot_v6 failed with rc={rc}")
    return buf.value.decode(), seconds.value, iterations.value


def legacy_cert_run(lib: ctypes.CDLL, target_digits: int) -> Tuple[int, float, int, int]:
    safe = ctypes.c_uint()
    seconds = ctypes.c_double()
    q_terms = ctypes.c_uint()
    r_terms = ctypes.c_uint()
    rc = lib.phase_current_law_cert_v6(
        target_digits,
        ctypes.byref(safe),
        ctypes.byref(seconds),
        ctypes.byref(q_terms),
        ctypes.byref(r_terms),
    )
    if rc != 0:
        raise RuntimeError(f"phase_current_law_cert_v6 failed with rc={rc}")
    return safe.value, seconds.value, q_terms.value, r_terms.value


def c_benchmark(fn, digits: int, reps: int) -> Tuple[float, float]:
    best = ctypes.c_double()
    mean = ctypes.c_double()
    rc = fn(digits, reps, ctypes.byref(best), ctypes.byref(mean))
    if rc != 0:
        raise RuntimeError(f"C benchmark failed with rc={rc}")
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


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


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


def packet_depth_exploration(max_depth: int = 4) -> List[PacketState]:
    rows: List[PacketState] = []
    for depth in range(1, max_depth + 1):
        u = fibonacci(depth + 1)
        v = fibonacci(depth + 2)
        N = u * v
        q = math.exp(-2.0 * math.pi / N)
        r = math.exp(-2.0 * N * math.pi)
        rows.append(
            PacketState(
                depth=depth,
                u=u,
                v=v,
                N=N,
                q_at_pi=q,
                r_at_pi=r,
                q_terms_tol_520=pentagonal_term_count(q, 528),
                r_terms_tol_520=pentagonal_term_count(r, 528),
                q_terms_tol_1000=pentagonal_term_count(q, 1008),
                r_terms_tol_1000=pentagonal_term_count(r, 1008),
                q_terms_tol_10000=pentagonal_term_count(q, 10008),
                r_terms_tol_10000=pentagonal_term_count(r, 10008),
            )
        )
    return rows


def first_mismatch(a: str, b: str) -> int | None:
    fa = a.split(".", 1)[1]
    fb = b.split(".", 1)[1]
    for idx, (ca, cb) in enumerate(zip(fa, fb), start=1):
        if ca != cb:
            return idx
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase native streaming spigot v6")
    parser.add_argument("--benchmark-520-reps", type=int, default=20)
    parser.add_argument("--benchmark-1000-reps", type=int, default=10)
    parser.add_argument("--benchmark-10000-reps", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    lib = load_lib()
    lib.phase_native_reset_v6()

    plan = {520: args.benchmark_520_reps, 1000: args.benchmark_1000_reps, 10000: args.benchmark_10000_reps}
    benchmark_results: Dict[str, object] = {"lengths": {}}
    verification_rows: Dict[int, Dict[str, object]] = {}
    hot_summaries: List[Dict[str, object]] = []

    for digits, reps in plan.items():
        hot_text, hot_one_run, hot_iters = hot_run(lib, digits)
        safe_digits, cert_seconds, q_terms, r_terms = legacy_cert_run(lib, digits)

        hot_best, hot_mean = c_benchmark(lib.phase_native_pi_hot_benchmark_v6, digits, reps)
        cert_best, cert_mean = c_benchmark(lib.phase_current_law_cert_benchmark_v6, digits, reps)
        both_best, both_mean = c_benchmark(lib.phase_native_pi_hot_with_cert_benchmark_v6, digits, reps)
        chud_best, chud_mean = benchmark_mean_seconds(chudnovsky_pi, digits, reps)
        ram_best, ram_mean = benchmark_mean_seconds(ramanujan_1914_pi, digits, reps)

        dec_chud = expand_in_base(chudnovsky_pi(digits), digits, base=10)
        dec_ram = expand_in_base(ramanujan_1914_pi(digits), digits, base=10)
        verification_rows[digits] = asdict(
            VerificationRow(
                target_digits=digits,
                candidate_correct_vs_chudnovsky=(hot_text == dec_chud),
                candidate_correct_vs_ramanujan_1914=(hot_text == dec_ram),
                first_mismatch_vs_chudnovsky=first_mismatch(hot_text, dec_chud),
                first_mismatch_vs_ramanujan_1914=first_mismatch(hot_text, dec_ram),
                legacy_cert_safe_digits=safe_digits,
                legacy_cert_seconds_one_run=cert_seconds,
                legacy_cert_q_terms=q_terms,
                legacy_cert_r_terms=r_terms,
            )
        )
        hot_summaries.append(asdict(HotSummary(digits, hot_one_run, hot_iters, hot_text[:82])))
        (outdir / f"pi_{digits}_decimal.txt").write_text(hot_text + "\n")
        (outdir / f"baseline_chudnovsky_{digits}.txt").write_text(dec_chud + "\n")
        (outdir / f"baseline_ramanujan_1914_{digits}.txt").write_text(dec_ram + "\n")

        frac = hot_text.split(".", 1)[1]
        write_csv(
            outdir / f"emission_trace_{digits}.csv",
            [asdict(EmissionRow(step=1, decimal_from=1, decimal_to=digits, newly_emitted=digits, digits_chunk=frac))],
        )

        benchmark_results["lengths"][str(digits)] = {
            "benchmark_repetitions": reps,
            "hot_path_seconds_one_run": hot_one_run,
            "hot_path_iterations": hot_iters,
            "hot_path_seconds_min": hot_best,
            "hot_path_seconds_mean": hot_mean,
            "hot_path_digits_per_second_mean": digits / hot_mean,
            "legacy_certificate_seconds_min": cert_best,
            "legacy_certificate_seconds_mean": cert_mean,
            "legacy_hot_plus_certificate_seconds_min": both_best,
            "legacy_hot_plus_certificate_seconds_mean": both_mean,
            "chudnovsky_seconds_min": chud_best,
            "chudnovsky_seconds_mean": chud_mean,
            "ramanujan_1914_seconds_min": ram_best,
            "ramanujan_1914_seconds_mean": ram_mean,
            "hot_vs_chudnovsky_speed_ratio_mean": hot_mean / chud_mean,
            "hot_vs_ramanujan_1914_speed_ratio_mean": hot_mean / ram_mean,
            "legacy_plus_cert_vs_target_0p0002": both_mean / 0.0002,
        }

    benchmark_results["headline"] = {
        "target_520_under_0p0002_hot_path": benchmark_results["lengths"]["520"]["hot_path_seconds_mean"] <= 0.0002,
        "legacy_current_law_with_legacy_certificate_under_0p0002": benchmark_results["lengths"]["520"][
            "legacy_hot_plus_certificate_seconds_mean"
        ] <= 0.0002,
        "structural_reason": "legacy current-law certification dominates the 520-digit wall-clock budget",
    }
    benchmark_results["impossibility_statement"] = {
        "machine_scope": "This statement is scoped to the current v6 C/MPFR regime on this machine.",
        "claim": "The old current-law regime with legacy residual certification does not clear 0.0002 s mean at 520 digits.",
        "evidence": {
            "hot_path_mean_520": benchmark_results["lengths"]["520"]["hot_path_seconds_mean"],
            "legacy_certificate_mean_520": benchmark_results["lengths"]["520"]["legacy_certificate_seconds_mean"],
            "legacy_hot_plus_certificate_mean_520": benchmark_results["lengths"]["520"]["legacy_hot_plus_certificate_seconds_mean"],
        },
    }

    write_csv(outdir / "packet_depth_exploration.csv", [asdict(x) for x in packet_depth_exploration(4)])
    (outdir / "benchmark_results.json").write_text(json.dumps(benchmark_results, indent=2) + "\n")
    (outdir / "verification_results.json").write_text(json.dumps(verification_rows, indent=2) + "\n")
    (outdir / "hot_summaries.json").write_text(json.dumps(hot_summaries, indent=2) + "\n")
    (outdir / "summary.txt").write_text(
        "\n".join(
            [
                "phase_native_streaming_spigot_bundle_v6",
                f"hot_520_mean={benchmark_results['lengths']['520']['hot_path_seconds_mean']}",
                f"legacy_plus_cert_520_mean={benchmark_results['lengths']['520']['legacy_hot_plus_certificate_seconds_mean']}",
                f"hot_1000_mean={benchmark_results['lengths']['1000']['hot_path_seconds_mean']}",
                f"hot_10000_mean={benchmark_results['lengths']['10000']['hot_path_seconds_mean']}",
                f"hot_520_vs_chud={benchmark_results['lengths']['520']['hot_vs_chudnovsky_speed_ratio_mean']}",
                f"hot_10000_vs_chud={benchmark_results['lengths']['10000']['hot_vs_chudnovsky_speed_ratio_mean']}",
                f"correct_520={verification_rows[520]['candidate_correct_vs_chudnovsky'] and verification_rows[520]['candidate_correct_vs_ramanujan_1914']}",
                f"correct_1000={verification_rows[1000]['candidate_correct_vs_chudnovsky'] and verification_rows[1000]['candidate_correct_vs_ramanujan_1914']}",
                f"correct_10000={verification_rows[10000]['candidate_correct_vs_chudnovsky'] and verification_rows[10000]['candidate_correct_vs_ramanujan_1914']}",
            ]
        )
        + "\n"
    )

    build_manifest(ROOT)


if __name__ == "__main__":
    main()
