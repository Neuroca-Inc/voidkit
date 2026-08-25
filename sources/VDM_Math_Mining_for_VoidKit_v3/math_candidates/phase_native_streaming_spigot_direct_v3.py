#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import csv
import hashlib
import json
import time

import mpmath as mp


@dataclass(frozen=True)
class PacketState:
    depth: int
    u: int
    v: int
    N: int
    delta: int
    block_word: str
    execution_coordinate: int

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IterationRow:
    step: int
    work_dps: int
    packet_depth: int
    u: int
    v: int
    N: int
    q_terms: int
    r_terms: int
    abs_update: str
    abs_residual: str
    error_bound: str
    safe_decimal_digits: int
    emitted_through: int

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EmissionRow:
    step: int
    decimal_from: int
    decimal_to: int
    newly_emitted: int
    digits_chunk: str

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    target_digits: int
    candidate_correct_vs_chudnovsky: bool
    candidate_correct_vs_ramanujan_1914: bool
    first_mismatch_vs_chudnovsky: int | None
    first_mismatch_vs_ramanujan_1914: int | None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def balanced_pair(depth: int) -> Tuple[int, int]:
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    return fibonacci(depth + 1), fibonacci(depth + 2)


def canonical_block_word(delta: int, W: int = 64) -> str:
    if not (0 <= delta <= W - 1):
        raise ValueError("delta must lie in [0, W-1]")
    return "R" * delta + "S" * (W - 1 - delta) + "T"


def execution_coordinate(delta: int, W: int = 64) -> int:
    return (3**delta - 1) // 2 + 2 * 3 ** (W - 1)


def packet_state(depth: int, W: int = 64) -> PacketState:
    u, v = balanced_pair(depth)
    delta = min(depth, W - 1)
    return PacketState(
        depth=depth,
        u=u,
        v=v,
        N=u * v,
        delta=delta,
        block_word=canonical_block_word(delta, W=W),
        execution_coordinate=execution_coordinate(delta, W=W),
    )


def packet_family(max_depth: int = 10, W: int = 64) -> List[PacketState]:
    return [packet_state(depth, W=W) for depth in range(max_depth + 1)]


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


def first_mismatch(a: str, b: str) -> int | None:
    fa = a.split(".", 1)[1]
    fb = b.split(".", 1)[1]
    for idx, (ca, cb) in enumerate(zip(fa, fb), start=1):
        if ca != cb:
            return idx
    return None


def pentagonal_bundle_fast(q: mp.mpf, tol_digits: int | None = None) -> Tuple[mp.mpf, mp.mpf, mp.mpf, int]:
    """
    Exact Euler pentagonal evaluation of (q;q)_∞ and its first two x-derivative coefficient sums.

    Returns
        P(q)   = (q;q)_∞
        S1(q)  = Σ c_a a q^a
        S2(q)  = Σ c_a a^2 q^a
        term_count
    where the generalized pentagonal coefficients are c_a ∈ {+1,-1}.
    """
    if not (mp.mpf("0") <= q < mp.mpf("1")):
        raise ValueError("q must lie in [0,1)")
    if tol_digits is None:
        tol_digits = mp.mp.dps + 10
    tol = mp.mpf(10) ** (-tol_digits)

    P = mp.mpf(1)
    S1 = mp.mpf(0)
    S2 = mp.mpf(0)

    k = 1
    t_minus = q
    t_plus = q * q

    power_cache = {1: q, 2: q * q}
    cur_n = 2
    q_cur = q * q

    def q_to(n: int) -> mp.mpf:
        nonlocal cur_n, q_cur
        while cur_n < n:
            q_cur *= q
            cur_n += 1
            power_cache[cur_n] = q_cur
        return power_cache[n]

    while True:
        if t_minus < tol and t_plus < tol:
            break

        a1 = k * (3 * k - 1) // 2
        a2 = a1 + k
        sign = -1 if (k & 1) else 1

        P += sign * (t_minus + t_plus)
        S1 += sign * (a1 * t_minus + a2 * t_plus)
        S2 += sign * (a1 * a1 * t_minus + a2 * a2 * t_plus)

        t_minus = t_plus * q_to(2 * k + 1)
        t_plus = t_minus * q_to(k + 1)
        k += 1
        if k > 1_000_000:
            raise RuntimeError("Pentagonal evaluation did not converge")

    return P, S1, S2, k - 1


def packet_bundle_pentagonal(N: int, x: mp.mpf, tol_digits: int | None = None) -> Tuple[mp.mpf, mp.mpf, mp.mpf, int, int]:
    """
    Native packet law and derivatives using only q = exp(-2x/N), r = exp(-2Nx),
    and the exact Euler pentagonal theorem for (q;q)_∞ and (r;r)_∞.

    No external π, no generic solver, no classical series oracle.
    """
    if N <= 1:
        raise ValueError("N must exceed 1")
    if x <= 0:
        raise ValueError("x must be positive")

    Nf = mp.mpf(N)
    q = mp.e ** (-2 * x / Nf)
    r = mp.e ** (-2 * Nf * x)

    Pq, Sq1, Sq2, tq = pentagonal_bundle_fast(q, tol_digits=tol_digits)
    Pr, Sr1, Sr2, tr = pentagonal_bundle_fast(r, tol_digits=tol_digits)

    q1 = (-2 / Nf) * Sq1
    q2 = (4 / (Nf * Nf)) * Sq2
    r1 = (-2 * Nf) * Sr1
    r2 = (4 * Nf * Nf) * Sr2

    F = mp.log(Nf) / 2 - mp.log(Pq) - (Nf - 1 / Nf) * x / 12 + mp.log(Pr)
    Fp = -(q1 / Pq) - (Nf - 1 / Nf) / 12 + (r1 / Pr)
    Fpp = -((q2 / Pq) - (q1 / Pq) ** 2) + ((r2 / Pr) - (r1 / Pr) ** 2)
    return F, Fp, Fpp, tq, tr


def newton_starter(N: int, x0: mp.mpf, starter_dps: int = 25) -> mp.mpf:
    old = mp.mp.dps
    try:
        mp.mp.dps = starter_dps
        F, Fp, _, _, _ = packet_bundle_pentagonal(N, x0, tol_digits=starter_dps + 10)
        return x0 - F / Fp
    finally:
        mp.mp.dps = old


def halley_update(F: mp.mpf, Fp: mp.mpf, Fpp: mp.mpf) -> mp.mpf:
    denom = 2 * Fp * Fp - F * Fpp
    if denom == 0:
        raise ZeroDivisionError("Halley denominator vanished")
    return 2 * F * Fp / denom


def error_bound_from_residual(N: int, residual: mp.mpf) -> mp.mpf:
    Nf = mp.mpf(N)
    slope_floor = (Nf - 1 / Nf) / 12
    return abs(residual) / slope_floor


def safe_digits_from_error_bound(error_bound: mp.mpf, base: int = 10) -> int:
    if error_bound <= 0:
        return 10**9
    return max(0, int(mp.floor(-mp.log(2 * error_bound, base))))


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


def benchmark_mean_seconds(fn, digits: int, reps: int) -> Tuple[float, float]:
    timings: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(digits)
        timings.append(time.perf_counter() - t0)
    return min(timings), sum(timings) / len(timings)




def compute_candidate_value(target_digits: int) -> mp.mpf:
    packet = packet_state(1)
    schedule = [25, 55, 110, 220, 520]
    x = newton_starter(packet.N, mp.mpf(3), starter_dps=schedule[0])
    for work_dps in schedule:
        mp.mp.dps = work_dps
        F, Fp, Fpp, _, _ = packet_bundle_pentagonal(packet.N, x, tol_digits=work_dps + 10)
        x = x - halley_update(F, Fp, Fpp)
    mp.mp.dps = max(540, target_digits + 20)
    return +x



def run_candidate_stream(target_digits: int, outdir: Path) -> Tuple[str, List[IterationRow], List[EmissionRow], PacketState, float]:
    outdir.mkdir(parents=True, exist_ok=True)

    packet = packet_state(1)  # (u,v)=(1,2), N=2
    schedule = [25, 55, 110, 220, 520]
    x = newton_starter(packet.N, mp.mpf(3), starter_dps=schedule[0])

    iteration_rows: List[IterationRow] = []
    emission_rows: List[EmissionRow] = []
    emitted_through = 0

    start = time.perf_counter()
    for step, work_dps in enumerate(schedule, start=1):
        mp.mp.dps = work_dps
        F, Fp, Fpp, tq, tr = packet_bundle_pentagonal(packet.N, x, tol_digits=work_dps + 10)
        dx = halley_update(F, Fp, Fpp)
        x = x - dx

        F_new, _, _, tq_new, tr_new = packet_bundle_pentagonal(packet.N, x, tol_digits=work_dps + 10)
        bound = error_bound_from_residual(packet.N, F_new)
        safe_dec = safe_digits_from_error_bound(bound, base=10)

        new_through = min(target_digits, safe_dec)
        if new_through > emitted_through:
            mp.mp.dps = max(work_dps, target_digits + 20)
            text_now = expand_in_base(x, new_through, base=10)
            chunk = text_now.split(".", 1)[1][emitted_through:new_through]
            emission_rows.append(
                EmissionRow(
                    step=step,
                    decimal_from=emitted_through + 1,
                    decimal_to=new_through,
                    newly_emitted=new_through - emitted_through,
                    digits_chunk=chunk,
                )
            )
            emitted_through = new_through

        iteration_rows.append(
            IterationRow(
                step=step,
                work_dps=work_dps,
                packet_depth=packet.depth,
                u=packet.u,
                v=packet.v,
                N=packet.N,
                q_terms=tq_new,
                r_terms=tr_new,
                abs_update=mp.nstr(abs(dx), n=min(50, work_dps)),
                abs_residual=mp.nstr(abs(F_new), n=min(50, work_dps)),
                error_bound=mp.nstr(bound, n=min(50, work_dps)),
                safe_decimal_digits=safe_dec,
                emitted_through=emitted_through,
            )
        )

    mp.mp.dps = max(540, target_digits + 20)
    candidate_text = expand_in_base(x, target_digits, base=10)
    total_seconds = time.perf_counter() - start

    (outdir / f"pi_{target_digits}_decimal.txt").write_text(candidate_text + "\n")
    write_csv(outdir / "iteration_trace.csv", [row.to_row() for row in iteration_rows])
    write_csv(outdir / "emission_trace.csv", [row.to_row() for row in emission_rows])
    write_csv(outdir / "packet_family.csv", [state.to_row() for state in packet_family(max_depth=10)])

    summary = {
        "target_digits": target_digits,
        "packet_depth": packet.depth,
        "packet_u": packet.u,
        "packet_v": packet.v,
        "packet_N": packet.N,
        "block_word": packet.block_word,
        "execution_coordinate": str(packet.execution_coordinate),
        "candidate_seconds_total": total_seconds,
        "final_emitted_digits": emitted_through,
        "final_prefix_64": candidate_text[:66],
    }
    (outdir / "candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return candidate_text, iteration_rows, emission_rows, packet, total_seconds


def verify_and_benchmark(candidate_text: str, target_digits: int, outdir: Path, candidate_seconds_total: float, benchmark_reps: int) -> Tuple[VerificationResult, Dict[str, float]]:
    chud_text = expand_in_base(chudnovsky_pi(target_digits + 20), target_digits, base=10)
    ram_text = expand_in_base(ramanujan_1914_pi(target_digits + 20), target_digits, base=10)

    verification = VerificationResult(
        target_digits=target_digits,
        candidate_correct_vs_chudnovsky=(candidate_text == chud_text),
        candidate_correct_vs_ramanujan_1914=(candidate_text == ram_text),
        first_mismatch_vs_chudnovsky=first_mismatch(candidate_text, chud_text),
        first_mismatch_vs_ramanujan_1914=first_mismatch(candidate_text, ram_text),
    )
    (outdir / "verification_results.json").write_text(json.dumps(verification.to_dict(), indent=2) + "\n")
    (outdir / f"baseline_chudnovsky_{target_digits}.txt").write_text(chud_text + "\n")
    (outdir / f"baseline_ramanujan_1914_{target_digits}.txt").write_text(ram_text + "\n")

    cand_min, cand_mean = benchmark_mean_seconds(compute_candidate_value, target_digits, benchmark_reps)
    chud_min, chud_mean = benchmark_mean_seconds(chudnovsky_pi, target_digits, benchmark_reps)
    ram_min, ram_mean = benchmark_mean_seconds(ramanujan_1914_pi, target_digits, benchmark_reps)

    benchmark = {
        "target_digits": target_digits,
        "benchmark_repetitions": benchmark_reps,
        "candidate_seconds_artifact_run": candidate_seconds_total,
        "candidate_seconds_min": cand_min,
        "candidate_seconds_mean": cand_mean,
        "chudnovsky_seconds_min": chud_min,
        "chudnovsky_seconds_mean": chud_mean,
        "ramanujan_1914_seconds_min": ram_min,
        "ramanujan_1914_seconds_mean": ram_mean,
        "candidate_digits_per_second_mean": target_digits / cand_mean,
        "chudnovsky_digits_per_second_mean": target_digits / chud_mean,
        "ramanujan_1914_digits_per_second_mean": target_digits / ram_mean,
        "candidate_vs_chudnovsky_speed_ratio_mean": cand_mean / chud_mean,
        "candidate_vs_ramanujan_1914_speed_ratio_mean": cand_mean / ram_mean,
    }
    (outdir / "benchmark_results.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    return verification, benchmark


def build_manifest(bundle_dir: Path) -> None:
    files = []
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(bundle_dir)
        files.append({
            "path": str(rel),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {"bundle": bundle_dir.name, "files": files}
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (bundle_dir / "SHA256SUMS.txt").open("w") as f:
        for entry in files:
            f.write(f"{entry['sha256']}  {entry['path']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized native Phase packet π streamer (Euler-pentagonal accelerator).")
    parser.add_argument("--target-digits", type=int, default=520)
    parser.add_argument("--outdir", type=Path, default=Path("phase_native_streaming_spigot_outputs_v3"))
    parser.add_argument("--benchmark-reps", type=int, default=15)
    args = parser.parse_args()

    candidate_text, iteration_rows, emission_rows, packet, total_seconds = run_candidate_stream(
        target_digits=args.target_digits,
        outdir=args.outdir,
    )
    verification, benchmark = verify_and_benchmark(
        candidate_text,
        args.target_digits,
        args.outdir,
        total_seconds,
        args.benchmark_reps,
    )

    summary_lines = [
        f"packet_depth={packet.depth}",
        f"packet_u={packet.u}",
        f"packet_v={packet.v}",
        f"packet_N={packet.N}",
        f"candidate_seconds_artifact_run={total_seconds}",
        f"candidate_seconds_mean={benchmark['candidate_seconds_mean']}",
        f"chudnovsky_seconds_mean={benchmark['chudnovsky_seconds_mean']}",
        f"ramanujan_1914_seconds_mean={benchmark['ramanujan_1914_seconds_mean']}",
        f"candidate_correct_vs_chudnovsky={verification.candidate_correct_vs_chudnovsky}",
        f"candidate_correct_vs_ramanujan_1914={verification.candidate_correct_vs_ramanujan_1914}",
        f"candidate_vs_chudnovsky_speed_ratio_mean={benchmark['candidate_vs_chudnovsky_speed_ratio_mean']}",
        f"candidate_vs_ramanujan_1914_speed_ratio_mean={benchmark['candidate_vs_ramanujan_1914_speed_ratio_mean']}",
        f"final_prefix_64={candidate_text[:66]}",
    ]
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
