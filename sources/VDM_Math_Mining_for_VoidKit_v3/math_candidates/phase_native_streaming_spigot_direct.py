#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import csv
import hashlib
import json
import math
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
    abs_update: str
    abs_residual: str
    error_bound: str
    safe_decimal_digits: int
    qp_terms_q: int
    qp_terms_r: int
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


@dataclass(frozen=True)
class BenchmarkResult:
    target_digits: int
    candidate_seconds_total: float
    chudnovsky_seconds_total: float
    ramanujan_1914_seconds_total: float
    candidate_digits_per_second: float
    chudnovsky_digits_per_second: float
    ramanujan_1914_digits_per_second: float
    candidate_vs_chudnovsky_speed_ratio: float
    candidate_vs_ramanujan_1914_speed_ratio: float

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
    if W < 1:
        raise ValueError("W must be positive")
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


def q_stats(q: mp.mpf, tol_digits: int | None = None) -> Tuple[mp.mpf, mp.mpf, mp.mpf, int]:
    """
    Return exact-in-practice statistics for the q-Pochhammer tail used by the direct packet law:

        log((q;q)_∞),   sum m q^m/(1-q^m),   sum m^2 q^m/(1-q^m)^2.

    All three are accumulated in one pass over q^m. This is the candidate path.
    """
    if not (mp.mpf("0") <= q < mp.mpf("1")):
        raise ValueError("q must lie in [0,1)")
    if tol_digits is None:
        tol_digits = mp.mp.dps + 10
    tol = mp.mpf(10) ** (-tol_digits)

    qm = mp.mpf(q)
    m = 1
    log_prod = mp.mpf("0")
    deriv_sum = mp.mpf("0")
    second_sum = mp.mpf("0")

    while qm > tol:
        inv = 1 / (1 - qm)
        log_prod += mp.log1p(-qm)
        deriv_sum += m * qm * inv
        second_sum += (m * m) * qm * (inv * inv)
        qm *= q
        m += 1
        if m > 10_000_000:
            raise RuntimeError("q-series accumulation did not converge in time")

    return log_prod, deriv_sum, second_sum, m - 1


def packet_exact_bundle(N: int, x: mp.mpf, tol_digits: int | None = None) -> Tuple[mp.mpf, mp.mpf, mp.mpf, Tuple[int, int]]:
    r"""
    Exact packet closure law on any balanced packet N = uv > 1:

        F_N(x) = 1/2 log N - log P_N(x) - ((N - N^{-1})/12) x + log P_N^*(x),

    with
        P_N(x)   = \prod_{m>=1} (1 - e^{-2mx/N}),
        P_N^*(x) = \prod_{m>=1} (1 - e^{-2Nmx}).

    The returned tuple is (F, F', F'', (terms_q, terms_r)).
    No external π value enters here.
    """
    if N <= 1:
        raise ValueError("exact packet law requires N > 1")
    if x <= 0:
        raise ValueError("x must be positive")

    Nf = mp.mpf(N)
    q = mp.e ** (-2 * x / Nf)
    r = mp.e ** (-2 * Nf * x)

    lq, dq, d2q, tq = q_stats(q, tol_digits=tol_digits)
    lr, dr, d2r, tr = q_stats(r, tol_digits=tol_digits)

    F = mp.log(Nf) / 2 - lq - (Nf - 1 / Nf) * x / 12 + lr
    Fp = -(2 / Nf) * dq - (Nf - 1 / Nf) / 12 + (2 * Nf) * dr
    Fpp = (4 / (Nf * Nf)) * d2q - (4 * Nf * Nf) * d2r
    return F, Fp, Fpp, (tq, tr)


def starter_newton_from_three(packet: PacketState, starter_dps: int = 30) -> mp.mpf:
    """
    Direct starter used by the native candidate path.

    The only seed is x0 = 3, i.e. an integer lower witness inside the packet-law bracket.
    One low-precision Newton correction is used to place the iterate in the fast Halley basin.
    """
    old_dps = mp.mp.dps
    try:
        mp.mp.dps = starter_dps
        x = mp.mpf(3)
        F, Fp, _, _ = packet_exact_bundle(packet.N, x, tol_digits=starter_dps + 10)
        return x - F / Fp
    finally:
        mp.mp.dps = old_dps


def adaptive_schedule(target_digits: int) -> List[int]:
    """
    Precision schedule tuned for the direct packet recurrence.
    For the current 500-digit target this yields [30, 60, 120, 240, 540].
    """
    if target_digits <= 0:
        raise ValueError("target_digits must be positive")
    schedule = [30, 60, 120, 240]
    final_dps = target_digits + 40
    if final_dps not in schedule:
        schedule.append(final_dps)
    return schedule


def safe_digits_from_error_bound(error_bound: mp.mpf, base: int = 10) -> int:
    """
    If |x - π| <= ε and ε < (1/2) base^{-d}, then the first d radix digits are stable.
    """
    if error_bound <= 0:
        return 10**9
    return max(0, int(mp.floor(-mp.log(2 * error_bound, base))))


def halley_update(F: mp.mpf, Fp: mp.mpf, Fpp: mp.mpf) -> mp.mpf:
    denom = 2 * Fp * Fp - F * Fpp
    if denom == 0:
        raise ZeroDivisionError("Halley denominator vanished")
    return 2 * F * Fp / denom


def error_bound_from_residual(packet: PacketState, residual: mp.mpf) -> mp.mpf:
    r"""
    A posteriori certification bound from strict monotonicity:

        F'_N(x) <= -(N - N^{-1})/12 < 0,
        |x - π| <= |F_N(x)| / ((N - N^{-1})/12).
    """
    Nf = mp.mpf(packet.N)
    slope_floor = (Nf - 1 / Nf) / 12
    return abs(residual) / slope_floor


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
    work_dps = max(mp.mp.dps, int(math.ceil(digits * math.log10(base))) + 25)
    with mp.workdps(work_dps):
        y = mp.mpf(x)
        n = int(mp.floor(y))
        frac = y - n
        int_part = int_to_base(n, base)
        out: List[str] = []
        for _ in range(digits):
            frac *= base
            d = int(mp.floor(frac))
            out.append(alphabet[d])
            frac -= d
        return int_part + "." + "".join(out)


def first_mismatch(a: str, b: str) -> int | None:
    fa = a.split(".", 1)[1]
    fb = b.split(".", 1)[1]
    for idx, (ca, cb) in enumerate(zip(fa, fb), start=1):
        if ca != cb:
            return idx
    return None


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


def run_direct_packet_stream(target_digits: int, outdir: Path) -> Tuple[str, List[IterationRow], List[EmissionRow], PacketState, float]:
    outdir.mkdir(parents=True, exist_ok=True)

    packet = packet_state(1)  # first nondegenerate balanced packet: (u,v)=(1,2), N=2
    x = starter_newton_from_three(packet, starter_dps=30)

    schedule = adaptive_schedule(target_digits)
    iteration_rows: List[IterationRow] = []
    emission_rows: List[EmissionRow] = []
    emitted_through = 0

    start = time.perf_counter()
    for step, work_dps in enumerate(schedule, start=1):
        mp.mp.dps = work_dps
        F, Fp, Fpp, _ = packet_exact_bundle(packet.N, x, tol_digits=work_dps + 10)
        dx = halley_update(F, Fp, Fpp)
        x = x - dx

        F_new, _, _, (terms_q, terms_r) = packet_exact_bundle(packet.N, x, tol_digits=work_dps + 10)
        bound = error_bound_from_residual(packet, F_new)
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
                abs_update=mp.nstr(abs(dx), n=min(50, work_dps)),
                abs_residual=mp.nstr(abs(F_new), n=min(50, work_dps)),
                error_bound=mp.nstr(bound, n=min(50, work_dps)),
                safe_decimal_digits=safe_dec,
                qp_terms_q=terms_q,
                qp_terms_r=terms_r,
                emitted_through=emitted_through,
            )
        )

    mp.mp.dps = target_digits + 20
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


def verify_and_benchmark(candidate_text: str, target_digits: int, outdir: Path) -> Tuple[VerificationResult, BenchmarkResult]:
    t0 = time.perf_counter()
    chud_text = expand_in_base(chudnovsky_pi(target_digits + 20), target_digits, base=10)
    chud_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    ram_text = expand_in_base(ramanujan_1914_pi(target_digits + 20), target_digits, base=10)
    ram_seconds = time.perf_counter() - t0

    verification = VerificationResult(
        target_digits=target_digits,
        candidate_correct_vs_chudnovsky=(candidate_text == chud_text),
        candidate_correct_vs_ramanujan_1914=(candidate_text == ram_text),
        first_mismatch_vs_chudnovsky=first_mismatch(candidate_text, chud_text),
        first_mismatch_vs_ramanujan_1914=first_mismatch(candidate_text, ram_text),
    )
    (outdir / "verification_results.json").write_text(json.dumps(verification.to_dict(), indent=2) + "\n")

    candidate_seconds = json.loads((outdir / "candidate_summary.json").read_text())["candidate_seconds_total"]
    benchmark = BenchmarkResult(
        target_digits=target_digits,
        candidate_seconds_total=float(candidate_seconds),
        chudnovsky_seconds_total=chud_seconds,
        ramanujan_1914_seconds_total=ram_seconds,
        candidate_digits_per_second=target_digits / float(candidate_seconds),
        chudnovsky_digits_per_second=target_digits / chud_seconds,
        ramanujan_1914_digits_per_second=target_digits / ram_seconds,
        candidate_vs_chudnovsky_speed_ratio=float(candidate_seconds) / chud_seconds,
        candidate_vs_ramanujan_1914_speed_ratio=float(candidate_seconds) / ram_seconds,
    )
    (outdir / "benchmark_results.json").write_text(json.dumps(benchmark.to_dict(), indent=2) + "\n")
    (outdir / f"baseline_chudnovsky_{target_digits}.txt").write_text(chud_text + "\n")
    (outdir / f"baseline_ramanujan_1914_{target_digits}.txt").write_text(ram_text + "\n")
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
    parser = argparse.ArgumentParser(description="Direct Phase packet π streamer (no generic root solver).")
    parser.add_argument("--target-digits", type=int, default=500)
    parser.add_argument("--outdir", type=Path, default=Path("phase_native_streaming_spigot_outputs"))
    args = parser.parse_args()

    candidate_text, iteration_rows, emission_rows, packet, total_seconds = run_direct_packet_stream(
        target_digits=args.target_digits,
        outdir=args.outdir,
    )
    verification, benchmark = verify_and_benchmark(candidate_text, args.target_digits, args.outdir)

    summary_lines = [
        f"packet_depth={packet.depth}",
        f"packet_u={packet.u}",
        f"packet_v={packet.v}",
        f"packet_N={packet.N}",
        f"candidate_seconds_total={total_seconds}",
        f"candidate_correct_vs_chudnovsky={verification.candidate_correct_vs_chudnovsky}",
        f"candidate_correct_vs_ramanujan_1914={verification.candidate_correct_vs_ramanujan_1914}",
        f"candidate_vs_chudnovsky_speed_ratio={benchmark.candidate_vs_chudnovsky_speed_ratio}",
        f"candidate_vs_ramanujan_1914_speed_ratio={benchmark.candidate_vs_ramanujan_1914_speed_ratio}",
        f"final_prefix_64={candidate_text[:66]}",
    ]
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
