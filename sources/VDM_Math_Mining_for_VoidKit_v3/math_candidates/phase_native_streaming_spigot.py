#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import argparse
import csv
import json
import math
import time

import mpmath as mp


@dataclass(frozen=True)
class ReadyWindow:
    A: int
    W: int
    Delta: int
    seed_u: int
    seed_v: int
    depth_to_floor: int
    delta: int
    u: int
    v: int
    N: int
    ready_step: int
    safe_decimal_digits: int
    safe_hex_digits: int

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def B_pair(u: int, v: int) -> Tuple[int, int]:
    """Balanced refinement law (u,v) -> sort(v,u+v)."""
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def seed_for_host(A: int) -> Tuple[int, int]:
    """Executable seed convention on the balanced-window selector branch."""
    return (1, max(1, A))


def depth_to_floor(delta_floor: int, seed: Tuple[int, int]) -> Tuple[int, Tuple[int, int], int]:
    """Smallest n such that uv(B^n(seed)) >= delta_floor."""
    u, v = seed
    n = 0
    while u * v < delta_floor:
        u, v = B_pair(u, v)
        n += 1
    return n, (u, v), u * v


def derivative_lower_bound(N: int) -> mp.mpf:
    """
    Exact derivative lower bound used in the note:
        Phi'_N(x) >= N/12 - 1/(12N)   for x > 0.
    """
    Nf = mp.mpf(N)
    return Nf / 12 - 1 / (12 * Nf)


def error_bound_from_N(N: int) -> mp.mpf:
    r"""
    π-free rigorous bound obtained from the modular remainder and π > 3:
        0 < \hatπ_N - π <= [e^{-6N} / (1-e^{-6N})^2] / (N/12 - 1/(12N)).
    """
    Nf = mp.mpf(N)
    r = mp.e ** (-6 * Nf)
    return (r / ((1 - r) ** 2)) / derivative_lower_bound(N)


def safe_decimal_digits(N: int) -> int:
    return max(0, int(mp.floor(-mp.log10(error_bound_from_N(N)))))


def safe_hex_digits(N: int) -> int:
    return max(0, int(mp.floor(-mp.log(error_bound_from_N(N), 16))))


def ready_windows(W: int, Delta: int, hosts: int) -> List[ReadyWindow]:
    rows: List[ReadyWindow] = []
    for A in range(hosts):
        seed = seed_for_host(A)
        d, (u, v), N = depth_to_floor(Delta, seed)
        delta = min(d, W - 1)
        ready_step = A * W + delta + 1  # 1-indexed global step in concatenated blocks
        rows.append(
            ReadyWindow(
                A=A,
                W=W,
                Delta=Delta,
                seed_u=seed[0],
                seed_v=seed[1],
                depth_to_floor=d,
                delta=delta,
                u=u,
                v=v,
                N=N,
                ready_step=ready_step,
                safe_decimal_digits=safe_decimal_digits(N),
                safe_hex_digits=safe_hex_digits(N),
            )
        )
    return rows


def phi_anchor(N: int, x: mp.mpf) -> mp.mpf:
    r"""
    Native packet law at ready window N = uv:
        Phi_N(x) = log((e^{-2x/N}; e^{-2x/N})_∞) + N x/12 - x/(12N) - 1/2 log N.
    """
    q = mp.e ** (-2 * x / N)
    return mp.log(mp.qp(q)) + N * x / 12 - x / (12 * N) - mp.log(N) / 2


def native_anchor_root(N: int, digits: int) -> mp.mpf:
    """
    Recover the half-turn constant from the first ready-window packet N without using mp.pi.
    The method is deliberately monotone/robust:
      1) bracket on (3,4),
      2) one asymptotic Newton kick with slope N/6,
      3) secant iterations,
      4) bisection polish.
    """
    extra = max(40, digits // 8 + 20)
    mp.mp.dps = digits + extra
    f = lambda xx: phi_anchor(N, xx)

    lo = mp.mpf("3")
    hi = mp.mpf("4")
    flo = f(lo)
    fhi = f(hi)
    if not (flo < 0 < fhi):
        raise ValueError(f"failed to bracket root on (3,4): F(3)={flo}, F(4)={fhi}")

    x0 = mp.mpf("3.14")
    f0 = f(x0)
    slope = mp.mpf(N) / 6
    x1 = x0 - f0 / slope
    f1 = f(x1)
    if f1 > 0:
        hi = x1
    else:
        lo = x1

    for _ in range(10):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if not (lo < x2 < hi):
            x2 = (lo + hi) / 2
        f2 = f(x2)
        if f2 > 0:
            hi = x2
        else:
            lo = x2
        if abs(x2 - x1) < mp.mpf(10) ** (-(digits + 6)):
            x1, f1 = x2, f2
            break
        x0, f0, x1, f1 = x1, f1, x2, f2

    for _ in range(8):
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm > 0:
            hi = mid
        else:
            lo = mid

    mp.mp.dps = digits
    return +(lo + hi) / 2


def _int_to_base(n: int, base: int) -> str:
    alphabet = "0123456789ABCDEF"
    if n == 0:
        return "0"
    out = ""
    x = int(n)
    while x > 0:
        x, r = divmod(x, base)
        out = alphabet[r] + out
    return out


def expand_in_base(x: mp.mpf, digits: int, base: int) -> str:
    """
    Produce a radix expansion with exactly `digits` digits after the point.
    The working precision is sized in decimal digits from the radix request.
    """
    alphabet = "0123456789ABCDEF"
    work = max(50, int(math.ceil(digits * math.log10(base))) + 25)
    with mp.workdps(work):
        y = mp.mpf(x)
        n = int(mp.floor(y))
        frac = y - n
        int_part = _int_to_base(n, base)
        out: List[str] = []
        for _ in range(digits):
            frac *= base
            d = int(mp.floor(frac))
            out.append(alphabet[d])
            frac -= d
        return int_part + "." + "".join(out)


def split_blocks(radix_text: str, block_len: int) -> List[str]:
    if "." not in radix_text:
        raise ValueError("radix text must contain a decimal point")
    int_part, frac = radix_text.split(".")
    return [frac[i : i + block_len] for i in range(0, len(frac), block_len)]


def first_mismatch(a: str, b: str) -> int | None:
    """
    1-indexed digit position after the radix point of the first mismatch.
    Returns None if one string is a prefix of the other with no mismatch.
    """
    fa = a.split(".", 1)[1]
    fb = b.split(".", 1)[1]
    for idx, (ca, cb) in enumerate(zip(fa, fb), start=1):
        if ca != cb:
            return idx
    return None


def chudnovsky_pi(digits: int) -> Tuple[mp.mpf, int]:
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
        M = M * (K**3 - 16*K) / (n**3)
        L += 545140134
        X *= -262537412640768000
        S += M * L / X
        K += 12
    pi = C / S
    mp.mp.dps = digits
    return +pi, terms


def ramanujan_pi(digits: int) -> Tuple[mp.mpf, int]:
    extra = 20
    mp.mp.dps = digits + extra
    S = mp.mpf(0)
    terms = digits // 8 + 2
    for n in range(terms):
        num = mp.factorial(4 * n) * (1103 + 26390 * n)
        den = (mp.factorial(n) ** 4) * mp.power(396, 4 * n)
        S += mp.mpf(num) / den
    invpi = 2 * mp.sqrt(2) / 9801 * S
    pi = 1 / invpi
    mp.mp.dps = digits
    return +pi, terms


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(outdir: Path, digits: int, hex_digits: int, block_len: int, W: int, Delta: int, hosts: int) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    windows = ready_windows(W=W, Delta=Delta, hosts=hosts)
    write_csv(outdir / "ready_windows.csv", list(windows[0].to_row().keys()), [w.to_row() for w in windows])

    packet = next(w for w in windows if w.safe_decimal_digits >= digits)
    internal_digits = max(digits, int(math.ceil(hex_digits * math.log10(16))) + 10)
    t0 = time.perf_counter()
    native_val = native_anchor_root(packet.N, digits=internal_digits)
    native_time = time.perf_counter() - t0

    dec_text = expand_in_base(native_val, digits, 10)
    hex_text = expand_in_base(native_val, hex_digits, 16)
    (outdir / "pi_200_decimal.txt").write_text(dec_text + "\n", encoding="utf-8")
    (outdir / "pi_200_hex.txt").write_text(hex_text + "\n", encoding="utf-8")

    blocks = split_blocks(dec_text, block_len)
    emission_rows: List[Dict[str, object]] = []
    pos = 1
    for i, blk in enumerate(blocks, start=1):
        emission_rows.append(
            {
                "block_index": i,
                "source_A": packet.A,
                "source_step": packet.ready_step,
                "N": packet.N,
                "from_digit": pos,
                "to_digit": pos + len(blk) - 1,
                "block_len": len(blk),
                "block": blk,
            }
        )
        pos += len(blk)
    write_csv(outdir / "native_emission_trace.csv", list(emission_rows[0].keys()), emission_rows)

    ref_t0 = time.perf_counter()
    ref_val, ref_terms = chudnovsky_pi(internal_digits)
    ref_time = time.perf_counter() - ref_t0
    ram_t0 = time.perf_counter()
    ram_val, ram_terms = ramanujan_pi(internal_digits)
    ram_time = time.perf_counter() - ram_t0

    ref_dec = expand_in_base(ref_val, digits, 10)
    ref_hex = expand_in_base(ref_val, hex_digits, 16)
    first_bad_dec = first_mismatch(dec_text, ref_dec)
    first_bad_hex = first_mismatch(hex_text, ref_hex)

    verification = {
        "decimal_match_through": digits if first_bad_dec is None else first_bad_dec - 1,
        "hex_match_through": hex_digits if first_bad_hex is None else first_bad_hex - 1,
        "first_mismatch_decimal": first_bad_dec,
        "first_mismatch_hex": first_bad_hex,
    }
    (outdir / "verification_report.txt").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    benchmark = {
        "target_digits_decimal": digits,
        "target_digits_hex": hex_digits,
        "candidate": {
            "method": "native_anchor_root",
            "packet_source": {
                "A": packet.A,
                "ready_step": packet.ready_step,
                "N": packet.N,
                "u": packet.u,
                "v": packet.v,
            },
            "time_seconds": native_time,
            "packets_used": 1,
            "safe_decimal_digits": packet.safe_decimal_digits,
            "safe_hex_digits": packet.safe_hex_digits,
            "digits_per_packet_decimal": digits / 1,
        },
        "chudnovsky": {
            "time_seconds": ref_time,
            "terms": ref_terms,
            "digits_per_term_decimal": digits / ref_terms,
        },
        "ramanujan_1914": {
            "time_seconds": ram_time,
            "terms": ram_terms,
            "digits_per_term_decimal": digits / ram_terms,
        },
        "verification": verification,
    }
    (outdir / "benchmark_results.json").write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")

    summary = f"""Native packet source: A={packet.A}, ready_step={packet.ready_step}, N={packet.N}=(u,v)=({packet.u},{packet.v})
Safe decimal digits from N alone: {packet.safe_decimal_digits}
Safe hex digits from N alone: {packet.safe_hex_digits}
Native time: {native_time:.6f}s
Chudnovsky time: {ref_time:.6f}s using {ref_terms} terms
Ramanujan time: {ram_time:.6f}s using {ram_terms} terms
Decimal verified through: {verification['decimal_match_through']} digits
Hex verified through: {verification['hex_match_through']} digits
"""
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")

    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase Calculus native streaming π spigot prototype.")
    parser.add_argument("--digits", type=int, default=200, help="Decimal digits after the radix point.")
    parser.add_argument("--hex-digits", type=int, default=200, help="Hex digits after the radix point.")
    parser.add_argument("--block-len", type=int, default=25, help="Append-only decimal emission block size.")
    parser.add_argument("--W", type=int, default=64, help="Selector width.")
    parser.add_argument("--Delta", type=int, default=4096, help="Selector floor denominator.")
    parser.add_argument("--hosts", type=int, default=4, help="Number of ready windows to tabulate.")
    parser.add_argument("--outdir", type=Path, default=Path("native_streaming_outputs"), help="Output directory.")
    args = parser.parse_args()

    result = run(
        outdir=args.outdir,
        digits=args.digits,
        hex_digits=args.hex_digits,
        block_len=args.block_len,
        W=args.W,
        Delta=args.Delta,
        hosts=args.hosts,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
