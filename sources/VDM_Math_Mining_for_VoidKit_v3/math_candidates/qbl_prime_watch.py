#!/usr/bin/env python3
"""QBL prime-pattern tracker.

This program uses only the clean primitive/domain laws:

    N_A = 6 * 2^A
    Q_A = N_A - 1

and, for B counts, the current terminal floor-crossing convention:

    Delta_A = 2^(12 * (2^(A+1) - 1))
    T_A = min m such that F_(m+1) F_(m+2) >= Delta_A
    B_A = T_A - T_(A-1), with T_(-1) = 0.

No R/S/T scheduler, fixed window, or macro grammar is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import mpmath as mp
from sympy import factorint, isprime
from sympy.ntheory import n_order


@dataclass(frozen=True)
class DomainPrimeRow:
    A: int
    q_count: int
    q_prime: bool
    b_count: int
    b_prime: bool
    b_double_correction: int | None

    @property
    def both_prime(self) -> bool:
        return self.q_prime and self.b_prime


def q_count(A: int) -> int:
    if A < 0:
        raise ValueError("A must be nonnegative")
    return 6 * (1 << A) - 1


def final_orientation_index(A: int) -> int:
    if A < 0:
        raise ValueError("A must be nonnegative")
    return 6 * ((1 << (A + 1)) - 1)


def final_capacity_exponent(A: int) -> int:
    """Return e_A where Delta_A = 2^e_A."""
    return 2 * final_orientation_index(A)


def _required_dps(A: int, guard: int = 100) -> int:
    # T_A is O(2^A), so its decimal length is approximately A*log10(2).
    return max(100, int((A + 2) * 0.30103) + guard)


def _log_fib_product(m: int) -> mp.mpf:
    """Exact real logarithm of F_(m+1) F_(m+2) via Binet's formula.

    F_n = phi^n / sqrt(5) * (1 - (-1)^n phi^(-2n)).
    The tiny correction is retained whenever it is resolvable at the active precision.
    """
    if m < 0:
        raise ValueError("m must be nonnegative")

    phi = (1 + mp.sqrt(5)) / 2
    log_phi = mp.log(phi)
    lead = (2 * m + 3) * log_phi - mp.log(5)

    def correction(n: int) -> mp.mpf:
        decay_log = -2 * n * log_phi
        # If the correction is far beneath the working precision, replacing it by
        # zero is smaller than the numerical comparison margin used below.
        if -decay_log > (mp.mp.dps + 20) * mp.log(10):
            return mp.mpf("0")
        x = mp.e ** decay_log
        return mp.log1p(-x if n % 2 == 0 else x)

    return lead + correction(m + 1) + correction(m + 2)


def cumulative_b_steps(A: int) -> int:
    """Certified threshold index T_A for the current floor-crossing convention."""
    if A < 0:
        return 0

    dps = _required_dps(A)
    while True:
        with mp.workdps(dps):
            phi = (1 + mp.sqrt(5)) / 2
            log_phi = mp.log(phi)
            log_delta = final_capacity_exponent(A) * mp.log(2)

            # Leading Binet root. The exact correction is exponentially tiny and
            # the candidate is then certified by the two adjacent inequalities.
            root = (log_delta + mp.log(5)) / (2 * log_phi) - mp.mpf("1.5")
            m = int(mp.ceil(root))

            while m > 0 and _log_fib_product(m - 1) >= log_delta:
                m -= 1
            while _log_fib_product(m) < log_delta:
                m += 1

            left_gap = log_delta - (_log_fib_product(m - 1) if m > 0 else mp.ninf)
            right_gap = _log_fib_product(m) - log_delta
            margin = mp.power(10, -(dps // 2))

            if left_gap > margin and right_gap > margin:
                return m

        dps *= 2
        if dps > 20000:
            raise ArithmeticError(f"Could not certify B threshold for A={A}")


def b_count(A: int) -> int:
    if A < 0:
        raise ValueError("A must be nonnegative")
    return cumulative_b_steps(A) - cumulative_b_steps(A - 1)


def scan(max_A: int) -> list[DomainPrimeRow]:
    if max_A < 0:
        raise ValueError("max_A must be nonnegative")

    totals = [cumulative_b_steps(A) for A in range(max_A + 1)]
    rows: list[DomainPrimeRow] = []
    prior = 0
    for A, total in enumerate(totals):
        b = total - prior
        q = q_count(A)
        correction = None if A == 0 else b - 2 * rows[-1].b_count
        rows.append(DomainPrimeRow(A, q, bool(isprime(q)), b, bool(isprime(b)), correction))
        prior = total
    return rows


def q_divisor_progressions(prime_limit: int) -> list[dict[str, int]]:
    """Find periodic A-classes on which small primes divide Q_A.

    For p not dividing 6, solve 2^A = 6^(-1) mod p. If a solution exists,
    all solutions repeat modulo ord_p(2).
    """
    from sympy import primerange

    result: list[dict[str, int]] = []
    for p in primerange(5, prime_limit + 1):
        inv6 = pow(6, -1, p)
        order = int(n_order(2, p))
        residue = None
        value = 1
        for A in range(order):
            if value == inv6:
                residue = A
                break
            value = (value * 2) % p
        if residue is not None:
            result.append({"prime": int(p), "residue": residue, "modulus": order})
    return result


def _fmt_factorization(n: int) -> str:
    if isprime(n):
        return "prime"
    factors = factorint(n)
    return " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in factors.items())


def render_text(rows: Iterable[DomainPrimeRow], factor_composites: bool = False) -> str:
    rows = list(rows)
    lines = [
        " A | Q_A | Q prime | B_A | B prime | d_A=B_A-2B_(A-1) | both",
        "---+-----+---------+-----+---------+---------------------+-----",
    ]
    for row in rows:
        q_text = str(row.q_count)
        b_text = str(row.b_count)
        if factor_composites:
            if not row.q_prime:
                q_text += f" ({_fmt_factorization(row.q_count)})"
            if not row.b_prime:
                b_text += f" ({_fmt_factorization(row.b_count)})"
        lines.append(
            f"{row.A:>2} | {q_text} | {'yes' if row.q_prime else 'no ':>7} | "
            f"{b_text} | {'yes' if row.b_prime else 'no ':>7} | "
            f"{str(row.b_double_correction) if row.b_double_correction is not None else '-':>19} | "
            f"{'YES' if row.both_prime else ''}"
        )

    q_indices = [r.A for r in rows if r.q_prime]
    b_indices = [r.A for r in rows if r.b_prime]
    both = [r.A for r in rows if r.both_prime]
    lines.extend(
        [
            "",
            f"Q-prime domains: {q_indices}",
            f"B-prime domains: {b_indices}",
            f"simultaneous-prime domains: {both}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan QBL domain counts for prime structure.")
    parser.add_argument("--max-a", type=int, default=100, help="largest domain A to scan")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--factor-composites",
        action="store_true",
        help="factor composite values in the table (best for small --max-a)",
    )
    parser.add_argument(
        "--q-sieve-primes",
        type=int,
        default=0,
        metavar="P",
        help="also emit all Q_A divisor progressions generated by primes <= P",
    )
    args = parser.parse_args()

    rows = scan(args.max_a)
    sieve = q_divisor_progressions(args.q_sieve_primes) if args.q_sieve_primes else []

    if args.json:
        payload = {
            "law": {
                "Q_A": "6*2^A - 1 = 3*2^(A+1) - 1",
                "Delta_A": "2^(12*(2^(A+1)-1))",
                "T_A": "min m: F_(m+1)*F_(m+2) >= Delta_A",
                "B_A": "T_A-T_(A-1)",
            },
            "rows": [{**asdict(r), "both_prime": r.both_prime} for r in rows],
            "q_divisor_progressions": sieve,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(rows, factor_composites=args.factor_composites))
        if sieve:
            print("\nQ divisor progressions A ≡ r (mod ord_p(2)):")
            for item in sieve:
                print(
                    f"  p={item['prime']}: A ≡ {item['residue']} "
                    f"(mod {item['modulus']})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
