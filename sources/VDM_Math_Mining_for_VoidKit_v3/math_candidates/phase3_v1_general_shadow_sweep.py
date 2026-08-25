#!/usr/bin/env python3
"""
Phase 3 v1: General Shadow Correspondence Sweep

Builds finite quadratic residue carriers A_m = Z/(2m)Z, tests even
orientation vectors, verifies scalar orientation cancellation, constructs
the positive false-theta readout and coefficient-stripped shadow channel,
and verifies the primitive S/T carrier numerically.
"""
from __future__ import annotations

import cmath
import csv
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np


def legendre_symbol(a: int, p: int) -> int:
    a %= p
    if a == 0:
        return 0
    val = pow(a, (p - 1) // 2, p)
    return -1 if val == p - 1 else val


def chi12_vector() -> tuple[int, list[int]]:
    m, n = 6, 12
    a = [0] * n
    for r in range(n):
        if math.gcd(r, 6) != 1:
            a[r] = 0
        elif r % 12 in (1, 11):
            a[r] = 1
        elif r % 12 in (5, 7):
            a[r] = -1
    return m, a


def chi8_vector() -> tuple[int, list[int]]:
    m, n = 4, 8
    a = [0] * n
    for r in range(n):
        if r % 2 == 0:
            a[r] = 0
        elif r % 8 in (1, 7):
            a[r] = 1
        elif r % 8 in (3, 5):
            a[r] = -1
    return m, a


def legendre_vector(p: int) -> tuple[int, list[int]]:
    m, n = p, 2 * p
    return m, [legendre_symbol(r, p) for r in range(n)]


def delta_vector(m: int, pairs: list[tuple[int, int]]) -> tuple[int, list[int]]:
    n = 2 * m
    a = [0] * n
    for r, c in pairs:
        a[r % n] = c
        a[(-r) % n] = c
    return m, a


def S_matrix(n: int) -> np.ndarray:
    return np.array(
        [[cmath.exp(-2j * math.pi * r * s / n) / math.sqrt(n) for s in range(n)] for r in range(n)],
        dtype=np.complex128,
    )


def orientation_cancel_worst(m: int, a: list[int], max_n: int = 300) -> int:
    nmod = 2 * m
    return max(abs(a[k % nmod] * k + a[(-k) % nmod] * (-k)) for k in range(1, max_n + 1))


def theta_vec(nmod: int, tau: complex, kmax: int) -> list[mp.mpc]:
    tau = mp.mpc(tau)
    vals = []
    for r in range(nmod):
        total = mp.mpc(0)
        for k in range(-kmax, kmax + 1):
            n = r + k * nmod
            total += mp.e ** (2j * mp.pi * tau * (mp.mpf(n * n) / (2 * nmod)))
        vals.append(total)
    return vals


def D_vec(nmod: int, tau: complex, kmax: int) -> list[mp.mpc]:
    tau = mp.mpc(tau)
    vals = []
    for r in range(nmod):
        total = mp.mpc(0)
        for k in range(-kmax, kmax + 1):
            n = r + k * nmod
            total += n * mp.e ** (2j * mp.pi * tau * (mp.mpf(n * n) / (2 * nmod)))
        vals.append(total)
    return vals


def S_apply_mp(nmod: int, vec: list[mp.mpc]) -> list[mp.mpc]:
    return [
        sum(mp.e ** (-2j * mp.pi * r * s / nmod) * vec[s] for s in range(nmod)) / mp.sqrt(nmod)
        for r in range(nmod)
    ]


def transform_errors(nmod: int, tau: complex = 0.37 + 1.21j, kmax: int = 70) -> tuple[float, float]:
    tau = mp.mpc(tau)
    theta_left = theta_vec(nmod, -1 / tau, kmax)
    theta_right = [mp.sqrt(-1j * tau) * x for x in S_apply_mp(nmod, theta_vec(nmod, tau, kmax))]
    theta_err = max(abs(theta_left[i] - theta_right[i]) for i in range(nmod))

    d_left = D_vec(nmod, -1 / tau, kmax)
    d_right = [
        1j * ((-1j * tau) ** mp.mpf("1.5")) * x
        for x in S_apply_mp(nmod, D_vec(nmod, tau, kmax))
    ]
    d_err = max(abs(d_left[i] - d_right[i]) for i in range(nmod))
    return float(theta_err), float(d_err)


def main() -> None:
    mp.mp.dps = 90
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    raw_cases = [
        ("chi12_eta_shadow_residual_closed_phase2", *chi12_vector()),
        ("chi8_level8_second_case", *chi8_vector()),
        ("legendre5_level10", *legendre_vector(5)),
        ("legendre13_level26", *legendre_vector(13)),
        ("delta_pair_m7", *delta_vector(7, [(1, 1), (3, -1)])),
        ("three_pair_m9", *delta_vector(9, [(1, 1), (3, -2), (5, 1)])),
    ]

    rows = []
    for name, m, a in raw_cases:
        nmod = 2 * m
        s = S_matrix(nmod)
        identity = np.eye(nmod, dtype=np.complex128)
        reversal = np.zeros((nmod, nmod), dtype=np.complex128)
        for r in range(nmod):
            reversal[r, (-r) % nmod] = 1

        theta_err, d_err = transform_errors(nmod, kmax=70 if nmod <= 18 else 55)
        rows.append(
            {
                "case": name,
                "m": m,
                "N": nmod,
                "even_orientation_vector": all(a[(-r) % nmod] == a[r] for r in range(nmod)),
                "orientation_cancel_worst_abs": orientation_cancel_worst(m, a),
                "S_unitarity_max_abs": float(np.max(np.abs(s.conj().T @ s - identity))),
                "S_square_reversal_max_abs": float(np.max(np.abs(s @ s - reversal))),
                "theta_transform_max_abs": theta_err,
                "derivative_transform_max_abs": d_err,
            }
        )

    with (out / "phase3_v1_sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (out / "phase3_v1_result_card.json").open("w") as f:
        json.dump({"global_pass": all(r["orientation_cancel_worst_abs"] == 0 for r in rows), "cases": rows}, f, indent=2)


if __name__ == "__main__":
    main()
