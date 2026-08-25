#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> None:
    x = sp.symbols("x", positive=True)
    N = sp.symbols("N", positive=True)
    Pq, Pr = sp.symbols("P_q P_r", positive=True)
    Sq1, Sq2, Sr1, Sr2 = sp.symbols("S_q1 S_q2 S_r1 S_r2", real=True)

    F_general = sp.Rational(1, 2) * sp.log(N) - sp.log(Pq) - (N - 1 / N) * x / 12 + sp.log(Pr)
    F_special = sp.Rational(1, 2) * sp.log(2) - sp.log(Pq) - x / 8 + sp.log(Pr)

    Fp_general = -((-2 / N) * Sq1 / Pq) - (N - 1 / N) / 12 + ((-2 * N) * Sr1 / Pr)
    Fp_special = Sq1 / Pq - sp.Rational(1, 8) - 4 * Sr1 / Pr

    Fpp_general = -((4 / N**2) * Sq2 / Pq - (((-2 / N) * Sq1 / Pq) ** 2)) + ((4 * N**2) * Sr2 / Pr - (((-2 * N) * Sr1 / Pr) ** 2))
    Fpp_special = -(Sq2 / Pq - (Sq1 / Pq) ** 2) + 16 * (Sr2 / Pr - (Sr1 / Pr) ** 2)

    checks = {
        "F_N2_specialization_zero": sp.simplify(F_general.subs(N, 2) - F_special) == 0,
        "Fp_N2_specialization_zero": sp.simplify(Fp_general.subs(N, 2) - Fp_special) == 0,
        "Fpp_N2_specialization_zero": sp.simplify(Fpp_general.subs(N, 2) - Fpp_special) == 0,
        "slope_floor_N2": sp.simplify(((N - 1 / N) / 12).subs(N, 2) - sp.Rational(1, 8)) == 0,
        "deeper_packet_N_values": [2, 6, 15, 40] == [1 * 2, 2 * 3, 3 * 5, 5 * 8],
    }

    out = {"checks": checks, "all_pass": all(bool(v) for v in checks.values())}
    root = Path(__file__).resolve().parent
    (root / "sympy_phase_native_streaming_spigot_direct_v5_output.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
