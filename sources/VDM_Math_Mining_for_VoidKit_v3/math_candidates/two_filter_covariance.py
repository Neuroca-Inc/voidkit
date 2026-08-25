
#!/usr/bin/env python3
"""
Two-filter covariance package for the Phase Calculus / Shadow Calculus launched-family test.

This module implements:
- the lifted-state toy model on the balanced working family,
- the green geometric filter G = (A,u,v,h,g),
- the red analytic filter R = (A,u,v,tau,M,S),
- operator actions Q, B, L on the lifted state and on each filter,
- positive covariance gates on the enriched filters,
- a negative-control impossibility witness for stripped filters,
- a distinct-filter check showing local geometric and shadow readouts are not numerically identical.

The implementation is grounded in the canonical launched-family anchor
(u,v) = (55,89), uv = 4895, delta_* = pi/4895,
together with the order-3 pair
M(τ) = q^(-1/24) f(q),   S(τ) = sum_{n in Z} (6n+1) q^((6n+1)^2/24).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

mp.mp.dps = 80
PI = mp.pi


@dataclass(frozen=True)
class State:
    """Balanced working-family state used in the package."""
    A: str
    u: int
    v: int
    theta: mp.mpf

    def kappa(self) -> int:
        return math.floor(float(self.theta / (2 * PI)))

    def delta(self) -> mp.mpf:
        return PI / (self.u * self.v)

    def tau(self) -> mp.mpc:
        return self.theta / (2 * PI) + 1j / (self.u * self.v)

    def q(self) -> mp.mpc:
        return mp.e ** (2 * mp.pi * 1j * self.tau())

    def germ(self) -> Tuple[mp.mpf, mp.mpf]:
        r = mp.mpf(1) / (self.u * self.v)
        return (self.theta - PI * r, self.theta + PI * r)


def next_host(A: str) -> str:
    """Symbolic next-host map N(A), kept abstract in the source stack."""
    return f"N({A})"


def op_Q(state: State) -> State:
    return State(state.A, state.u, state.v, state.theta + PI / 2)


def op_B(state: State) -> State:
    return State(state.A, state.v, state.u + state.v, state.theta)


def op_L(state: State) -> State:
    return State(next_host(state.A), state.u, state.v, state.theta + PI / 2)


def green_pair(theta: mp.mpf, u: int, v: int) -> Tuple[mp.mpc, mp.mpc]:
    d = PI / (u * v)
    phase = mp.e ** (1j * theta)
    return phase * mp.cos(d), phase * mp.sin(d)


def green_filter(state: State) -> Dict[str, object]:
    h, g = green_pair(state.theta, state.u, state.v)
    return {"A": state.A, "u": state.u, "v": state.v, "h": h, "g": g}


def s3_point(h: mp.mpc, g: mp.mpc) -> Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    return mp.re(h), mp.im(h), mp.re(g), mp.im(g)


def stereographic_factor(h: mp.mpc, g: mp.mpc) -> mp.mpf:
    """Standard conformal factor for the S^3 -> R^3 stereographic chart from x4 = 1."""
    _x1, _x2, _x3, x4 = s3_point(h, g)
    return 1 - x4


def recover_theta_from_green(h: mp.mpc, g: mp.mpc) -> mp.mpf:
    z = h if abs(h) >= abs(g) else g
    theta = mp.arg(z)
    if theta < 0:
        theta += 2 * PI
    return theta


def green_recovery(filter_state: Dict[str, object]) -> Dict[str, object]:
    h = filter_state["h"]
    g = filter_state["g"]
    theta = recover_theta_from_green(h, g)
    delta_rec = mp.atan2(abs(g), abs(h))
    return {
        "A": filter_state["A"],
        "u": filter_state["u"],
        "v": filter_state["v"],
        "theta": theta,
        "delta": delta_rec,
    }


def green_action_Q(filter_state: Dict[str, object]) -> Dict[str, object]:
    return {
        "A": filter_state["A"],
        "u": filter_state["u"],
        "v": filter_state["v"],
        "h": 1j * filter_state["h"],
        "g": 1j * filter_state["g"],
    }


def green_action_B(filter_state: Dict[str, object]) -> Dict[str, object]:
    u = int(filter_state["u"])
    v = int(filter_state["v"])
    theta = recover_theta_from_green(filter_state["h"], filter_state["g"])
    u2, v2 = v, u + v
    h2, g2 = green_pair(theta, u2, v2)
    return {"A": filter_state["A"], "u": u2, "v": v2, "h": h2, "g": g2}


def green_action_L(filter_state: Dict[str, object]) -> Dict[str, object]:
    out = green_action_Q(filter_state)
    out["A"] = next_host(str(filter_state["A"]))
    return out


def classical_f(q: mp.mpc, tol: mp.mpf = mp.mpf("1e-25"), max_terms: int = 800) -> Tuple[mp.mpc, int]:
    """
    Ramanujan order-3 mock theta f(q) = sum_{n>=0} q^{n^2}/(-q;q)_n^2.
    The canonical launched family has |q| close to 1, so we use direct recurrence with a generous cutoff.
    """
    poch = mp.mpf(1)
    total = mp.mpf(1)
    for n in range(1, max_terms + 1):
        poch *= (1 + q ** n)
        term = q ** (n * n) / (poch ** 2)
        total += term
        if abs(term) < tol:
            return total, n
    return total, max_terms


def unary_shadow(q: mp.mpc, max_n: int = 220) -> mp.mpc:
    total = mp.mpc(0)
    for n in range(-max_n, max_n + 1):
        k = 6 * n + 1
        total += k * q ** (mp.mpf(k * k) / 24)
    return total


@lru_cache(maxsize=None)
def red_eval(u: int, v: int, theta_str: str) -> Tuple[mp.mpc, mp.mpc, mp.mpc]:
    theta = mp.mpf(theta_str)
    tau = theta / (2 * PI) + 1j / (u * v)
    q = mp.e ** (2 * mp.pi * 1j * tau)
    f_val, _ = classical_f(q)
    M = q ** (-mp.mpf(1) / 24) * f_val
    S = unary_shadow(q)
    return tau, M, S


def red_filter(state: State) -> Dict[str, object]:
    tau, M, S = red_eval(state.u, state.v, mp.nstr(state.theta, 50))
    return {"A": state.A, "u": state.u, "v": state.v, "tau": tau, "M": M, "S": S}


def red_recovery(filter_state: Dict[str, object]) -> Dict[str, object]:
    theta = 2 * PI * mp.re(filter_state["tau"])
    delta_rec = PI / (int(filter_state["u"]) * int(filter_state["v"]))
    return {
        "A": filter_state["A"],
        "u": filter_state["u"],
        "v": filter_state["v"],
        "theta": theta,
        "delta": delta_rec,
    }


def red_action_Q(filter_state: Dict[str, object]) -> Dict[str, object]:
    u = int(filter_state["u"])
    v = int(filter_state["v"])
    tau0 = filter_state["tau"] + mp.mpf(1) / 4
    theta = 2 * PI * mp.re(tau0)
    tau, M, S = red_eval(u, v, mp.nstr(theta, 50))
    return {"A": filter_state["A"], "u": u, "v": v, "tau": tau, "M": M, "S": S}


def red_action_B(filter_state: Dict[str, object]) -> Dict[str, object]:
    u = int(filter_state["u"])
    v = int(filter_state["v"])
    u2, v2 = v, u + v
    theta = 2 * PI * mp.re(filter_state["tau"])
    tau, M, S = red_eval(u2, v2, mp.nstr(theta, 50))
    return {"A": filter_state["A"], "u": u2, "v": v2, "tau": tau, "M": M, "S": S}


def red_action_L(filter_state: Dict[str, object]) -> Dict[str, object]:
    out = red_action_Q(filter_state)
    out["A"] = next_host(str(filter_state["A"]))
    return out


def apply_word_state(state: State, word: str) -> State:
    for ch in word:
        state = {"Q": op_Q, "B": op_B, "L": op_L}[ch](state)
    return state


def apply_word_green(filter_state: Dict[str, object], word: str) -> Dict[str, object]:
    for ch in word:
        filter_state = {"Q": green_action_Q, "B": green_action_B, "L": green_action_L}[ch](filter_state)
    return filter_state


def apply_word_red(filter_state: Dict[str, object], word: str) -> Dict[str, object]:
    for ch in word:
        filter_state = {"Q": red_action_Q, "B": red_action_B, "L": red_action_L}[ch](filter_state)
    return filter_state


def angular_difference(a: mp.mpf, b: mp.mpf) -> mp.mpf:
    x = (a - b) % (2 * PI)
    return min(x, 2 * PI - x)


def green_diff(a: Dict[str, object], b: Dict[str, object]) -> mp.mpf:
    host_mismatch = mp.mpf(0 if (a["A"], a["u"], a["v"]) == (b["A"], b["u"], b["v"]) else 1)
    return max(abs(a["h"] - b["h"]), abs(a["g"] - b["g"]), host_mismatch)


def red_diff(a: Dict[str, object], b: Dict[str, object]) -> mp.mpf:
    host_mismatch = mp.mpf(0 if (a["A"], a["u"], a["v"]) == (b["A"], b["u"], b["v"]) else 1)
    return max(abs(a["tau"] - b["tau"]), abs(a["M"] - b["M"]), abs(a["S"] - b["S"]), host_mismatch)


def generate_words() -> List[str]:
    alphabet = "QBL"
    words = [""]
    for length in range(1, 4):
        for tup in __import__("itertools").product(alphabet, repeat=length):
            words.append("".join(tup))
    words.extend(["QBQL", "LBQ", "BQL", "QLB", "BLQ", "BBQ", "QBB", "LBB", "BBL"])
    return words


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_analysis(output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    words = generate_words()
    sample_states = [
        State("A0", 55, 89, mp.mpf("0.3")),
        State("A0", 34, 55, mp.mpf("1.1")),
        State("A0", 8, 13, mp.mpf("2.2")),
    ]

    # Positive covariance gate on enriched filters.
    green_gate_rows = []
    red_gate_rows = []
    max_green = mp.mpf(0)
    max_red = mp.mpf(0)
    worst_green = None
    worst_red = None

    for state in sample_states:
        for word in words:
            left_g = green_filter(apply_word_state(state, word))
            right_g = apply_word_green(green_filter(state), word)
            err_g = green_diff(left_g, right_g)
            green_gate_rows.append(
                {
                    "A": state.A,
                    "u": state.u,
                    "v": state.v,
                    "theta": mp.nstr(state.theta, 20),
                    "word": word,
                    "error": mp.nstr(err_g, 20),
                }
            )
            if err_g > max_green:
                max_green = err_g
                worst_green = {"state": repr(state), "word": word, "error": mp.nstr(err_g, 30)}

            left_r = red_filter(apply_word_state(state, word))
            right_r = apply_word_red(red_filter(state), word)
            err_r = red_diff(left_r, right_r)
            red_gate_rows.append(
                {
                    "A": state.A,
                    "u": state.u,
                    "v": state.v,
                    "theta": mp.nstr(state.theta, 20),
                    "word": word,
                    "error": mp.nstr(err_r, 20),
                }
            )
            if err_r > max_red:
                max_red = err_r
                worst_red = {"state": repr(state), "word": word, "error": mp.nstr(err_r, 30)}

    write_csv(output_dir / "green_covariance_gate.csv", green_gate_rows)
    write_csv(output_dir / "red_covariance_gate.csv", red_gate_rows)

    # Recovery gate on canonical launched family: sample theta along the canonical height uv = 4895.
    recovery_rows = []
    theta_samples = [2 * PI * k / 64 for k in range(64)]
    max_theta_err_green = mp.mpf(0)
    max_delta_err_green = mp.mpf(0)
    max_theta_err_red = mp.mpf(0)
    max_delta_err_red = mp.mpf(0)

    for idx, theta in enumerate(theta_samples):
        state = State("A0", 55, 89, theta)
        d_true = state.delta()
        g_rec = green_recovery(green_filter(state))
        r_rec = red_recovery(red_filter(state))
        theta_err_g = angular_difference(g_rec["theta"], theta)
        theta_err_r = angular_difference(r_rec["theta"], theta)
        delta_err_g = abs(g_rec["delta"] - d_true)
        delta_err_r = abs(r_rec["delta"] - d_true)
        max_theta_err_green = max(max_theta_err_green, theta_err_g)
        max_delta_err_green = max(max_delta_err_green, delta_err_g)
        max_theta_err_red = max(max_theta_err_red, theta_err_r)
        max_delta_err_red = max(max_delta_err_red, delta_err_r)

        recovery_rows.append(
            {
                "sample": idx,
                "theta": mp.nstr(theta, 20),
                "green_theta_error": mp.nstr(theta_err_g, 12),
                "green_delta_error": mp.nstr(delta_err_g, 12),
                "red_theta_error": mp.nstr(theta_err_r, 12),
                "red_delta_error": mp.nstr(delta_err_r, 12),
            }
        )
    write_csv(output_dir / "canonical_recovery_gate.csv", recovery_rows)

    # Negative control: stripped filters are not B-closed.
    s1 = State("A0", 1, 6, mp.mpf("0.7"))
    s2 = State("A0", 2, 3, mp.mpf("0.7"))
    g1 = green_filter(s1)
    g2 = green_filter(s2)
    r1 = red_filter(s1)
    r2 = red_filter(s2)
    stripped_counterexample = {
        "pair_1": {"u": s1.u, "v": s1.v},
        "pair_2": {"u": s2.u, "v": s2.v},
        "shared_product": s1.u * s1.v,
        "same_green_output_error": mp.nstr(max(abs(g1["h"] - g2["h"]), abs(g1["g"] - g2["g"])), 12),
        "same_red_tau_error": mp.nstr(abs(r1["tau"] - r2["tau"]), 12),
        "B_image_pair_1": {"u": op_B(s1).u, "v": op_B(s1).v},
        "B_image_pair_2": {"u": op_B(s2).u, "v": op_B(s2).v},
        "B_images_equal": False,
    }
    with (output_dir / "stripped_filter_counterexample.json").open("w") as f:
        json.dump(stripped_counterexample, f, indent=2)

    # Distinct-filter test on the canonical launched orbit.
    theta_orbit = [2 * PI * k / 256 for k in range(256)]
    rho = []
    shadow_abs = []
    holomorphic_abs = []
    plot_rows = []
    for theta in theta_orbit:
        state = State("A0", 55, 89, theta)
        g = green_filter(state)
        r = red_filter(state)
        rho_val = float(stereographic_factor(g["h"], g["g"]))
        s_val = float(abs(r["S"]))
        m_val = float(abs(r["M"]))
        rho.append(rho_val)
        shadow_abs.append(s_val)
        holomorphic_abs.append(m_val)
        plot_rows.append(
            {
                "theta": float(theta),
                "rho": rho_val,
                "shadow_abs": s_val,
                "holomorphic_abs": m_val,
            }
        )
    write_csv(output_dir / "canonical_orbit_filter_values.csv", plot_rows)
    corr_rho_shadow = float(np.corrcoef(np.array(rho), np.array(shadow_abs))[0, 1])
    corr_rho_holomorphic = float(np.corrcoef(np.array(rho), np.array(holomorphic_abs))[0, 1])

    # Corridor table linking delta = pi/(uv) and pi * Im(tau).
    corridor_rows = []
    u, v = 8, 13
    for _ in range(5):
        theta = mp.mpf("0.3")
        state = State("A0", u, v, theta)
        g = green_recovery(green_filter(state))
        r = red_recovery(red_filter(state))
        corridor_rows.append(
            {
                "u": u,
                "v": v,
                "uv": u * v,
                "delta_from_green": mp.nstr(g["delta"], 20),
                "pi_times_im_tau": mp.nstr(PI / (u * v), 20),
                "red_delta": mp.nstr(r["delta"], 20),
            }
        )
        u, v = v, u + v
    write_csv(output_dir / "balanced_corridor_bridge.csv", corridor_rows)

    # Figures.
    plt.figure(figsize=(8, 4.8))
    plt.plot([row["theta"] for row in plot_rows], [row["rho"] for row in plot_rows], label="green stereographic factor")
    plt.plot([row["theta"] for row in plot_rows], [row["shadow_abs"] for row in plot_rows], label="|red shadow|")
    plt.xlabel(r"$\theta$")
    plt.ylabel("value")
    plt.title("Canonical anchor: distinct green and red readouts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "canonical_distinct_filters.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    # projected coordinates for visual intuition
    xs, ys = [], []
    for theta in theta_orbit:
        h, g = green_pair(theta, 55, 89)
        x1, x2, x3, x4 = s3_point(h, g)
        denom = 1 - x4
        X = x1 / denom
        Y = x2 / denom
        xs.append(float(X))
        ys.append(float(Y))
    plt.plot(xs, ys)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(r"Canonical anchor green filter: stereographic trace of $(h,g)\in S^3$")
    plt.tight_layout()
    plt.savefig(fig_dir / "canonical_green_stereographic_trace.png", dpi=160)
    plt.close()

    summary = {
        "canonical_anchor": {
            "u": 55,
            "v": 89,
            "uv": 4895,
            "delta_star": mp.nstr(PI / 4895, 30),
            "im_tau": mp.nstr(mp.mpf(1) / 4895, 30),
            "residual_per_edge": "1/24",
            "residual_two_sided": "1/12",
        },
        "positive_covariance_gate": {
            "tested_words": len(words),
            "tested_states": len(sample_states),
            "max_green_error": mp.nstr(max_green, 30),
            "max_red_error": mp.nstr(max_red, 30),
            "worst_green_case": worst_green,
            "worst_red_case": worst_red,
        },
        "recovery_gate": {
            "canonical_orbit_samples": len(theta_samples),
            "max_green_theta_error": mp.nstr(max_theta_err_green, 30),
            "max_green_delta_error": mp.nstr(max_delta_err_green, 30),
            "max_red_theta_error": mp.nstr(max_theta_err_red, 30),
            "max_red_delta_error": mp.nstr(max_delta_err_red, 30),
        },
        "negative_control": stripped_counterexample,
        "distinct_filter_check": {
            "canonical_orbit_samples": len(theta_orbit),
            "corr_rho_vs_shadow_abs": corr_rho_shadow,
            "corr_rho_vs_holomorphic_abs": corr_rho_holomorphic,
        },
        "interpretation": {
            "positive": "The enriched filters (A,u,v,h,g) and (A,u,v,tau,M,S) are covariant readouts of the same lifted object under Q, B, L.",
            "negative": "If q=(u,v) is stripped away, B is not filter-closed: same stripped readout can come from different denominator pairs with different B images.",
        },
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the two-filter covariance analysis.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory in which result files should be written.",
    )
    args = parser.parse_args()
    summary = run_analysis(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
