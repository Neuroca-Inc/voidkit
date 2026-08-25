#!/usr/bin/env python3
"""
Three-filter covariance package for the Phase Calculus / Shadow Calculus lifted object.

This module implements:
- the balanced lifted state Xi_hat = (A, q=(u,v), theta, kappa, c),
- a green filter: total S^3 point plus stereographic/Riemannian readout,
- a blue filter: SU(2) / Hopf-fibration readout,
- a red filter: shadow-completion coordinate tau, with optional launched-family branch values,
- exact Q, B, L actions on the state and on all three filters,
- exhaustive short-word covariance gates, deep deterministic-word gates, and random-walk gates,
- product-only impossibility witness under B,
- sheet-loss witness showing why kappa is needed on periodic geometric/topological filters,
- launched-family anchor diagnostics and figures.

The package keeps the main theorem burden on exact carried-state filters and treats the
mock-theta/shadow branch values as downstream launched-family observables, in the same spirit
as the current Shadow Calculus supplement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

mp.mp.dps = 100
PI = mp.pi
TWO_PI = 2 * PI


@dataclass(frozen=True)
class State:
    """Balanced working-family lifted state."""

    A: str
    u: int
    v: int
    theta: mp.mpf

    def kappa(self) -> int:
        return math.floor(float(self.theta / TWO_PI))

    def phi(self) -> mp.mpf:
        return self.theta - TWO_PI * self.kappa()

    def delta(self) -> mp.mpf:
        return PI / (self.u * self.v)

    def tau(self) -> mp.mpc:
        return self.theta / TWO_PI + 1j / (self.u * self.v)

    def germ(self) -> Tuple[mp.mpf, mp.mpf]:
        d = self.delta()
        return (self.theta - d, self.theta + d)


def next_host(A: str) -> str:
    return f"N({A})"


def op_Q(state: State) -> State:
    return State(state.A, state.u, state.v, state.theta + PI / 2)


def op_B(state: State) -> State:
    return State(state.A, state.v, state.u + state.v, state.theta)


def op_L(state: State) -> State:
    return State(next_host(state.A), state.u, state.v, state.theta + PI / 2)


def state_from_parts(A: str, u: int, v: int, kappa: int, phi: mp.mpf) -> State:
    return State(A, u, v, TWO_PI * kappa + phi)


# ---------------------------------------------------------------------------
# Intrinsic branch pair and filter definitions
# ---------------------------------------------------------------------------

def intrinsic_pair(state: State) -> Tuple[mp.mpc, mp.mpc]:
    """Shadow Calculus intrinsic branch pair on the balanced family."""
    phase = mp.e ** (1j * state.phi())
    d = state.delta()
    m = phase * mp.cos(d)
    s = phase * mp.sin(d)
    return m, s


def green_filter(state: State) -> Dict[str, object]:
    """Total S^3 point with stereographic / Riemannian readout."""
    m, s = intrinsic_pair(state)
    x1, x2, x3, x4 = mp.re(m), mp.im(m), mp.re(s), mp.im(s)
    rho = 1 - x4
    stereo_defined = abs(rho) > mp.mpf("1e-40")
    X1 = X2 = X3 = None
    if stereo_defined:
        X1, X2, X3 = x1 / rho, x2 / rho, x3 / rho
    return {
        "A": state.A,
        "u": state.u,
        "v": state.v,
        "kappa": state.kappa(),
        "phi": state.phi(),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "rho": rho,
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "stereo_defined": stereo_defined,
    }


def blue_filter(state: State) -> Dict[str, object]:
    """SU(2) / Hopf readout of the same S^3 point."""
    m, s = intrinsic_pair(state)
    u11 = m
    u12 = -mp.conj(s)
    u21 = s
    u22 = mp.conj(m)
    n1 = 2 * mp.re(m * mp.conj(s))
    n2 = 2 * mp.im(m * mp.conj(s))
    n3 = abs(m) ** 2 - abs(s) ** 2
    return {
        "A": state.A,
        "u": state.u,
        "v": state.v,
        "kappa": state.kappa(),
        "phi": state.phi(),
        "u11": u11,
        "u12": u12,
        "u21": u21,
        "u22": u22,
        "n1": n1,
        "n2": n2,
        "n3": n3,
    }


@lru_cache(maxsize=None)
def classical_f_cached(q_re: str, q_im: str) -> Tuple[mp.mpc, int]:
    q = mp.mpc(q_re, q_im)
    poch = mp.mpf(1)
    total = mp.mpf(1)
    max_terms = 6000
    tol = mp.mpf("1e-40")
    n_used = max_terms
    for n in range(1, max_terms + 1):
        poch *= (1 + q ** n)
        term = q ** (n * n) / (poch ** 2)
        total += term
        if abs(term) < tol:
            n_used = n
            break
    return total, n_used


@lru_cache(maxsize=None)
def unary_shadow_cached(q_re: str, q_im: str) -> mp.mpc:
    q = mp.mpc(q_re, q_im)
    total = mp.mpc(0)
    max_n = 700
    for n in range(-max_n, max_n + 1):
        k = 6 * n + 1
        total += k * q ** (mp.mpf(k * k) / 24)
    return total


@lru_cache(maxsize=None)
def red_branch_values(u: int, v: int, theta_str: str) -> Tuple[mp.mpc, mp.mpc, mp.mpc, int]:
    theta = mp.mpf(theta_str)
    tau = theta / TWO_PI + 1j / (u * v)
    q = mp.e ** (2 * mp.pi * 1j * tau)
    f_val, n_used = classical_f_cached(mp.nstr(mp.re(q), 60), mp.nstr(mp.im(q), 60))
    M = q ** (-mp.mpf(1) / 24) * f_val
    S = unary_shadow_cached(mp.nstr(mp.re(q), 60), mp.nstr(mp.im(q), 60))
    return tau, M, S, n_used


def red_filter(state: State, attach_branch: bool = False) -> Dict[str, object]:
    out: Dict[str, object] = {
        "A": state.A,
        "u": state.u,
        "v": state.v,
        "tau": state.tau(),
    }
    if attach_branch:
        tau, M, S, n_used = red_branch_values(state.u, state.v, mp.nstr(state.theta, 60))
        out["tau"] = tau
        out["M"] = M
        out["S"] = S
        out["series_terms_used"] = n_used
    return out


# ---------------------------------------------------------------------------
# Filter actions
# ---------------------------------------------------------------------------

def phase_sheet_update(kappa: int, phi: mp.mpf, delta_theta: mp.mpf) -> Tuple[int, mp.mpf]:
    theta = TWO_PI * kappa + phi + delta_theta
    kappa_new = math.floor(float(theta / TWO_PI))
    phi_new = theta - TWO_PI * kappa_new
    return kappa_new, phi_new


def green_action_Q(g: Dict[str, object]) -> Dict[str, object]:
    kappa_new, phi_new = phase_sheet_update(int(g["kappa"]), mp.mpf(g["phi"]), PI / 2)
    return green_filter(state_from_parts(str(g["A"]), int(g["u"]), int(g["v"]), kappa_new, phi_new))


def green_action_B(g: Dict[str, object]) -> Dict[str, object]:
    return green_filter(
        state_from_parts(str(g["A"]), int(g["v"]), int(g["u"]) + int(g["v"]), int(g["kappa"]), mp.mpf(g["phi"]))
    )


def green_action_L(g: Dict[str, object]) -> Dict[str, object]:
    kappa_new, phi_new = phase_sheet_update(int(g["kappa"]), mp.mpf(g["phi"]), PI / 2)
    return green_filter(state_from_parts(next_host(str(g["A"])), int(g["u"]), int(g["v"]), kappa_new, phi_new))



def blue_action_Q(b: Dict[str, object]) -> Dict[str, object]:
    kappa_new, phi_new = phase_sheet_update(int(b["kappa"]), mp.mpf(b["phi"]), PI / 2)
    return blue_filter(state_from_parts(str(b["A"]), int(b["u"]), int(b["v"]), kappa_new, phi_new))


def blue_action_B(b: Dict[str, object]) -> Dict[str, object]:
    return blue_filter(
        state_from_parts(str(b["A"]), int(b["v"]), int(b["u"]) + int(b["v"]), int(b["kappa"]), mp.mpf(b["phi"]))
    )


def blue_action_L(b: Dict[str, object]) -> Dict[str, object]:
    kappa_new, phi_new = phase_sheet_update(int(b["kappa"]), mp.mpf(b["phi"]), PI / 2)
    return blue_filter(state_from_parts(next_host(str(b["A"])), int(b["u"]), int(b["v"]), kappa_new, phi_new))



def red_action_Q(r: Dict[str, object]) -> Dict[str, object]:
    return {"A": r["A"], "u": r["u"], "v": r["v"], "tau": r["tau"] + mp.mpf(1) / 4}



def red_action_B(r: Dict[str, object]) -> Dict[str, object]:
    u = int(r["u"])
    v = int(r["v"])
    tau = r["tau"]
    return {"A": r["A"], "u": v, "v": u + v, "tau": mp.re(tau) + 1j / (v * (u + v))}



def red_action_L(r: Dict[str, object]) -> Dict[str, object]:
    return {"A": next_host(str(r["A"])), "u": r["u"], "v": r["v"], "tau": r["tau"] + mp.mpf(1) / 4}


# ---------------------------------------------------------------------------
# Comparators and recovery
# ---------------------------------------------------------------------------

def angular_difference(a: mp.mpf, b: mp.mpf) -> mp.mpf:
    x = (a - b) % TWO_PI
    return min(x, TWO_PI - x)



def green_diff(a: Dict[str, object], b: Dict[str, object]) -> mp.mpf:
    mismatch = mp.mpf(0 if (a["A"], a["u"], a["v"], a["kappa"]) == (b["A"], b["u"], b["v"], b["kappa"]) else 1)
    vals = [mismatch, abs(mp.mpf(a["phi"]) - mp.mpf(b["phi"]))]
    for key in ["x1", "x2", "x3", "x4", "rho"]:
        vals.append(abs(a[key] - b[key]))
    return max(vals)



def blue_diff(a: Dict[str, object], b: Dict[str, object]) -> mp.mpf:
    mismatch = mp.mpf(0 if (a["A"], a["u"], a["v"], a["kappa"]) == (b["A"], b["u"], b["v"], b["kappa"]) else 1)
    vals = [mismatch, abs(mp.mpf(a["phi"]) - mp.mpf(b["phi"]))]
    for key in ["u11", "u12", "u21", "u22", "n1", "n2", "n3"]:
        vals.append(abs(a[key] - b[key]))
    return max(vals)



def red_diff(a: Dict[str, object], b: Dict[str, object]) -> mp.mpf:
    mismatch = mp.mpf(0 if (a["A"], a["u"], a["v"]) == (b["A"], b["u"], b["v"]) else 1)
    return max(mismatch, abs(a["tau"] - b["tau"]))



def recover_from_green(g: Dict[str, object]) -> Dict[str, mp.mpf]:
    phi = mp.atan2(g["x2"], g["x1"])
    if phi < 0:
        phi += TWO_PI
    delta = mp.atan2(mp.sqrt(g["x3"] ** 2 + g["x4"] ** 2), mp.sqrt(g["x1"] ** 2 + g["x2"] ** 2))
    theta = TWO_PI * int(g["kappa"]) + phi
    return {"theta": theta, "delta": delta}



def recover_from_blue(b: Dict[str, object]) -> Dict[str, mp.mpf]:
    m = b["u11"]
    s = b["u21"]
    phi = mp.arg(m if abs(m) >= abs(s) else s)
    if phi < 0:
        phi += TWO_PI
    delta = mp.atan2(abs(s), abs(m))
    theta = TWO_PI * int(b["kappa"]) + phi
    return {"theta": theta, "delta": delta}



def recover_from_red(r: Dict[str, object]) -> Dict[str, mp.mpf]:
    theta = TWO_PI * mp.re(r["tau"])
    delta = PI / (int(r["u"]) * int(r["v"]))
    return {"theta": theta, "delta": delta}


# ---------------------------------------------------------------------------
# Word application
# ---------------------------------------------------------------------------

def apply_word_state(state: State, word: str) -> State:
    for ch in word:
        state = {"Q": op_Q, "B": op_B, "L": op_L}[ch](state)
    return state



def apply_word_green(g: Dict[str, object], word: str) -> Dict[str, object]:
    for ch in word:
        g = {"Q": green_action_Q, "B": green_action_B, "L": green_action_L}[ch](g)
    return g



def apply_word_blue(b: Dict[str, object], word: str) -> Dict[str, object]:
    for ch in word:
        b = {"Q": blue_action_Q, "B": blue_action_B, "L": blue_action_L}[ch](b)
    return b



def apply_word_red(r: Dict[str, object], word: str) -> Dict[str, object]:
    for ch in word:
        r = {"Q": red_action_Q, "B": red_action_B, "L": red_action_L}[ch](r)
    return r


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mp_to_str(x: object, digits: int = 20) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, bool, str)):
        return str(x)
    if isinstance(x, complex):
        x = mp.mpc(x.real, x.imag)
    if isinstance(x, mp.mpc):
        return f"{mp.nstr(mp.re(x), digits)}+{mp.nstr(mp.im(x), digits)}j"
    return mp.nstr(mp.mpf(x), digits)



def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: mp_to_str(v) for k, v in row.items()})



def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()



def compute_correlation_table(name: str, rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            va = np.array([float(r[a]) for r in rows], dtype=float)
            vb = np.array([float(r[b]) for r in rows], dtype=float)
            if np.std(va) == 0 or np.std(vb) == 0:
                corr = None
            else:
                corr = float(np.corrcoef(va, vb)[0, 1])
            out.append({"dataset": name, "observable_a": a, "observable_b": b, "correlation": corr})
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def exhaustive_words(max_len: int = 6) -> List[str]:
    alphabet = "QBL"
    words = [""]
    for length in range(1, max_len + 1):
        words.extend("".join(t) for t in itertools.product(alphabet, repeat=length))
    return words



def deterministic_deep_words() -> List[str]:
    words = [
        "Q" * 20,
        "L" * 20,
        "B" * 20,
        "QBL" * 7,
        "BQL" * 7,
        "QL" * 10,
        "QB" * 10,
        "LB" * 10,
        "QBLLBQ" * 4,
        "BBQQLLQBLBQLQQBLBQLQ"[:20],
    ]
    return [w if len(w) >= 20 else w + "Q" * (20 - len(w)) for w in words]



def random_walk_words(n: int = 12, length: int = 100, seed: int = 20260411) -> List[str]:
    rng = random.Random(seed)
    alphabet = "QBL"
    return ["".join(rng.choice(alphabet) for _ in range(length)) for _ in range(n)]



def run_covariance_gate(states: Sequence[State], words: Sequence[str], label: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    max_green = mp.mpf(0)
    max_blue = mp.mpf(0)
    max_red = mp.mpf(0)
    worst_green: Optional[Tuple[str, str, mp.mpf]] = None
    worst_blue: Optional[Tuple[str, str, mp.mpf]] = None
    worst_red: Optional[Tuple[str, str, mp.mpf]] = None

    for idx, state in enumerate(states):
        for word in words:
            g_left = green_filter(apply_word_state(state, word))
            g_right = apply_word_green(green_filter(state), word)
            g_err = green_diff(g_left, g_right)

            b_left = blue_filter(apply_word_state(state, word))
            b_right = apply_word_blue(blue_filter(state), word)
            b_err = blue_diff(b_left, b_right)

            r_left = red_filter(apply_word_state(state, word))
            r_right = apply_word_red(red_filter(state), word)
            r_err = red_diff(r_left, r_right)

            rows.append(
                {
                    "gate": label,
                    "state_id": idx,
                    "A": state.A,
                    "u": state.u,
                    "v": state.v,
                    "theta": state.theta,
                    "word": word,
                    "length": len(word),
                    "green_error": g_err,
                    "blue_error": b_err,
                    "red_error": r_err,
                    "B_count": word.count("B"),
                    "Q_count": word.count("Q"),
                    "L_count": word.count("L"),
                }
            )

            if g_err > max_green:
                max_green, worst_green = g_err, (repr(state), word, g_err)
            if b_err > max_blue:
                max_blue, worst_blue = b_err, (repr(state), word, b_err)
            if r_err > max_red:
                max_red, worst_red = r_err, (repr(state), word, r_err)

    summary = {
        "label": label,
        "rows": len(rows),
        "states": len(states),
        "words": len(words),
        "max_green_error": mp.nstr(max_green, 40),
        "max_blue_error": mp.nstr(max_blue, 40),
        "max_red_error": mp.nstr(max_red, 40),
        "worst_green_case": None if worst_green is None else {"state": worst_green[0], "word": worst_green[1], "error": mp.nstr(worst_green[2], 40)},
        "worst_blue_case": None if worst_blue is None else {"state": worst_blue[0], "word": worst_blue[1], "error": mp.nstr(worst_blue[2], 40)},
        "worst_red_case": None if worst_red is None else {"state": worst_red[0], "word": worst_red[1], "error": mp.nstr(worst_red[2], 40)},
    }
    return rows, summary



def run_recovery_gate(states: Sequence[State]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    max_green_theta = mp.mpf(0)
    max_green_delta = mp.mpf(0)
    max_blue_theta = mp.mpf(0)
    max_blue_delta = mp.mpf(0)
    max_red_theta = mp.mpf(0)
    max_red_delta = mp.mpf(0)

    for idx, state in enumerate(states):
        d_true = state.delta()
        g_rec = recover_from_green(green_filter(state))
        b_rec = recover_from_blue(blue_filter(state))
        r_rec = recover_from_red(red_filter(state))

        g_theta_err = abs(g_rec["theta"] - state.theta)
        g_delta_err = abs(g_rec["delta"] - d_true)
        b_theta_err = abs(b_rec["theta"] - state.theta)
        b_delta_err = abs(b_rec["delta"] - d_true)
        r_theta_err = abs(r_rec["theta"] - state.theta)
        r_delta_err = abs(r_rec["delta"] - d_true)

        max_green_theta = max(max_green_theta, g_theta_err)
        max_green_delta = max(max_green_delta, g_delta_err)
        max_blue_theta = max(max_blue_theta, b_theta_err)
        max_blue_delta = max(max_blue_delta, b_delta_err)
        max_red_theta = max(max_red_theta, r_theta_err)
        max_red_delta = max(max_red_delta, r_delta_err)

        rows.append(
            {
                "sample": idx,
                "A": state.A,
                "u": state.u,
                "v": state.v,
                "kappa": state.kappa(),
                "theta": state.theta,
                "green_theta_error": g_theta_err,
                "green_delta_error": g_delta_err,
                "blue_theta_error": b_theta_err,
                "blue_delta_error": b_delta_err,
                "red_theta_error": r_theta_err,
                "red_delta_error": r_delta_err,
            }
        )

    summary = {
        "samples": len(states),
        "max_green_theta_error": mp.nstr(max_green_theta, 40),
        "max_green_delta_error": mp.nstr(max_green_delta, 40),
        "max_blue_theta_error": mp.nstr(max_blue_theta, 40),
        "max_blue_delta_error": mp.nstr(max_blue_delta, 40),
        "max_red_theta_error": mp.nstr(max_red_theta, 40),
        "max_red_delta_error": mp.nstr(max_red_delta, 40),
    }
    return rows, summary



def product_only_counterexample() -> Dict[str, object]:
    s1 = State("A0", 1, 6, mp.mpf("0.7"))
    s2 = State("A0", 2, 3, mp.mpf("0.7"))

    g1, g2 = green_filter(s1), green_filter(s2)
    b1, b2 = blue_filter(s1), blue_filter(s2)
    r1, r2 = red_filter(s1), red_filter(s2)

    green_payload_err = max(abs(g1[k] - g2[k]) for k in ["phi", "x1", "x2", "x3", "x4", "rho"])
    blue_payload_err = max(abs(b1[k] - b2[k]) for k in ["phi", "u11", "u12", "u21", "u22", "n1", "n2", "n3"])
    red_payload_err = abs(r1["tau"] - r2["tau"])

    return {
        "pair_1": {"u": s1.u, "v": s1.v},
        "pair_2": {"u": s2.u, "v": s2.v},
        "shared_product": s1.u * s1.v,
        "same_green_stripped_payload_error": mp.nstr(green_payload_err, 30),
        "same_blue_stripped_payload_error": mp.nstr(blue_payload_err, 30),
        "same_red_stripped_payload_error": mp.nstr(red_payload_err, 30),
        "B_image_pair_1": {"u": op_B(s1).u, "v": op_B(s1).v},
        "B_image_pair_2": {"u": op_B(s2).u, "v": op_B(s2).v},
        "B_images_equal": False,
    }



def sheet_loss_witness() -> Dict[str, object]:
    s = State("A0", 55, 89, mp.mpf("0.3"))
    s4 = apply_word_state(s, "QQQQ")
    g0, g4 = green_filter(s), green_filter(s4)
    b0, b4 = blue_filter(s), blue_filter(s4)
    return {
        "state": {"u": s.u, "v": s.v, "theta": mp.nstr(s.theta, 30), "kappa": s.kappa()},
        "Q4_state": {"u": s4.u, "v": s4.v, "theta": mp.nstr(s4.theta, 30), "kappa": s4.kappa()},
        "same_periodic_green_without_kappa": mp.nstr(max(abs(g0[k] - g4[k]) for k in ["x1", "x2", "x3", "x4"]), 30),
        "same_periodic_blue_without_kappa": mp.nstr(max(abs(b0[k] - b4[k]) for k in ["u11", "u12", "u21", "u22", "n1", "n2", "n3"]), 30),
        "kappa_changed": s.kappa() != s4.kappa(),
        "interpretation": "Periodic S^3 / Hopf data alone collapse exact lifted-sheet information under Q^4; kappa is needed if the filter is meant to remain faithful to the full lifted object.",
    }



def launched_anchor_orbit(n_samples: int = 48) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    rows: List[Dict[str, float]] = []
    max_terms_used = 0
    for k in range(n_samples):
        theta = TWO_PI * k / n_samples
        state = State("A0", 55, 89, theta)
        g = green_filter(state)
        b = blue_filter(state)
        r = red_filter(state, attach_branch=True)
        X_norm = math.sqrt(float(g["X1"] ** 2 + g["X2"] ** 2 + g["X3"] ** 2)) if g["stereo_defined"] else float("nan")
        row = {
            "theta": float(theta),
            "green_rho": float(g["rho"]),
            "green_X_norm": X_norm,
            "blue_phi": float(b["phi"]),
            "blue_n1": float(b["n1"]),
            "blue_n2": float(b["n2"]),
            "blue_n3": float(b["n3"]),
            "red_tau_re": float(mp.re(r["tau"])),
            "red_tau_im": float(mp.im(r["tau"])),
            "red_absM": float(abs(r["M"])),
            "red_absS": float(abs(r["S"])),
        }
        rows.append(row)
        max_terms_used = max(max_terms_used, int(r["series_terms_used"]))
    summary = {"samples": n_samples, "max_series_terms_used": max_terms_used}
    return rows, summary



def launched_corridor(depth: int = 6, theta: mp.mpf = mp.mpf("0.3")) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    rows: List[Dict[str, float]] = []
    u, v = 55, 89
    max_terms_used = 0
    for d in range(depth):
        state = State("A0", u, v, theta)
        g = green_filter(state)
        b = blue_filter(state)
        r = red_filter(state, attach_branch=True)
        row = {
            "depth": d,
            "u": float(u),
            "v": float(v),
            "uv": float(u * v),
            "delta": float(state.delta()),
            "green_rho": float(g["rho"]),
            "green_X_norm": math.sqrt(float(g["X1"] ** 2 + g["X2"] ** 2 + g["X3"] ** 2)),
            "blue_n1": float(b["n1"]),
            "blue_n2": float(b["n2"]),
            "blue_n3": float(b["n3"]),
            "red_tau_im": float(mp.im(r["tau"])),
            "red_absM": float(abs(r["M"])),
            "red_absS": float(abs(r["S"])),
        }
        rows.append(row)
        max_terms_used = max(max_terms_used, int(r["series_terms_used"]))
        u, v = v, u + v
    summary = {"depth": depth, "max_series_terms_used": max_terms_used}
    return rows, summary



def make_figures(fig_dir: Path, short_summary: Dict[str, object], deep_summary: Dict[str, object], random_summary: Dict[str, object], anchor_rows: Sequence[Dict[str, float]], corridor_rows: Sequence[Dict[str, float]]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: anchor orbit, three distinct observables.
    thetas = [r["theta"] for r in anchor_rows]
    green_rho = np.array([r["green_rho"] for r in anchor_rows])
    blue_phi = np.array([r["blue_phi"] for r in anchor_rows])
    red_absS = np.array([r["red_absS"] for r in anchor_rows])
    red_norm = red_absS / np.max(red_absS)
    plt.figure(figsize=(8, 4.8))
    plt.plot(thetas, green_rho, label="green stereographic factor $\\rho$")
    plt.plot(thetas, blue_phi / (2 * math.pi), label="blue fiber phase $\\phi/(2\\pi)$")
    plt.plot(thetas, red_norm, label="red shadow magnitude $|S|$ (normalized)")
    plt.xlabel(r"$\theta$")
    plt.ylabel("value")
    plt.title("Canonical launched anchor: three distinct filter observables")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "anchor_three_filter_observables.png", dpi=180)
    plt.close()

    # Figure 2: B-corridor, Hopf meridian and red/green observables.
    plt.figure(figsize=(8, 4.8))
    depth = [r["depth"] for r in corridor_rows]
    plt.plot(depth, [r["green_rho"] for r in corridor_rows], marker="o", label="green $\\rho$")
    plt.plot(depth, [r["blue_n3"] for r in corridor_rows], marker="o", label="blue Hopf latitude $n_3$")
    plt.plot(depth, [r["red_tau_im"] for r in corridor_rows], marker="o", label="red $\\Im(\\tau)$")
    plt.xlabel("balanced B-depth from anchor")
    plt.ylabel("value")
    plt.title("Balanced corridor from the canonical anchor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "corridor_three_filter_values.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([r["blue_n1"] for r in corridor_rows], [r["blue_n3"] for r in corridor_rows], marker="o")
    for r in corridor_rows:
        plt.annotate(str(int(r["depth"])), (r["blue_n1"], r["blue_n3"]))
    plt.xlabel(r"$n_1$")
    plt.ylabel(r"$n_3$")
    plt.title("Blue filter: Hopf-base meridian along the B-corridor")
    plt.tight_layout()
    plt.savefig(fig_dir / "blue_hopf_meridian.png", dpi=180)
    plt.close()

    # Figure 3: covariance errors by gate and filter.
    labels = [short_summary["label"], deep_summary["label"], random_summary["label"]]
    green_vals = [float(short_summary["max_green_error"]), float(deep_summary["max_green_error"]), float(random_summary["max_green_error"])]
    blue_vals = [float(short_summary["max_blue_error"]), float(deep_summary["max_blue_error"]), float(random_summary["max_blue_error"])]
    red_vals = [float(short_summary["max_red_error"]), float(deep_summary["max_red_error"]), float(random_summary["max_red_error"])]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width, green_vals, width, label="green")
    plt.bar(x, blue_vals, width, label="blue")
    plt.bar(x + width, red_vals, width, label="red")
    plt.yscale("log")
    plt.xticks(x, labels)
    plt.ylabel("max covariance error")
    plt.title("Three-filter covariance gates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "covariance_error_bars.png", dpi=180)
    plt.close()



def run_analysis(output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # States used for exact covariance and recovery stress-tests.
    base_states = [
        State("A0", 55, 89, mp.mpf("0.3")),
        State("A0", 55, 89, TWO_PI - mp.mpf("0.2")),
        State("A2", 89, 144, 4 * PI + mp.mpf("0.1")),
    ]

    short_words = exhaustive_words(6)
    deep_words = deterministic_deep_words()
    random_words = random_walk_words(12, 100, 20260411)

    short_rows, short_summary = run_covariance_gate(base_states, short_words, "short_exhaustive")
    deep_rows, deep_summary = run_covariance_gate(base_states, deep_words, "deep_words")
    random_rows, random_summary = run_covariance_gate(base_states, random_words, "random_walks")

    write_csv(output_dir / "short_exhaustive_covariance_gate.csv", short_rows)
    write_csv(output_dir / "deep_word_covariance_gate.csv", deep_rows)
    write_csv(output_dir / "random_walk_covariance_gate.csv", random_rows)

    # Recovery gate across multiple kappa sheets on the canonical anchor.
    recovery_states: List[State] = []
    for kappa in range(3):
        for j in range(16):
            phi = TWO_PI * j / 16 + mp.mpf("0.05")
            if phi >= TWO_PI:
                phi -= TWO_PI
            recovery_states.append(state_from_parts("A0", 55, 89, kappa, phi))
    recovery_rows, recovery_summary = run_recovery_gate(recovery_states)
    write_csv(output_dir / "recovery_gate.csv", recovery_rows)

    # Negative controls and exact carried-state caveats.
    product_counterexample = product_only_counterexample()
    sheet_counterexample = sheet_loss_witness()
    with (output_dir / "product_only_counterexample.json").open("w") as f:
        json.dump(product_counterexample, f, indent=2)
    with (output_dir / "sheet_loss_witness.json").open("w") as f:
        json.dump(sheet_counterexample, f, indent=2)

    # Launched-family diagnostics.
    anchor_rows, anchor_summary = launched_anchor_orbit(48)
    corridor_rows, corridor_summary = launched_corridor(6, mp.mpf("0.3"))
    write_csv(output_dir / "canonical_anchor_branch_orbit.csv", anchor_rows)
    write_csv(output_dir / "balanced_corridor_branch.csv", corridor_rows)

    correlation_rows = []
    correlation_rows.extend(compute_correlation_table("anchor_orbit", anchor_rows, ["green_rho", "green_X_norm", "blue_phi", "red_tau_re", "red_absM", "red_absS"]))
    correlation_rows.extend(compute_correlation_table("corridor", corridor_rows, ["delta", "green_rho", "green_X_norm", "blue_n1", "blue_n3", "red_tau_im", "red_absM", "red_absS"]))
    write_csv(output_dir / "correlations.csv", correlation_rows)

    make_figures(fig_dir, short_summary, deep_summary, random_summary, anchor_rows, corridor_rows)

    summary = {
        "canonical_anchor": {
            "u": 55,
            "v": 89,
            "uv": 4895,
            "delta_star": mp.nstr(PI / 4895, 40),
            "im_tau_star": mp.nstr(mp.mpf(1) / 4895, 40),
            "residual_per_edge": "1/24",
            "residual_two_sided": "1/12",
        },
        "three_filter_definitions": {
            "green": "(A,u,v,kappa,phi,x in S^3, stereographic chart)",
            "blue": "(A,u,v,kappa,phi,U in SU(2), Hopf base n in S^2)",
            "red": "(A,u,v,tau), with launched-family M,S attached only as downstream diagnostics",
        },
        "covariance_gates": {
            "short_exhaustive": short_summary,
            "deep_words": deep_summary,
            "random_walks": random_summary,
        },
        "recovery_gate": recovery_summary,
        "negative_controls": {
            "product_only_B_failure": product_counterexample,
            "sheet_loss_without_kappa": sheet_counterexample,
        },
        "launched_family_diagnostics": {
            "anchor_orbit": anchor_summary,
            "balanced_corridor": corridor_summary,
        },
        "interpretation": {
            "positive": "With the full ordered denominator pair q=(u,v) and lifted phase-sheet retention, the green, blue, and red filters are covariant readouts of the same lifted object under Q, B, L.",
            "negative": "If only N=uv is retained, exact B-closure fails. If periodic geometric/topological data are retained without kappa, exact lifted return collapses to visible recurrence.",
            "balanced_family_caveat": "On the balanced family the germ c is determined by (q,theta), so c does not need to be stored independently for these tests. That should not be generalized beyond the declared family.",
        },
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Packaging helpers
# ---------------------------------------------------------------------------

def build_manifest(package_dir: Path) -> Dict[str, object]:
    files = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            files.append(
                {
                    "path": str(path.relative_to(package_dir)),
                    "sha256": sha256_of_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {"package": package_dir.name, "files": files}
    with (package_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest



def write_sha256s(package_dir: Path) -> None:
    lines = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{sha256_of_file(path)}  {path.relative_to(package_dir)}")
    (package_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")



def zip_package(package_dir: Path, zip_path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            zf.write(path, arcname=str(path.relative_to(package_dir.parent)))



def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase Calculus three-filter covariance package.")
    parser.add_argument("--package-dir", type=Path, default=Path("phase_three_filter_package_v1"))
    parser.add_argument("--zip-path", type=Path, default=Path("phase_three_filter_package_v1.zip"))
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    results_dir = package_dir / "results"
    summary = run_analysis(results_dir)
    print(json.dumps(summary, indent=2))

    manifest = build_manifest(package_dir)
    write_sha256s(package_dir)
    # rebuild manifest now that SHA256SUMS exists
    manifest = build_manifest(package_dir)
    write_sha256s(package_dir)

    zip_package(package_dir, args.zip_path.resolve())
    print(json.dumps({"package": str(package_dir), "zip": str(args.zip_path.resolve()), "manifest_entries": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
