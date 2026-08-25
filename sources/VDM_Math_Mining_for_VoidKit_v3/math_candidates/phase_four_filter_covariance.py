
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import math
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

mp.mp.dps = 100

PI = mp.pi
TAU = 2 * PI
QUARTER = PI / 2
TWO_THIRDS_PI = 2 * PI / 3

BASE_COUNTS: Dict[str, int] = {"u": 56, "c": 72, "a": 56, "g": 56}
BASE_ORDER: Tuple[str, ...] = ("u", "c", "a", "g")
BASE_FOCUS_ORDER: Tuple[str, ...] = ("u", "c", "a", "g")

# Three standard IUPAC bifurcations explicitly listed in the attached preprint.
BIFURCATIONS: Dict[int, Dict[str, Any]] = {
    0: {"positive": "m", "negative": "k", "positive_bases": ("c", "a"), "negative_bases": ("u", "g")},
    1: {"positive": "y", "negative": "r", "positive_bases": ("c", "u"), "negative_bases": ("a", "g")},
    2: {"positive": "s", "negative": "w", "positive_bases": ("c", "g"), "negative_bases": ("u", "a")},
}


def mpf(x: Any) -> mp.mpf:
    return mp.mpf(x)


def normalize_angle(theta: mp.mpf) -> mp.mpf:
    out = mp.fmod(theta, TAU)
    if out < 0:
        out += TAU
    return out


def phase_arg(z: complex | mp.mpc) -> mp.mpf:
    a = mp.atan2(mp.im(z), mp.re(z))
    if a < 0:
        a += TAU
    return a


def abs_diff(a: Any, b: Any) -> mp.mpf:
    if isinstance(a, (mp.mpf, int)) and isinstance(b, (mp.mpf, int)):
        return mp.fabs(mpf(a) - mpf(b))
    if isinstance(a, (float,)) or isinstance(b, (float,)):
        return mp.fabs(mpf(a) - mpf(b))
    if isinstance(a, (complex, mp.mpc)) or isinstance(b, (complex, mp.mpc)):
        return mp.fabs(a - b)
    if isinstance(a, str) or isinstance(b, str):
        return mp.mpf("0") if a == b else mp.mpf("1")
    if isinstance(a, bool) or isinstance(b, bool):
        return mp.mpf("0") if a == b else mp.mpf("1")
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        if not keys:
            return mp.mpf("0")
        return max(abs_diff(a.get(k), b.get(k)) for k in keys)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return mp.mpf("1")
        if not a:
            return mp.mpf("0")
        return max(abs_diff(x, y) for x, y in zip(a, b))
    return mp.mpf("0") if a == b else mp.mpf("1")


def serialize_value(x: Any) -> Any:
    if isinstance(x, mp.mpf):
        return str(x)
    if isinstance(x, mp.mpc):
        return {"re": str(mp.re(x)), "im": str(mp.im(x))}
    if isinstance(x, complex):
        return {"re": str(x.real), "im": str(x.imag)}
    if isinstance(x, dict):
        return {k: serialize_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serialize_value(v) for v in x]
    return x


@dataclass(frozen=True)
class State:
    A: int
    u: int
    v: int
    theta: mp.mpf

    def __post_init__(self) -> None:
        if self.u <= 0 or self.v <= 0:
            raise ValueError("u and v must be positive")
        if self.u > self.v:
            raise ValueError("state must remain on ordered balanced branch u <= v")

    @property
    def kappa(self) -> int:
        return int(mp.floor(self.theta / TAU))

    @property
    def phi(self) -> mp.mpf:
        return normalize_angle(self.theta)

    @property
    def N(self) -> int:
        return self.u * self.v

    @property
    def delta(self) -> mp.mpf:
        return PI / self.N

    @property
    def r(self) -> mp.mpf:
        return mp.mpf(1) / self.N

    @property
    def germ(self) -> Tuple[mp.mpf, mp.mpf]:
        return (self.theta - self.delta, self.theta + self.delta)

    def phase_focus(self) -> str:
        idx = int(mp.floor(self.phi / QUARTER)) % 4
        return BASE_FOCUS_ORDER[idx]


def state_from_sheet(A: int, u: int, v: int, kappa: int, phi: mp.mpf) -> State:
    return State(A=A, u=u, v=v, theta=mpf(kappa) * TAU + phi)


def host_step(A: int) -> int:
    return (A + 1) % 3


def Q_state(s: State) -> State:
    return State(A=s.A, u=s.u, v=s.v, theta=s.theta + QUARTER)


def B_pair(u: int, v: int) -> Tuple[int, int]:
    return tuple(sorted((v, u + v)))  # type: ignore[return-value]


def B_state(s: State) -> State:
    u, v = B_pair(s.u, s.v)
    return State(A=s.A, u=u, v=v, theta=s.theta)


def L_state(s: State) -> State:
    return State(A=host_step(s.A), u=s.u, v=s.v, theta=s.theta + QUARTER)


OPS = {"Q": Q_state, "B": B_state, "L": L_state}


def apply_word(word: str, s: State) -> State:
    out = s
    for ch in word:
        out = OPS[ch](out)
    return out


def words_upto(max_len: int) -> List[str]:
    out = [""]
    for n in range(1, max_len + 1):
        out.extend("".join(chars) for chars in product("QBL", repeat=n))
    return out


def deep_words() -> List[str]:
    return [
        "Q" * 20,
        "L" * 20,
        "B" * 20,
        "QL" * 10,
        "QB" * 10,
        "BL" * 10,
        "QLB" * 7,
        "BQL" * 7,
        "LLQQBB" * 4,
        "QBLLQBBLQQLBQQBLLQL",  # length 20
    ]


def random_words(num_words: int = 12, length: int = 100, seed: int = 7) -> List[str]:
    rng = random.Random(seed)
    alphabet = ("Q", "B", "L")
    return ["".join(rng.choice(alphabet) for _ in range(length)) for _ in range(num_words)]


def intrinsic_edge_phases(s: State) -> Tuple[mp.mpc, mp.mpc]:
    eplus = mp.e ** (1j * (s.theta + s.delta))
    eminus = mp.e ** (1j * (s.theta - s.delta))
    return eplus, eminus


def intrinsic_branch_pair(s: State) -> Tuple[mp.mpc, mp.mpc]:
    eplus, eminus = intrinsic_edge_phases(s)
    m = (eplus + eminus) / 2
    sig = (eplus - eminus) / (2j)
    return m, sig


def green_filter(s: State) -> Dict[str, Any]:
    m, sig = intrinsic_branch_pair(s)
    x = (mp.re(m), mp.im(m), mp.re(sig), mp.im(sig))
    rho = 1 - x[3]
    if mp.fabs(rho) > mp.mpf("1e-80"):
        stereo = (x[0] / rho, x[1] / rho, x[2] / rho)
    else:
        stereo = ("chart_singularity", "chart_singularity", "chart_singularity")
    return {
        "A": s.A,
        "u": s.u,
        "v": s.v,
        "kappa": s.kappa,
        "phi": s.phi,
        "delta": s.delta,
        "m": m,
        "s": sig,
        "x": x,
        "rho": rho,
        "stereo": stereo,
    }


def blue_filter(s: State) -> Dict[str, Any]:
    m, sig = intrinsic_branch_pair(s)
    U = (
        (m, -mp.conj(sig)),
        (sig, mp.conj(m)),
    )
    n1 = 2 * mp.re(m * mp.conj(sig))
    n2 = 2 * mp.im(m * mp.conj(sig))
    n3 = mp.fabs(m) ** 2 - mp.fabs(sig) ** 2
    return {
        "A": s.A,
        "u": s.u,
        "v": s.v,
        "kappa": s.kappa,
        "phi": s.phi,
        "delta": s.delta,
        "U": U,
        "n": (n1, n2, n3),
    }


def red_filter(s: State) -> Dict[str, Any]:
    tau = mp.mpc(s.theta / TAU, mp.mpf(1) / (s.u * s.v))
    return {
        "A": s.A,
        "u": s.u,
        "v": s.v,
        "tau": tau,
    }


def host_phase(A: int) -> mp.mpf:
    return TWO_THIRDS_PI * A


def yellow_spinor(s: State) -> Tuple[mp.mpc, mp.mpc]:
    chi = s.phi + host_phase(s.A)
    return (mp.cos(s.delta), mp.e ** (1j * chi) * mp.sin(s.delta))


def qgt_projected_numeric(s: State) -> Dict[str, Any]:
    psi = yellow_spinor(s)
    chi = s.phi + host_phase(s.A)
    dphi = (mp.mpc(0), 1j * mp.e ** (1j * chi) * mp.sin(s.delta))
    ddelta = (-mp.sin(s.delta), mp.e ** (1j * chi) * mp.cos(s.delta))

    p00 = 1 - psi[0] * mp.conj(psi[0])
    p01 = -psi[0] * mp.conj(psi[1])
    p10 = -psi[1] * mp.conj(psi[0])
    p11 = 1 - psi[1] * mp.conj(psi[1])

    def apply_P(v: Tuple[mp.mpc, mp.mpc]) -> Tuple[mp.mpc, mp.mpc]:
        return (p00 * v[0] + p01 * v[1], p10 * v[0] + p11 * v[1])

    def inner(u: Tuple[mp.mpc, mp.mpc], v: Tuple[mp.mpc, mp.mpc]) -> mp.mpc:
        return mp.conj(u[0]) * v[0] + mp.conj(u[1]) * v[1]

    P_dphi = apply_P(dphi)
    P_ddelta = apply_P(ddelta)

    q11 = inner(dphi, P_dphi)
    q12 = inner(dphi, P_ddelta)
    q21 = inner(ddelta, P_dphi)
    q22 = inner(ddelta, P_ddelta)

    Q = ((q11, q12), (q21, q22))
    g = ((mp.re(q11), mp.re(q12)), (mp.re(q21), mp.re(q22)))
    Omega = ((mp.mpf("0"), -2 * mp.im(q12)), (-2 * mp.im(q21), mp.mpf("0")))

    # closed forms for exactness checks
    q11_closed = mp.sin(s.delta) ** 2 * mp.cos(s.delta) ** 2
    q12_closed = -1j * mp.sin(2 * s.delta) / 2
    q21_closed = 1j * mp.sin(2 * s.delta) / 2
    q22_closed = mp.mpf("1")
    closed_resid = max(
        mp.fabs(q11 - q11_closed),
        mp.fabs(q12 - q12_closed),
        mp.fabs(q21 - q21_closed),
        mp.fabs(q22 - q22_closed),
    )
    return {
        "psi": psi,
        "Q": Q,
        "g": g,
        "Omega": Omega,
        "closed_form_residual": closed_resid,
    }


def build_biological_slots(A: int, phi: mp.mpf) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rule = BIFURCATIONS[A]
    focus = BASE_FOCUS_ORDER[int(mp.floor(phi / QUARTER)) % 4]
    slots: List[Dict[str, Any]] = []
    idx = 0
    for base in BASE_ORDER:
        for local_idx in range(BASE_COUNTS[base]):
            if base in rule["positive_bases"]:
                bucket = rule["positive"]
                coarse = 128
            else:
                bucket = rule["negative"]
                coarse = 112
            slots.append(
                {
                    "slot_id": idx,
                    "base": base,
                    "bifurcation_family": f"{rule['positive']}/{rule['negative']}",
                    "coarse_bucket": coarse,
                    "bucket_label": bucket,
                    "base_focus": focus,
                    "local_idx": local_idx,
                }
            )
            idx += 1

    positive_pair = tuple(BASE_COUNTS[b] for b in rule["positive_bases"])
    negative_pair = tuple(BASE_COUNTS[b] for b in rule["negative_bases"])
    # canonical ordering so the 72 sits first in the 128 bucket if present
    positive_pair_sorted = tuple(sorted(positive_pair, reverse=True))
    negative_pair_sorted = tuple(sorted(negative_pair, reverse=True))
    summary = {
        "family": f"{rule['positive']}/{rule['negative']}",
        "positive_label": rule["positive"],
        "negative_label": rule["negative"],
        "positive_bases": rule["positive_bases"],
        "negative_bases": rule["negative_bases"],
        "positive_count": sum(BASE_COUNTS[b] for b in rule["positive_bases"]),
        "negative_count": sum(BASE_COUNTS[b] for b in rule["negative_bases"]),
        "positive_pair_sorted": positive_pair_sorted,
        "negative_pair_sorted": negative_pair_sorted,
        "base_focus": focus,
        "focus_count": BASE_COUNTS[focus],
        "base_counts": dict(BASE_COUNTS),
        "total_acceptors": sum(BASE_COUNTS.values()),
    }
    return slots, summary


def yellow_filter(s: State) -> Dict[str, Any]:
    qgt = qgt_projected_numeric(s)
    slots, bio = build_biological_slots(s.A, s.phi)
    slot_hash = hashlib.sha256(json.dumps(slots, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "A": s.A,
        "u": s.u,
        "v": s.v,
        "kappa": s.kappa,
        "phi": s.phi,
        "delta": s.delta,
        "t": 2 * PI / (s.u * s.v),
        "residual_per_edge": mp.mpf(1) / 24,
        "residual_two_sided": mp.mpf(1) / 12,
        "psi": qgt["psi"],
        "Q": qgt["Q"],
        "g": qgt["g"],
        "Omega": qgt["Omega"],
        "qgt_closed_form_residual": qgt["closed_form_residual"],
        "biological": bio,
        "biological_hash": slot_hash,
    }


FILTERS = {
    "green": green_filter,
    "blue": blue_filter,
    "red": red_filter,
    "yellow": yellow_filter,
}


def state_from_readout(filter_name: str, readout: Dict[str, Any]) -> State:
    if filter_name == "red":
        tau = readout["tau"]
        theta = TAU * mp.re(tau)
        u = int(readout["u"])
        v = int(readout["v"])
        A = int(readout["A"])
        return State(A=A, u=u, v=v, theta=theta)
    A = int(readout["A"])
    u = int(readout["u"])
    v = int(readout["v"])
    kappa = int(readout["kappa"])
    phi = mpf(readout["phi"])
    return state_from_sheet(A, u, v, kappa, phi)


def apply_word_to_filter(filter_name: str, word: str, readout: Dict[str, Any]) -> Dict[str, Any]:
    s = state_from_readout(filter_name, readout)
    return FILTERS[filter_name](apply_word(word, s))


def filter_error(a: Dict[str, Any], b: Dict[str, Any]) -> mp.mpf:
    return abs_diff(a, b)


def canonical_states() -> List[State]:
    return [
        State(A=0, u=55, v=89, theta=mpf("0.3")),
        State(A=1, u=34, v=55, theta=2 * PI - mpf("0.2")),
        State(A=2, u=89, v=144, theta=4 * PI + mpf("0.1")),
    ]


def canonical_anchor_states_for_recovery() -> List[State]:
    phis = [TAU * mpf(k) / 16 for k in range(16)]
    states = []
    for kappa in range(3):
        for phi in phis:
            states.append(state_from_sheet(A=0, u=55, v=89, kappa=kappa, phi=phi))
    return states


def recover_green(readout: Dict[str, Any]) -> Tuple[mp.mpf, mp.mpf]:
    m = readout["m"]
    s = readout["s"]
    phi = phase_arg(m)
    theta = TAU * readout["kappa"] + phi
    delta = mp.atan2(mp.fabs(s), mp.fabs(m))
    return theta, delta


def recover_blue(readout: Dict[str, Any]) -> Tuple[mp.mpf, mp.mpf]:
    m = readout["U"][0][0]
    s = readout["U"][1][0]
    phi = phase_arg(m)
    theta = TAU * readout["kappa"] + phi
    delta = mp.atan2(mp.fabs(s), mp.fabs(m))
    return theta, delta


def recover_red(readout: Dict[str, Any]) -> Tuple[mp.mpf, mp.mpf]:
    tau = readout["tau"]
    theta = TAU * mp.re(tau)
    delta = PI / (readout["u"] * readout["v"])
    return theta, delta


def recover_yellow(readout: Dict[str, Any]) -> Tuple[mp.mpf, mp.mpf]:
    psi1 = readout["psi"][1]
    phi = phase_arg(psi1) - host_phase(readout["A"])
    phi = normalize_angle(phi)
    theta = TAU * readout["kappa"] + phi
    omega12 = readout["Omega"][0][1]
    delta = mp.asin(omega12) / 2
    return theta, delta


RECOVER = {
    "green": recover_green,
    "blue": recover_blue,
    "red": recover_red,
    "yellow": recover_yellow,
}


def stripped_product_payloads(s: State) -> Dict[str, Dict[str, Any]]:
    N = s.N
    delta = PI / N
    # stripped red keeps only product-derived imaginary height and visible phase modulo 2π
    red = {"A": s.A, "N": N, "phi": s.phi, "tau_mod1": s.phi / TAU, "tau_im": mp.mpf(1) / N}
    green = {"A": s.A, "N": N, "phi": s.phi, "delta": delta}
    blue = {"A": s.A, "N": N, "phi": s.phi, "delta": delta}
    y = yellow_filter(State(A=s.A, u=1, v=N, theta=s.phi))
    yellow = {
        "A": s.A,
        "N": N,
        "phi": s.phi,
        "delta": delta,
        "t": y["t"],
        "residual_per_edge": y["residual_per_edge"],
        "residual_two_sided": y["residual_two_sided"],
        "Omega12": y["Omega"][0][1],
        "g11": y["g"][0][0],
        "biology_family": y["biological"]["family"],
        "biology_hash": y["biological_hash"],
    }
    return {"green": green, "blue": blue, "red": red, "yellow": yellow}


def stripped_periodic_payloads(s: State) -> Dict[str, Dict[str, Any]]:
    green = green_filter(State(A=s.A, u=s.u, v=s.v, theta=s.phi))
    blue = blue_filter(State(A=s.A, u=s.u, v=s.v, theta=s.phi))
    red = {"A": s.A, "u": s.u, "v": s.v, "tau_mod1": s.phi / TAU, "tau_im": mp.mpf(1) / (s.u * s.v)}
    yellow = yellow_filter(State(A=s.A, u=s.u, v=s.v, theta=s.phi))
    return {"green": green, "blue": blue, "red": red, "yellow": yellow}


def corridor_pairs(depth: int = 5) -> List[Tuple[int, int]]:
    pair = (8, 13)
    out = [pair]
    for _ in range(depth - 1):
        pair = B_pair(*pair)
        out.append(pair)
    return out


def anchor_orbit_states() -> List[State]:
    return [State(A=A, u=55, v=89, theta=mpf("0.3") + j * QUARTER) for A in range(3) for j in range(4)]


def biological_gate_rows() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor_rows: List[Dict[str, Any]] = []
    corridor_rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []

    # Anchor orbit
    for s in anchor_orbit_states():
        y = yellow_filter(s)
        bio = y["biological"]
        anchor_rows.append(
            {
                "A": s.A,
                "u": s.u,
                "v": s.v,
                "phi": str(s.phi),
                "base_focus": bio["base_focus"],
                "family": bio["family"],
                "total_acceptors": bio["total_acceptors"],
                "positive_count": bio["positive_count"],
                "negative_count": bio["negative_count"],
                "positive_pair": f"{bio['positive_pair_sorted']}",
                "negative_pair": f"{bio['negative_pair_sorted']}",
                "mismatch_total": bio["total_acceptors"] - 240,
                "mismatch_128": bio["positive_count"] - 128,
                "mismatch_112": bio["negative_count"] - 112,
                "mismatch_pos_pair": str(tuple(x - y for x, y in zip(bio["positive_pair_sorted"], (72, 56)))),
                "mismatch_neg_pair": str(tuple(x - y for x, y in zip(bio["negative_pair_sorted"], (56, 56)))),
            }
        )

    # Balanced corridor
    for i, (u, v) in enumerate(corridor_pairs()):
        for A in range(3):
            s = State(A=A, u=u, v=v, theta=TAU * mpf(i) / 8)
            y = yellow_filter(s)
            bio = y["biological"]
            corridor_rows.append(
                {
                    "depth_index": i,
                    "A": A,
                    "u": u,
                    "v": v,
                    "uv": u * v,
                    "phi": str(s.phi),
                    "family": bio["family"],
                    "base_focus": bio["base_focus"],
                    "total_acceptors": bio["total_acceptors"],
                    "positive_count": bio["positive_count"],
                    "negative_count": bio["negative_count"],
                    "positive_pair": f"{bio['positive_pair_sorted']}",
                    "negative_pair": f"{bio['negative_pair_sorted']}",
                    "mismatch_total": bio["total_acceptors"] - 240,
                    "mismatch_128": bio["positive_count"] - 128,
                    "mismatch_112": bio["negative_count"] - 112,
                    "mismatch_pos_pair": str(tuple(x - y for x, y in zip(bio["positive_pair_sorted"], (72, 56)))),
                    "mismatch_neg_pair": str(tuple(x - y for x, y in zip(bio["negative_pair_sorted"], (56, 56)))),
                }
            )

    # Example explicit slots on the canonical anchor for each host
    for A in range(3):
        slots, bio = build_biological_slots(A, mpf("0.3"))
        for row in slots:
            slot_rows.append({"A": A, **row})

    return pd.DataFrame(anchor_rows), pd.DataFrame(corridor_rows), pd.DataFrame(slot_rows)


def qgt_corridor_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, (u, v) in enumerate(corridor_pairs()):
        for A in range(3):
            s = State(A=A, u=u, v=v, theta=TAU * mpf(i) / 10 + mpf("0.3"))
            y = yellow_filter(s)
            rows.append(
                {
                    "depth_index": i,
                    "A": A,
                    "u": u,
                    "v": v,
                    "uv": u * v,
                    "delta": str(s.delta),
                    "t": str(y["t"]),
                    "residual_per_edge": str(y["residual_per_edge"]),
                    "residual_two_sided": str(y["residual_two_sided"]),
                    "g_phiphi": str(y["g"][0][0]),
                    "g_deltadelta": str(y["g"][1][1]),
                    "Omega_phidelta": str(y["Omega"][0][1]),
                    "qgt_closed_form_residual": str(y["qgt_closed_form_residual"]),
                }
            )
    return pd.DataFrame(rows)


def pairwise_correlation_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, (u, v) in enumerate(corridor_pairs()):
        for A in range(3):
            for j in range(16):
                s = State(A=A, u=u, v=v, theta=TAU * j / 16 + mpf(i) * TAU / 32)
                g = green_filter(s)
                b = blue_filter(s)
                r = red_filter(s)
                y = yellow_filter(s)
                rows.append(
                    {
                        "A": A,
                        "u": u,
                        "v": v,
                        "uv": u * v,
                        "phi": float(s.phi),
                        "green_rho": float(g["rho"]),
                        "green_x4": float(g["x"][3]),
                        "blue_n3": float(b["n"][2]),
                        "red_tau_im": float(mp.im(r["tau"])),
                        "yellow_Omega12": float(y["Omega"][0][1]),
                        "yellow_g11": float(y["g"][0][0]),
                        "yellow_focus_index": BASE_FOCUS_ORDER.index(y["biological"]["base_focus"]),
                    }
                )
    df = pd.DataFrame(rows)
    corr = df[
        ["green_rho", "green_x4", "blue_n3", "red_tau_im", "yellow_Omega12", "yellow_g11", "yellow_focus_index"]
    ].corr()
    corr = corr.reset_index().rename(columns={"index": "observable"})
    return corr


def gate_dataframe(words: Sequence[str], states: Sequence[State], label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    maxima = {name: mp.mpf("0") for name in FILTERS}
    worst: Dict[str, Dict[str, Any] | None] = {name: None for name in FILTERS}

    for s in states:
        for word in words:
            s2 = apply_word(word, s)
            row: Dict[str, Any] = {
                "gate": label,
                "state_A": s.A,
                "state_u": s.u,
                "state_v": s.v,
                "state_theta": str(s.theta),
                "word": word,
            }
            for name, filt in FILTERS.items():
                direct = filt(s2)
                pushed = apply_word_to_filter(name, word, filt(s))
                err = filter_error(direct, pushed)
                row[f"{name}_error"] = str(err)
                if err > maxima[name]:
                    maxima[name] = err
                    worst[name] = {"state": repr(s), "word": word, "error": str(err)}
            rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "label": label,
        "rows": len(rows),
        "states": len(states),
        "words": len(words),
        "max_green_error": str(maxima["green"]),
        "max_blue_error": str(maxima["blue"]),
        "max_red_error": str(maxima["red"]),
        "max_yellow_error": str(maxima["yellow"]),
        "worst_green_case": worst["green"],
        "worst_blue_case": worst["blue"],
        "worst_red_case": worst["red"],
        "worst_yellow_case": worst["yellow"],
    }
    return df, summary


def recovery_gate(states: Sequence[State]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    maxima: Dict[str, mp.mpf] = {}
    for name in RECOVER:
        maxima[f"{name}_theta"] = mp.mpf("0")
        maxima[f"{name}_delta"] = mp.mpf("0")

    for s in states:
        row: Dict[str, Any] = {
            "A": s.A,
            "u": s.u,
            "v": s.v,
            "theta": str(s.theta),
            "delta": str(s.delta),
            "kappa": s.kappa,
            "phi": str(s.phi),
        }
        for name, filt in FILTERS.items():
            rtheta, rdelta = RECOVER[name](filt(s))
            terr = mp.fabs(rtheta - s.theta)
            derr = mp.fabs(rdelta - s.delta)
            row[f"{name}_theta_error"] = str(terr)
            row[f"{name}_delta_error"] = str(derr)
            maxima[f"{name}_theta"] = max(maxima[f"{name}_theta"], terr)
            maxima[f"{name}_delta"] = max(maxima[f"{name}_delta"], derr)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "samples": len(states),
        "max_green_theta_error": str(maxima["green_theta"]),
        "max_green_delta_error": str(maxima["green_delta"]),
        "max_blue_theta_error": str(maxima["blue_theta"]),
        "max_blue_delta_error": str(maxima["blue_delta"]),
        "max_red_theta_error": str(maxima["red_theta"]),
        "max_red_delta_error": str(maxima["red_delta"]),
        "max_yellow_theta_error": str(maxima["yellow_theta"]),
        "max_yellow_delta_error": str(maxima["yellow_delta"]),
    }
    return df, summary


def negative_controls() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # product-only
    s1 = State(A=0, u=1, v=6, theta=mpf("0.3"))
    s2 = State(A=0, u=2, v=3, theta=mpf("0.3"))

    row_product: Dict[str, Any] = {
        "pair_1": "(1,6)",
        "pair_2": "(2,3)",
        "shared_product": 6,
        "B_image_pair_1": str((B_state(s1).u, B_state(s1).v)),
        "B_image_pair_2": str((B_state(s2).u, B_state(s2).v)),
        "B_images_equal": False,
    }
    prod_payload_1 = stripped_product_payloads(s1)
    prod_payload_2 = stripped_product_payloads(s2)
    for name in FILTERS:
        row_product[f"same_{name}_stripped_payload_error"] = str(abs_diff(prod_payload_1[name], prod_payload_2[name]))

    # sheet-loss
    s = State(A=1, u=55, v=89, theta=mpf("0.3"))
    s_q4 = apply_word("QQQQ", s)
    row_sheet: Dict[str, Any] = {
        "state": repr(s),
        "Q4_state": repr(s_q4),
        "kappa_changed": s.kappa != s_q4.kappa,
    }
    period_1 = stripped_periodic_payloads(s)
    period_2 = stripped_periodic_payloads(s_q4)
    for name in FILTERS:
        row_sheet[f"same_periodic_{name}_without_kappa"] = str(abs_diff(period_1[name], period_2[name]))
    product_df = pd.DataFrame([row_product])
    sheet_df = pd.DataFrame([row_sheet])

    summary = {
        "product_only_B_failure": serialize_value(row_product),
        "sheet_loss_without_kappa": serialize_value(row_sheet),
    }
    return product_df, sheet_df, summary


def plot_gate_errors(summary: Dict[str, Any], output_path: Path) -> None:
    gates = ["short_exhaustive", "deep_words", "random_walks"]
    filters = ["green", "blue", "red", "yellow"]
    values = {f: [] for f in filters}
    for gate in gates:
        d = summary["covariance_gates"][gate]
        for f in filters:
            v = mpf(d[f"max_{f}_error"])
            if v == 0:
                v = mp.mpf("1e-120")
            values[f].append(float(v))
    x = range(len(gates))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for off, f in zip(offsets, filters):
        ax.bar([i + off * width for i in x], values[f], width=width, label=f)
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(gates, rotation=15)
    ax.set_ylabel("max covariance error")
    ax.set_title("Four-filter covariance gate errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_qgt_corridor(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    uv = [int(x) for x in df["uv"]]
    g11 = [float(x) for x in df["g_phiphi"]]
    omega = [float(x) for x in df["Omega_phidelta"]]
    ax.plot(uv, omega, marker="o", label="Omega_{phi,delta}")
    ax.plot(uv, g11, marker="s", label="g_{phi,phi}")
    ax.set_xscale("log")
    ax.set_xlabel("u v")
    ax.set_ylabel("yellow QGT scalar component")
    ax.set_title("Yellow QGT components on balanced corridor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_biological_decomposition(anchor_df: pd.DataFrame, output_path: Path) -> None:
    # one bar per host family: positive bucket broken into (72,56), negative into (56,56)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    families = []
    pos1 = []
    pos2 = []
    neg1 = []
    neg2 = []
    for A in range(3):
        row = anchor_df[anchor_df["A"] == A].iloc[0]
        families.append(row["family"])
        p = eval(row["positive_pair"])
        n = eval(row["negative_pair"])
        pos1.append(p[0]); pos2.append(p[1]); neg1.append(n[0]); neg2.append(n[1])
    x = list(range(len(families)))
    ax.bar(x, pos1, label="128 bucket first part")
    ax.bar(x, pos2, bottom=pos1, label="128 bucket second part")
    bottoms = [p1 + p2 for p1, p2 in zip(pos1, pos2)]
    ax.bar(x, neg1, bottom=bottoms, label="112 bucket first part")
    ax.bar(x, neg2, bottom=[b + n1 for b, n1 in zip(bottoms, neg1)], label="112 bucket second part")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("acceptor count")
    ax.set_title("Yellow biological decomposition by host family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correlations(corr_df: pd.DataFrame, output_path: Path) -> None:
    df = corr_df.set_index("observable")
    mat = df.values
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title("Anchor/corridor observable correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_analysis(output_dir: Path | str) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    states = canonical_states()

    short_df, short_summary = gate_dataframe(words_upto(6), states, "short_exhaustive")
    deep_df, deep_summary = gate_dataframe(deep_words(), states, "deep_words")
    random_df, random_summary = gate_dataframe(random_words(), states, "random_walks")

    rec_df, rec_summary = recovery_gate(canonical_anchor_states_for_recovery())

    product_df, sheet_df, neg_summary = negative_controls()

    bio_anchor_df, bio_corridor_df, slot_df = biological_gate_rows()
    qgt_df = qgt_corridor_rows()
    corr_df = pairwise_correlation_rows()

    short_df.to_csv(output_dir / "short_exhaustive_covariance_gate.csv", index=False)
    deep_df.to_csv(output_dir / "deep_word_covariance_gate.csv", index=False)
    random_df.to_csv(output_dir / "random_walk_covariance_gate.csv", index=False)
    rec_df.to_csv(output_dir / "recovery_gate.csv", index=False)
    product_df.to_csv(output_dir / "negative_control_product_only.csv", index=False)
    sheet_df.to_csv(output_dir / "negative_control_sheet_loss.csv", index=False)
    bio_anchor_df.to_csv(output_dir / "biological_gate_anchor.csv", index=False)
    bio_corridor_df.to_csv(output_dir / "biological_gate_corridor.csv", index=False)
    slot_df.to_csv(output_dir / "biological_projection_slots.csv", index=False)
    qgt_df.to_csv(output_dir / "yellow_qgt_corridor.csv", index=False)
    corr_df.to_csv(output_dir / "pairwise_correlations.csv", index=False)

    summary: Dict[str, Any] = {
        "canonical_anchor": {
            "u": 55,
            "v": 89,
            "uv": 4895,
            "delta_star": str(PI / 4895),
            "im_tau_star": str(mp.mpf(1) / 4895),
            "residual_per_edge": "1/24",
            "residual_two_sided": "1/12",
        },
        "four_filter_definitions": {
            "green": "(A,u,v,kappa,phi,x in S^3, stereographic chart)",
            "blue": "(A,u,v,kappa,phi,U in SU(2), Hopf base n in S^2)",
            "red": "(A,u,v,tau), with completion coordinate retained",
            "yellow": "(A,u,v,kappa,phi, residual law, host-phased spinor, QGT split, biological 240-slot scaffold)",
        },
        "covariance_gates": {
            "short_exhaustive": short_summary,
            "deep_words": deep_summary,
            "random_walks": random_summary,
        },
        "recovery_gate": rec_summary,
        "negative_controls": neg_summary,
        "biological_gates": {
            "anchor_rows": len(bio_anchor_df),
            "corridor_rows": len(bio_corridor_df),
            "slot_rows": len(slot_df),
            "anchor_total_mismatch_max": int(bio_anchor_df["mismatch_total"].abs().max()),
            "corridor_total_mismatch_max": int(bio_corridor_df["mismatch_total"].abs().max()),
            "anchor_128_mismatch_max": int(bio_anchor_df["mismatch_128"].abs().max()),
            "corridor_128_mismatch_max": int(bio_corridor_df["mismatch_128"].abs().max()),
            "anchor_112_mismatch_max": int(bio_anchor_df["mismatch_112"].abs().max()),
            "corridor_112_mismatch_max": int(bio_corridor_df["mismatch_112"].abs().max()),
            "anchor_positive_pair_examples": sorted(set(bio_anchor_df["positive_pair"]))[:3],
            "anchor_negative_pair_examples": sorted(set(bio_anchor_df["negative_pair"]))[:3],
        },
        "yellow_qgt_bridge": {
            "corridor_rows": len(qgt_df),
            "max_qgt_closed_form_residual": str(max(mpf(x) for x in qgt_df["qgt_closed_form_residual"])),
            "g_phiphi_range": [str(min(mpf(x) for x in qgt_df["g_phiphi"])), str(max(mpf(x) for x in qgt_df["g_phiphi"]))],
            "Omega_phidelta_range": [str(min(mpf(x) for x in qgt_df["Omega_phidelta"])), str(max(mpf(x) for x in qgt_df["Omega_phidelta"]))],
        },
        "interpretation": {
            "positive": "With the full ordered denominator pair q=(u,v), the host label A, and lifted phase-sheet retention, green, blue, red, and yellow remain covariant readouts of the same lifted object under Q, B, L on the declared balanced-window family.",
            "negative": "If only N=uv is retained, exact B-closure still fails. If kappa is dropped, periodic payloads still collapse exact lifted return to visible recurrence, now including the yellow QGT / biology layer.",
            "yellow_scope": "The yellow bridge is package-level and candidate-theorem level: it numerically validates one clean four-filter unification using the CF00/CF01 quotient machinery and the Halitsky-Klitzing-Moxness 240-acceptor decomposition. It is not a proof that the biological filter is uniquely forced by the current canon.",
        },
    }

    plot_gate_errors(summary, figures_dir / "four_filter_gate_errors.png")
    plot_qgt_corridor(qgt_df, figures_dir / "yellow_qgt_corridor.png")
    plot_biological_decomposition(bio_anchor_df, figures_dir / "biological_decomposition.png")
    plot_correlations(corr_df, figures_dir / "correlation_heatmap.png")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(serialize_value(summary), f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the four-filter covariance analysis.")
    parser.add_argument("output_dir", nargs="?", default="results", help="Directory for analysis artifacts")
    args = parser.parse_args()
    run_analysis(Path(args.output_dir))
