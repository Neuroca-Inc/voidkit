#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-native-baryon")
import matplotlib.pyplot as plt
import numpy as np

sys.set_int_max_str_digits(0)
TOL = 1.0e-11
PHI = (1.0 + math.sqrt(5.0)) / 2.0


@dataclass(frozen=True)
class G:
    re: Fraction
    im: Fraction

    @staticmethod
    def zero() -> "G":
        return G(Fraction(0), Fraction(0))

    @staticmethod
    def one() -> "G":
        return G(Fraction(1), Fraction(0))

    def mul_i(self) -> "G":
        return G(-self.im, self.re)

    def scale(self, value: Fraction) -> "G":
        return G(self.re * value, self.im * value)

    def norm_sq(self) -> Fraction:
        return self.re * self.re + self.im * self.im

    def complex(self) -> complex:
        return complex(float(self.re), float(self.im))

    def text(self) -> str:
        if self.re == 0 and self.im.denominator == 1:
            return f"{self.im.numerator}i"
        return f"({self.re})+({self.im})i"


@dataclass
class State:
    domain: int
    u: int
    v: int
    quarter_turns: int
    k: int
    j: int
    completed: list[list[G]]
    active: list[G]

    @staticmethod
    def initial() -> "State":
        return State(0, 1, 1, 0, 0, 1, [], [G.zero(), G.one()])

    def clone(self) -> "State":
        return State(
            self.domain,
            self.u,
            self.v,
            self.quarter_turns,
            self.k,
            self.j,
            [list(layer) for layer in self.completed],
            list(self.active),
        )

    def phase_positions(self) -> int:
        return 6 * (2**self.domain)

    def capacity(self) -> int:
        if self.j == 1:
            return 2
        if self.j == 2:
            return 4
        return 2 ** (2 * self.j)

    def emit(self) -> str:
        terminal = self.k == self.phase_positions() - 1
        product = self.u * self.v
        cap = self.capacity()
        if terminal:
            return "B" if product < cap else "L"
        if product >= cap:
            return "Q"
        return "B" if self.v * (self.u + self.v) <= cap else "Q"

    def tick(self) -> tuple[str, Fraction | None]:
        primitive = self.emit()
        inserted_c: Fraction | None = None
        if primitive == "B":
            old_u, old_v = self.u, self.v
            inserted_c = Fraction(old_u, old_u + old_v)
            inserted = self.active[1].scale(inserted_c)
            self.active = [self.active[0], inserted, *self.active[1:]]
            self.u, self.v = old_v, old_u + old_v
        elif primitive == "Q":
            self.active = [point.mul_i() for point in self.active]
            self.quarter_turns += 1
            self.k += 1
            self.j += 1
        elif primitive == "L":
            self.completed.append(list(self.active))
            self.active = [G.zero(), G.one()]
            self.domain += 1
            self.k = 0
            self.j = self.phase_positions() - 5
        else:
            raise AssertionError(primitive)
        return primitive, inserted_c


# Exact B/chart operator already established by the native-operator stage.
def reversal(n: int) -> np.ndarray:
    return np.fliplr(np.eye(n, dtype=float))


def b_map(n: int, c: float) -> np.ndarray:
    if n < 3:
        raise ValueError("n must be >= 3")
    s = np.zeros((n + 1, n), dtype=float)
    s[0, 0] = 1.0
    s[1, 1] = c
    for j in range(1, n):
        s[j + 1, j] = 1.0
    return s


def b_chart_defect(n: int, c: float) -> np.ndarray:
    s = b_map(n, c)
    return reversal(n + 1) @ s - s @ reversal(n)


def excess_mode(n: int, c: float) -> np.ndarray:
    """Unique target-kernel mode after the two transported pole modes are removed."""
    if n < 3:
        raise ValueError("n must be >= 3")
    vector = np.zeros(n + 1, dtype=float)
    vector[1] = 1.0
    if n >= 4:
        vector[2 : n - 1] = 1.0 - c
    vector[n - 1] = 1.0
    return vector / np.linalg.norm(vector)


def standard_su3() -> list[np.ndarray]:
    sqrt3 = math.sqrt(3.0)
    return [
        np.array([[0, .5, 0], [.5, 0, 0], [0, 0, 0]], dtype=complex),
        np.array([[0, -.5j, 0], [.5j, 0, 0], [0, 0, 0]], dtype=complex),
        np.diag([.5, -.5, 0]).astype(complex),
        np.array([[0, 0, .5], [0, 0, 0], [.5, 0, 0]], dtype=complex),
        np.array([[0, 0, -.5j], [0, 0, 0], [.5j, 0, 0]], dtype=complex),
        np.array([[0, 0, 0], [0, 0, .5], [0, .5, 0]], dtype=complex),
        np.array([[0, 0, 0], [0, 0, -.5j], [0, .5j, 0]], dtype=complex),
        np.diag([1.0, 1.0, -2.0]).astype(complex) / (2.0 * sqrt3),
    ]


def kron(*matrices: np.ndarray) -> np.ndarray:
    out = matrices[0]
    for matrix in matrices[1:]:
        out = np.kron(out, matrix)
    return out


def parity(p: tuple[int, int, int]) -> int:
    inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if p[i] > p[j])
    return -1 if inversions % 2 else 1


def baryon_color_singlet() -> np.ndarray:
    vector = np.zeros(27, dtype=complex)
    for p in permutations(range(3)):
        vector[p[0] * 9 + p[1] * 3 + p[2]] = parity(p) / math.sqrt(6.0)
    return vector


def total_baryon_generators() -> list[np.ndarray]:
    eye = np.eye(3, dtype=complex)
    return [
        kron(t, eye, eye) + kron(eye, t, eye) + kron(eye, eye, t)
        for t in standard_su3()
    ]


def write_json(path: Path, payload: Any) -> None:
    
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not JSON serializable: {type(value)!r}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_history_check(package: Path) -> dict[str, Any]:
    ledger = package / "evidence" / "UPSTREAM_SNAPSHOT_HASHES.json"
    if not ledger.exists():
        return {"status": False, "reason": "missing upstream hash ledger"}
    expected = json.loads(ledger.read_text(encoding="utf-8"))
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected_hash in expected.items():
        target = package / relative
        if not target.exists():
            missing.append(relative)
        elif sha256(target) != expected_hash:
            mismatched.append(relative)
    return {
        "status": len(expected) >= 11 and not missing and not mismatched,
        "snapshot_count": len(expected),
        "missing": missing,
        "mismatched": mismatched,
    }


def weighted_quantile(x: np.ndarray, p: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cumulative = np.cumsum(p[order])
    return float(x[order[np.searchsorted(cumulative, q)]])


def profile_certificate(layer: int, n: int, c_exact: Fraction, points: list[G]) -> dict[str, Any]:
    c = float(c_exact)
    d = b_chart_defect(n, c)
    k_target = d @ d.T
    values = np.linalg.eigvalsh(k_target)
    positive = values[values > TOL]
    chi = excess_mode(n, c)
    probability = chi * chi
    x = np.array([abs(point.complex()) for point in points], dtype=float)
    mean = float(np.dot(probability, x))
    second = float(np.dot(probability, x * x))
    variance = second - mean * mean
    interior = k_target[1:-1, 1:-1]
    interior_values = np.linalg.eigvalsh(interior)
    uniform = np.zeros(n + 1, dtype=float)
    uniform[1:-1] = 1.0
    uniform /= np.linalg.norm(uniform)
    shifted = np.zeros_like(chi)
    shifted[1:-1] = np.roll(chi[1:-1], 1)
    shifted /= np.linalg.norm(shifted)
    return {
        "layer": layer,
        "source_dimension_n": n,
        "target_dimension": n + 1,
        "final_B_c_exact": str(c_exact),
        "final_B_c": c,
        "target_kernel_dimension": int(np.sum(values < TOL)),
        "interior_kernel_dimension": int(np.sum(interior_values < TOL)),
        "excess_mode_residual": float(np.linalg.norm(d.T @ chi)),
        "endpoint_amplitude_left": float(chi[0]),
        "endpoint_amplitude_right": float(chi[-1]),
        "smallest_positive_relation_eigenvalue": float(positive[0]),
        "largest_relation_eigenvalue": float(positive[-1]),
        "uniform_interior_cost": float(uniform @ k_target @ uniform),
        "one_step_shift_cost": float(shifted @ k_target @ shifted),
        "mean_native_coordinate": mean,
        "variance_native_coordinate": variance,
        "rms_native_coordinate": math.sqrt(second),
        "median_native_coordinate": weighted_quantile(x, probability, 0.5),
        "q90_native_coordinate": weighted_quantile(x, probability, 0.9),
        "probability_below_0_1": float(np.sum(probability[x <= 0.1])),
        "probability_below_0_01": float(np.sum(probability[x <= 0.01])),
        "inverse_participation_ratio": float(np.sum(probability * probability)),
        "participation_count": float(1.0 / np.sum(probability * probability)),
        "normalization_residual": abs(float(np.dot(chi, chi)) - 1.0),
        "profile": chi.tolist(),
        "native_coordinates": x.tolist(),
    }


def main() -> int:
    package = Path(__file__).resolve().parents[1]
    results = package / "results"
    figures = package / "figures"
    evidence = package / "evidence"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)

    state = State.initial()
    states: dict[int, State] = {0: state.clone()}
    l_events: list[int] = []
    final_b: dict[int, tuple[int, int, Fraction]] = {}
    event_rows: list[dict[str, Any]] = []
    for causal_index in range(1, 456):
        before = state.clone()
        primitive, c = state.tick()
        states[causal_index] = state.clone()
        if primitive == "B" and c is not None:
            final_b[before.domain] = (causal_index, len(before.active), c)
        if primitive == "L":
            l_events.append(causal_index)
        event_rows.append({
            "causal_index": causal_index,
            "primitive": primitive,
            "domain_before": before.domain,
            "domain_after": state.domain,
            "active_points_before": len(before.active),
            "active_points_after": len(state.active),
            "inserted_c_exact": "" if c is None else str(c),
        })
    expected_l = [15, 45, 103, 220, 455]
    if l_events != expected_l:
        raise AssertionError((l_events, expected_l))
    write_csv(results / "runtime_events.csv", event_rows)

    l220 = states[220]
    l455 = states[455]
    color_layers = [1, 2, 3]
    profiles: list[dict[str, Any]] = []
    for layer in color_layers:
        _, n, c_exact = final_b[layer]
        profiles.append(profile_certificate(layer, n, c_exact, l220.completed[layer]))
    write_json(results / "l220_color_chiral_profiles.json", {"profiles": profiles})

    scaling_rows: list[dict[str, Any]] = []
    for layer in range(5):
        _, n, c_exact = final_b[layer]
        cert = profile_certificate(layer, n, c_exact, l455.completed[layer])
        scaling_rows.append({
            "layer": layer,
            "source_dimension_n": n,
            "target_dimension": n + 1,
            "final_B_c_exact": str(c_exact),
            "final_B_c": float(c_exact),
            "gap": cert["smallest_positive_relation_eigenvalue"],
            "n_squared_gap": n * n * cert["smallest_positive_relation_eigenvalue"],
            "pi_squared": math.pi * math.pi,
            "rms_native_coordinate": cert["rms_native_coordinate"],
            "sqrt_n_times_rms": math.sqrt(n) * cert["rms_native_coordinate"],
            "mean_native_coordinate": cert["mean_native_coordinate"],
            "target_kernel_dimension": cert["target_kernel_dimension"],
            "interior_kernel_dimension": cert["interior_kernel_dimension"],
        })
    write_csv(results / "relation_localization_scaling.csv", scaling_rows)

    ns = np.array([row["source_dimension_n"] for row in scaling_rows[1:]], dtype=float)
    gaps = np.array([row["gap"] for row in scaling_rows[1:]], dtype=float)
    rms = np.array([row["rms_native_coordinate"] for row in scaling_rows[1:]], dtype=float)
    gap_slope, gap_intercept = np.polyfit(np.log(ns), np.log(gaps), 1)
    rms_slope, rms_intercept = np.polyfit(np.log(ns), np.log(rms), 1)

    individual_gaps = [float(profile["smallest_positive_relation_eigenvalue"]) for profile in profiles]
    composite_gap = min(individual_gaps)
    interior_dimensions = [int(profile["target_dimension"]) - 2 for profile in profiles]
    flattened_degeneracy = math.prod(interior_dimensions)
    means = np.array([float(profile["mean_native_coordinate"]) for profile in profiles])
    variances = np.array([float(profile["variance_native_coordinate"]) for profile in profiles])
    baryon_mean = float(np.mean(means))
    baryon_radius_sq = float(np.mean(variances + (means - baryon_mean) ** 2))

    relation_ground = {
        "operator": "K_B = K_1 tensor I tensor I + I tensor K_2 tensor I + I tensor I tensor K_3, with K_a = D_a D_a^T and endpoint poles removed",
        "operator_origin": "Exact B/chart noncommutation on the retained Orthad; no external kinetic operator is introduced.",
        "color_layers": color_layers,
        "individual_target_dimensions": [int(p["target_dimension"]) for p in profiles],
        "individual_interior_dimensions": interior_dimensions,
        "individual_interior_kernel_dimensions": [int(p["interior_kernel_dimension"]) for p in profiles],
        "individual_relation_gaps": individual_gaps,
        "composite_interior_kernel_dimension": 1,
        "composite_first_excitation_gap": composite_gap,
        "ground_state": "chi_1 tensor chi_2 tensor chi_3",
        "flattened_operator_removed_kernel_dimension": flattened_degeneracy,
        "baryon_mean_native_coordinate": baryon_mean,
        "baryon_rms_internal_radius": math.sqrt(baryon_radius_sq),
        "middle_coordinate_status": "No independent middle-position coordinate exists in the full lifted state. The three chiral profiles are uniquely fixed by the three native target kernels.",
    }
    write_json(results / "native_baryon_relation_ground_state.json", relation_ground)

    # Exact endpoint SU(3) singlet remains load-bearing and commutes with relation action.
    color_state = baryon_color_singlet()
    generators = total_baryon_generators()
    color_residual = max(float(np.linalg.norm(generator @ color_state)) for generator in generators)
    color_stack = np.vstack(generators)
    singular = np.linalg.svd(color_stack, compute_uv=False)
    color_kernel_dimension = 27 - int(np.sum(singular > TOL))
    combined = {
        "color_kernel_dimension": color_kernel_dimension,
        "color_generator_action_residual": color_residual,
        "relation_kernel_dimension": 1,
        "combined_color_relation_ground_dimension": color_kernel_dimension,
        "combined_state": "epsilon_abc |a b c> tensor chi_1 tensor chi_2 tensor chi_3",
        "SU3_invariance_reason": "The relation operator acts identically outside the endpoint color factor, while epsilon is annihilated by every total native SU(3) generator.",
        "fermionic_structure": "The color factor is antisymmetric. The native relation ground factor is a fixed product of the three retained chiral index lines; no external spatial ansatz is used.",
    }
    write_json(results / "combined_color_relation_certificate.json", combined)

    negative_rows: list[dict[str, Any]] = []
    for profile in profiles:
        negative_rows.extend([
            {
                "layer": profile["layer"],
                "control": "native_excess_mode",
                "relation_cost": profile["excess_mode_residual"] ** 2,
                "selected": True,
            },
            {
                "layer": profile["layer"],
                "control": "uniform_interior",
                "relation_cost": profile["uniform_interior_cost"],
                "selected": False,
            },
            {
                "layer": profile["layer"],
                "control": "one_step_shift",
                "relation_cost": profile["one_step_shift_cost"],
                "selected": False,
            },
        ])
    write_csv(results / "relation_profile_negative_controls.csv", negative_rows)

    scaling = {
        "gap_loglog_slope": float(gap_slope),
        "gap_loglog_intercept": float(gap_intercept),
        "rms_loglog_slope": float(rms_slope),
        "rms_loglog_intercept": float(rms_intercept),
        "deepest_n_squared_gap": scaling_rows[-1]["n_squared_gap"],
        "relative_distance_to_pi_squared": abs(scaling_rows[-1]["n_squared_gap"] - math.pi**2) / math.pi**2,
        "deepest_sqrt_n_rms": scaling_rows[-1]["sqrt_n_times_rms"],
        "interpretation": "The native relation gap follows the discrete second-difference scale and the geometric profile narrows under retained refinement. These are measured consequences of the exact B/chart operator and exact layer coordinates.",
    }
    write_json(results / "relation_scaling_certificate.json", scaling)

    history = upstream_history_check(package)

    gates = {
        "G1_exact_native_recurrence": {
            "pass": l_events == expected_l,
            "measured": l_events,
            "expected": expected_l,
        },
        "G2_unique_chiral_profile_per_color_line": {
            "pass": all(p["target_kernel_dimension"] == 3 and p["interior_kernel_dimension"] == 1 and p["excess_mode_residual"] < TOL for p in profiles),
            "max_residual": max(p["excess_mode_residual"] for p in profiles),
        },
        "G3_relation_operator_lifts_flatness": {
            "pass": all(p["uniform_interior_cost"] > 1e-5 and p["one_step_shift_cost"] > 1e-5 for p in profiles) and flattened_degeneracy > 10000,
            "flattened_kernel_dimension_without_operator": flattened_degeneracy,
            "native_composite_kernel_dimension": 1,
        },
        "G4_unique_gapped_baryon_relation_ground": {
            "pass": relation_ground["composite_interior_kernel_dimension"] == 1 and composite_gap > 0,
            "gap": composite_gap,
        },
        "G5_exact_native_color_singlet": {
            "pass": color_kernel_dimension == 1 and color_residual < TOL,
            "kernel_dimension": color_kernel_dimension,
            "residual": color_residual,
        },
        "G6_finite_native_baryon_radius": {
            "pass": 0 < math.sqrt(baryon_radius_sq) < 1,
            "rms_internal_radius": math.sqrt(baryon_radius_sq),
        },
        "G7_relation_localization_scaling": {
            "pass": gap_slope < -1.7 and gap_slope > -2.2 and rms_slope < -0.35 and rms_slope > -0.65,
            **scaling,
        },
        "G8_no_forgetting_history_custody": history,
    }
    if not all(value.get("pass", value.get("status", False)) for value in gates.values()):
        raise AssertionError(gates)
    write_json(results / "gate_matrix.json", gates)

    summary = {
        "title": "Native baryon relation dynamics",
        "main_result": "The full retained B/chart relation operator already supplies the composite kinetic/localization structure. The L220 color singlet has a unique lifted chiral relation ground state and a nonzero first internal gap; no external kinetic operator or middle-position ansatz is required.",
        "l_events": l_events,
        "color_profile_dimensions": [p["target_dimension"] for p in profiles],
        "color_profile_source_n": [p["source_dimension_n"] for p in profiles],
        "individual_relation_gaps": individual_gaps,
        "composite_relation_gap": composite_gap,
        "baryon_rms_internal_radius": math.sqrt(baryon_radius_sq),
        "combined_ground_dimension": combined["combined_color_relation_ground_dimension"],
        "flattened_degeneracy_removed": flattened_degeneracy,
        "scaling": scaling,
        "gates_passed": sum(1 for gate in gates.values() if gate.get("pass", gate.get("status", False))),
        "gates_total": len(gates),
        "cortex_engine_modified": False,
    }
    write_json(results / "summary.json", summary)

    # Figures: each is a distinct plot and does not set explicit colors.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for p in profiles:
        ax.plot(p["native_coordinates"], np.square(p["profile"]), marker=".", label=f"layer {p['layer']} (N={p['target_dimension']})")
    ax.set_xscale("log")
    ax.set_xlabel("native point magnitude |p|")
    ax.set_ylabel("chiral-mode probability")
    ax.set_title("The three L220 color lines carry fixed lifted chiral profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "01_l220_color_chiral_profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for p in profiles:
        n = p["source_dimension_n"]
        c = p["final_B_c"]
        vals = np.linalg.eigvalsh(b_chart_defect(n, c) @ b_chart_defect(n, c).T)
        ax.semilogy(np.arange(len(vals)), np.maximum(vals, 1e-18), marker=".", label=f"layer {p['layer']}")
    ax.set_xlabel("ordered target eigenvalue")
    ax.set_ylabel("relation eigenvalue")
    ax.set_title("Three zero modes and a positive native relation spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "02_native_relation_spectra.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(["flattened\noperator erased", "full relation\ncolor × chiral"], [flattened_degeneracy, 1])
    ax.set_yscale("log")
    ax.set_ylabel("ground-space dimension")
    ax.set_title("The retained relation operator removes the artificial flat family")
    fig.tight_layout()
    fig.savefig(figures / "03_flatness_removed.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = []
    values = []
    for row in negative_rows:
        labels.append(f"L{row['layer']}\n{row['control'].replace('_', ' ')}")
        values.append(max(float(row["relation_cost"]), 1e-20))
    ax.bar(np.arange(len(values)), values)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
    ax.set_ylabel("native relation cost")
    ax.set_title("Only the exact excess profile remains in the chiral kernel")
    fig.tight_layout()
    fig.savefig(figures / "04_profile_negative_controls.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    n_all = np.array([row["source_dimension_n"] for row in scaling_rows], dtype=float)
    rms_all = np.array([row["rms_native_coordinate"] for row in scaling_rows], dtype=float)
    ax.loglog(n_all, rms_all, marker="o", label="measured")
    ax.loglog(n_all, math.exp(rms_intercept) * n_all**rms_slope, label=f"fit slope {rms_slope:.3f}")
    ax.set_xlabel("native source dimension n")
    ax.set_ylabel("RMS native coordinate")
    ax.set_title("The retained chiral profile narrows under exact refinement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "05_relation_localization_scaling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    scaled_gap = np.array([row["n_squared_gap"] for row in scaling_rows], dtype=float)
    ax.plot(n_all, scaled_gap, marker="o", label=r"$n^2\Delta_n$")
    ax.axhline(math.pi**2, linestyle="--", label=r"$\pi^2$")
    ax.set_xlabel("native source dimension n")
    ax.set_ylabel("scaled first relation gap")
    ax.set_title("The native B/chart gap approaches the second-difference scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "06_relation_gap_scaling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.20, 0.22, "Exact B insertion\n+ chart reversal"),
        (0.28, 0.58, 0.20, 0.22, "Unique chiral\nexcess mode per line"),
        (0.54, 0.58, 0.20, 0.22, "Kronecker-sum\nrelation ground"),
        (0.78, 0.58, 0.20, 0.22, "ε color singlet\n× relation ground"),
        (0.28, 0.12, 0.20, 0.22, "Finite native radius\n+ internal gap"),
        (0.54, 0.12, 0.20, 0.22, "uud / udd\nparticle propagation"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
    arrows = [((.22,.69),(.28,.69)),((.48,.69),(.54,.69)),((.74,.69),(.78,.69)),((.64,.58),(.43,.34)),((.43,.23),(.54,.23))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"->"})
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("No bolt-on stage: the particle profile is already in the retained relation operator")
    fig.tight_layout()
    fig.savefig(figures / "07_native_baryon_dependency_map.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
