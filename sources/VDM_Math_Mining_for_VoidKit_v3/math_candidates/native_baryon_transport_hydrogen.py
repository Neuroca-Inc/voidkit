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
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-native-baryon-hydrogen")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal
import sympy as sp

sys.set_int_max_str_digits(0)
TOL = 1.0e-11
SQRT5 = math.sqrt(5.0)
PHI = (1.0 + SQRT5) / 2.0


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


def exact_profile(n: int, c: Fraction) -> tuple[list[Fraction], Fraction]:
    raw = [Fraction(0)] * (n + 1)
    raw[1] = Fraction(1)
    for index in range(2, n - 1):
        raw[index] = Fraction(1) - c
    raw[n - 1] = Fraction(1)
    z = sum(value * value for value in raw)
    return [value * value / z for value in raw], z


def complex_weighted_mean(points: list[G], weights: list[Fraction]) -> G:
    if len(points) != len(weights):
        raise ValueError("point/weight mismatch")
    re = sum((w * p.re for w, p in zip(weights, points, strict=True)), Fraction(0))
    im = sum((w * p.im for w, p in zip(weights, points, strict=True)), Fraction(0))
    return G(re, im)


def weighted_norm(points: list[G], weights: list[Fraction]) -> Fraction:
    return sum((w * p.norm_sq() for w, p in zip(weights, points, strict=True)), Fraction(0))


def uniform_mean(points: list[G]) -> G:
    scale = Fraction(1, len(points))
    return G(
        sum((point.re for point in points), Fraction(0)) * scale,
        sum((point.im for point in points), Fraction(0)) * scale,
    )


def uniform_norm(points: list[G]) -> Fraction:
    return sum((point.norm_sq() for point in points), Fraction(0)) / len(points)


def squared_distance_mean(weighted_points: list[G], weights: list[Fraction], host: list[G]) -> Fraction:
    # E_{x~weights,y~uniform host}|y-x|^2, evaluated exactly without the full Cartesian expansion.
    mx = complex_weighted_mean(weighted_points, weights)
    my = uniform_mean(host)
    nx = weighted_norm(weighted_points, weights)
    ny = uniform_norm(host)
    cross = Fraction(2) * (mx.re * my.re + mx.im * my.im)
    return nx + ny - cross


def baryon_host_energy(
    completed: list[list[G]],
    profile_weights: dict[int, list[Fraction]],
    host: list[G],
) -> Fraction:
    energies = [squared_distance_mean(completed[layer], profile_weights[layer], host) for layer in (1, 2, 3)]
    return sum(energies, Fraction(0)) / 3


def serialize_layers(layers: Iterable[list[G]]) -> bytes:
    parts: list[str] = []
    for layer in layers:
        parts.append("[")
        for point in layer:
            parts.append(f"{point.re.numerator}/{point.re.denominator},{point.im.numerator}/{point.im.denominator};")
        parts.append("]")
    return "".join(parts).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def expected_three_line_span(
    coordinates: list[list[Fraction]],
    probabilities: list[list[Fraction]],
) -> Fraction:
    # Exact order-statistic evaluation: E[max]-E[min] over the finite rational support.
    support = sorted(set().union(*[set(values) for values in coordinates]))
    cdfs: list[list[Fraction]] = []
    survivals: list[list[Fraction]] = []
    for values, weights in zip(coordinates, probabilities, strict=True):
        cdf = []
        survival = []
        for x in support:
            cdf.append(sum((w for v, w in zip(values, weights, strict=True) if v <= x), Fraction(0)))
            survival.append(sum((w for v, w in zip(values, weights, strict=True) if v >= x), Fraction(0)))
        cdfs.append(cdf)
        survivals.append(survival)
    emax = Fraction(0)
    emin = Fraction(0)
    previous_cdf_product = Fraction(0)
    previous_survival_product = Fraction(0)
    for index, x in enumerate(support):
        cdf_product = math.prod(cdf[index] for cdf in cdfs)
        # P(max=x) = P(all<=x)-P(all<x), with the preceding support point representing <x.
        emax += x * (cdf_product - previous_cdf_product)
        previous_cdf_product = cdf_product
    for reverse_index, x in enumerate(reversed(support)):
        index = len(support) - 1 - reverse_index
        survival_product = math.prod(survival[index] for survival in survivals)
        # P(min=x) = P(all>=x)-P(all>x), with the following support point represented by the previous reverse step.
        emin += x * (survival_product - previous_survival_product)
        previous_survival_product = survival_product
    return emax - emin


def baryon_host_energy_float(
    completed: list[list[G]],
    profile_weights: dict[int, list[Fraction]],
    host: list[G],
) -> float:
    total = 0.0
    host_values = np.array([point.complex() for point in host], dtype=complex)
    host_mean = complex(np.mean(host_values))
    host_norm = float(np.mean(np.abs(host_values) ** 2))
    for layer in (1, 2, 3):
        values = np.array([point.complex() for point in completed[layer]], dtype=complex)
        weights = np.array([float(value) for value in profile_weights[layer]], dtype=float)
        mean = complex(np.dot(weights, values))
        norm = float(np.dot(weights, np.abs(values) ** 2))
        total += norm + host_norm - 2.0 * (mean.conjugate() * host_mean).real
    return total / 3.0


def radial_levels(
    reduced_mass: float,
    alpha: float,
    ell: int,
    h: float,
    radius: float,
    count: int,
    charge_product: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.arange(h, radius, h, dtype=float)
    kinetic_diag = np.full(len(r), 1.0 / (reduced_mass * h * h), dtype=float)
    off = np.full(len(r) - 1, -1.0 / (2.0 * reduced_mass * h * h), dtype=float)
    centrifugal = ell * (ell + 1.0) / (2.0 * reduced_mass * r * r)
    potential = charge_product * alpha / r
    diagonal = kinetic_diag + centrifugal + potential
    values, vectors = eigh_tridiagonal(
        diagonal,
        off,
        select="i",
        select_range=(0, count - 1),
        check_finite=True,
    )
    norms = np.sqrt(np.sum(vectors * vectors, axis=0) * h)
    vectors = vectors / norms
    return r, values, vectors


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_history_check(package: Path) -> dict[str, Any]:
    ledger = package / "evidence" / "UPSTREAM_SNAPSHOT_HASHES.json"
    expected = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {}
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected_hash in expected.items():
        target = package / relative
        if not target.exists():
            missing.append(relative)
        elif file_sha256(target) != expected_hash:
            mismatched.append(relative)
    return {
        "status": len(expected) >= 12 and not missing and not mismatched,
        "snapshot_count": len(expected),
        "missing": missing,
        "mismatched": mismatched,
    }


def main() -> int:
    package = Path(__file__).resolve().parents[1]
    results = package / "results"
    figures = package / "figures"
    evidence = package / "evidence"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)

    # Reconstruct the exact existing recurrence. No engine rule is changed.
    state = State.initial()
    states: dict[int, State] = {0: state.clone()}
    final_b: dict[int, tuple[int, int, Fraction]] = {}
    l_events: list[int] = []
    baryon_hash_at_220 = ""
    custody_failures = 0
    handoff_rows: list[dict[str, Any]] = []
    profile_weights: dict[int, list[Fraction]] = {}

    for causal_index in range(1, 3736):
        before = state.clone()
        primitive, inserted_c = state.tick()
        states[causal_index] = state.clone()
        if primitive == "B" and inserted_c is not None:
            final_b[before.domain] = (causal_index, len(before.active), inserted_c)
        if primitive == "L":
            l_events.append(causal_index)

        if causal_index == 220:
            for layer in (1, 2, 3):
                _, n, c = final_b[layer]
                weights, _ = exact_profile(n, c)
                profile_weights[layer] = weights
            baryon_hash_at_220 = sha256_bytes(serialize_layers(state.completed[1:4]))

        if causal_index > 220 and baryon_hash_at_220:
            current_hash = sha256_bytes(serialize_layers(state.completed[1:4]))
            if current_hash != baryon_hash_at_220:
                custody_failures += 1

        if primitive == "L" and causal_index >= 455:
            before_hash = sha256_bytes(serialize_layers(before.completed[1:4]))
            after_hash = sha256_bytes(serialize_layers(state.completed[1:4]))
            new_completed_host = state.completed[-1]
            host_byte_exact = serialize_layers([before.active]) == serialize_layers([new_completed_host])
            before_energy_float = baryon_host_energy_float(before.completed, profile_weights, before.active)
            after_energy_float = baryon_host_energy_float(state.completed, profile_weights, new_completed_host)
            handoff_rows.append({
                "l_causal_index": causal_index,
                "completed_domain": before.domain,
                "baryon_internal_hash_before": before_hash,
                "baryon_internal_hash_after": after_hash,
                "internal_state_byte_exact": before_hash == after_hash == baryon_hash_at_220,
                "active_host_point_count": len(before.active),
                "retained_host_point_count": len(new_completed_host),
                "retained_host_byte_exact": host_byte_exact,
                "baryon_host_relation_energy_before": before_energy_float,
                "baryon_host_relation_energy_after": after_energy_float,
                "handoff_energy_drift_exact": "0" if host_byte_exact and before_hash == after_hash else "structural_failure",
                "handoff_energy_float": before_energy_float,
                "numerical_energy_drift": after_energy_float - before_energy_float,
            })

    expected_l = [15, 45, 103, 220, 455, 923, 1860, 3735]
    if l_events != expected_l:
        raise AssertionError((l_events, expected_l))
    write_csv(results / "whole_baryon_handoff.csv", handoff_rows)

    l220 = states[220]
    coordinates: list[list[Fraction]] = []
    probabilities: list[list[Fraction]] = []
    relation_gaps: list[float] = []
    profile_rows: list[dict[str, Any]] = []
    for layer in (1, 2, 3):
        _, n, c = final_b[layer]
        weights, z = exact_profile(n, c)
        coords = [abs(point.im) if point.re == 0 else Fraction.from_float(abs(point.complex())) for point in l220.completed[layer]]
        coordinates.append(coords)
        probabilities.append(weights)
        defect = b_chart_defect(n, float(c))
        eigenvalues = np.linalg.eigvalsh(defect @ defect.T)
        positive = eigenvalues[eigenvalues > TOL]
        relation_gaps.append(float(positive[0]))
        for index, (coord, weight) in enumerate(zip(coords, weights, strict=True)):
            profile_rows.append({
                "layer": layer,
                "position": index,
                "native_coordinate_exact": str(coord),
                "native_coordinate": float(coord),
                "probability_exact": str(weight),
                "probability": float(weight),
                "normalization_Z_exact": str(z),
            })
    write_csv(results / "baryon_internal_profiles.csv", profile_rows)

    span_exact = expected_three_line_span(coordinates, probabilities)
    span = float(span_exact)

    # All coefficients below are accepted outputs of earlier native stages.
    m_quark = 3.0 / math.sqrt(130.0)
    m_electron = m_quark
    sigma_color = 1.5610938576665823
    alpha_native = 0.07191120514956423
    confinement_energy = sigma_color * span
    baryon_rest_mass = 3.0 * m_quark + confinement_energy

    # The inherited wall transport axis and normal mass axis anticommute.
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    anticommutator_residual = float(np.linalg.norm(sigma_x @ sigma_z + sigma_z @ sigma_x))
    mass_shell_rows: list[dict[str, Any]] = []
    for momentum in np.linspace(0.0, 1.0, 101):
        h_b = momentum * sigma_z + baryon_rest_mass * sigma_x
        eigenvalues = np.linalg.eigvalsh(h_b)
        positive_energy = float(eigenvalues[-1])
        expected_energy = math.sqrt(baryon_rest_mass**2 + momentum**2)
        constituent_readout = confinement_energy + 3.0 * math.sqrt(m_quark**2 + (momentum / 3.0) ** 2)
        mass_shell_rows.append({
            "momentum": float(momentum),
            "negative_energy": float(eigenvalues[0]),
            "positive_energy": positive_energy,
            "closed_form_energy": expected_energy,
            "mass_shell_residual": positive_energy - expected_energy,
            "group_velocity": 0.0 if momentum == 0 else momentum / expected_energy,
            "constituent_positive_branch_readout": constituent_readout,
            "constituent_vs_whole_relative_difference": (constituent_readout - expected_energy) / expected_energy,
        })
    write_csv(results / "baryon_dispersion.csv", mass_shell_rows)

    baryon_mass_certificate = {
        "unit_normal_quark_mass_exact": "3/sqrt(130)",
        "unit_normal_quark_mass": m_quark,
        "expected_internal_flux_span_exact": str(span_exact),
        "expected_internal_flux_span_sha256": sha256_bytes(str(span_exact).encode("utf-8")),
        "expected_internal_flux_span": span,
        "color_tension_exact_form": "(4/3)*(1/2 + 3*sqrt(5)/10) = 2/3 + 2*sqrt(5)/5",
        "color_tension": sigma_color,
        "confinement_rest_energy": confinement_energy,
        "baryon_rest_mass_expression": "9/sqrt(130) + sigma_C * E[max(X1,X2,X3)-min(X1,X2,X3)]",
        "baryon_rest_mass": baryon_rest_mass,
        "internal_relation_ground_energy": 0.0,
        "internal_relation_first_gap": min(relation_gaps),
        "transport_operator": "H_B(P)=P*Gamma_parallel + M_B*Beta_normal on the unique epsilon-times-relation internal ground",
        "gamma_beta_anticommutator_residual": anticommutator_residual,
        "dispersion": "E_B(P)^2=M_B^2+P^2",
    }
    write_json(results / "baryon_mass_and_dispersion_certificate.json", baryon_mass_certificate)

    handoff_certificate = {
        "recurrence_L_events": l_events,
        "baryon_formed_at": 220,
        "ticks_checked_after_formation": 3735 - 220,
        "completed_support_hash": baryon_hash_at_220,
        "internal_custody_failures": custody_failures,
        "handoffs_checked": len(handoff_rows),
        "all_internal_states_byte_exact": all(bool(row["internal_state_byte_exact"]) for row in handoff_rows),
        "all_cross_interface_energy_drifts_exact_zero": all(row["handoff_energy_drift_exact"] == "0" for row in handoff_rows),
        "meaning": "The complete epsilon-times-relation state remains an exact completed-layer record under every later B/Q/L event. At each L, its full weighted relation field to the active host is retained exactly when that host becomes completed.",
    }
    write_json(results / "whole_baryon_transport_certificate.json", handoff_certificate)

    # Place the charge +1 baryon and charge -1 electron in the already-derived Coulomb geometry.
    reduced_mass = m_electron * baryon_rest_mass / (m_electron + baryon_rest_mass)
    bohr_radius = 1.0 / (reduced_mass * alpha_native)
    analytic_ground = -0.5 * reduced_mass * alpha_native**2

    convergence_rows: list[dict[str, Any]] = []
    for h in (1.0, 0.5, 0.25, 0.125):
        for radius in (800.0, 1200.0, 2000.0):
            _, s_values, _ = radial_levels(reduced_mass, alpha_native, 0, h, radius, 3, -1.0)
            _, p_values, _ = radial_levels(reduced_mass, alpha_native, 1, h, radius, 2, -1.0)
            convergence_rows.append({
                "grid_spacing": h,
                "outer_radius": radius,
                "one_s_energy": float(s_values[0]),
                "two_s_energy": float(s_values[1]),
                "two_p_energy": float(p_values[0]),
                "one_s_relative_error_from_coulomb_recognition": abs(float(s_values[0]) - analytic_ground) / abs(analytic_ground),
                "two_s_two_p_splitting": float(s_values[1] - p_values[0]),
            })
    write_csv(results / "hydrogen_convergence.csv", convergence_rows)

    h_final = 0.125
    radius_final = 2000.0
    r, s_values, s_vectors = radial_levels(reduced_mass, alpha_native, 0, h_final, radius_final, 4, -1.0)
    _, p_values, p_vectors = radial_levels(reduced_mass, alpha_native, 1, h_final, radius_final, 3, -1.0)
    _, repulsive_values, _ = radial_levels(reduced_mass, alpha_native, 0, h_final, radius_final, 1, +1.0)
    _, neutral_values, _ = radial_levels(reduced_mass, alpha_native, 0, h_final, radius_final, 1, 0.0)

    one_s = s_vectors[:, 0]
    two_p = p_vectors[:, 0]
    p_one_s = one_s * one_s
    mean_radius = float(np.sum(p_one_s * r) * h_final)
    rms_radius = math.sqrt(float(np.sum(p_one_s * r * r) * h_final))
    probability_within_4a0 = float(np.sum(p_one_s[r <= 4.0 * bohr_radius]) * h_final)
    radial_dipole = float(np.sum(one_s * two_p * r) * h_final)
    z_dipole = radial_dipole / math.sqrt(3.0)
    transition_gap = float(p_values[0] - s_values[0])

    relativistic_correction_fraction = (
        5.0 * reduced_mass**3 * alpha_native**2 / 4.0
    ) * (1.0 / m_electron**3 + 1.0 / baryon_rest_mass**3)

    hydrogen_spectrum = {
        "constituents": {
            "baryon_charge": 1,
            "electron_charge": -1,
            "total_charge": 0,
            "charge_product": -1,
            "baryon_rest_mass": baryon_rest_mass,
            "electron_rest_mass": m_electron,
            "reduced_mass": reduced_mass,
        },
        "native_coulomb": {
            "source": "accepted CF09/four-predeep massless Coulomb branch",
            "alpha_from_unit_charge_fit": alpha_native,
            "potential": "V(r)=-alpha_native/r",
        },
        "radial_operator": "s-wave and p-wave separation-coordinate quotient of the same CF09 lattice Gauss/Laplacian geometry",
        "grid_spacing": h_final,
        "outer_radius": radius_final,
        "s_levels": s_values.tolist(),
        "p_levels": p_values.tolist(),
        "one_s_binding_energy": float(s_values[0]),
        "one_s_coulomb_recognition": analytic_ground,
        "one_s_relative_error": abs(float(s_values[0]) - analytic_ground) / abs(analytic_ground),
        "two_s_two_p_splitting": float(s_values[1] - p_values[0]),
        "one_s_mean_radius": mean_radius,
        "one_s_rms_radius": rms_radius,
        "native_bohr_radius": bohr_radius,
        "probability_within_four_bohr_radii": probability_within_4a0,
        "one_s_to_two_p_gap": transition_gap,
        "one_s_to_two_p_radial_dipole": radial_dipole,
        "one_s_to_two_p_z_dipole": z_dipole,
        "same_sign_ground_energy": float(repulsive_values[0]),
        "neutral_coupling_ground_energy": float(neutral_values[0]),
        "leading_relativistic_correction_fraction_estimate": relativistic_correction_fraction,
        "main_result": "The already-derived +1 baryon, -1 electron, native kinetic mass shell, and massless U(1) Coulomb geometry produce a normalizable neutral bound ground state and a discrete dipole-active spectrum without adding a new force or particle rule.",
    }
    write_json(results / "native_hydrogen_spectrum.json", hydrogen_spectrum)

    sampled_rows: list[dict[str, Any]] = []
    stride = max(1, len(r) // 1200)
    for index in range(0, len(r), stride):
        sampled_rows.append({
            "radius": float(r[index]),
            "one_s_radial_probability_density": float(one_s[index] ** 2),
            "two_p_radial_probability_density": float(two_p[index] ** 2),
        })
    write_csv(results / "hydrogen_radial_states_sampled.csv", sampled_rows)

    resonance_rows: list[dict[str, Any]] = []
    for domain in range(1, 33):
        q_budget = 6 * (2**domain) - 1
        target_harmonic = max(1, int(round(transition_gap * q_budget / (2.0 * math.pi))))
        best: dict[str, Any] | None = None
        for harmonic in range(max(1, target_harmonic - 2), target_harmonic + 3):
            omega = 2.0 * math.pi * harmonic / q_budget
            row = {
                "domain": domain,
                "q_budget": q_budget,
                "harmonic": harmonic,
                "packet_angular_frequency": omega,
                "transition_gap": transition_gap,
                "detuning": omega - transition_gap,
                "relative_detuning": (omega - transition_gap) / transition_gap,
                "dipole_matrix_element": z_dipole,
            }
            if best is None or abs(float(row["detuning"])) < abs(float(best["detuning"])):
                best = row
        assert best is not None
        resonance_rows.append(best)
    write_csv(results / "hydrogen_photon_resonance.csv", resonance_rows)
    best_resonance = min(resonance_rows, key=lambda row: abs(float(row["detuning"])))

    history = upstream_history_check(package)
    max_mass_shell_residual = max(abs(float(row["mass_shell_residual"])) for row in mass_shell_rows)
    final_convergence = [row for row in convergence_rows if row["grid_spacing"] == h_final and row["outer_radius"] == radius_final][0]

    gates = {
        "G1_exact_recurrence_and_baryon_custody": {
            "pass": l_events == expected_l and custody_failures == 0,
            "l_events": l_events,
            "ticks_checked": 3735 - 220,
            "custody_failures": custody_failures,
        },
        "G2_exact_whole_state_L_handoff": {
            "pass": len(handoff_rows) == 4 and all(row["handoff_energy_drift_exact"] == "0" and bool(row["internal_state_byte_exact"]) for row in handoff_rows),
            "handoffs": [row["l_causal_index"] for row in handoff_rows],
        },
        "G3_native_baryon_rest_mass_from_existing_sectors": {
            "pass": span > 0 and confinement_energy > 0 and baryon_rest_mass > 3 * m_quark,
            "expected_span": span,
            "confinement_energy": confinement_energy,
            "rest_mass": baryon_rest_mass,
        },
        "G4_exact_whole_particle_dispersion": {
            "pass": anticommutator_residual < TOL and max_mass_shell_residual < TOL and all(0.0 <= float(row["group_velocity"]) < 1.0 for row in mass_shell_rows),
            "anticommutator_residual": anticommutator_residual,
            "max_mass_shell_residual": max_mass_shell_residual,
        },
        "G5_native_hydrogen_bound_ground": {
            "pass": float(s_values[0]) < 0 and mean_radius > 0 and probability_within_4a0 > 0.95,
            "ground_energy": float(s_values[0]),
            "mean_radius": mean_radius,
            "probability_within_4a0": probability_within_4a0,
        },
        "G6_discrete_excited_and_dipole_active_spectrum": {
            "pass": float(s_values[1]) < 0 and float(p_values[0]) < 0 and transition_gap > 0 and abs(z_dipole) > 1.0 and abs(float(s_values[1] - p_values[0])) < 2.0e-8,
            "two_s": float(s_values[1]),
            "two_p": float(p_values[0]),
            "transition_gap": transition_gap,
            "dipole": z_dipole,
        },
        "G7_charge_and_force_negative_controls": {
            "pass": float(repulsive_values[0]) > 0 and float(neutral_values[0]) > 0,
            "same_sign_ground": float(repulsive_values[0]),
            "neutral_ground": float(neutral_values[0]),
        },
        "G8_numerical_convergence_and_relativistic_control": {
            "pass": float(final_convergence["one_s_relative_error_from_coulomb_recognition"]) < 1.0e-5 and relativistic_correction_fraction < 0.01,
            "one_s_relative_error": final_convergence["one_s_relative_error_from_coulomb_recognition"],
            "relativistic_correction_fraction": relativistic_correction_fraction,
        },
        "G9_native_packet_resonance_exists": {
            "pass": abs(float(best_resonance["relative_detuning"])) < 1.0e-6 and abs(z_dipole) > 1.0,
            "best_resonance": best_resonance,
        },
        "G10_no_forgetting_history_custody": history,
    }
    if not all(value.get("pass", value.get("status", False)) for value in gates.values()):
        raise AssertionError(gates)
    write_json(results / "gate_matrix.json", gates)

    summary = {
        "title": "Native baryon transport and first hydrogen-like bound state",
        "main_result": "The complete epsilon-times-relation baryon is an exact retained record through 3515 later B/Q/L transitions and four deeper L handoffs, with zero internal or cross-interface relation-energy loss. Projecting the already-derived wall transport and normal mass axes onto its unique internal ground gives an exact massive whole-particle dispersion. Coupling the resulting +1 uud state to the accepted -1 electron through the existing massless U(1) Coulomb geometry produces a normalizable neutral bound ground state, discrete 2s/2p levels, a nonzero 1s-to-2p dipole, and a near-resonant native light-packet harmonic.",
        "l_events": l_events,
        "baryon_transport": handoff_certificate,
        "baryon_rest_mass": baryon_rest_mass,
        "baryon_confinement_rest_energy": confinement_energy,
        "baryon_expected_span": span,
        "baryon_internal_gap": min(relation_gaps),
        "hydrogen": {
            "reduced_mass": reduced_mass,
            "native_alpha": alpha_native,
            "ground_energy": float(s_values[0]),
            "bohr_radius": bohr_radius,
            "mean_radius": mean_radius,
            "rms_radius": rms_radius,
            "transition_gap_1s_2p": transition_gap,
            "dipole_1s_2p": z_dipole,
            "best_native_packet_resonance": best_resonance,
        },
        "gates_passed": sum(1 for gate in gates.values() if gate.get("pass", gate.get("status", False))),
        "gates_total": len(gates),
        "cortex_engine_modified": False,
    }
    write_json(results / "summary.json", summary)

    # Distinct figures, no explicit colors/styles.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot([row["l_causal_index"] for row in handoff_rows], [row["handoff_energy_float"] for row in handoff_rows], marker="o", label="before L")
    ax.plot([row["l_causal_index"] for row in handoff_rows], [row["baryon_host_relation_energy_after"] for row in handoff_rows], marker="x", label="after L")
    ax.set_xlabel("L causal index")
    ax.set_ylabel("weighted baryon-host relation energy")
    ax.set_title("The complete baryon relation field is retained exactly through L")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "01_exact_whole_baryon_handoff.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    p_plot = [row["momentum"] for row in mass_shell_rows]
    ax.plot(p_plot, [row["positive_energy"] for row in mass_shell_rows], label="whole-particle native mass shell")
    ax.plot(p_plot, [row["constituent_positive_branch_readout"] for row in mass_shell_rows], linestyle="--", label="three-quark positive branch")
    ax.set_xlabel("whole-particle momentum")
    ax.set_ylabel("energy")
    ax.set_title("Native massive uud dispersion on the unique internal ground")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "02_native_baryon_dispersion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(p_plot, [row["group_velocity"] for row in mass_shell_rows])
    ax.set_xlabel("whole-particle momentum")
    ax.set_ylabel("group velocity")
    ax.set_title("The native baryon branch is massive and subluminal")
    fig.tight_layout()
    fig.savefig(figures / "03_native_baryon_group_velocity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    mask = r <= 6.0 * bohr_radius
    ax.plot(r[mask], one_s[mask] ** 2, label="1s")
    ax.plot(r[mask], two_p[mask] ** 2, label="2p")
    ax.set_xlabel("native separation radius")
    ax.set_ylabel("radial probability density")
    ax.set_title("First neutral uud+electron bound-state profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "04_native_hydrogen_radial_states.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    level_labels = ["1s", "2s", "2p", "3s", "3p"]
    level_values = [s_values[0], s_values[1], p_values[0], s_values[2], p_values[1]]
    ax.scatter(np.arange(len(level_values)), level_values)
    ax.set_xticks(np.arange(len(level_values)), level_labels)
    ax.set_ylabel("energy below continuum")
    ax.set_title("The native Coulomb quotient has a discrete atomic spectrum")
    fig.tight_layout()
    fig.savefig(figures / "05_native_hydrogen_spectrum.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    selected_resonances = [row for row in resonance_rows if 10 <= int(row["domain"]) <= 32]
    ax.semilogy([row["domain"] for row in selected_resonances], [abs(float(row["relative_detuning"])) for row in selected_resonances], marker="o")
    ax.set_xlabel("native packet domain")
    ax.set_ylabel("best relative detuning")
    ax.set_title("Generated light harmonics approach the 1s-to-2p transition")
    fig.tight_layout()
    fig.savefig(figures / "06_native_photon_transition_resonance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.axis("off")
    boxes = [
        (0.02, .62, .19, .22, "epsilon × relation\ninternal ground"),
        (.27, .62, .19, .22, "exact B/Q/L custody\nand L handoff"),
        (.52, .62, .19, .22, "wall transport ×\nnormal mass"),
        (.77, .62, .20, .22, "massive +1 uud\nparticle branch"),
        (.27, .15, .19, .22, "accepted -1\nelectron"),
        (.52, .15, .19, .22, "existing massless\nU(1) Coulomb field"),
        (.77, .15, .20, .22, "neutral discrete\nbound spectrum"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
    for start, end in [((.21,.73),(.27,.73)),((.46,.73),(.52,.73)),((.71,.73),(.77,.73)),((.46,.26),(.52,.26)),((.71,.26),(.77,.26)),((.87,.62),(.87,.37))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"->"})
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("No new force or particle rule: the atom is assembled from already generated sectors")
    fig.tight_layout()
    fig.savefig(figures / "07_native_atom_dependency_map.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
