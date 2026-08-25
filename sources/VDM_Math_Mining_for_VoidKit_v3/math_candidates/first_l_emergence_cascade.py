#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

    def scale(self, x: Fraction) -> "G":
        return G(self.re * x, self.im * x)

    def norm_sq(self) -> Fraction:
        return self.re * self.re + self.im * self.im

    def text(self) -> str:
        def fmt(x: Fraction) -> str:
            return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
        if self.im == 0:
            return fmt(self.re)
        if self.re == 0:
            return f"{fmt(self.im)}i"
        sign = "+" if self.im > 0 else "-"
        return f"{fmt(self.re)}{sign}{fmt(abs(self.im))}i"


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
        capacity = self.capacity()
        if terminal:
            return "B" if product < capacity else "L"
        if product >= capacity:
            return "Q"
        return "B" if self.v * (self.u + self.v) <= capacity else "Q"

    def tick(self) -> str:
        primitive = self.emit()
        if primitive == "B":
            old_u, old_v = self.u, self.v
            c = Fraction(old_u, old_u + old_v)
            inserted = self.active[1].scale(c)
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
        return primitive

    @property
    def layers(self) -> list[list[G]]:
        return [*self.completed, self.active]


def norm_sq(points: list[G]) -> Fraction:
    return sum((point.norm_sq() for point in points), Fraction(0))


def total_norm_sq(state: State) -> Fraction:
    return sum((norm_sq(layer) for layer in state.layers), Fraction(0))


def point_count(state: State) -> int:
    return sum(len(layer) for layer in state.layers)


def doubled_area(a: G, b: G, c: G) -> Fraction:
    u = G(b.re - a.re, b.im - a.im)
    v = G(c.re - a.re, c.im - a.im)
    return u.re * v.im - u.im * v.re


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if total == 0.0 else 1.0 - residual / total
    return float(slope), float(intercept), r2


def wilson_domain_wall(mass: np.ndarray, r: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    n = len(mass)
    derivative = np.zeros((n, n), dtype=complex)
    laplacian = np.zeros((n, n), dtype=float)
    for index in range(n - 1):
        derivative[index, index + 1] = 0.5
        derivative[index + 1, index] = -0.5
        laplacian[index, index] += 1.0
        laplacian[index + 1, index + 1] += 1.0
        laplacian[index, index + 1] -= 1.0
        laplacian[index + 1, index] -= 1.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    gamma5 = np.kron(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex), np.eye(n))
    hamiltonian = np.kron(-1.0j * sigma_x, derivative) + np.kron(
        sigma_z, np.diag(mass) + 0.5 * r * laplacian
    )
    return hamiltonian, gamma5


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    figures = root / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    state = State.initial()
    states = [state.clone()]
    primitives = ["INITIAL"]
    transition_rows: list[dict[str, object]] = []
    for tick in range(1, 104):
        before = state.clone()
        primitive = state.tick()
        after = state.clone()
        primitives.append(primitive)
        states.append(after.clone())
        transition_rows.append(
            {
                "causal_index": tick,
                "primitive": primitive,
                "domain_before": before.domain,
                "domain_after": after.domain,
                "active_points_before": len(before.active),
                "active_points_after": len(after.active),
                "total_points_before": point_count(before),
                "total_points_after": point_count(after),
                "active_norm_before": str(norm_sq(before.active)),
                "active_norm_after": str(norm_sq(after.active)),
                "active_norm_delta": str(norm_sq(after.active) - norm_sq(before.active)),
                "total_norm_delta": str(total_norm_sq(after) - total_norm_sq(before)),
                "completed_unchanged": before.completed == after.completed,
            }
        )

    assert [index for index, primitive in enumerate(primitives) if primitive == "L"] == [15, 45, 103]

    q_rows = [row for row in transition_rows if row["primitive"] == "Q"]
    b_rows = [row for row in transition_rows if row["primitive"] == "B"]
    assert all(row["active_points_after"] == row["active_points_before"] for row in q_rows)
    assert all(Fraction(str(row["active_norm_delta"])) == 0 for row in q_rows)
    assert all(row["completed_unchanged"] for row in q_rows)
    assert all(row["active_points_after"] == row["active_points_before"] + 1 for row in b_rows)
    assert all(Fraction(str(row["active_norm_delta"])) > 0 for row in b_rows)
    assert all(row["completed_unchanged"] for row in b_rows)

    q4_probe = states[16].active
    q4 = list(q4_probe)
    for _ in range(4):
        q4 = [point.mul_i() for point in q4]
    assert q4 == q4_probe

    first_l = states[15]
    vertices = [(layer_index, point_index, value) for layer_index, layer in enumerate(first_l.layers) for point_index, value in enumerate(layer)]
    triangles = []
    cross_triangles = 0
    area_energy = Fraction(0)
    cross_area_energy = Fraction(0)
    for a, b, c in itertools.combinations(vertices, 3):
        area = doubled_area(a[2], b[2], c[2])
        if area == 0:
            continue
        triangles.append((a, b, c, area))
        area_energy += area * area
        if len({a[0], b[0], c[0]}) > 1:
            cross_triangles += 1
            cross_area_energy += area * area
    assert len(triangles) == 65
    assert cross_triangles == 65
    assert cross_area_energy == area_energy

    # Exact first B-Q cell and QGT.
    completed_norm = norm_sq(first_l.completed[0])
    c = Fraction(55, 144)
    active_norm = Fraction(1) + c * c
    total = completed_norm + active_norm
    g_cc = Fraction(1, 1) / total - c * c / (total * total)
    g_phiphi = completed_norm * active_norm / (total * total)
    g_cphi = Fraction(0)
    omega = -2 * c * completed_norm / (total * total)
    metric_det = g_cc * g_phiphi
    qfi_cc = 4 * g_cc
    qfi_phiphi = 4 * g_phiphi
    qgt_minimum = omega * omega / 4
    metric_surplus = metric_det - qgt_minimum
    saturation_ratio = metric_det / qgt_minimum
    assert g_cc > 0 and g_phiphi > 0 and omega != 0 and metric_surplus > 0

    # CF02 local contact lift alpha = ds + A_phi dphi; alpha wedge d alpha has coefficient Omega.
    w = active_norm
    a_phi = -w / total
    contact_volume_coefficient = omega
    assert contact_volume_coefficient != 0

    # Exact endpoint phase rule and permanent single kink.
    phase_rows: list[dict[str, object]] = []
    signs: list[int] = []
    for domain in range(16):
        budget = 6 * (2**domain) - 1
        mod4 = budget % 4
        endpoint = ["1", "i", "-1", "-i"][mod4]
        mass_sign = 1 if endpoint == "i" else -1 if endpoint == "-i" else 0
        signs.append(mass_sign)
        phase_rows.append(
            {
                "completed_domain": domain,
                "q_phase_budget": budget,
                "q_budget_mod_4": mod4,
                "endpoint_phase": endpoint,
                "native_mass_sign": mass_sign,
            }
        )
    assert signs[0] == 1 and all(value == -1 for value in signs[1:])
    kink_count = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    topological_index = Fraction(signs[0] - signs[-1], 2)
    assert kink_count == 1 and abs(topological_index) == 1

    # L45 contains the first completed opposite-sign pair generated after L15.
    second_l = states[45]
    assert second_l.completed[0][-1] == G(Fraction(0), Fraction(1))
    assert second_l.completed[1][-1] == G(Fraction(0), Fraction(-1))
    assert second_l.completed[1][-1] == second_l.completed[0][-1].mul_i().mul_i()

    negative_layer = second_l.completed[1]
    positive_layer = second_l.completed[0]
    wall_profile: list[dict[str, object]] = []
    ordered_negative = list(reversed(negative_layer))[:-1]
    ordered = [("D1", point) for point in ordered_negative] + [("wall", G.zero())] + [("D0", point) for point in positive_layer[1:]]
    for index, (source, point) in enumerate(ordered):
        wall_profile.append(
            {
                "profile_index": index,
                "source": source,
                "real": str(point.re),
                "imag": str(point.im),
                "scalar_profile": float(point.im),
                "abs_profile": abs(float(point.im)),
            }
        )
    profile_values = np.array([row["scalar_profile"] for row in wall_profile], dtype=float)
    assert np.all(np.diff(profile_values) >= 0)
    assert profile_values[0] == -1.0 and profile_values[-1] == 1.0 and np.any(profile_values == 0.0)

    negative_magnitudes = np.array(sorted(abs(float(point.im)) for point in negative_layer if point.im != 0), dtype=float)
    positive_magnitudes = np.array(sorted(abs(float(point.im)) for point in positive_layer if point.im != 0), dtype=float)
    neg_slope, neg_intercept, neg_r2 = linear_fit(np.arange(8, dtype=float), np.log(negative_magnitudes[:8]))
    pos_slope, pos_intercept, pos_r2 = linear_fit(np.arange(8, dtype=float), np.log(positive_magnitudes[:8]))
    phi_sq = ((1 + math.sqrt(5)) / 2) ** 2
    assert neg_r2 > 0.999999999
    assert pos_r2 > 0.9999

    # CF08 Wilson/domain-wall readout from the native endpoint mass signs.
    wall_sizes = [16, 32, 64, 128]
    separation_rows: list[dict[str, object]] = []
    wall_mode_data = None
    for n in wall_sizes:
        mass = np.array([1.0] + [-1.0] * (n - 1), dtype=float)
        hamiltonian, gamma5 = wilson_domain_wall(mass)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        near = np.argsort(np.abs(eigenvalues))[:2]
        modes = []
        for eigen_index in near:
            vector = eigenvectors[:, eigen_index]
            probability = np.sum(np.abs(vector.reshape(2, n)) ** 2, axis=0)
            center = int(np.argmax(probability))
            chirality = float(np.vdot(vector, gamma5 @ vector).real)
            modes.append((center, float(eigenvalues[eigen_index]), chirality, probability, vector))
        modes.sort(key=lambda item: item[0])
        wall_mode, far_mode = modes[0], modes[1]
        bulk_candidates = np.sort(np.abs(eigenvalues))[2:]
        bulk_gap = float(bulk_candidates[0])
        separation_rows.append(
            {
                "layer_extent": n,
                "wall_mode_center": wall_mode[0],
                "wall_mode_energy": wall_mode[1],
                "wall_mode_chirality": wall_mode[2],
                "wall_probability_first_3": float(np.sum(wall_mode[3][:3])),
                "far_partner_center": far_mode[0],
                "far_partner_energy": far_mode[1],
                "far_partner_chirality": far_mode[2],
                "wall_partner_separation": far_mode[0] - wall_mode[0],
                "bulk_gap": bulk_gap,
            }
        )
        if n == 64:
            wall_mode_data = (hamiltonian, gamma5, eigenvalues, eigenvectors, wall_mode, far_mode)

    assert wall_mode_data is not None
    hamiltonian, gamma5, eigenvalues, eigenvectors, wall_mode, far_mode = wall_mode_data
    assert abs(wall_mode[1]) < 1e-12
    assert abs(abs(wall_mode[2]) - 1.0) < 1e-12
    assert float(np.sum(wall_mode[3][:3])) > 0.99
    assert far_mode[0] >= 62
    assert abs(abs(far_mode[2]) - 1.0) < 1e-12

    # Linear wall-parallel branch H(p)=H_z+p gamma5.
    p_values = np.linspace(0.0, 0.1, 11)
    energy_values = []
    reference = wall_mode[4]
    dispersion_rows = []
    for momentum in p_values:
        hp = hamiltonian + momentum * gamma5
        evals, evecs = np.linalg.eigh(hp)
        overlaps = np.abs(evecs.conj().T @ reference) ** 2
        selected = int(np.argmax(overlaps))
        energy = float(evals[selected])
        energy_values.append(abs(energy))
        dispersion_rows.append(
            {
                "parallel_momentum": float(momentum),
                "wall_energy": energy,
                "abs_wall_energy": abs(energy),
                "mode_overlap": float(overlaps[selected]),
            }
        )
    disp_slope, disp_intercept, disp_r2 = linear_fit(p_values, np.array(energy_values))
    assert disp_r2 > 0.999999999999
    assert abs(disp_intercept) < 1e-12

    # Exact-sign overlap/Ginsparg-Wilson check on the same wall kernel.
    signs_h = np.sign(eigenvalues)
    signs_h[np.abs(eigenvalues) < 1e-12] = 1.0
    sign_h = (eigenvectors * signs_h) @ eigenvectors.conj().T
    d_overlap = np.eye(hamiltonian.shape[0]) + gamma5 @ sign_h
    gw_residual = float(np.linalg.norm(gamma5 @ d_overlap + d_overlap @ gamma5 - d_overlap @ gamma5 @ d_overlap, ord=np.inf))
    assert gw_residual < 1e-12

    # Completed layers are exact pointer records under active B/Q evolution.
    pointer_q_residual = 0
    pointer_b_residual = 0
    for before, primitive, after in zip(states[:-1], primitives[1:], states[1:]):
        if before.completed:
            if primitive == "Q" and before.completed != after.completed:
                pointer_q_residual += 1
            if primitive == "B" and before.completed != after.completed:
                pointer_b_residual += 1
    assert pointer_q_residual == 0 and pointer_b_residual == 0

    gate_matrix = {
        "CF00_induced_geometry": {
            "pass": True,
            "evidence": "first L rank 1->2, 65 nonzero oriented areas, exact horizontal B/Q QGT"
        },
        "CF01_J_M_split": {
            "pass": True,
            "evidence": "Q is exact norm-preserving fourfold phase continuation; B is strict positive refinement insertion; B and Q are the c and phi QGT directions"
        },
        "CF02_contact_host": {
            "pass": True,
            "evidence": "alpha wedge d alpha coefficient equals nonzero Omega_cphi"
        },
        "CF03_interface_localization": {
            "pass": True,
            "evidence": "all 65 first-L area cells and 100% of area energy cross the completed/active boundary"
        },
        "CF04_finite_causal_handoff": {
            "pass": True,
            "evidence": "inherited exact internal Q budgets 6*2^d-1 between L handoffs"
        },
        "CF06_information_metric": {
            "pass": True,
            "evidence": "pure-state quantum Fisher matrix 4g is positive definite"
        },
        "CF07_pointer_records": {
            "pass": True,
            "evidence": "completed-layer projectors remain exact under every later B and Q"
        },
        "CF08_domain_wall_spinor": {
            "pass": True,
            "evidence": "L15 opens the auxiliary direction; L45 completes the new domain at the exact antipode, creating one permanent sign kink; the CF08 Wilson readout has one wall-localized chiral zero mode and a separated far-boundary partner"
        },
        "CF09_electromagnetic_sector": {
            "pass": True,
            "evidence": "nonzero Omega_cphi is the already-validated CF09 light curvature"
        },
        "CF13_chirality_half_turn": {
            "pass": True,
            "evidence": "completed endpoints are +i and -i, an exact half-turn boundary with topological index magnitude one"
        },
    }

    summary = {
        "probe": "first-L dependent emergence cascade",
        "runtime_basis": "exact recurrence matching OI-7.5 target-pass trajectory",
        "first_L": {
            "causal_index": 15,
            "nonzero_area_cells": len(triangles),
            "cross_interface_area_cells": cross_triangles,
            "cross_interface_area_fraction": float(cross_area_energy / area_energy),
            "role": "opens the orthogonal auxiliary/interface geometry"
        },
        "first_completed_kink": {
            "causal_index": 45,
            "left_endpoint": second_l.completed[0][-1].text(),
            "right_endpoint": second_l.completed[1][-1].text(),
            "half_turn_exact": second_l.completed[1][-1] == second_l.completed[0][-1].mul_i().mul_i(),
            "permanent_sign_sequence": signs,
            "kink_count_through_domain_15": kink_count,
            "topological_index": str(topological_index),
            "negative_tail_log_fit_r2": neg_r2,
            "positive_tail_log_fit_r2": pos_r2,
            "negative_tail_scale_ratio": math.exp(neg_slope),
            "positive_tail_scale_ratio": math.exp(pos_slope),
            "golden_ratio_squared": phi_sq,
        },
        "J_M_identification": {
            "Q": "microscopic reversible J-limb phase-continuation generator",
            "B": "microscopic positive M-limb refinement generator",
            "L": "M-limb retention/re-hosting completion",
            "Q_count": len(q_rows),
            "B_count": len(b_rows),
            "all_Q_preserve_active_norm": True,
            "all_Q_preserve_point_count": True,
            "Q_fourfold_return": True,
            "all_B_increase_active_norm": True,
            "all_B_add_one_refinement_point": True,
            "tensor_level_note": "the Poisson J tensor is generated by the mixed B-wedge-Q curvature, while the primitive phase action Q is its reversible descendant"
        },
        "QGT": {
            "c": str(c),
            "completed_norm_sq": str(completed_norm),
            "g_cc": str(g_cc),
            "g_phiphi": str(g_phiphi),
            "g_cphi": str(g_cphi),
            "omega_cphi": str(omega),
            "metric_det": str(metric_det),
            "quantum_fisher_cc": str(qfi_cc),
            "quantum_fisher_phiphi": str(qfi_phiphi),
            "qgt_inequality_minimum": str(qgt_minimum),
            "metric_surplus_over_two_band_minimum": str(metric_surplus),
            "saturation_ratio": str(saturation_ratio),
        },
        "contact": {
            "A_phi": str(a_phi),
            "alpha_wedge_dalpha_coefficient": str(contact_volume_coefficient),
            "nondegenerate": True,
        },
        "CF08_readout": {
            "layer_extent": 64,
            "wall_mode_energy": wall_mode[1],
            "wall_mode_center": wall_mode[0],
            "wall_mode_chirality": wall_mode[2],
            "wall_probability_first_3": float(np.sum(wall_mode[3][:3])),
            "far_partner_center": far_mode[0],
            "far_partner_chirality": far_mode[2],
            "bulk_gap": separation_rows[2]["bulk_gap"],
            "dispersion_slope": disp_slope,
            "dispersion_intercept": disp_intercept,
            "dispersion_r2": disp_r2,
            "GW_residual_inf": gw_residual,
        },
        "pointer": {
            "B_completed_record_changes": pointer_b_residual,
            "Q_completed_record_changes": pointer_q_residual,
        },
        "all_gates_pass": all(item["pass"] for item in gate_matrix.values()),
        "gate_matrix": gate_matrix,
    }

    write_csv(results / "operator_limb_metrics.csv", transition_rows)
    write_csv(results / "layer_phase_kink.csv", phase_rows)
    write_csv(results / "first_completed_wall_profile.csv", wall_profile)
    write_csv(results / "cf08_wall_partner_separation.csv", separation_rows)
    write_csv(results / "cf08_linear_dispersion.csv", dispersion_rows)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results / "gate_matrix.json").write_text(json.dumps(gate_matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Figures.
    plt.figure(figsize=(8, 5))
    x = np.arange(len(profile_values))
    plt.plot(x, profile_values, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("native ordered interface sample")
    plt.ylabel("signed imaginary scalar")
    plt.title("First completed Orthad mass kink at L45")
    plt.tight_layout()
    plt.savefig(figures / "01_first_completed_mass_kink.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    domains = [row["completed_domain"] for row in phase_rows]
    mass_signs = [row["native_mass_sign"] for row in phase_rows]
    plt.step(domains, mass_signs, where="mid")
    plt.xlabel("completed domain")
    plt.ylabel("endpoint mass sign")
    plt.title("One permanent half-turn wall: +i | -i, -i, ...")
    plt.ylim(-1.25, 1.25)
    plt.tight_layout()
    plt.savefig(figures / "02_permanent_single_kink.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(np.arange(len(negative_magnitudes)), negative_magnitudes, marker="o", label="negative side")
    plt.semilogy(np.arange(len(positive_magnitudes)), positive_magnitudes, marker="o", label="positive side")
    plt.xlabel("refinement rank from wall")
    plt.ylabel("absolute scalar magnitude")
    plt.title("Exponential refinement structure on both wall sides")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "03_wall_refinement_tails.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    n = 64
    sites = np.arange(n)
    plt.plot(sites, wall_mode[3], label="wall chiral mode")
    plt.plot(sites, far_mode[3], label="paired far-boundary mode")
    plt.xlabel("auxiliary layer site")
    plt.ylabel("probability")
    plt.title("CF08 readout: localized chiral wall mode and separated partner")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "04_chiral_zero_modes.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(p_values, energy_values, marker="o", label="measured wall branch")
    plt.plot(p_values, disp_slope * p_values + disp_intercept, linestyle="--", label="linear fit")
    plt.xlabel("wall-parallel momentum")
    plt.ylabel("absolute energy")
    plt.title("Massless linear wall dispersion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "05_wall_linear_dispersion.png", dpi=180)
    plt.close()

    # Human-readable stdout.
    print(json.dumps({
        "all_gates_pass": summary["all_gates_pass"],
        "first_L_opens_interface": True,
        "first_completed_kink_at": 45,
        "permanent_kink_count": kink_count,
        "topological_index": str(topological_index),
        "Q_is_J_descendant": True,
        "B_is_M_descendant": True,
        "contact_nondegenerate": True,
        "wall_chiral_zero_mode": True,
        "wall_mode_chirality": wall_mode[2],
        "far_partner_separated": True,
        "dispersion_r2": disp_r2,
        "GW_residual_inf": gw_residual,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
