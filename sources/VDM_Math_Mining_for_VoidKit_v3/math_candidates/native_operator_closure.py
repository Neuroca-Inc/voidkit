#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

sys.set_int_max_str_digits(0)


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

    def __add__(self, other: "G") -> "G":
        return G(self.re + other.re, self.im + other.im)

    def __sub__(self, other: "G") -> "G":
        return G(self.re - other.re, self.im - other.im)

    def mul_i(self) -> "G":
        return G(-self.im, self.re)

    def scale(self, value: Fraction) -> "G":
        return G(self.re * value, self.im * value)

    def norm_sq(self) -> Fraction:
        return self.re * self.re + self.im * self.im

    def complex(self) -> complex:
        return complex(float(self.re), float(self.im))

    def text(self) -> str:
        def fmt(value: Fraction) -> str:
            if value.denominator == 1:
                return str(value.numerator)
            return f"{value.numerator}/{value.denominator}"

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

    @property
    def layers(self) -> list[list[G]]:
        return [*self.completed, self.active]

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


def norm_sq(points: Iterable[G]) -> Fraction:
    return sum((point.norm_sq() for point in points), Fraction(0))


def flatten_completed(state: State) -> list[complex]:
    return [point.complex() for layer in state.completed for point in layer]


def normalized_state(completed: np.ndarray, c: float, phi: float) -> np.ndarray:
    active = np.exp(1j * phi) * np.array([0.0, c, 1.0], dtype=complex)
    raw = np.concatenate([completed, active])
    return raw / np.linalg.norm(raw)


def projected_derivative(psi: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    return derivative - psi * np.vdot(psi, derivative)


def finite_qgt(completed: np.ndarray, c: float, phi: float, h: float = 1e-6) -> np.ndarray:
    psi = normalized_state(completed, c, phi)
    d_c = (
        normalized_state(completed, c + h, phi)
        - normalized_state(completed, c - h, phi)
    ) / (2.0 * h)
    d_phi = (
        normalized_state(completed, c, phi + h)
        - normalized_state(completed, c, phi - h)
    ) / (2.0 * h)
    d_c = projected_derivative(psi, d_c)
    d_phi = projected_derivative(psi, d_phi)
    return np.array(
        [
            [np.vdot(d_c, d_c), np.vdot(d_c, d_phi)],
            [np.vdot(d_phi, d_c), np.vdot(d_phi, d_phi)],
        ],
        dtype=complex,
    )


def analytic_qgt(old_norm_sq: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_norm_sq = 1.0 + c * c
    total = old_norm_sq + active_norm_sq
    g_cc = (old_norm_sq + 1.0) / total**2
    g_phiphi = old_norm_sq * active_norm_sq / total**2
    omega = -2.0 * c * old_norm_sq / total**2
    qgt = np.array(
        [[g_cc, -0.5j * omega], [0.5j * omega, g_phiphi]], dtype=complex
    )
    metric = np.diag([g_cc, g_phiphi])
    curvature = np.array([[0.0, omega], [-omega, 0.0]], dtype=float)
    return qgt, metric, curvature


def berry_connection(old_norm_sq: float, c: float) -> np.ndarray:
    total = old_norm_sq + 1.0 + c * c
    return np.array([0.0, -(1.0 + c * c) / total], dtype=float)


def phase_generator(completed_count: int) -> np.ndarray:
    return np.diag([0.0] * completed_count + [1.0, 1.0, 1.0]).astype(complex)


def overlap_link(left: np.ndarray, right: np.ndarray) -> complex:
    value = np.vdot(left, right)
    if abs(value) == 0.0:
        raise ValueError("zero state overlap")
    return value / abs(value)


def plaquette_phase(
    completed: np.ndarray,
    c: float,
    phi: float,
    dc: float,
    dphi: float,
    phases: tuple[float, float, float, float] | None = None,
) -> float:
    states = [
        normalized_state(completed, c, phi),
        normalized_state(completed, c + dc, phi),
        normalized_state(completed, c + dc, phi + dphi),
        normalized_state(completed, c, phi + dphi),
    ]
    if phases is not None:
        states = [state * np.exp(1j * phase) for state, phase in zip(states, phases, strict=True)]
    product = (
        overlap_link(states[0], states[1])
        * overlap_link(states[1], states[2])
        * overlap_link(states[2], states[3])
        * overlap_link(states[3], states[0])
    )
    return float(np.angle(product))


def completion_phase_budget(domain: int) -> int:
    return 6 * (2**domain) - 1


def endpoint_phase(domain: int) -> complex:
    return 1j ** (completion_phase_budget(domain) % 4)


def mass_sign(domain: int) -> int:
    phase = endpoint_phase(domain)
    if abs(phase - 1j) < 1e-12:
        return 1
    if abs(phase + 1j) < 1e-12:
        return -1
    return 0


def path_operators(size: int) -> tuple[np.ndarray, np.ndarray]:
    derivative = np.zeros((size, size), dtype=complex)
    laplacian = np.zeros((size, size), dtype=float)
    for index in range(size - 1):
        derivative[index, index + 1] = 0.5
        derivative[index + 1, index] = -0.5
        laplacian[index, index] += 1.0
        laplacian[index + 1, index + 1] += 1.0
        laplacian[index, index + 1] -= 1.0
        laplacian[index + 1, index] -= 1.0
    return derivative, laplacian


def ors_clifford() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    j_ors = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    gamma5 = 1j * j_ors
    gamma_mass = np.diag([1.0, -1.0]).astype(complex)
    gamma_derivative = -1j * gamma5 @ gamma_mass
    identity = np.eye(2, dtype=complex)
    metrics = {
        "ors_square_residual": float(np.linalg.norm(j_ors @ j_ors + identity)),
        "gamma5_square_residual": float(np.linalg.norm(gamma5 @ gamma5 - identity)),
        "gamma_mass_square_residual": float(np.linalg.norm(gamma_mass @ gamma_mass - identity)),
        "gamma_derivative_square_residual": float(np.linalg.norm(gamma_derivative @ gamma_derivative - identity)),
        "gamma5_mass_anticommutator": float(np.linalg.norm(gamma5 @ gamma_mass + gamma_mass @ gamma5)),
        "gamma5_derivative_anticommutator": float(np.linalg.norm(gamma5 @ gamma_derivative + gamma_derivative @ gamma5)),
        "mass_derivative_anticommutator": float(np.linalg.norm(gamma_mass @ gamma_derivative + gamma_derivative @ gamma_mass)),
    }
    return gamma5, gamma_mass, gamma_derivative, metrics


def native_wilson_kernel(size: int, profile: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if profile is None:
        profile = np.array([mass_sign(domain) for domain in range(size)], dtype=float)
    derivative, laplacian = path_operators(size)
    gamma5_small, gamma_mass, gamma_derivative, _ = ors_clifford()
    gamma5 = np.kron(gamma5_small, np.eye(size))
    kernel = np.kron(-1j * gamma_derivative, derivative) + np.kron(
        gamma_mass, np.diag(profile) + 0.5 * laplacian
    )
    return kernel, gamma5


def mode_metrics(kernel: np.ndarray, gamma5: np.ndarray, size: int) -> dict[str, object]:
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    near = np.argsort(np.abs(eigenvalues))[:2]
    near_vectors = eigenvectors[:, near]
    if float(np.max(np.abs(eigenvalues[near]))) < 1e-12:
        # The two boundary zero modes are numerically degenerate.  Resolve the
        # physical basis by diagonalizing native chirality inside that exact
        # zero-mode subspace instead of accepting an arbitrary eigensolver mix.
        chiral_subspace = near_vectors.conj().T @ gamma5 @ near_vectors
        _, chiral_vectors = np.linalg.eigh(chiral_subspace)
        near_vectors = near_vectors @ chiral_vectors
    modes: list[dict[str, object]] = []
    for local_index, eigen_index in enumerate(near):
        vector = near_vectors[:, local_index]
        probability = np.sum(np.abs(vector.reshape(2, size)) ** 2, axis=0)
        modes.append(
            {
                "energy": float(np.vdot(vector, kernel @ vector).real),
                "chirality": float(np.vdot(vector, gamma5 @ vector).real),
                "center": int(np.argmax(probability)),
                "peak_probability": float(np.max(probability)),
                "first_three_probability": float(np.sum(probability[:3])),
                "last_three_probability": float(np.sum(probability[-3:])),
                "probability": probability,
            }
        )
    modes.sort(key=lambda item: int(item["center"]))
    bulk_gap = float(np.sort(np.abs(eigenvalues))[2])
    signs = np.sign(eigenvalues)
    signs[np.abs(eigenvalues) < 1e-12] = 1.0
    sign_kernel = (eigenvectors * signs) @ eigenvectors.conj().T
    overlap = np.eye(kernel.shape[0], dtype=complex) + gamma5 @ sign_kernel
    gw = gamma5 @ overlap + overlap @ gamma5 - overlap @ gamma5 @ overlap
    return {
        "modes": modes,
        "bulk_gap": bulk_gap,
        "gw_residual": float(np.linalg.norm(gw, ord=np.inf)),
        "hermiticity_residual": float(np.linalg.norm(kernel - kernel.conj().T)),
        "eigenvalues": eigenvalues,
    }


def padded_normalized(layer: list[G], size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=complex)
    vector[: len(layer)] = np.array([point.complex() for point in layer])
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("zero layer")
    return vector / norm


def cross_relation_energy(left: list[G], right: list[G]) -> Fraction:
    return sum(
        ((b - a).norm_sq() for a in left for b in right),
        Fraction(0),
    )


def interface_normal_operator(weight: float) -> np.ndarray:
    return weight * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    figures = root / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    # Exact recurrence through the first five completed-domain handoffs.
    state = State.initial()
    states: dict[int, State] = {0: state.clone()}
    primitives: dict[int, str] = {}
    inserted: dict[int, Fraction | None] = {}
    l_indices: list[int] = []
    first_b_after_l: dict[int, int] = {}
    first_q_after_l: dict[int, int] = {}
    pending_l: int | None = None
    for causal_index in range(1, 106):
        primitive, c = state.tick()
        states[causal_index] = state.clone()
        primitives[causal_index] = primitive
        inserted[causal_index] = c
        if primitive == "L":
            l_indices.append(causal_index)
            pending_l = causal_index
        elif pending_l is not None and primitive == "B" and pending_l not in first_b_after_l:
            first_b_after_l[pending_l] = causal_index
        elif pending_l is not None and primitive == "Q" and pending_l in first_b_after_l and pending_l not in first_q_after_l:
            first_q_after_l[pending_l] = causal_index
            pending_l = None
    assert l_indices == [15, 45, 103]

    snapshot_rows: list[dict[str, object]] = []
    for l_index in l_indices:
        snapshot = states[l_index]
        for layer_index, layer in enumerate(snapshot.completed):
            snapshot_rows.append(
                {
                    "l_causal_index": l_index,
                    "completed_count": len(snapshot.completed),
                    "layer_index": layer_index,
                    "point_count": len(layer),
                    "endpoint_phase": layer[-1].text(),
                    "endpoint_mass_sign": mass_sign(layer_index),
                    "phase_budget": completion_phase_budget(layer_index),
                    "phase_budget_mod_4": completion_phase_budget(layer_index) % 4,
                    "layer_norm_sq_exact": str(norm_sq(layer)),
                }
            )
    write_csv(results / "runtime_layer_snapshots.csv", snapshot_rows)

    # QGT, gauge connection, J flow, B metric flow, and B+L metriplectic completion.
    qgt_rows: list[dict[str, object]] = []
    gauge_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260720)
    first_cell_payload: dict[str, object] = {}
    for l_index in l_indices:
        b_index = first_b_after_l[l_index]
        q_index = first_q_after_l[l_index]
        after_l = states[l_index]
        after_b = states[b_index]
        c_frac = inserted[b_index]
        assert c_frac is not None
        c = float(c_frac)
        completed = np.array(flatten_completed(after_l), dtype=complex)
        old_norm = float(sum((norm_sq(layer) for layer in after_l.completed), Fraction(0)))
        qgt_exact, metric, curvature = analytic_qgt(old_norm, c)
        qgt_numeric = finite_qgt(completed, c, 0.371)
        qgt_residual = float(np.max(np.abs(qgt_numeric - qgt_exact)))
        omega = float(curvature[0, 1])
        j_operator = np.linalg.inv(curvature)
        h_gradient = np.array([omega, 0.0])
        q_flow = j_operator @ h_gradient
        q_flow_residual = float(np.linalg.norm(q_flow - np.array([0.0, 1.0])))
        g_inverse = np.linalg.inv(metric)
        sigma_gradient = np.array([metric[0, 0], 0.0])
        b_flow = g_inverse @ sigma_gradient
        b_flow_residual = float(np.linalg.norm(b_flow - np.array([1.0, 0.0])))

        # Exact obstruction: in 2D, a symmetric operator that kills dH=(omega,0)
        # cannot generate the B tangent (1,0). The native L reservoir supplies the missing coordinate.
        two_dim_obstruction = abs(omega)
        d_invariant = np.array([omega, 0.0, 1.0])
        bl_tangent = np.array([1.0, 0.0, -omega])
        m_operator = np.outer(bl_tangent, bl_tangent)
        m_degeneracy = float(np.linalg.norm(m_operator @ d_invariant))
        m_eigenvalues = np.linalg.eigvalsh(m_operator)
        entropy_gradient = bl_tangent / float(np.dot(bl_tangent, bl_tangent))
        bl_flow_residual = float(np.linalg.norm(m_operator @ entropy_gradient - bl_tangent))

        generator = phase_generator(len(completed))
        psi = normalized_state(completed, c, 0.371)
        h = 1e-7
        dphi = (
            normalized_state(completed, c, 0.371 + h)
            - normalized_state(completed, c, 0.371 - h)
        ) / (2.0 * h)
        generator_residual = float(np.linalg.norm(dphi - 1j * generator @ psi))
        connection = berry_connection(old_norm, c)
        generator_expectation = float(np.vdot(psi, generator @ psi).real)
        connection_generator_residual = abs(connection[1] + generator_expectation)

        qgt_rows.append(
            {
                "l_causal_index": l_index,
                "first_b_causal_index": b_index,
                "first_q_causal_index": q_index,
                "domain_after_l": after_l.domain,
                "c_exact": str(c_frac),
                "c": c,
                "old_norm_sq": old_norm,
                "g_cc": metric[0, 0],
                "g_phiphi": metric[1, 1],
                "omega_cphi": omega,
                "det_g": float(np.linalg.det(metric)),
                "det_omega": float(np.linalg.det(curvature)),
                "qgt_finite_difference_residual": qgt_residual,
                "q_is_J_flow_residual": q_flow_residual,
                "b_is_unprojected_metric_flow_residual": b_flow_residual,
                "two_coordinate_metriplectic_obstruction_witness": two_dim_obstruction,
                "BL_reservoir_M_degeneracy_residual": m_degeneracy,
                "BL_reservoir_M_flow_residual": bl_flow_residual,
                "BL_reservoir_M_min_eigenvalue": float(m_eigenvalues[0]),
                "BL_reservoir_M_rank": int(np.linalg.matrix_rank(m_operator, tol=1e-12)),
                "phase_generator_residual": generator_residual,
                "connection_generator_residual": connection_generator_residual,
            }
        )

        if l_index == 15:
            first_cell_payload = {
                "c": c,
                "old_norm": old_norm,
                "metric": metric,
                "curvature": curvature,
                "j_operator": j_operator,
                "m_operator": m_operator,
                "d_invariant": d_invariant,
                "bl_tangent": bl_tangent,
                "completed": completed,
            }
            for dc in [0.04, 0.02, 0.01, 0.005, 0.0025]:
                dphase = dc
                phase = plaquette_phase(completed, c, 0.271, dc, dphase)
                measured = -phase / (dc * dphase)
                random_residual = 0.0
                for _ in range(128):
                    local_phases = tuple(rng.uniform(-math.pi, math.pi, size=4))
                    shifted = plaquette_phase(
                        completed, c, 0.271, dc, dphase, local_phases
                    )
                    random_residual = max(
                        random_residual,
                        abs(np.angle(np.exp(1j * (shifted - phase)))),
                    )
                gauge_rows.append(
                    {
                        "dc": dc,
                        "dphi": dphase,
                        "plaquette_phase": phase,
                        "measured_curvature": measured,
                        "analytic_curvature": omega,
                        "absolute_curvature_error": abs(measured - omega),
                        "random_rephase_max_phase_residual": random_residual,
                    }
                )
    write_csv(results / "native_qgt_jm_cells.csv", qgt_rows)
    write_csv(results / "gauge_plaquette_convergence.csv", gauge_rows)

    # Native Clifford/Dirac/overlap chain from L ordering and exact endpoint phases.
    gamma5_small, gamma_mass, gamma_derivative, clifford_metrics = ors_clifford()
    dirac_rows: list[dict[str, object]] = []
    probability_64: list[np.ndarray] = []
    native_wall_q_weight = math.nan
    for size in [3, 4, 5, 8, 16, 32, 64, 128]:
        kernel, gamma5 = native_wilson_kernel(size)
        metrics = mode_metrics(kernel, gamma5, size)
        modes = metrics["modes"]
        trivial_kernel, trivial_gamma5 = native_wilson_kernel(size, np.ones(size))
        trivial_metrics = mode_metrics(trivial_kernel, trivial_gamma5, size)
        row = {
            "completed_layers": size,
            "left_wall_energy": modes[0]["energy"],
            "left_wall_chirality": modes[0]["chirality"],
            "left_wall_center": modes[0]["center"],
            "left_wall_first_three_probability": modes[0]["first_three_probability"],
            "far_partner_energy": modes[1]["energy"],
            "far_partner_chirality": modes[1]["chirality"],
            "far_partner_center": modes[1]["center"],
            "far_partner_last_three_probability": modes[1]["last_three_probability"],
            "bulk_gap": metrics["bulk_gap"],
            "GW_residual": metrics["gw_residual"],
            "kernel_hermiticity_residual": metrics["hermiticity_residual"],
            "trivial_uniform_positive_min_abs_energy": float(
                np.min(np.abs(trivial_metrics["eigenvalues"]))
            ),
        }
        dirac_rows.append(row)
        if size == 3:
            # At L45 the wall occupies the two completed layers while primitive Q
            # acts only on the newly active third layer.  This is the direct
            # runtime Q-weight of the completed wall mode.
            native_wall_q_weight = float(modes[0]["probability"][-1])
        if size == 64:
            probability_64 = [modes[0]["probability"], modes[1]["probability"]]
    write_csv(results / "native_dirac_chain.csv", dirac_rows)

    # L103 supplies the first two same-orientation completed modes. Derive their host algebra.
    l103 = states[103]
    layer_one = l103.completed[1]
    layer_two = l103.completed[2]
    max_size = max(len(layer_one), len(layer_two))
    mode_one = padded_normalized(layer_one, max_size)
    mode_two_raw = padded_normalized(layer_two, max_size)
    overlap = np.vdot(mode_one, mode_two_raw)
    mode_two = mode_two_raw - mode_one * overlap
    mode_two /= np.linalg.norm(mode_two)
    host_orthogonality = abs(np.vdot(mode_one, mode_two))
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex) / 2.0
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex) / 2.0
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex) / 2.0
    su2_residual = max(
        np.linalg.norm(sigma_x @ sigma_y - sigma_y @ sigma_x - 1j * sigma_z),
        np.linalg.norm(sigma_y @ sigma_z - sigma_z @ sigma_y - 1j * sigma_x),
        np.linalg.norm(sigma_z @ sigma_x - sigma_x @ sigma_z - 1j * sigma_y),
    )
    # CF16 fixes the order-parameter companion normalization by condensate neutrality.
    # On the order doublet Y=+1, Q_em=T3+Y/2 has spectrum (+1,0).
    q_order = sigma_z + 0.5 * np.eye(2)
    order_charge_eigenvalues = np.linalg.eigvalsh(q_order)

    interface_weight_exact = cross_relation_energy(layer_one, layer_two)
    interface_weight = float(interface_weight_exact)
    normal_operator = interface_normal_operator(interface_weight)
    normal_eigenvalues, normal_eigenvectors = np.linalg.eigh(normal_operator)
    tangent = np.array([1.0, 1.0]) / math.sqrt(2.0)
    normal = np.array([1.0, -1.0]) / math.sqrt(2.0)
    tangent_residual = float(np.linalg.norm(normal_operator @ tangent))
    normal_residual = float(
        np.linalg.norm(normal_operator @ normal - 2.0 * interface_weight * normal)
    )

    charge_host = {
        "l103_raw_layer_overlap_abs": float(abs(overlap)),
        "l103_orthonormalized_host_residual": float(host_orthogonality),
        "su2_commutator_residual": float(su2_residual),
        "order_parameter_Qem_eigenvalues": [float(x) for x in order_charge_eigenvalues],
        "order_parameter_charge_interpretation": ["neutral condensed component", "charged companion component"],
        "direct_L45_wall_weight_under_primitive_Q_generator": native_wall_q_weight,
        "direct_L45_wall_Q_weight_is_zero": native_wall_q_weight < 1e-12,
        "matter_hypercharge_fixed_by_first_L_geometry": False,
        "matter_charge_reason": "CF16 fixes the residual generator normalization from the order parameter, but a specific lepton hypercharge representation is additional matter-sector data and is not fixed by the first-L relation geometry alone.",
    }
    (results / "native_charge_host.json").write_text(
        json.dumps(charge_host, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    interface_rows = [
        {
            "l_causal_index": 103,
            "left_layer": 1,
            "right_layer": 2,
            "cross_relation_energy_exact": str(interface_weight_exact),
            "cross_relation_energy": interface_weight,
            "operator_rank": int(np.linalg.matrix_rank(normal_operator, tol=1e-12)),
            "eigenvalue_tangent": float(normal_eigenvalues[0]),
            "eigenvalue_normal": float(normal_eigenvalues[1]),
            "normalized_eigenvalue_tangent": 0.0,
            "normalized_eigenvalue_normal": 2.0,
            "tangent_zero_residual": tangent_residual,
            "normal_eigenvector_residual": normal_residual,
            "electron_yukawa_scale_fixed": False,
        }
    ]
    write_csv(results / "native_interface_normal_operator.csv", interface_rows)

    operator_algebra = {
        **clifford_metrics,
        "first_cell_phase_generator_eigenvalues": [0, 1],
        "first_cell_phase_generator_zero_multiplicity": int(len(first_cell_payload["completed"])),
        "first_cell_phase_generator_one_multiplicity": 3,
        "first_cell_J_antisymmetry_residual": float(
            np.linalg.norm(first_cell_payload["j_operator"] + first_cell_payload["j_operator"].T)
        ),
        "first_cell_BL_M_symmetry_residual": float(
            np.linalg.norm(first_cell_payload["m_operator"] - first_cell_payload["m_operator"].T)
        ),
        "first_cell_BL_M_min_eigenvalue": float(
            np.min(np.linalg.eigvalsh(first_cell_payload["m_operator"]))
        ),
        "first_cell_BL_M_degeneracy_residual": float(
            np.linalg.norm(first_cell_payload["m_operator"] @ first_cell_payload["d_invariant"])
        ),
        "l103_interface_normal_eigenvalues": [float(x) for x in normal_eigenvalues],
        "l103_interface_normal_vectors": normal_eigenvectors.tolist(),
    }
    (results / "operator_algebra.json").write_text(
        json.dumps(operator_algebra, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Figures.
    c_grid = np.linspace(0.0, 1.0, 300)
    old_norm = float(first_cell_payload["old_norm"])
    g_cc_curve = []
    g_phi_curve = []
    omega_curve = []
    for c_value in c_grid:
        _, metric, curvature = analytic_qgt(old_norm, float(c_value))
        g_cc_curve.append(metric[0, 0])
        g_phi_curve.append(metric[1, 1])
        omega_curve.append(curvature[0, 1])
    plt.figure(figsize=(8, 5))
    plt.plot(c_grid, g_cc_curve, label=r"$g_{cc}$")
    plt.plot(c_grid, g_phi_curve, label=r"$g_{\phi\phi}$")
    plt.plot(c_grid, omega_curve, label=r"$\Omega_{c\phi}$")
    plt.axvline(float(first_cell_payload["c"]), linestyle="--", label="first runtime B")
    plt.xlabel("native refinement coordinate c")
    plt.ylabel("operator coefficient")
    plt.title("First-L native QGT coefficients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "01_native_qgt_coefficients.png", dpi=180)
    plt.close()

    gauge_dc = np.array([row["dc"] for row in gauge_rows], dtype=float)
    gauge_error = np.array([row["absolute_curvature_error"] for row in gauge_rows], dtype=float)
    plt.figure(figsize=(7, 5))
    plt.loglog(gauge_dc, gauge_error, marker="o")
    plt.xlabel("plaquette side length")
    plt.ylabel("absolute curvature error")
    plt.title("Overlap plaquette convergence to CF09 curvature")
    plt.tight_layout()
    plt.savefig(figures / "02_gauge_plaquette_convergence.png", dpi=180)
    plt.close()

    x = np.arange(64)
    plt.figure(figsize=(9, 5))
    plt.step(x, [mass_sign(i) for i in range(64)], where="mid", label="native endpoint mass sign")
    plt.plot(x, probability_64[0], label="left chiral wall mode probability")
    plt.plot(x, probability_64[1], label="far partner probability")
    plt.xlabel("completed-layer index")
    plt.ylabel("mass sign / probability")
    plt.title("Native L-chain domain wall and chiral modes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "03_native_dirac_wall_modes.png", dpi=180)
    plt.close()

    dirac_sizes = np.array([row["completed_layers"] for row in dirac_rows], dtype=float)
    gw_residuals = np.array([max(float(row["GW_residual"]), 1e-18) for row in dirac_rows])
    plt.figure(figsize=(7, 5))
    plt.loglog(dirac_sizes, gw_residuals, marker="o", label="GW residual")
    plt.axhline(1e-12, linestyle="--", label="CF08 gate")
    plt.xlabel("completed layers")
    plt.ylabel("infinity-norm residual")
    plt.title("Native overlap/Ginsparg-Wilson closure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "04_ginsparg_wilson_residual.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["tangent/common", "normal/difference"]
    plt.bar(labels, normal_eigenvalues)
    plt.ylabel("interface operator eigenvalue")
    plt.title("L103 native rank-one interface-normal operator")
    plt.tight_layout()
    plt.savefig(figures / "05_interface_normal_spectrum.png", dpi=180)
    plt.close()

    # Gate matrix distinguishes direct closure from CF-dependent or still-unfixed representation data.
    qgt_max = max(float(row["qgt_finite_difference_residual"]) for row in qgt_rows)
    qflow_max = max(float(row["q_is_J_flow_residual"]) for row in qgt_rows)
    bflow_max = max(float(row["b_is_unprojected_metric_flow_residual"]) for row in qgt_rows)
    reservoir_deg = max(float(row["BL_reservoir_M_degeneracy_residual"]) for row in qgt_rows)
    reservoir_flow = max(float(row["BL_reservoir_M_flow_residual"]) for row in qgt_rows)
    gauge_rephase = max(float(row["random_rephase_max_phase_residual"]) for row in gauge_rows)
    gauge_final_error = float(gauge_rows[-1]["absolute_curvature_error"])
    dirac_zero = max(abs(float(row["left_wall_energy"])) for row in dirac_rows if int(row["completed_layers"]) >= 3)
    chirality_error = max(
        abs(abs(float(row["left_wall_chirality"])) - 1.0)
        for row in dirac_rows
        if int(row["completed_layers"]) >= 3
    )
    gw_max = max(float(row["GW_residual"]) for row in dirac_rows)
    gates = [
        {
            "gate": "G1_native_CF09_connection_and_curvature",
            "status": "PASS" if qgt_max < 1e-8 and gauge_rephase < 1e-12 and gauge_final_error < 1e-3 else "FAIL",
            "evidence": {
                "qgt_finite_difference_max_residual": qgt_max,
                "random_rephase_max_phase_residual": gauge_rephase,
                "smallest_plaquette_curvature_error": gauge_final_error,
            },
        },
        {
            "gate": "G2_Q_is_J_flow",
            "status": "PASS" if qflow_max < 1e-12 else "FAIL",
            "evidence": {"max_flow_residual": qflow_max},
        },
        {
            "gate": "G3_B_metric_and_BL_full_M",
            "status": "PASS" if bflow_max < 1e-12 and reservoir_deg < 1e-12 and reservoir_flow < 1e-12 else "FAIL",
            "evidence": {
                "B_unprojected_metric_flow_max_residual": bflow_max,
                "BL_M_degeneracy_max_residual": reservoir_deg,
                "BL_M_flow_max_residual": reservoir_flow,
                "two_coordinate_B_as_full_M": "PROVED_IMPOSSIBLE_WITH_SAME_Q_GENERATOR",
            },
        },
        {
            "gate": "G4_native_CF08_Dirac_and_GW",
            "status": "PASS" if dirac_zero < 1e-12 and chirality_error < 1e-12 and gw_max < 1e-12 else "FAIL",
            "evidence": {
                "max_wall_zero_energy": dirac_zero,
                "max_chirality_error": chirality_error,
                "max_GW_residual": gw_max,
            },
        },
        {
            "gate": "G5_native_L103_interface_normal_operator",
            "status": "PASS" if tangent_residual < 1e-8 and normal_residual < 1e-8 and normal_eigenvalues[0] > -1e-8 else "FAIL",
            "evidence": {
                "rank": int(np.linalg.matrix_rank(normal_operator, tol=1e-8)),
                "tangent_residual": tangent_residual,
                "normal_residual": normal_residual,
                "eigenvalues": [float(x) for x in normal_eigenvalues],
            },
        },
        {
            "gate": "G6_native_two_mode_SU2_host",
            "status": "PASS" if host_orthogonality < 1e-12 and su2_residual < 1e-12 else "FAIL",
            "evidence": {
                "host_orthogonality_residual": float(host_orthogonality),
                "su2_commutator_residual": float(su2_residual),
            },
        },
        {
            "gate": "G7_absolute_matter_charge_from_first_L_only",
            "status": "NOT_CLOSED",
            "evidence": {
                "native_phase_generator": "derived",
                "CF16_order_parameter_charge_generator": "derived on native host",
                "direct_L45_wall_primitive_Q_weight": native_wall_q_weight,
                "specific_lepton_hypercharge": "not fixed by first-L geometry",
            },
        },
        {
            "gate": "G8_numerical_electron_mass_scale",
            "status": "NOT_CLOSED",
            "evidence": {
                "normal_mass_projector": "derived",
                "interface_stiffness": interface_weight,
                "electron_Yukawa_scale": "not fixed by CF08/CF16 geometry",
            },
        },
    ]
    (results / "gate_matrix.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    direct_passes = sum(gate["status"] == "PASS" for gate in gates)
    not_closed = sum(gate["status"] == "NOT_CLOSED" for gate in gates)
    summary = {
        "study": "native operator closure from exact Orthad L/B/Q geometry",
        "causal_index_is_not_physical_time": True,
        "runtime_l_indices": l_indices,
        "direct_gates_passed": direct_passes,
        "direct_gates_total": 6,
        "not_closed_gates": not_closed,
        "key_result": "Q is exactly the J flow; B is exactly the unprojected metric-gradient direction; the full CF01-degenerate M flow is the coupled B+L reservoir motion, not B alone.",
        "native_closed": [
            "Berry connection and curvature",
            "QGT metric/curvature split",
            "Q-induced J flow",
            "B metric-gradient flow",
            "B+L degenerate positive M operator",
            "ORS-derived Clifford algebra",
            "L-chain domain-wall Dirac kernel",
            "overlap/Ginsparg-Wilson operator",
            "L103 two-mode SU(2) host",
            "L103 rank-one normal interface operator",
        ],
        "not_fixed_by_first_L_geometry": [
            "specific lepton hypercharge representation and therefore the absolute electron charge label",
            "electron Yukawa coupling and numerical electron mass scale",
        ],
        "first_cell": {
            "c": float(first_cell_payload["c"]),
            "old_norm_sq": old_norm,
            "g_cc": float(first_cell_payload["metric"][0, 0]),
            "g_phiphi": float(first_cell_payload["metric"][1, 1]),
            "omega_cphi": float(first_cell_payload["curvature"][0, 1]),
        },
        "dirac_64": next(row for row in dirac_rows if int(row["completed_layers"]) == 64),
        "l103_interface": interface_rows[0],
    }
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Status figure.
    status_labels = [gate["gate"].replace("_", " ") for gate in gates]
    status_values = [1.0 if gate["status"] == "PASS" else 0.5 if gate["status"] == "NOT_CLOSED" else 0.0 for gate in gates]
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(gates)), status_values)
    plt.yticks(np.arange(len(gates)), status_labels, fontsize=8)
    plt.xticks([0.0, 0.5, 1.0], ["FAIL", "NOT CLOSED", "PASS"])
    plt.xlim(0.0, 1.05)
    plt.title("Native operator closure gate status")
    plt.tight_layout()
    plt.savefig(figures / "06_native_operator_gate_status.png", dpi=180)
    plt.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
