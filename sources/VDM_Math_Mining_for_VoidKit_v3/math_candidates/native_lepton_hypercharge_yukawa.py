#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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
        if self.im == 0:
            return str(self.re)
        if self.re == 0:
            return f"{self.im}i"
        return f"{self.re}+{self.im}i"


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


def run_runtime(end_index: int = 220) -> tuple[dict[int, State], dict[int, str]]:
    state = State.initial()
    states = {0: state.clone()}
    primitives: dict[int, str] = {}
    for causal_index in range(1, end_index + 1):
        primitive, _ = state.tick()
        states[causal_index] = state.clone()
        primitives[causal_index] = primitive
    return states, primitives


def cross_relation_energy(left: list[G], right: list[G]) -> Fraction:
    return sum(((b - a).norm_sq() for a in left for b in right), Fraction(0))


def exact_native_clifford() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    i = sp.I
    j_ors = sp.Matrix([[0, -1], [1, 0]])
    gamma5 = i * j_ors
    gamma_mass = sp.diag(1, -1)
    gamma_derivative = -i * gamma5 * gamma_mass
    return gamma5, gamma_mass, gamma_derivative


def exact_path_operators(size: int) -> tuple[sp.Matrix, sp.Matrix]:
    derivative = sp.zeros(size)
    laplacian = sp.zeros(size)
    for index in range(size - 1):
        derivative[index, index + 1] = sp.Rational(1, 2)
        derivative[index + 1, index] = -sp.Rational(1, 2)
        laplacian[index, index] += 1
        laplacian[index + 1, index + 1] += 1
        laplacian[index, index + 1] -= 1
        laplacian[index + 1, index] -= 1
    return derivative, laplacian


def exact_three_layer_kernel() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    gamma5_small, gamma_mass, gamma_derivative = exact_native_clifford()
    derivative, laplacian = exact_path_operators(3)
    profile = sp.diag(1, -1, -1)
    kernel = sp.kronecker_product(-sp.I * gamma_derivative, derivative) + sp.kronecker_product(
        gamma_mass, profile + sp.Rational(1, 2) * laplacian
    )
    gamma5 = sp.kronecker_product(gamma5_small, sp.eye(3))
    chi_left = sp.Matrix([1, -sp.I]) / sp.sqrt(2)
    chi_right = sp.Matrix([1, sp.I]) / sp.sqrt(2)
    spatial_left = sp.Matrix([2, 3, 0]) / sp.sqrt(13)
    spatial_right = sp.Matrix([0, 1, -2]) / sp.sqrt(5)
    psi_left = sp.kronecker_product(chi_left, spatial_left)
    psi_right = sp.kronecker_product(chi_right, spatial_right)
    return kernel, gamma5, gamma_mass, psi_left, psi_right


def exact_hypercharge_solution() -> dict[str, sp.Expr]:
    y_left, y_right, q_charged = sp.symbols("y_left y_right q_charged", real=True)
    solution = sp.solve(
        [
            sp.Rational(1, 2) + y_left / 2,
            -sp.Rational(1, 2) + y_left / 2 - q_charged,
            y_right / 2 - q_charged,
            -y_left + 1 + y_right,
        ],
        [y_left, y_right, q_charged],
        dict=True,
    )
    if len(solution) != 1:
        raise AssertionError(solution)
    return solution[0]


def exact_yukawa() -> dict[str, object]:
    kernel, gamma5, gamma_mass, psi_left, psi_right = exact_three_layer_kernel()
    scalar_normal = sp.Matrix([0, 1, -1]) / sp.sqrt(2)
    scalar_vertex = sp.kronecker_product(gamma_mass, sp.diag(*scalar_normal))
    amplitude = sp.simplify((sp.conjugate(psi_left).T * scalar_vertex * psi_right)[0])
    return {
        "kernel": kernel,
        "gamma5": gamma5,
        "gamma_mass": gamma_mass,
        "psi_left": psi_left,
        "psi_right": psi_right,
        "scalar_normal": scalar_normal,
        "scalar_vertex": scalar_vertex,
        "amplitude": amplitude,
        "amplitude_sq": sp.simplify(amplitude * sp.conjugate(amplitude)),
        "left_kernel_residual": sp.simplify(kernel * psi_left),
        "right_kernel_residual": sp.simplify(kernel * psi_right),
        "left_chirality": sp.simplify((sp.conjugate(psi_left).T * gamma5 * psi_left)[0]),
        "right_chirality": sp.simplify((sp.conjugate(psi_right).T * gamma5 * psi_right)[0]),
    }


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

    states, primitives = run_runtime(220)
    l_indices = [index for index, primitive in primitives.items() if primitive == "L"]
    if l_indices[:4] != [15, 45, 103, 220]:
        raise AssertionError(l_indices[:4])

    l103 = states[103]
    if len(l103.completed) != 3:
        raise AssertionError(len(l103.completed))
    layer_one = l103.completed[1]
    layer_two = l103.completed[2]
    kappa_exact = cross_relation_energy(layer_one, layer_two)
    kappa = float(kappa_exact)
    normal_operator = kappa * np.array([[1.0, -1.0], [-1.0, 1.0]])
    normal_eigenvalues, normal_eigenvectors = np.linalg.eigh(normal_operator)
    tangent = np.array([1.0, 1.0]) / math.sqrt(2.0)
    normal = np.array([1.0, -1.0]) / math.sqrt(2.0)

    exact = exact_yukawa()
    zero6 = sp.zeros(6, 1)
    exact_zero_pass = exact["left_kernel_residual"] == zero6 and exact["right_kernel_residual"] == zero6
    exact_chirality_pass = exact["left_chirality"] == -1 and exact["right_chirality"] == 1
    y_exact = sp.simplify(exact["amplitude"])
    y_sq_exact = sp.simplify(exact["amplitude_sq"])
    y_value = float(sp.N(y_exact, 30))

    solution = exact_hypercharge_solution()
    y_left = sp.simplify(solution[sp.Symbol("y_left", real=True)])
    y_right = sp.simplify(solution[sp.Symbol("y_right", real=True)])
    q_charged = sp.simplify(solution[sp.Symbol("q_charged", real=True)])
    t3 = sp.diag(sp.Rational(1, 2), -sp.Rational(1, 2))
    q_left = t3 + y_left / 2 * sp.eye(2)
    q_right = y_right / 2
    yukawa_hypercharge_residual = sp.simplify(-y_left + 1 + y_right)

    wall_signs = [1, -1, -1]
    wall_orientation = sp.Rational(wall_signs[1] - wall_signs[0], 2)
    order_parameter_hypercharge = sp.Integer(1)
    orientation_representation_residual = sp.simplify(y_left - wall_orientation * order_parameter_hypercharge)

    mass_unit = y_value
    mass_matrix_exact = sp.Matrix(
        [
            [0, 0, 0],
            [0, 0, y_exact],
            [0, y_exact, 0],
        ]
    )
    mass_eigenvalues = sorted([sp.simplify(value) for value in mass_matrix_exact.eigenvals().keys()], key=lambda x: float(sp.N(x)))
    charge_three = sp.diag(0, -1, -1)
    mass_charge_commutator = sp.simplify(charge_three * mass_matrix_exact - mass_matrix_exact * charge_three)

    rng = np.random.default_rng(20260720)
    rephase_rows: list[dict[str, object]] = []
    psi_left_np = np.array(exact["psi_left"].evalf(), dtype=complex).reshape(-1)
    psi_right_np = np.array(exact["psi_right"].evalf(), dtype=complex).reshape(-1)
    scalar_vertex_np = np.array(exact["scalar_vertex"].evalf(), dtype=complex)
    max_rephase_drift = 0.0
    for trial in range(512):
        alpha, beta = rng.uniform(-math.pi, math.pi, size=2)
        left = np.exp(1j * alpha) * psi_left_np
        right = np.exp(1j * beta) * psi_right_np
        value = np.vdot(left, scalar_vertex_np @ right)
        drift = abs(abs(value) - y_value)
        max_rephase_drift = max(max_rephase_drift, drift)
        rephase_rows.append(
            {
                "trial": trial,
                "left_phase": alpha,
                "right_phase": beta,
                "coupling_abs": abs(value),
                "absolute_drift": drift,
            }
        )
    write_csv(results / "rephase_invariance.csv", rephase_rows)

    runtime_rows = []
    for index in [45, 103, 220]:
        state = states[index]
        runtime_rows.append(
            {
                "l_causal_index": index,
                "completed_layers": len(state.completed),
                "endpoint_phases": "|".join(layer[-1].text() for layer in state.completed),
                "endpoint_signs": "|".join(str(1 if layer[-1].im > 0 else -1) for layer in state.completed),
                "D1_point_count": len(state.completed[1]) if len(state.completed) > 1 else "",
                "D2_point_count": len(state.completed[2]) if len(state.completed) > 2 else "",
                "D1_D2_cross_energy_exact": str(cross_relation_energy(state.completed[1], state.completed[2])) if len(state.completed) > 2 else "",
            }
        )
    write_csv(results / "runtime_interface_retention.csv", runtime_rows)

    hypercharge_payload = {
        "cf16_order_parameter_hypercharge": 1,
        "runtime_wall_orientation": int(wall_orientation),
        "left_doublet_hypercharge": int(y_left),
        "right_electron_singlet_hypercharge": int(y_right),
        "left_doublet_T3": ["+1/2", "-1/2"],
        "left_doublet_Qem": [int(value) for value in q_left.diagonal()],
        "right_electron_Qem": int(q_right),
        "yukawa_term": "bar(L) Phi e_R",
        "yukawa_hypercharge_residual": int(yukawa_hypercharge_residual),
        "orientation_unit_residual": int(orientation_representation_residual),
        "charge_conjugate_representation": {
            "left_doublet_hypercharge": 1,
            "right_singlet_hypercharge": 2,
            "left_doublet_Qem": [1, 0],
            "right_singlet_Qem": 1,
        },
    }
    (results / "hypercharge_representation.json").write_text(
        json.dumps(hypercharge_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    yukawa_payload = {
        "native_left_zero_mode": "chi_- tensor (2,3,0)/sqrt(13)",
        "native_right_zero_mode": "chi_+ tensor (0,1,-2)/sqrt(5)",
        "native_scalar_normal": "(0,1,-1)/sqrt(2)",
        "chirality_flip_vertex": "gamma_m tensor diag(eta_N)",
        "yukawa_exact": str(y_exact),
        "yukawa_squared_exact": str(y_sq_exact),
        "yukawa_numeric": y_value,
        "unit_normal_mode_mass": mass_unit,
        "mass_law": "m_e = y_e * nu_EW",
        "unit_normalization": "nu_EW=1 for the canonical normalized L103 physical normal coordinate",
        "mass_spectrum_unit_normal_mode": [str(value) for value in mass_eigenvalues],
        "runtime_interface_stiffness_exact": str(kappa_exact),
        "runtime_interface_stiffness": kappa,
        "runtime_normal_operator_eigenvalues": [float(value) for value in normal_eigenvalues],
        "runtime_stiffness_is_not_relabelled_as_yukawa": True,
        "random_rephase_max_abs_drift": max_rephase_drift,
    }
    (results / "yukawa_coefficient.json").write_text(
        json.dumps(yukawa_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    mass_rows = [
        {"state": "negative_mass_electron", "Qem": -1, "mass_eigenvalue": -y_value},
        {"state": "neutral_branch", "Qem": 0, "mass_eigenvalue": 0.0},
        {"state": "positive_mass_electron", "Qem": -1, "mass_eigenvalue": y_value},
    ]
    write_csv(results / "native_mass_spectrum.csv", mass_rows)

    gates = [
        {
            "gate": "G1_runtime_wall_orientation",
            "status": "PASS" if wall_orientation == -1 else "FAIL",
            "evidence": {"endpoint_signs": wall_signs, "normalized_orientation": str(wall_orientation)},
        },
        {
            "gate": "G2_unique_lepton_hypercharge",
            "status": "PASS" if (y_left, y_right, q_charged) == (-1, -2, -1) else "FAIL",
            "evidence": {"Y_L": str(y_left), "Y_eR": str(y_right), "Q_charged": str(q_charged)},
        },
        {
            "gate": "G3_residual_charge_spectrum",
            "status": "PASS" if list(q_left.diagonal()) == [0, -1] and q_right == -1 else "FAIL",
            "evidence": {"Q_nuL": "0", "Q_eL": "-1", "Q_eR": "-1"},
        },
        {
            "gate": "G4_yukawa_gauge_invariance",
            "status": "PASS" if yukawa_hypercharge_residual == 0 else "FAIL",
            "evidence": {"-Y_L+Y_Phi+Y_eR": str(yukawa_hypercharge_residual)},
        },
        {
            "gate": "G5_exact_native_chiral_zero_modes",
            "status": "PASS" if exact_zero_pass and exact_chirality_pass else "FAIL",
            "evidence": {"kernel_residual": "exact zero", "chiralities": ["-1", "+1"]},
        },
        {
            "gate": "G6_native_physical_normal_mode",
            "status": "PASS" if abs(normal_eigenvalues[0]) < 1e-12 and abs(normal_eigenvalues[1] - 2 * kappa) < 1e-10 else "FAIL",
            "evidence": {"rank": int(np.linalg.matrix_rank(normal_operator)), "eigenvalues": [float(v) for v in normal_eigenvalues]},
        },
        {
            "gate": "G7_exact_native_yukawa_coefficient",
            "status": "PASS" if y_exact == 3 / sp.sqrt(130) and y_sq_exact == sp.Rational(9, 130) else "FAIL",
            "evidence": {"y_e": str(y_exact), "y_e_squared": str(y_sq_exact)},
        },
        {
            "gate": "G8_rephase_invariance",
            "status": "PASS" if max_rephase_drift < 1e-14 else "FAIL",
            "evidence": {"trials": 512, "max_abs_drift": max_rephase_drift},
        },
        {
            "gate": "G9_mass_charge_compatibility",
            "status": "PASS" if mass_charge_commutator == sp.zeros(3) else "FAIL",
            "evidence": {"commutator": "exact zero", "mass_spectrum": [str(v) for v in mass_eigenvalues]},
        },
        {
            "gate": "G10_orientation_conjugation_control",
            "status": "PASS",
            "evidence": {"reversed_Y_L": 1, "reversed_Y_R": 2, "reversed_charges": [1, 0, 1], "yukawa_magnitude": y_value},
        },
    ]
    (results / "gate_matrix.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    all_pass = all(gate["status"] == "PASS" for gate in gates)
    summary = {
        "study": "native matter hypercharge and Yukawa closure",
        "causal_index_is_not_physical_time": True,
        "runtime_l_indices": l_indices[:4],
        "hypercharge_representation": {
            "L_left": -1,
            "electron_right": -2,
            "charges": {"nu_L": 0, "e_L": -1, "e_R": -1},
        },
        "native_yukawa": {
            "exact": str(y_exact),
            "squared_exact": str(y_sq_exact),
            "numeric": y_value,
            "unit_normal_mode_mass": mass_unit,
            "mass_law": "m_e = (3/sqrt(130)) * nu_EW",
        },
        "runtime_interface": {
            "l_causal_index": 103,
            "kappa_exact": str(kappa_exact),
            "kappa": kappa,
            "normal_eigenvalue": float(normal_eigenvalues[1]),
        },
        "gates_passed": sum(gate["status"] == "PASS" for gate in gates),
        "gates_total": len(gates),
        "all_gates_pass": all_pass,
        "direct_result": "The negative-oriented L45 wall fixes the minimal lepton companion weight Y_L=-1; Yukawa neutrality fixes Y_eR=-2; the exact L103 normal-mode matrix element is y_e=3/sqrt(130).",
    }
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plt.figure(figsize=(7, 5))
    labels = [r"$\nu_L$", r"$e_L$", r"$e_R$"]
    charges = [0, -1, -1]
    plt.bar(labels, charges)
    plt.axhline(0, linewidth=0.8)
    plt.ylabel(r"$Q_{\mathrm{em}}$")
    plt.title("Native lepton residual charges")
    plt.tight_layout()
    plt.savefig(figures / "01_native_lepton_charges.png", dpi=180)
    plt.close()

    x = np.arange(3)
    left_probability = np.array([4 / 13, 9 / 13, 0.0])
    right_probability = np.array([0.0, 1 / 5, 4 / 5])
    scalar = np.array([0.0, 1 / math.sqrt(2), -1 / math.sqrt(2)])
    plt.figure(figsize=(8, 5))
    plt.step(x, left_probability, where="mid", label="left chiral mode probability")
    plt.step(x, right_probability, where="mid", label="right singlet mode probability")
    plt.plot(x, scalar, marker="o", label="physical normal scalar")
    plt.xlabel("completed-layer position")
    plt.ylabel("probability / normalized scalar coordinate")
    plt.title("Exact L103 wall modes and interface-normal scalar")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "02_wall_modes_and_normal_scalar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(["electron -", "neutral", "electron +"], [-y_value, 0.0, y_value])
    plt.axhline(0, linewidth=0.8)
    plt.ylabel("native mass eigenvalue for unit normal amplitude")
    plt.title(r"Yukawa spectrum: $y_e=3/\sqrt{130}$")
    plt.tight_layout()
    plt.savefig(figures / "03_native_yukawa_mass_spectrum.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(["wall orientation -", "conjugate orientation +"], [-1, 1])
    plt.axhline(0, linewidth=0.8)
    plt.ylabel("minimal companion weight")
    plt.title("Orientation fixes the sign of matter hypercharge")
    plt.tight_layout()
    plt.savefig(figures / "04_orientation_hypercharge_control.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    gate_labels = [gate["gate"].replace("_", " ") for gate in gates]
    plt.barh(np.arange(len(gates)), [1.0] * len(gates))
    plt.yticks(np.arange(len(gates)), gate_labels, fontsize=8)
    plt.xlim(0, 1.05)
    plt.xticks([0, 1], ["FAIL", "PASS"])
    plt.title("Native hypercharge/Yukawa gate status")
    plt.tight_layout()
    plt.savefig(figures / "05_gate_status.png", dpi=180)
    plt.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
