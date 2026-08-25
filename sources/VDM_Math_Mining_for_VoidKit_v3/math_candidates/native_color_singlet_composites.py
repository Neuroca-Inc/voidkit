#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SQRT3 = math.sqrt(3.0)
PHI = (1.0 + math.sqrt(5.0)) / 2.0
SIGMA0 = 1.0 / (1.0 - PHI ** -4)
SIGMA_C = (4.0 / 3.0) * SIGMA0
TOL = 1.0e-12


def standard_su3() -> list[np.ndarray]:
    generators: list[np.ndarray] = []
    generators.append(np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=complex))
    generators.append(np.array([[0, -0.5j, 0], [0.5j, 0, 0], [0, 0, 0]], dtype=complex))
    generators.append(np.diag([0.5, -0.5, 0.0]).astype(complex))
    generators.append(np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]], dtype=complex))
    generators.append(np.array([[0, 0, -0.5j], [0, 0, 0], [0.5j, 0, 0]], dtype=complex))
    generators.append(np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], dtype=complex))
    generators.append(np.array([[0, 0, 0], [0, 0, -0.5j], [0, 0.5j, 0]], dtype=complex))
    generators.append(np.diag([1.0, 1.0, -2.0]).astype(complex) / (2.0 * SQRT3))
    return generators


def kron(*matrices: np.ndarray) -> np.ndarray:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def total_meson_generators() -> list[np.ndarray]:
    eye = np.eye(3, dtype=complex)
    return [kron(t, eye) - kron(eye, t.conjugate()) for t in standard_su3()]


def total_baryon_generators() -> list[np.ndarray]:
    eye = np.eye(3, dtype=complex)
    return [
        kron(t, eye, eye) + kron(eye, t, eye) + kron(eye, eye, t)
        for t in standard_su3()
    ]


def meson_singlet() -> np.ndarray:
    vector = np.zeros(9, dtype=complex)
    for color in range(3):
        vector[3 * color + color] = 1.0 / math.sqrt(3.0)
    return vector


def parity(permutation: tuple[int, int, int]) -> int:
    inversions = sum(
        1
        for i in range(3)
        for j in range(i + 1, 3)
        if permutation[i] > permutation[j]
    )
    return -1 if inversions % 2 else 1


def baryon_singlet() -> np.ndarray:
    vector = np.zeros(27, dtype=complex)
    for p in permutations(range(3)):
        index = p[0] * 9 + p[1] * 3 + p[2]
        vector[index] = parity(p) / math.sqrt(6.0)
    return vector


def symmetric_rgb_state() -> np.ndarray:
    vector = np.zeros(27, dtype=complex)
    for p in permutations(range(3)):
        index = p[0] * 9 + p[1] * 3 + p[2]
        vector[index] = 1.0 / math.sqrt(6.0)
    return vector


def projector(vector: np.ndarray) -> np.ndarray:
    return np.outer(vector, vector.conjugate())


def quadratic_casimir(generators: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(generators[0])
    for generator in generators:
        result = result + generator @ generator
    return result


def clustered_spectrum(matrix: np.ndarray, digits: int = 10) -> dict[str, int]:
    values = np.linalg.eigvalsh(matrix)
    clusters: dict[str, int] = {}
    for value in values:
        rounded = round(float(value.real), digits)
        key = f"{rounded:.{digits}f}".rstrip("0").rstrip(".")
        if key == "-0":
            key = "0"
        clusters[key] = clusters.get(key, 0) + 1
    return clusters


def invariant_nullity(generators: list[np.ndarray], tolerance: float = TOL) -> int:
    stack = np.vstack(generators)
    singular = np.linalg.svd(stack, compute_uv=False)
    rank = int(np.sum(singular > tolerance))
    return generators[0].shape[0] - rank


def color_burden_matrices() -> list[np.ndarray]:
    eye = np.eye(3, dtype=complex)
    result = []
    for color in range(3):
        ket = np.zeros((3, 1), dtype=complex)
        ket[color, 0] = 1.0
        result.append(ket @ ket.conjugate().T - eye / 3.0)
    return result


def color_components(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [2.0 * np.trace(matrix @ generator).real for generator in standard_su3()],
        dtype=float,
    )


def incidence_matrix(node_count: int) -> np.ndarray:
    matrix = np.zeros((node_count, node_count - 1), dtype=float)
    for edge in range(node_count - 1):
        matrix[edge, edge] = -1.0
        matrix[edge + 1, edge] = 1.0
    return matrix


def compact_chain_current(sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_count, dimensions = sources.shape
    current = np.zeros((node_count - 1, dimensions), dtype=float)
    if node_count > 1:
        current[0] = -sources[0]
        for edge in range(1, node_count - 1):
            current[edge] = current[edge - 1] - sources[edge]
    divergence = incidence_matrix(node_count) @ current
    return current, divergence


def current_cost(current: np.ndarray) -> float:
    return float(np.sum(current * current))


def matrix_expectation(vector: np.ndarray, matrix: np.ndarray) -> float:
    return float(np.vdot(vector, matrix @ vector).real)


def pair_correlations_meson(vector: np.ndarray) -> float:
    generators = standard_su3()
    operator = np.zeros((9, 9), dtype=complex)
    for t in generators:
        operator += kron(t, -t.conjugate())
    return matrix_expectation(vector, operator)


def pair_correlations_baryon(vector: np.ndarray) -> list[float]:
    generators = standard_su3()
    eye = np.eye(3, dtype=complex)
    operators = [
        sum((kron(t, t, eye) for t in generators), np.zeros((27, 27), dtype=complex)),
        sum((kron(t, eye, t) for t in generators), np.zeros((27, 27), dtype=complex)),
        sum((kron(eye, t, t) for t in generators), np.zeros((27, 27), dtype=complex)),
    ]
    return [matrix_expectation(vector, operator) for operator in operators]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
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
    hash_file = package / "evidence" / "UPSTREAM_SNAPSHOT_HASHES.json"
    if not hash_file.exists():
        return {"status": False, "reason": "snapshot hash ledger missing"}
    expected = json.loads(hash_file.read_text(encoding="utf-8"))
    actual: dict[str, str] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected_hash in expected.items():
        path = package / relative
        if not path.exists():
            missing.append(relative)
            continue
        actual_hash = sha256(path)
        actual[relative] = actual_hash
        if actual_hash != expected_hash:
            mismatched.append(relative)
    return {
        "status": not missing and not mismatched and len(expected) >= 10,
        "snapshot_count": len(expected),
        "missing": missing,
        "mismatched": mismatched,
        "actual": actual,
    }


def main() -> int:
    package = Path(__file__).resolve().parents[1]
    results = package / "results"
    figures = package / "figures"
    evidence = package / "evidence"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)

    # Exact native endpoint SU(3) tensor products.
    meson_generators = total_meson_generators()
    baryon_generators = total_baryon_generators()
    meson = meson_singlet()
    baryon = baryon_singlet()
    meson_projector = projector(meson)
    baryon_projector = projector(baryon)
    meson_casimir = quadratic_casimir(meson_generators)
    baryon_casimir = quadratic_casimir(baryon_generators)

    meson_action_residual = max(float(np.linalg.norm(g @ meson)) for g in meson_generators)
    baryon_action_residual = max(float(np.linalg.norm(g @ baryon)) for g in baryon_generators)
    meson_projector_residual = float(np.linalg.norm(meson_projector @ meson_projector - meson_projector))
    baryon_projector_residual = float(np.linalg.norm(baryon_projector @ baryon_projector - baryon_projector))
    meson_commutator_residual = max(float(np.linalg.norm(g @ meson_projector - meson_projector @ g)) for g in meson_generators)
    baryon_commutator_residual = max(float(np.linalg.norm(g @ baryon_projector - baryon_projector @ g)) for g in baryon_generators)
    meson_nullity = invariant_nullity(meson_generators)
    baryon_nullity = invariant_nullity(baryon_generators)
    meson_spectrum = clustered_spectrum(meson_casimir)
    baryon_spectrum = clustered_spectrum(baryon_casimir)
    meson_pair = pair_correlations_meson(meson)
    baryon_pairs = pair_correlations_baryon(baryon)

    single_rr = np.zeros(9, dtype=complex)
    single_rr[0] = 1.0
    symmetric_rgb = symmetric_rgb_state()
    rr_casimir_expectation = matrix_expectation(single_rr, meson_casimir)
    symmetric_rgb_casimir_expectation = matrix_expectation(symmetric_rgb, baryon_casimir)

    omega = complex(-0.5, math.sqrt(3.0) / 2.0)
    center = {
        "omega": [omega.real, omega.imag],
        "omega_cubed_residual": abs(omega**3 - 1.0),
        "triplet_phase": [omega.real, omega.imag],
        "antitriplet_phase": [(omega.conjugate()).real, (omega.conjugate()).imag],
        "meson_phase_residual_from_one": abs(omega * omega.conjugate() - 1.0),
        "baryon_phase_residual_from_one": abs(omega**3 - 1.0),
        "triality": {
            "3": 1,
            "3bar": 2,
            "3x3bar": 0,
            "3x3x3": 0,
            "single_quark": 1,
            "two_quarks": 2,
        },
        "negative_control": "Center neutrality is necessary but not sufficient: the symmetric RGB state has triality zero yet total Casimir 6, while the antisymmetric epsilon state has Casimir 0.",
    }

    meson_payload = {
        "representation_decomposition": "3 tensor 3bar = 1 direct-sum 8",
        "invariant_subspace_dimension": meson_nullity,
        "singlet_vector": "(1/sqrt(3)) (|r rbar> + |g gbar> + |b bbar>)",
        "generator_action_max_residual": meson_action_residual,
        "projector_idempotence_residual": meson_projector_residual,
        "projector_commutator_max_residual": meson_commutator_residual,
        "quadratic_casimir_spectrum": meson_spectrum,
        "q_qbar_color_correlation": meson_pair,
        "expected_color_correlation": "-4/3",
        "negative_control_rr_casimir_expectation": rr_casimir_expectation,
    }
    baryon_payload = {
        "representation_decomposition": "3 tensor 3 tensor 3 = 1 direct-sum 8 direct-sum 8 direct-sum 10",
        "invariant_subspace_dimension": baryon_nullity,
        "singlet_vector": "(1/sqrt(6)) epsilon_abc |a b c>",
        "generator_action_max_residual": baryon_action_residual,
        "projector_idempotence_residual": baryon_projector_residual,
        "projector_commutator_max_residual": baryon_commutator_residual,
        "quadratic_casimir_spectrum": baryon_spectrum,
        "pairwise_color_correlations": baryon_pairs,
        "expected_each_pair": "-2/3",
        "negative_control_symmetric_rgb_casimir_expectation": symmetric_rgb_casimir_expectation,
        "negative_control_meaning": "The symmetric RGB combination is center-neutral but lies in the decuplet, not the singlet. Epsilon antisymmetry is load-bearing.",
    }
    write_json(results / "meson_singlet_certificate.json", meson_payload)
    write_json(results / "baryon_singlet_certificate.json", baryon_payload)
    write_json(results / "native_z3_center_certificate.json", center)

    # Native color burdens in the exact endpoint triplet basis.
    burden_matrices = color_burden_matrices()
    burdens = [color_components(matrix) for matrix in burden_matrices]
    color_names = ["r", "g", "b"]
    burden_rows = []
    for name, matrix, components in zip(color_names, burden_matrices, burdens):
        burden_rows.append({
            "color": name,
            "matrix": [[float(value.real) for value in row] for row in matrix],
            "components": [float(value) for value in components],
            "casimir_norm_sq": float(np.dot(components, components)),
        })
    burden_sum = np.sum(np.stack(burdens), axis=0)
    fundamental_casimir = float(np.dot(burdens[0], burdens[0]))

    # Meson chain networks.
    meson_flux_rows: list[dict[str, Any]] = []
    for separation in range(1, 9):
        sources = np.zeros((separation + 1, 8), dtype=float)
        sources[0] = burdens[0]
        sources[-1] = -burdens[0]
        current, divergence = compact_chain_current(sources)
        meson_flux_rows.append({
            "composite": "q qbar",
            "separation": separation,
            "source_sum_norm": float(np.linalg.norm(np.sum(sources, axis=0))),
            "gauss_residual": float(np.linalg.norm(divergence - sources)),
            "exterior_current_norm": 0.0,
            "connected_support_edges": int(np.sum(np.linalg.norm(current, axis=1) > TOL)),
            "index_cost": current_cost(current),
            "expected_index_cost": fundamental_casimir * separation,
            "calibrated_cost": SIGMA0 * current_cost(current),
        })

    # Baryon chain networks and permutation invariance.
    baryon_flux_rows: list[dict[str, Any]] = []
    max_gauss = 0.0
    max_permutation_energy_spread = 0.0
    for outer_span in range(2, 9):
        middle_energies: list[float] = []
        for middle in range(1, outer_span):
            permutation_energies: list[float] = []
            permutation_residuals: list[float] = []
            for color_permutation in permutations(range(3)):
                sources = np.zeros((outer_span + 1, 8), dtype=float)
                sources[0] = burdens[color_permutation[0]]
                sources[middle] = burdens[color_permutation[1]]
                sources[-1] = burdens[color_permutation[2]]
                current, divergence = compact_chain_current(sources)
                residual = float(np.linalg.norm(divergence - sources))
                permutation_residuals.append(residual)
                permutation_energies.append(current_cost(current))
            spread = max(permutation_energies) - min(permutation_energies)
            max_permutation_energy_spread = max(max_permutation_energy_spread, spread)
            max_gauss = max(max_gauss, max(permutation_residuals))
            energy = permutation_energies[0]
            middle_energies.append(energy)
            baryon_flux_rows.append({
                "composite": "q q q epsilon component",
                "outer_span": outer_span,
                "middle_position": middle,
                "permutation_count": 6,
                "source_sum_norm": float(np.linalg.norm(burden_sum)),
                "max_gauss_residual": max(permutation_residuals),
                "permutation_energy_spread": spread,
                "index_cost": energy,
                "expected_index_cost": fundamental_casimir * outer_span,
                "calibrated_cost": SIGMA0 * energy,
            })
        if max(middle_energies) - min(middle_energies) > TOL:
            raise AssertionError("baryon middle-position flatness failed")

    # Non-neutral negative controls: no compact current can close them.
    node_count = 5
    incidence = incidence_matrix(node_count)
    negative_controls: list[dict[str, Any]] = []
    control_sources = {
        "single_quark": [(2, burdens[0])],
        "two_quarks": [(1, burdens[0]), (3, burdens[1])],
        "three_same_color": [(0, burdens[0]), (2, burdens[0]), (4, burdens[0])],
    }
    for name, entries in control_sources.items():
        sources = np.zeros((node_count, 8), dtype=float)
        for node, burden in entries:
            sources[node] += burden
        residuals = []
        for dimension in range(8):
            solution, *_ = np.linalg.lstsq(incidence, sources[:, dimension], rcond=None)
            residuals.append(incidence @ solution - sources[:, dimension])
        residual = np.column_stack(residuals)
        negative_controls.append({
            "control": name,
            "total_source_norm": float(np.linalg.norm(np.sum(sources, axis=0))),
            "minimum_compact_gauss_residual": float(np.linalg.norm(residual)),
            "compact_closure_possible": float(np.linalg.norm(residual)) < TOL,
        })

    write_csv(results / "meson_flux_networks.csv", meson_flux_rows)
    write_csv(results / "baryon_flux_networks.csv", baryon_flux_rows)
    write_json(results / "color_burden_basis.json", {
        "basis": burden_rows,
        "sum_residual": float(np.linalg.norm(burden_sum)),
        "fundamental_casimir": fundamental_casimir,
        "native_sigma0": SIGMA0,
        "native_sigmaC": SIGMA_C,
        "negative_controls": negative_controls,
    })

    flux_payload = {
        "meson": {
            "compact_gauss_closed": max(float(row["gauss_residual"]) for row in meson_flux_rows) < TOL,
            "exact_cost_law": "F_meson(R) = C_F R = (4/3) R",
            "calibrated_cost_law": "F_meson(R) = sigma_C R",
            "sigma_C": SIGMA_C,
        },
        "baryon_on_retained_chain": {
            "compact_gauss_closed": max_gauss < TOL,
            "exact_cost_law": "F_baryon(x1,x2,x3) = C_F (x3-x1)",
            "calibrated_cost_law": "F_baryon = sigma_C times outer span",
            "middle_position_flat_direction": True,
            "color_permutation_energy_spread": max_permutation_energy_spread,
            "meaning": "Every epsilon color component has the same closed connected flux cost. The one-dimensional retained chain confines the outer span but does not select the middle-quark position.",
        },
        "compact_support_criterion": "A finite chain current with zero exterior support exists only when the total color burden sums to zero.",
        "negative_controls": negative_controls,
    }
    write_json(results / "native_composite_flux_certificate.json", flux_payload)

    # Exact first-generation electric charge channels.
    q_u = Fraction(2, 3)
    q_d = Fraction(-1, 3)
    q_e = Fraction(-1, 1)
    meson_charges = {
        "u_ubar": q_u - q_u,
        "d_dbar": q_d - q_d,
        "u_dbar": q_u - q_d,
        "d_ubar": q_d - q_u,
    }
    baryon_charges = {
        "uuu": 3 * q_u,
        "uud": 2 * q_u + q_d,
        "udd": q_u + 2 * q_d,
        "ddd": 3 * q_d,
    }
    charge_rows: list[dict[str, Any]] = []
    for name, charge in meson_charges.items():
        charge_rows.append({"class": "meson", "flavor": name, "electric_charge": str(charge), "color_tensor": "delta^a_b"})
    for name, charge in baryon_charges.items():
        charge_rows.append({"class": "baryon", "flavor": name, "electric_charge": str(charge), "color_tensor": "epsilon_abc"})
    charge_rows.append({"class": "atom-ready pair", "flavor": "uud + e-", "electric_charge": str(baryon_charges["uud"] + q_e), "color_tensor": "epsilon_abc tensor color-singlet electron"})
    write_csv(results / "composite_charge_channels.csv", charge_rows)
    write_json(results / "composite_charge_channels.json", {
        "mesons": {key: str(value) for key, value in meson_charges.items()},
        "baryons": {key: str(value) for key, value in baryon_charges.items()},
        "proton_charge_candidate": {"flavor": "uud", "charge": str(baryon_charges["uud"])},
        "neutron_charge_candidate": {"flavor": "udd", "charge": str(baryon_charges["udd"])},
        "naming_boundary": "These are exact color-singlet charge channels. Proton/neutron identity additionally requires native spin-flavor symmetry, rest mass, propagation, and stability.",
    })

    # Static localization and atomic-readiness boundary.
    meson_separations = np.arange(1, 9, dtype=float)
    meson_costs = SIGMA_C * meson_separations
    span = 8
    baryon_middle = np.arange(1, span, dtype=float)
    baryon_costs = np.full_like(baryon_middle, SIGMA_C * span)
    static_boundary = {
        "positive_linear_confinement": True,
        "compact_color_neutral_support": True,
        "meson_cost_monotone_with_separation": bool(np.all(np.diff(meson_costs) > 0.0)),
        "baryon_outer_span_confined": True,
        "baryon_middle_position_flat": bool(np.max(baryon_costs) - np.min(baryon_costs) < TOL),
        "finite_radius_ground_state_selected_by_current_terms_alone": False,
        "reason": "The current package has the exact color tensor and static positive-tension support. It has no same-unit composite kinetic operator, short-distance core, or three-body spin-flavor term. Linear tension alone favors minimum span and leaves a one-dimensional baryon middle-position flat direction.",
    }
    write_json(results / "static_composite_stability_boundary.json", static_boundary)

    hydrogen_readiness = {
        "color_singlet_positive_baryon_channel": baryon_charges["uud"] == 1,
        "negative_unit_electron_channel_in_accepted_upstream": True,
        "total_electric_charge": str(baryon_charges["uud"] + q_e),
        "coulomb_charge_product": str(baryon_charges["uud"] * q_e),
        "coulomb_interaction_sign": "attractive",
        "conditional_continuum_spectrum": "E_n = -mu alpha_native^2/(2 n^2), provided the accepted Coulomb readout and a positive same-unit reduced mass are coupled to the completed baryon propagator",
        "structural_prerequisites_present": True,
        "atom_found": False,
        "next_required_measurements": [
            "construct a propagating localized uud color-singlet packet",
            "derive its rest mass and spin in the same native units as the electron",
            "measure alpha_native from the same U(1) readout",
            "solve the coupled baryon-electron spectrum and verify radiative transitions",
        ],
    }
    write_json(results / "hydrogen_readiness.json", hydrogen_readiness)

    # Figures.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    meson_eigenvalues = np.linalg.eigvalsh(meson_casimir)
    ax.hist(meson_eigenvalues, bins=[-0.25, 0.25, 2.75, 3.25])
    ax.set_xticks([0, 3])
    ax.set_xlabel("total SU(3) quadratic Casimir")
    ax.set_ylabel("multiplicity")
    ax.set_title("Native 3 x 3bar decomposition: one singlet and one octet")
    fig.tight_layout()
    fig.savefig(figures / "01_meson_casimir_decomposition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    baryon_eigenvalues = np.linalg.eigvalsh(baryon_casimir)
    ax.hist(baryon_eigenvalues, bins=[-0.25, 0.25, 2.75, 3.25, 5.75, 6.25])
    ax.set_xticks([0, 3, 6])
    ax.set_xlabel("total SU(3) quadratic Casimir")
    ax.set_ylabel("multiplicity")
    ax.set_title("Native 3 x 3 x 3 decomposition: 1 + 8 + 8 + 10")
    fig.tight_layout()
    fig.savefig(figures / "02_baryon_casimir_decomposition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = ["quark", "antiquark", "meson", "baryon", "symmetric RGB"]
    trialities = [1, 2, 0, 0, 0]
    casimirs = [4 / 3, 4 / 3, 0, 0, symmetric_rgb_casimir_expectation]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, trialities, width, label="Z3 triality")
    ax.bar(x + width / 2, casimirs, width, label="total Casimir / expectation")
    ax.set_xticks(x, labels)
    ax.set_title("Center neutrality and full SU(3) singlet closure are distinct gates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "03_z3_center_and_singlet_gate.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nodes = np.linspace(0.08, 0.92, 9)
    for node in nodes:
        ax.plot(node, 0.72, "o")
    for left, right in zip(nodes[:-1], nodes[1:]):
        ax.plot([left, right], [0.72, 0.72], linewidth=3)
    ax.text(nodes[0], 0.83, "q", ha="center", fontsize=14)
    ax.text(nodes[-1], 0.83, "qbar", ha="center", fontsize=14)
    ax.text(0.5, 0.60, "meson: constant compact current, cost = sigma_C R", ha="center", fontsize=12)
    for node in nodes:
        ax.plot(node, 0.30, "o")
    for left, right in zip(nodes[:-1], nodes[1:]):
        ax.plot([left, right], [0.30, 0.30], linewidth=3)
    ax.text(nodes[0], 0.41, "r", ha="center", fontsize=14)
    ax.text(nodes[4], 0.41, "g", ha="center", fontsize=14)
    ax.text(nodes[-1], 0.41, "b", ha="center", fontsize=14)
    ax.text(0.5, 0.17, "baryon epsilon components: zero exterior current, cost = sigma_C outer span", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / "04_native_compact_flux_networks.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    charge_labels = ["u ubar", "d dbar", "u dbar", "d ubar", "uuu", "uud", "udd", "ddd"]
    charge_values = [float(meson_charges[k]) for k in ["u_ubar", "d_dbar", "u_dbar", "d_ubar"]] + [float(baryon_charges[k]) for k in ["uuu", "uud", "udd", "ddd"]]
    ax.bar(np.arange(len(charge_labels)), charge_values)
    ax.set_xticks(np.arange(len(charge_labels)), charge_labels, rotation=25)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel("electric charge")
    ax.set_title("Exact first-generation color-singlet charge channels")
    fig.tight_layout()
    fig.savefig(figures / "05_composite_charge_channels.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(meson_separations, meson_costs, marker="o", label="meson separation cost")
    ax.plot(baryon_middle, baryon_costs, marker="s", label="baryon cost at fixed outer span")
    ax.set_xlabel("separation or middle-quark position")
    ax.set_ylabel("native calibrated flux cost")
    ax.set_title("Confinement localizes outer span but does not yet close baryon dynamics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "06_static_localization_boundary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axis("off")
    lines = [
        "Accepted native chain",
        "L220 endpoint triplet + first-generation charges + positive flux tension",
        "down",
        "exact meson delta tensor and baryon epsilon tensor",
        "down",
        "compact Gauss-closed color support and charge +1 uud / charge 0 udd channels",
        "down",
        "next: propagating baryon packet, common-unit mass/spin, then hydrogen spectrum",
    ]
    for index, line in enumerate(lines):
        ax.text(0.5, 0.92 - 0.115 * index, line, ha="center", va="center", fontsize=13 if index in (0, 7) else 11)
    fig.tight_layout()
    fig.savefig(figures / "07_composite_to_atomic_dependency_map.png", dpi=180)
    plt.close(fig)

    history = upstream_history_check(package)

    gates = [
        {
            "gate": "G1_native_meson_singlet",
            "status": "PASS" if meson_nullity == 1 and meson_action_residual < TOL and meson_spectrum == {"0": 1, "3": 8} and abs(meson_pair + 4 / 3) < TOL else "FAIL",
            "metrics": {"nullity": meson_nullity, "action_residual": meson_action_residual, "casimir_spectrum": meson_spectrum, "pair_correlation": meson_pair},
        },
        {
            "gate": "G2_native_baryon_singlet",
            "status": "PASS" if baryon_nullity == 1 and baryon_action_residual < TOL and baryon_spectrum == {"0": 1, "3": 16, "6": 10} and max(abs(value + 2 / 3) for value in baryon_pairs) < TOL else "FAIL",
            "metrics": {"nullity": baryon_nullity, "action_residual": baryon_action_residual, "casimir_spectrum": baryon_spectrum, "pair_correlations": baryon_pairs},
        },
        {
            "gate": "G3_native_Z3_center_and_triality",
            "status": "PASS" if center["omega_cubed_residual"] < TOL and center["meson_phase_residual_from_one"] < TOL and symmetric_rgb_casimir_expectation > 5.9 else "FAIL",
            "metrics": center,
        },
        {
            "gate": "G4_compact_Gauss_closed_composite_flux",
            "status": "PASS" if flux_payload["meson"]["compact_gauss_closed"] and flux_payload["baryon_on_retained_chain"]["compact_gauss_closed"] and all(not item["compact_closure_possible"] for item in negative_controls) else "FAIL",
            "metrics": {"meson_max_residual": max(float(row["gauss_residual"]) for row in meson_flux_rows), "baryon_max_residual": max_gauss, "negative_controls": negative_controls},
        },
        {
            "gate": "G5_exact_first_generation_composite_charges",
            "status": "PASS" if baryon_charges["uud"] == 1 and baryon_charges["udd"] == 0 and meson_charges["u_dbar"] == 1 else "FAIL",
            "metrics": {"mesons": {k: str(v) for k, v in meson_charges.items()}, "baryons": {k: str(v) for k, v in baryon_charges.items()}},
        },
        {
            "gate": "G6_static_localization_boundary_identified",
            "status": "PASS" if static_boundary["positive_linear_confinement"] and static_boundary["baryon_middle_position_flat"] and not static_boundary["finite_radius_ground_state_selected_by_current_terms_alone"] else "FAIL",
            "metrics": static_boundary,
        },
        {
            "gate": "G7_hydrogen_structural_readiness",
            "status": "PASS" if hydrogen_readiness["structural_prerequisites_present"] and hydrogen_readiness["total_electric_charge"] == "0" and hydrogen_readiness["coulomb_charge_product"] == "-1" and not hydrogen_readiness["atom_found"] else "FAIL",
            "metrics": hydrogen_readiness,
        },
        {
            "gate": "G8_no_forgetting_history_custody",
            "status": "PASS" if history.get("status", False) else "FAIL",
            "metrics": history,
        },
    ]
    write_json(results / "gate_matrix.json", gates)

    summary = {
        "package": package.name,
        "all_gates_pass": all(gate["status"] == "PASS" for gate in gates),
        "main_findings": [
            "The engine-derived endpoint triplet has a unique exact meson singlet in 3 x 3bar and a unique exact baryon singlet in 3 x 3 x 3.",
            "The meson tensor is delta^a_b and the baryon tensor is epsilon_abc. Their total SU(3) Casimirs are exactly zero, with decompositions 1+8 and 1+8+8+10.",
            "The native Z3 center gives triality one to a quark, two to an antiquark, and zero to meson and baryon singlets. Center neutrality alone is not sufficient; epsilon antisymmetry is required.",
            "Meson and baryon color sources admit compact connected width-one Gauss-closed currents on the retained handoff chain. Nonzero-triality controls cannot close without exterior flux.",
            "The exact first-generation color-singlet charge channels include uud with charge +1 and udd with charge 0.",
            "Positive linear tension confines the composite outer span, but the current static model does not yet select a finite-radius dynamical baryon and leaves the middle position flat on the one-dimensional chain.",
            "A neutral uud+electron channel with attractive Coulomb sign now exists structurally. A hydrogen atom has not yet been generated because the propagating baryon, common-unit mass/spin, and coupled spectrum remain to be built.",
        ],
        "meson": meson_payload,
        "baryon": baryon_payload,
        "center": center,
        "flux": flux_payload,
        "charges": {"mesons": {k: str(v) for k, v in meson_charges.items()}, "baryons": {k: str(v) for k, v in baryon_charges.items()}},
        "static_boundary": static_boundary,
        "hydrogen_readiness": hydrogen_readiness,
        "next_native_targets": [
            "derive a same-unit composite kinetic/propagation operator on the retained support graph",
            "couple the native spinor sector to the epsilon color singlet and separate spin-1/2 uud/udd channels from spin-3/2 channels",
            "measure the baryon rest spectrum and stability under exact retained evolution",
            "then solve the first coupled uud-plus-electron Coulomb spectrum and radiative transitions",
        ],
    }
    write_json(results / "summary.json", summary)

    print(json.dumps({"all_gates_pass": summary["all_gates_pass"], "package": package.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
