from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


TOLERANCE = 1.0e-12
GRID_SAMPLES = 73
GRID_MIN = -10.0 * math.pi
GRID_MAX = 10.0 * math.pi


def rotation_numeric(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def distorted_rotation_numeric(theta: float, eps: float = 0.05) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -(1.0 + eps) * s], [s, c]], dtype=float)


def frob_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def centered_slope(f, x: float, h: float = 1.0e-6) -> float:
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x)


def main() -> None:
    theta, phi = sp.symbols("theta phi", real=True)

    R_theta = sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])
    R_phi = sp.Matrix([[sp.cos(phi), -sp.sin(phi)], [sp.sin(phi), sp.cos(phi)]])
    R_sum = sp.Matrix(
        [[sp.cos(theta + phi), -sp.sin(theta + phi)], [sp.sin(theta + phi), sp.cos(theta + phi)]]
    )

    symbolic_composition = sp.simplify(R_sum - R_theta * R_phi)
    symbolic_half_turn = sp.simplify(R_theta.subs(theta, sp.pi) + sp.eye(2))
    symbolic_euler = sp.simplify(sp.exp(sp.I * sp.pi) + 1)
    symbolic_generator_square = sp.Matrix([[0, -1], [1, 0]]) ** 2 + sp.eye(2)

    angles = np.linspace(GRID_MIN, GRID_MAX, GRID_SAMPLES)
    composition_residual_max = max(
        frob_norm(rotation_numeric(a + b) - rotation_numeric(a) @ rotation_numeric(b))
        for a in angles
        for b in angles
    )
    orthogonality_residual_max = max(
        frob_norm(rotation_numeric(a).T @ rotation_numeric(a) - np.eye(2)) for a in angles
    )
    determinant_residual_max = max(abs(float(np.linalg.det(rotation_numeric(a))) - 1.0) for a in angles)
    halfturn_residual = frob_norm(rotation_numeric(math.pi) + np.eye(2))
    quarterturn_residual = frob_norm(rotation_numeric(math.pi / 2.0) - np.array([[0, -1], [1, 0]], dtype=float))
    euler_residual = float(abs(np.exp(1j * math.pi) + 1.0))
    large_angle_recurrence_residual = frob_norm(rotation_numeric(20.0 * math.pi) - np.eye(2))

    distorted_composition_residual_max = max(
        frob_norm(
            distorted_rotation_numeric(a + b)
            - distorted_rotation_numeric(a) @ distorted_rotation_numeric(b)
        )
        for a in angles
        for b in angles
    )
    off_target_halfturn_control = frob_norm(rotation_numeric(math.pi / 2.0) + np.eye(2))

    sin_slope_at_0 = centered_slope(math.sin, 0.0)
    sin_slope_at_pi = centered_slope(math.sin, math.pi)
    nonperiodic_single_cone_boundary_mismatch = 2.0 * math.pi

    z = np.linspace(-20.0, 20.0, 100001)
    wall_envelope = sech(z) / math.sqrt(2.0)
    wall_norm_squared = float(np.trapezoid(wall_envelope * wall_envelope, z))
    wall_norm_residual = abs(wall_norm_squared - 1.0)
    sech_normalization_factor = float(1.0 / math.sqrt(2.0))

    metrics = {
        "declared_tolerance": TOLERANCE,
        "grid": {
            "samples": GRID_SAMPLES,
            "min": GRID_MIN,
            "max": GRID_MAX,
        },
        "symbolic": {
            "composition_zero": bool(symbolic_composition == sp.zeros(2)),
            "half_turn_zero": bool(symbolic_half_turn == sp.zeros(2)),
            "euler_zero": bool(symbolic_euler == 0),
            "generator_square_zero": bool(symbolic_generator_square == sp.zeros(2)),
        },
        "positive_checks": {
            "composition_residual_max": float(composition_residual_max),
            "orthogonality_residual_max": float(orthogonality_residual_max),
            "determinant_residual_max": float(determinant_residual_max),
            "halfturn_residual": float(halfturn_residual),
            "quarterturn_residual": float(quarterturn_residual),
            "euler_residual": float(euler_residual),
            "large_angle_recurrence_residual": float(large_angle_recurrence_residual),
            "wall_norm_squared": wall_norm_squared,
            "wall_norm_residual": wall_norm_residual,
            "sech_normalization_factor": sech_normalization_factor,
        },
        "negative_controls": {
            "distorted_rotation_composition_residual_max": float(distorted_composition_residual_max),
            "off_target_halfturn_control_R_pi_over_2_plus_I": float(off_target_halfturn_control),
            "nonperiodic_single_cone_boundary_mismatch": float(nonperiodic_single_cone_boundary_mismatch),
        },
        "nielsen_pairing_toy": {
            "sin_slope_at_0": float(sin_slope_at_0),
            "sin_slope_at_pi": float(sin_slope_at_pi),
        },
        "acceptance_summary": {
            "all_positive_checks_below_tolerance": bool(
                composition_residual_max < TOLERANCE
                and orthogonality_residual_max < TOLERANCE
                and determinant_residual_max < TOLERANCE
                and halfturn_residual < TOLERANCE
                and quarterturn_residual < TOLERANCE
                and euler_residual < TOLERANCE
                and large_angle_recurrence_residual < TOLERANCE
                and wall_norm_residual < TOLERANCE
            ),
            "controls_far_from_tolerance": bool(
                distorted_composition_residual_max > 1.0e-3
                and off_target_halfturn_control > 1.0
                and nonperiodic_single_cone_boundary_mismatch > 1.0
            ),
        },
    }

    out = Path(__file__).resolve().parents[1] / "reports" / "cf13_sympy_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
