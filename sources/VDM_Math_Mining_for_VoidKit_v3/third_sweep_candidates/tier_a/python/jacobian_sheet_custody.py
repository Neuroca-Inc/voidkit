#!/usr/bin/env python3
"""Exact Jacobian sheet-custody benchmark.

The polynomial map is external mathematics. Phase Calculus contributes the
retained-state interpretation and the projection-ablation comparison.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import sympy as sp

x, y, z, lam = sp.symbols("x y z lam", nonzero=False)
u = 1 + x * y
A = sp.expand(u**3 * z + y**2 * u * (4 + 3 * x * y))
B = sp.expand(y + 3 * x * u**2 * z + 3 * x * y**2 * (4 + 3 * x * y))
C = sp.expand(2 * x - 3 * x**2 * y - x**3 * z)
F = sp.Matrix([A, B, C])

v = sp.expand(x * y)
t = sp.expand(x**2 * z)
gamma = sp.expand(1 - sp.Rational(3, 2) * v - sp.Rational(1, 2) * t)
w = sp.expand((1 + v) * gamma)
phi = sp.expand(w**2 - w**3)
P_target = sp.expand(B * C / 4)
Q_target = sp.expand(A * C**2 / 4)

P0 = (sp.Rational(0), sp.Rational(0), -sp.Rational(1, 4))
PPLUS = (sp.Rational(1), -sp.Rational(3, 2), sp.Rational(13, 2))
PMINUS = (-sp.Rational(1), sp.Rational(3, 2), sp.Rational(13, 2))
POINTS = {"p0": P0, "p_plus": PPLUS, "p_minus": PMINUS}


@dataclass(frozen=True)
class Witness:
    name: str
    input: tuple[str, str, str]
    output: tuple[str, str, str]
    v: str
    t: str
    gamma: str
    w: str
    x_oriented: str
    orientation_sign: int


def exact_tuple(values: Iterable[sp.Expr]) -> tuple[str, ...]:
    return tuple(str(sp.simplify(value)) for value in values)


def eval_expr(expr: sp.Expr | sp.Matrix, point: tuple[sp.Rational, sp.Rational, sp.Rational]):
    substitutions = {x: point[0], y: point[1], z: point[2]}
    if isinstance(expr, sp.MatrixBase):
        return tuple(sp.simplify(value) for value in expr.subs(substitutions))
    return sp.simplify(expr.subs(substitutions))


def sign_of(value: sp.Expr) -> int:
    value = sp.simplify(value)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def make_witness(name: str, point: tuple[sp.Rational, sp.Rational, sp.Rational]) -> Witness:
    output = eval_expr(F, point)
    x_value = point[0]
    return Witness(
        name=name,
        input=exact_tuple(point),
        output=exact_tuple(output),
        v=str(eval_expr(v, point)),
        t=str(eval_expr(t, point)),
        gamma=str(eval_expr(gamma, point)),
        w=str(eval_expr(w, point)),
        x_oriented=str(x_value),
        orientation_sign=sign_of(x_value),
    )


def equivalence_classes(records: list[Witness], fields: tuple[str, ...]) -> list[list[str]]:
    groups: dict[tuple[Any, ...], list[str]] = {}
    for record in records:
        key = tuple(getattr(record, field) for field in fields)
        groups.setdefault(key, []).append(record.name)
    return sorted((sorted(names) for names in groups.values()), key=lambda names: (len(names), names))


def verify() -> dict[str, Any]:
    determinant = sp.factor(F.jacobian((x, y, z)).det())
    fiber_identity = sp.factor(phi - (w * P_target - Q_target))

    scaled = F.subs({x: lam * x, y: y / lam, z: z / lam**2}, simultaneous=True)
    expected_scaled = sp.Matrix([A / lam**2, B / lam, lam * C])
    scaling_residuals = [sp.factor(sp.together(a - b)) for a, b in zip(scaled, expected_scaled)]
    w_scaled = sp.factor(sp.together(w.subs({x: lam * x, y: y / lam, z: z / lam**2}, simultaneous=True) - w))

    records = [make_witness(name, point) for name, point in POINTS.items()]
    outputs = {record.output for record in records}

    projection_classes = equivalence_classes(records, ("output",))
    root_sheet_classes = equivalence_classes(records, ("output", "w"))
    full_classes = equivalence_classes(records, ("output", "w", "x_oriented"))
    orientation_erased_classes = equivalence_classes(records, ("output", "w", "orientation_sign"))

    checks = {
        "J1_constant_jacobian": determinant == -2,
        "J2_three_point_collision": len(outputs) == 1 and next(iter(outputs)) == ("-1/4", "0", "0"),
        "J3_weighted_fiber_identity": fiber_identity == 0,
        "J4_output_only_collapses_three_to_one": len(projection_classes) == 1,
        "J5_root_sheet_separates_flat_from_curved": len(root_sheet_classes) == 2,
        "J6_oriented_coordinate_separates_all_three": len(full_classes) == 3,
        "J7_weighted_scaling_covariance": all(residual == 0 for residual in scaling_residuals),
        "J8_w_is_scaling_invariant": w_scaled == 0,
        "negative_control_abs_orientation_collapses_lobes": len(orientation_erased_classes) == 3,
    }

    # The negative control above deliberately uses sign, which still separates p+ and p-.
    # A true orientation-erasing control uses |x| and must collapse the two curved lobes.
    abs_groups: dict[tuple[Any, ...], list[str]] = {}
    for record in records:
        key = (record.output, record.w, str(abs(sp.Rational(record.x_oriented))))
        abs_groups.setdefault(key, []).append(record.name)
    abs_classes = sorted((sorted(vs) for vs in abs_groups.values()), key=lambda names: (len(names), names))
    checks["negative_control_absolute_x_collapses_curved_lobes"] = len(abs_classes) == 2
    checks.pop("negative_control_abs_orientation_collapses_lobes")

    return {
        "schema_id": "phase-calculus.jacobian-sheet-custody-certificate",
        "schema_version": 1,
        "map": {
            "A": str(A),
            "B": str(B),
            "C": str(C),
            "jacobian_determinant": str(determinant),
        },
        "weighted_coordinates": {
            "v": str(v),
            "t": str(t),
            "gamma": str(gamma),
            "w": str(w),
            "fiber_identity_residual": str(fiber_identity),
            "fiber_equation": "w^2 - w^3 = w*(B*C/4) - A*C^2/4",
        },
        "witnesses": [asdict(record) for record in records],
        "ablation": {
            "visible_output_only": projection_classes,
            "visible_plus_root_sheet_w": root_sheet_classes,
            "visible_plus_w_plus_oriented_x": full_classes,
            "visible_plus_w_plus_absolute_x_negative_control": abs_classes,
            "distinguishable_class_counts": [
                {"retained_fields": "(A,B,C)", "count": len(projection_classes)},
                {"retained_fields": "(A,B,C,w)", "count": len(root_sheet_classes)},
                {"retained_fields": "(A,B,C,w,x)", "count": len(full_classes)},
            ],
        },
        "scaling": {
            "law": "F(lambda*x, y/lambda, z/lambda^2) = (A/lambda^2, B/lambda, lambda*C)",
            "component_residuals": [str(value) for value in scaling_residuals],
            "w_invariance_residual": str(w_scaled),
        },
        "checks": checks,
        "final_result": "PASS" if all(checks.values()) else "FAIL",
    }


def write_outputs(base: Path) -> dict[str, Any]:
    result = verify()
    base.mkdir(parents=True, exist_ok=True)
    (base / "exact_certificates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (base / "ablation_ladder.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["retained_fields", "count"])
        writer.writeheader()
        writer.writerows(result["ablation"]["distinguishable_class_counts"])
    with (base / "collision_witnesses.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = result["witnesses"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> int:
    here = Path(__file__).resolve()
    release_root = here.parents[2]
    result = write_outputs(release_root / "results")
    for name, passed in result["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"FINAL_RESULT: {result['final_result']}")
    return 0 if result["final_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
