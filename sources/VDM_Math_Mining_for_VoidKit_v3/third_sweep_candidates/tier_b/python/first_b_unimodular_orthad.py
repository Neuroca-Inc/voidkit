from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import sympy as sp


def matrix_list(matrix: sp.Matrix) -> list[list[int]]:
    return [[int(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)]


def calculate() -> dict:
    J = sp.Matrix([[0, 1], [-1, 0]])
    ell = sp.Matrix([0, 1])
    r = sp.Matrix([1, 1])

    C0_plus = sp.Matrix.hstack(ell, r)
    C0_minus = sp.Matrix.hstack(r, ell)

    R_B = sp.Matrix([[1, 1], [0, 1]])
    m = ell + r

    C1_plus = C0_plus * R_B
    C1_minus = C0_minus * R_B

    Omega0_plus = C0_plus.T * J * C0_plus
    Omega0_minus = C0_minus.T * J * C0_minus
    T0_plus_to_minus = C0_minus.T * J * C0_plus
    T0_minus_to_plus = C0_plus.T * J * C0_minus

    Omega1_plus = C1_plus.T * J * C1_plus
    Omega1_minus = C1_minus.T * J * C1_minus
    T1_plus_to_minus = C1_minus.T * J * C1_plus
    T1_minus_to_plus = C1_plus.T * J * C1_minus

    q0 = (int(ell[1]), int(r[1]))
    q1 = tuple(sorted((int(r[1]), int(ell[1] + r[1]))))
    frame_q1_plus = tuple(int(C1_plus[1, j]) for j in range(2))
    frame_q1_minus = tuple(int(C1_minus[1, j]) for j in range(2))

    parent_width = Fraction(1, 1)
    left_width = Fraction(1, 2)
    right_width = Fraction(1, 2)

    wrong_midpoint_numerator = sp.Matrix([0, 2])
    wrong_midpoint_denominator = sp.Matrix([1, 3])
    same_orientation_minus = C0_plus
    one_chart_only_minus = C0_minus

    checks = {
        "seed_adjacent_relation_abs": abs(int((ell.T * J * r)[0])) == 1,
        "B_shear_unimodular": int(R_B.det()) == 1,
        "mediant_vector": m == sp.Matrix([1, 2]),
        "primitive_pair_update": q0 == (1, 1) and q1 == (1, 2),
        "plus_frame_pair_update": frame_q1_plus == (1, 2),
        "minus_frame_pair_update": frame_q1_minus == (1, 2),
        "common_relation_preserved": R_B.T * J * R_B == J,
        "plus_chart_update": Omega1_plus == R_B.T * Omega0_plus * R_B,
        "minus_chart_update": Omega1_minus == R_B.T * Omega0_minus * R_B,
        "plus_to_minus_transfer_update": (
            T1_plus_to_minus == R_B.T * T0_plus_to_minus * R_B
        ),
        "minus_to_plus_transfer_update": (
            T1_minus_to_plus == R_B.T * T0_minus_to_plus * R_B
        ),
        "counter_orientation": Omega1_minus == -Omega1_plus,
        "directed_transfer_reciprocity": (
            T1_minus_to_plus == -T1_plus_to_minus.T
        ),
        "child_determinant_left": abs(int((ell.T * J * m)[0])) == 1,
        "child_determinant_right": abs(int((m.T * J * r)[0])) == 1,
        "width_retention": left_width + right_width == parent_width,
    }

    negative_controls = {
        "wrong_midpoint_numerator_rejected": not (
            abs(int((ell.T * J * wrong_midpoint_numerator)[0])) == 1
            and abs(int((wrong_midpoint_numerator.T * J * r)[0])) == 1
        ),
        "wrong_midpoint_denominator_rejected": not (
            abs(int((ell.T * J * wrong_midpoint_denominator)[0])) == 1
            and abs(int((wrong_midpoint_denominator.T * J * r)[0])) == 1
            and int(wrong_midpoint_denominator[1]) == q1[1]
        ),
        "same_orientation_rejected": (
            same_orientation_minus.T * J * same_orientation_minus
            != -Omega0_plus
        ),
        "one_chart_only_rejected": (
            tuple(int(one_chart_only_minus[1, j]) for j in range(2))
            != frame_q1_plus
        ),
    }

    result = {
        "claim_scope": "first-B unimodular alternating Orthad sector",
        "inputs": {
            "ell": [0, 1],
            "r": [1, 1],
            "q0": [1, 1],
            "selected_primitive": "B",
        },
        "recurrence": {
            "J": matrix_list(J),
            "R_B": matrix_list(R_B),
            "det_R_B": int(R_B.det()),
        },
        "outputs": {
            "m": [int(m[0]), int(m[1])],
            "mediant": "1/2",
            "q1": list(q1),
            "C0_plus": matrix_list(C0_plus),
            "C0_minus": matrix_list(C0_minus),
            "C1_plus": matrix_list(C1_plus),
            "C1_minus": matrix_list(C1_minus),
            "Omega0_plus": matrix_list(Omega0_plus),
            "Omega0_minus": matrix_list(Omega0_minus),
            "Omega1_plus": matrix_list(Omega1_plus),
            "Omega1_minus": matrix_list(Omega1_minus),
            "T0_plus_to_minus": matrix_list(T0_plus_to_minus),
            "T0_minus_to_plus": matrix_list(T0_minus_to_plus),
            "T1_plus_to_minus": matrix_list(T1_plus_to_minus),
            "T1_minus_to_plus": matrix_list(T1_minus_to_plus),
            "parent_width": str(parent_width),
            "left_child_width": str(left_width),
            "right_child_width": str(right_width),
        },
        "checks": checks,
        "negative_controls": negative_controls,
        "computed_verdict": (
            "PASS"
            if all(checks.values()) and all(negative_controls.values())
            else "FAIL"
        ),
    }
    return result


def main() -> None:
    result = calculate()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["computed_verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
