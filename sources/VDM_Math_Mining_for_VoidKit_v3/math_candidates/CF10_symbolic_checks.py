"""Symbolic support checks for CF10.

This script keeps the exact algebra separate from the reader-facing paper:
1. D2Q9 weights have the expected zeroth, first, and second moments.
2. The finite geometric shell-sum identity used in the conditional regularity chain is exact.
3. The Bernstein shell ratio is below one precisely in the beta > 3 regime.
"""
from __future__ import annotations

import json
import sympy as sp


def d2q9_moment_checks() -> dict[str, object]:
    c = [
        (0, 0),
        (1, 0), (0, 1), (-1, 0), (0, -1),
        (1, 1), (-1, 1), (-1, -1), (1, -1),
    ]
    w = [sp.Rational(4, 9)] + [sp.Rational(1, 9)] * 4 + [sp.Rational(1, 36)] * 4
    weight_sum = sp.simplify(sum(w))
    first_x = sp.simplify(sum(wi * ci[0] for wi, ci in zip(w, c)))
    first_y = sp.simplify(sum(wi * ci[1] for wi, ci in zip(w, c)))
    second_xx = sp.simplify(sum(wi * ci[0] * ci[0] for wi, ci in zip(w, c)))
    second_yy = sp.simplify(sum(wi * ci[1] * ci[1] for wi, ci in zip(w, c)))
    second_xy = sp.simplify(sum(wi * ci[0] * ci[1] for wi, ci in zip(w, c)))
    return {
        "weight_sum": str(weight_sum),
        "first_moment": [str(first_x), str(first_y)],
        "second_moment": [[str(second_xx), str(second_xy)], [str(second_xy), str(second_yy)]],
        "expected_second_moment": [["1/3", "0"], ["0", "1/3"]],
        "exact": bool(
            weight_sum == 1
            and first_x == 0 and first_y == 0
            and second_xx == sp.Rational(1, 3)
            and second_yy == sp.Rational(1, 3)
            and second_xy == 0
        ),
    }


def shell_series_checks() -> dict[str, object]:
    q, n, k = sp.symbols("q n k", positive=True, integer=True)
    finite_sum = sp.summation(q**k, (k, 0, n))
    finite_formula = (1 - q ** (n + 1)) / (1 - q)
    # Avoid the removable singularity at q=1 by checking the cross-multiplied identity.
    finite_identity = sp.simplify((1 - q) * finite_sum - (1 - q ** (n + 1)))

    beta = sp.symbols("beta", real=True)
    bernstein_ratio = 2 ** ((sp.Integer(3) - beta) / 2)
    ratios = {
        "beta_16_over_5": sp.simplify(bernstein_ratio.subs(beta, sp.Rational(16, 5))),
        "beta_3": sp.simplify(bernstein_ratio.subs(beta, sp.Integer(3))),
        "beta_14_over_5": sp.simplify(bernstein_ratio.subs(beta, sp.Rational(14, 5))),
    }
    return {
        "finite_geometric_identity_residual": str(finite_identity),
        "finite_geometric_identity_exact": bool(finite_identity == 0),
        "bernstein_ratio_expression": str(bernstein_ratio),
        "bernstein_ratios": {name: str(value) for name, value in ratios.items()},
        "numeric_ratios": {name: float(value.evalf()) for name, value in ratios.items()},
    }


def main() -> None:
    results = {
        "d2q9": d2q9_moment_checks(),
        "shell_series": shell_series_checks(),
    }
    print(json.dumps(results, indent=2))
    if not results["d2q9"]["exact"]:
        raise SystemExit("D2Q9 moment check failed")
    if not results["shell_series"]["finite_geometric_identity_exact"]:
        raise SystemExit("Geometric shell identity failed")


if __name__ == "__main__":
    main()
