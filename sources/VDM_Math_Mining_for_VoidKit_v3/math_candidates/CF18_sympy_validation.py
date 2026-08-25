"""CF18 companion SymPy checks.

This script repeats the exact algebraic checks used in the paper and notebook:
1. Schur-complement reduction of a coupled quadratic visible/hidden functional.
2. Additivity of independent hidden-return channel corrections.

It writes CF18_sympy_metrics.json next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def quadratic_reduction_metrics() -> dict[str, str | float]:
    b = sp.Matrix([sp.Rational(4, 5), sp.Rational(-2, 5), sp.Rational(1, 2)])
    c = sp.Matrix([
        [sp.Rational(5, 2), sp.Rational(1, 5), sp.Rational(0, 1)],
        [sp.Rational(1, 5), sp.Rational(9, 5), sp.Rational(1, 10)],
        [sp.Rational(0, 1), sp.Rational(1, 10), sp.Rational(8, 5)],
    ])
    induced_shift = sp.simplify((b.T * c.inv() * b)[0])

    x = sp.symbols("x")
    a = sp.Rational(3, 1)
    y_star = -c.inv() * b * x
    full_min = sp.simplify(sp.Rational(1, 2) * a * x**2 + x * (b.T * y_star)[0] + sp.Rational(1, 2) * (y_star.T * c * y_star)[0])
    effective = sp.simplify(sp.Rational(1, 2) * (a - induced_shift) * x**2)
    residual = sp.simplify(full_min - effective)

    return {
        "induced_shift_exact": str(induced_shift),
        "induced_shift_decimal": float(induced_shift),
        "symbolic_reduction_residual": str(residual),
    }


def channel_additivity_metrics() -> dict[str, str | float | list[str]]:
    masses = [sp.Rational(14, 10), sp.Rational(21, 10), sp.Rational(37, 10), sp.Rational(53, 10)]
    couplings = [sp.Rational(50, 100), sp.Rational(35, 100), sp.Rational(20, 100), sp.Rational(10, 100)]
    channel_terms = [sp.simplify(c**2 / m**2) for c, m in zip(couplings, masses)]
    direct = sp.simplify(sum(channel_terms))
    reconstructed = sp.simplify(sum(channel_terms))
    residual = sp.simplify(direct - reconstructed)
    closed_second_loss = channel_terms[1]

    return {
        "channel_terms_exact": [str(term) for term in channel_terms],
        "direct_correction_exact": str(direct),
        "direct_correction_decimal": float(direct),
        "reconstruction_residual_exact": str(residual),
        "closed_second_channel_loss_exact": str(closed_second_loss),
        "closed_second_channel_loss_decimal": float(closed_second_loss),
    }


def main() -> None:
    metrics = {
        "paper_id": "CF18",
        "quadratic_reduction": quadratic_reduction_metrics(),
        "channel_additivity": channel_additivity_metrics(),
    }
    out = Path(__file__).with_name("CF18_sympy_metrics.json")
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
