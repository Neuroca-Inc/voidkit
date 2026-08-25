import json
from fractions import Fraction
from pathlib import Path

import sympy as sp

OUT = Path('/mnt/data/shadow_residual_v59_correction/outputs')
OUT.mkdir(parents=True, exist_ok=True)


def chi12(n: int) -> int:
    r = n % 12
    if r in (1, 11):
        return 1
    if r in (5, 7):
        return -1
    return 0


def positive_half_coeffs(max_exp: int) -> dict[int, int]:
    d: dict[int, int] = {}
    n = 1
    while n * n <= max_exp:
        c = chi12(n)
        if c:
            d[n * n] = d.get(n * n, 0) + c
        n += 1
    return d


def shadow_residual_coeffs(max_exp: int) -> dict[int, int]:
    d: dict[int, int] = {}
    n = 1
    while n * n <= max_exp:
        c = chi12(n)
        if c:
            d[n * n] = d.get(n * n, 0) + c * n
        n += 1
    return d


def eta_product_coeffs(max_exp: int) -> dict[int, int]:
    t = sp.Symbol('t')
    expr = t
    max_m = max_exp // 24 + 2
    for m in range(1, max_m + 1):
        expr = sp.series(expr * (1 - t ** (24 * m)), t, 0, max_exp + 1).removeO().expand()
    poly = sp.Poly(expr, t)
    return {int(exp[0]): int(coeff) for exp, coeff in poly.terms() if exp[0] <= max_exp and coeff != 0}


def bilateral_derivative_symmetric(max_n: int) -> int:
    return sum(chi12(n) * n for n in range(-max_n, max_n + 1))


def bilateral_shadow_symmetric(max_exp: int) -> dict[int, int]:
    d: dict[int, int] = {}
    nmax = int(max_exp ** 0.5)
    for n in range(-nmax, nmax + 1):
        if n == 0:
            continue
        c = chi12(n)
        if c and n * n <= max_exp:
            d[n * n] = d.get(n * n, 0) + c
    return d


def main() -> None:
    max_exp = 1600
    pos = positive_half_coeffs(max_exp)
    eta = eta_product_coeffs(max_exp)
    residual = shadow_residual_coeffs(max_exp)
    bilateral_shadow = bilateral_shadow_symmetric(max_exp)

    all_exps = sorted(set(pos) | set(eta))
    mismatches = [(e, pos.get(e, 0), eta.get(e, 0)) for e in all_exps if pos.get(e, 0) != eta.get(e, 0)]

    derivative_checks = {N: bilateral_derivative_symmetric(N) for N in [1, 5, 12, 37, 89, 144]}
    derivative_pass = all(v == 0 for v in derivative_checks.values())

    bilateral_normalization_mismatches = []
    for e, c in bilateral_shadow.items():
        if c != 2 * pos.get(e, 0):
            bilateral_normalization_mismatches.append((e, c, pos.get(e, 0)))

    summary = {
        'max_t_exponent_checked': max_exp,
        'eta_positive_half_coefficients_match': len(mismatches) == 0,
        'eta_positive_half_mismatch_count': len(mismatches),
        'bilateral_derivative_cancels_for_symmetric_windows': derivative_pass,
        'bilateral_derivative_checks': derivative_checks,
        'bilateral_n0_theta_is_twice_positive_half': len(bilateral_normalization_mismatches) == 0,
        'bilateral_normalization_mismatch_count': len(bilateral_normalization_mismatches),
        'positive_half_nonzero_terms_checked': len(pos),
        'shadow_residual_nonzero_terms_checked': len(residual),
        'first_positive_half_terms_t_exponent_coeff': sorted(pos.items())[:16],
        'first_shadow_residual_terms_t_exponent_coeff': sorted(residual.items())[:16],
        'status': 'PASS' if len(mismatches) == 0 and derivative_pass and len(bilateral_normalization_mismatches) == 0 and len(residual) > 0 else 'FAIL'
    }

    (OUT / 'shadow_residual_eta_summary_v59.json').write_text(json.dumps(summary, indent=2))

    lines = ['series, t_exponent, q_exponent, coefficient']
    for e, c in sorted(pos.items()):
        lines.append(f'eta_positive_half,{e},{Fraction(e,24)},{c}')
    for e, c in sorted(residual.items()):
        lines.append(f'shadow_residual_R,{e},{Fraction(e,24)},{c}')
    (OUT / 'shadow_residual_eta_coefficients_v59.csv').write_text('\n'.join(lines) + '\n')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
