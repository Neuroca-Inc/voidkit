#!/usr/bin/env python3
"""CF07 retained-measurement companion checks.

The checks are intentionally small and explicit. They attack the formal burden of
CF07: non-injective visible projection, state-only selector failure under branch
symmetry, PSD metric contraction, entropy-producing dephasing, pointer residuals,
neutral frequency convergence, non-neutral response normalization, and finite-
resolution sinc suppression.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


def trace_norm_2x2_hermitian(a: complex, b: complex, d: complex) -> float:
    """Trace norm of [[a,b],[conj(b),d]] for Hermitian 2x2."""
    mat = np.array([[a, b], [np.conjugate(b), d]], dtype=complex)
    vals = np.linalg.eigvalsh(mat)
    return float(np.sum(np.abs(vals)))


def kl(f: np.ndarray, p: np.ndarray) -> float:
    mask = f > 0
    return float(np.sum(f[mask] * np.log(f[mask] / p[mask])))


def main() -> dict[str, float]:
    results: dict[str, float] = {}

    # 1. Projection non-injectivity: two retained fibers share one rho.
    rho_visible = (sp.Rational(1, 2), sp.Rational(1, 2))
    retained_a = (rho_visible, "env=A", "branch=left")
    retained_b = (rho_visible, "env=B", "branch=right")
    results["visible_collision_count"] = float(int(retained_a != retained_b and retained_a[0] == retained_b[0]))

    # 2. State-only selector no-go witness: selector chooses 0, swap sends to 1.
    selector_value = 0
    swap_value = 1 - selector_value
    results["branch_swap_covariance_residual"] = float(abs(selector_value - swap_value))

    # 3. M-limb PSD contraction matrix.
    Gamma = sp.Matrix([[1, -1, 0], [0, 1, -1]])
    M = Gamma.T * Gamma
    eigs = [ev for ev in M.eigenvals().keys()]
    results["M_min_eigenvalue"] = float(min([sp.N(ev) for ev in eigs]))

    # 4. Entropy-producing dephasing path.
    gamma = 2.0
    t_grid = np.linspace(0.0, 4.0, 4001)
    entropies = []
    D_numeric = []
    D_formula = []
    for t in t_grid:
        off = math.exp(-gamma * t) / 2.0
        rho = np.array([[0.5, off], [off, 0.5]], dtype=float)
        vals = np.linalg.eigvalsh(rho)
        vals = np.maximum(vals, 1e-300)
        entropies.append(float(-np.sum(vals * np.log(vals))))
        diff = np.array([[0.0, off], [off, 0.0]], dtype=float)
        D_numeric.append(0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(diff)))))
        D_formula.append(0.5 * math.exp(-gamma * t))
    entropies = np.array(entropies)
    results["entropy_sign_violations"] = float(np.sum(np.diff(entropies) < -1e-12))
    results["dephasing_formula_residual"] = float(np.max(np.abs(np.array(D_numeric) - np.array(D_formula))))
    eps_dec = 0.05
    tau_exact = math.log(0.5 / eps_dec) / gamma
    tau_grid = t_grid[np.where(np.array(D_numeric) <= eps_dec)[0][0]]
    results["tau_grid_error"] = float(abs(tau_grid - tau_exact))

    # 5. Pointer commutator residuals.
    sigma_z = sp.Matrix([[1, 0], [0, -1]])
    Pz0 = sp.Matrix([[1, 0], [0, 0]])
    Pz1 = sp.Matrix([[0, 0], [0, 1]])
    # x-like basis projectors
    sx_plus = sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 2), sp.Rational(1, 2)]])
    sx_minus = sp.Matrix([[sp.Rational(1, 2), -sp.Rational(1, 2)], [-sp.Rational(1, 2), sp.Rational(1, 2)]])

    def spectral_norm_sympy_2(mat: sp.Matrix) -> float:
        arr = np.array(mat.tolist(), dtype=float)
        return float(np.linalg.norm(arr, 2))

    results["pointer_residual_z"] = float(max(spectral_norm_sympy_2(sigma_z * P - P * sigma_z) for P in [Pz0, Pz1]))
    results["pointer_residual_x"] = float(max(spectral_norm_sympy_2(sigma_z * P - P * sigma_z) for P in [sx_plus, sx_minus]))

    # 6. Neutral retained-frequency law simulation.
    rng = np.random.default_rng(12345)
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    N = 200_000
    sample = rng.multinomial(N, w) / N
    results["neutral_frequency_max_error"] = float(np.max(np.abs(sample - w)))

    # KL sampling summaries.
    p = np.array([0.2, 0.3, 0.5], dtype=float)
    wrong = np.array([0.05, 0.25, 0.70], dtype=float)
    Nkl = 20_000
    kls_good = []
    kls_wrong = []
    for _ in range(300):
        f_good = rng.multinomial(Nkl, p) / Nkl
        f_wrong = rng.multinomial(Nkl, wrong) / Nkl
        kls_good.append(kl(f_good, p))
        kls_wrong.append(kl(f_wrong, p))
    results["median_KL_neutral"] = float(np.median(kls_good))
    results["median_KL_wrong"] = float(np.median(kls_wrong))

    Ns = np.array([500, 1000, 2000, 5000, 10000, 20000, 50000], dtype=int)
    means = []
    for n in Ns:
        vals = [kl(rng.multinomial(int(n), p) / int(n), p) for _ in range(2000)]
        means.append(float(np.mean(vals)))
    means = np.array(means)
    slope, intercept = np.polyfit(np.log(Ns), np.log(means), 1)
    expected = (len(p) - 1) / (2 * Ns)
    results["KL_loglog_slope"] = float(slope)
    results["KL_planning_max_relative_error"] = float(np.max(np.abs(means - expected) / expected))

    # 7. Non-neutral apparatus response.
    S = np.array([[0.85, 0.10, 0.05], [0.10, 0.80, 0.15], [0.05, 0.10, 0.80]], dtype=float)
    # columns sum to one
    p_meas = S @ w
    results["nonneutral_normalization_error"] = float(abs(np.sum(p_meas) - 1.0))
    results["nonneutral_min_weight"] = float(np.min(p_meas))

    # 8. Finite-resolution sinc suppression: numerical quadrature top-hat vs sinc.
    dx = 3.7
    ks = np.linspace(0.1, 12.0, 200)
    residuals = []
    xs = np.linspace(-dx / 2, dx / 2, 20001)
    W = np.ones_like(xs) / dx
    for k in ks:
        numeric = np.trapezoid(W * np.exp(1j * k * xs), xs)
        analytic = np.sinc(k * dx / (2 * np.pi))
        residuals.append(abs(numeric - analytic))
    results["sinc_quadrature_residual"] = float(np.max(residuals))

    # 9. Born recognition symbolic identity in a two-sector diagonal chart.
    r, a, b = sp.symbols("r a b", nonnegative=True, real=True)
    rho = sp.diag(r, 1 - r)
    Pi0 = sp.diag(1, 0)
    Pi1 = sp.diag(0, 1)
    born_sum = sp.trace(Pi0 * rho) + sp.trace(Pi1 * rho)
    results["born_trace_normalization_symbolic"] = float(sp.simplify(born_sum - 1))

    return results


if __name__ == "__main__":
    results = main()
    out = Path("/mnt/data/cf07_work/cf07_sympy_attack_results.json")
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
