from __future__ import annotations

import math

import numpy as np
import pytest

from voidkit.algebra.free_group import commutator_word, free_reduce, inverse_word
from voidkit.algebra.heisenberg import HeisenbergState, commutator, inverse, multiply
from voidkit.algebra.permutation import Permutation
from voidkit.dynamics.logistic import exact_step, invariant_q
from voidkit.numerical.interval_roots import bisect_real, lambertw_real_bracket, x_plus_sin_bracket
from voidkit.numerical.spectral import spectral_gradient, spectral_laplacian
from voidkit.spatial.yinyang import create_component_grid, yang_to_yin, yin_to_yang
from voidkit.wave.klein_gordon import energy_norm_delta, verlet_step


def _cartesian(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )


def test_bisect_real_sqrt2_and_invalid_bracket() -> None:
    root, left, right, steps = bisect_real(lambda x: x * x - 2.0, 1.0, 2.0, tolerance=1e-13)
    assert abs(root - math.sqrt(2.0)) < 2e-13
    assert left <= math.sqrt(2.0) <= right
    assert steps > 0
    with pytest.raises(ValueError):
        bisect_real(lambda x: x * x + 1.0, -1.0, 1.0, tolerance=1e-6)
    with pytest.raises(ValueError):
        bisect_real(lambda x: x, -1.0, 1.0, tolerance=0.0)


def test_special_root_brackets_change_sign() -> None:
    for y, branch in [(2.0, 0), (-0.1, 0), (-0.1, -1)]:
        left, right = lambertw_real_bracket(y, branch)
        fl = left * math.exp(left) - y
        fr = right * math.exp(right) - y
        assert fl == 0.0 or fr == 0.0 or fl * fr <= 0.0
    y = 4.25
    left, right = x_plus_sin_bracket(y)
    f = lambda x: x + math.sin(x) - y
    assert f(left) <= 0.0 <= f(right)


def test_permutation_composition_inverse_and_cycles() -> None:
    p = Permutation((1, 2, 0, 3))
    ident = Permutation.identity(4)
    assert p.compose(p.inverse()) == ident
    assert p.inverse().compose(p) == ident
    assert p.cycles() == ((0, 1, 2),)
    with pytest.raises(ValueError):
        Permutation((0, 0))


def test_free_group_reduction_and_inverse() -> None:
    word = ("a", "b", "b^-1", "a^-1", "c")
    assert free_reduce(word) == ("c",)
    base = ("a", "b", "c^-1")
    assert free_reduce(base + inverse_word(base)) == ()
    assert commutator_word("a", "b") == ("a", "b", "a^-1", "b^-1")


def test_heisenberg_group_inverse_and_commutator_center() -> None:
    identity = HeisenbergState(0, 0, 0)
    a = HeisenbergState(2, 3, 5)
    b = HeisenbergState(-4, 7, 1)
    assert multiply(a, inverse(a)) == identity
    assert multiply(inverse(a), a) == identity
    c = commutator(a, b)
    assert c.m == 0 and c.n == 0
    assert c.omega == a.m * b.n - b.m * a.n


def test_spectral_derivatives_on_periodic_mode() -> None:
    n = 128
    length = 2.0 * math.pi
    dx = length / n
    x = np.arange(n) * dx
    values = np.sin(3.0 * x)
    grad = spectral_gradient(values, dx)
    lap = spectral_laplacian(values, dx)
    assert np.max(np.abs(grad - 3.0 * np.cos(3.0 * x))) < 2e-12
    assert np.max(np.abs(lap + 9.0 * values)) < 2e-11


def test_kg_helpers_zero_distance_and_bounded_short_step_energy_error() -> None:
    n = 128
    length = 2.0 * math.pi
    dx = length / n
    x = np.arange(n) * dx
    phi = np.sin(x)
    momentum = np.zeros_like(phi)
    assert energy_norm_delta(phi, momentum, phi, momentum, dx, 1.0, 0.5) == 0.0

    def energy(p: np.ndarray, q: np.ndarray) -> float:
        grad = spectral_gradient(p, dx)
        return 0.5 * float(np.sum(q*q + grad*grad + 0.25*p*p) * dx)

    e0 = energy(phi, momentum)
    p, q = phi.copy(), momentum.copy()
    for _ in range(100):
        p, q = verlet_step(p, q, 1e-3, dx, 1.0, 0.5)
    assert abs(energy(p, q) - e0) / e0 < 1e-6


def test_yinyang_roundtrip_in_cartesian_coordinates() -> None:
    theta, phi = create_component_grid(17, 31)
    theta_e, phi_e = yin_to_yang(theta, phi)
    theta_n, phi_n = yang_to_yin(theta_e, phi_e)
    original = _cartesian(theta, phi)
    recovered = _cartesian(theta_n, phi_n)
    assert np.max(np.abs(original - recovered)) < 2e-15


def test_logistic_exact_flow_composition_and_invariant() -> None:
    r = 0.15
    u = 0.25
    w0 = np.array([0.12, 0.3, 0.55])
    full = exact_step(w0, r, u, 10.0)
    split = w0.copy()
    for _ in range(1000):
        split = exact_step(split, r, u, 0.01)
    assert np.max(np.abs(full - split)) < 5e-13
    q0 = invariant_q(w0, r, u, 0.0)
    q1 = invariant_q(full, r, u, 10.0)
    assert np.max(np.abs(q1 - q0)) < 5e-13


def test_logistic_linear_limit() -> None:
    w0 = np.array([0.1, 0.5])
    r = 0.25
    dt = 1.0
    result = exact_step(w0, r, 0.0, dt)
    expected = w0 * np.exp(r * dt)
    assert np.max(np.abs(result - expected)) < 1e-14
