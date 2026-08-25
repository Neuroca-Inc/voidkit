import math

import numpy as np
import pytest
from scipy.special import gamma

from voidkit.causal_inference import calculate_transfer_entropy
from voidkit.dynamical_systems import analyze_stability, find_fixed_points
from voidkit.fractal_analysis import (
    calculate_fractal_dimension,
    generate_fractal_spike_train,
)
from voidkit.fractional_calculus import caputo_derivative
from voidkit.info_theory import information_bottleneck
from voidkit.soc_analysis import detect_neuronal_avalanches
from voidkit.stochastic import gillespie_simulation
from voidkit.time_series import calculate_autocorrelation


def test_caputo_constant_is_zero():
    result = caputo_derivative(np.ones(32), alpha=0.5, dt=0.1)
    assert np.allclose(result, 0.0)


def test_caputo_linear_matches_closed_form():
    alpha = 0.4
    dt = 0.05
    t = np.arange(40, dtype=float) * dt
    result = caputo_derivative(t, alpha=alpha, dt=dt)
    expected = t ** (1.0 - alpha) / gamma(2.0 - alpha)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_fixed_point_solver_rejects_non_roots_and_deduplicates():
    none = find_fixed_points(lambda x: x**2 + 1.0, [np.array([0.0]), np.array([2.0])])
    assert none.shape == (0, 1)

    roots = find_fixed_points(
        lambda x: x**2 - 1.0,
        [np.array([-2.0]), np.array([2.0]), np.array([0.5])],
    )
    assert roots.shape == (2, 1)
    assert np.sort(roots[:, 0]) == pytest.approx([-1.0, 1.0])


def test_nonhyperbolic_stability_is_not_reported_as_center():
    result = analyze_stability(np.diag([-1.0, 0.0]))
    assert result["hyperbolic"] is False
    assert result["stability_type"] == "Nonhyperbolic (Linearization Inconclusive)"


def test_gillespie_never_executes_after_horizon():
    rng = np.random.default_rng(7)
    times, states = gillespie_simulation(
        np.array([0], dtype=int),
        lambda _: np.array([1e-6]),
        np.array([[1]], dtype=int),
        t_max=0.01,
        rng=rng,
    )
    assert np.all(times <= 0.01)
    assert times.tolist() == [0.0]
    assert states[:, 0].tolist() == [0]


def test_box_counting_recovers_line_and_plane_dimensions():
    x = np.linspace(0.0, 1.0, 4096, endpoint=False)
    line = np.column_stack([x, np.zeros_like(x)])
    assert calculate_fractal_dimension(line) == pytest.approx(1.0, abs=0.05)

    axis = np.linspace(0.0, 1.0, 64, endpoint=False)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    plane = np.column_stack([xx.ravel(), yy.ravel()])
    assert calculate_fractal_dimension(plane) == pytest.approx(2.0, abs=0.05)


def test_fractal_spike_train_handles_nonintegral_duration_grid():
    rng = np.random.default_rng(1)
    spikes = generate_fractal_spike_train(1.5, 10.0, 100.0, 1.0, dt=0.3, rng=rng)
    assert spikes.ndim == 1
    assert np.all(spikes >= 0.0)
    assert np.all(spikes < 1.0)


def test_time_zero_spike_forms_an_avalanche():
    result = detect_neuronal_avalanches(np.array([0.0]), bin_width=1.0)
    assert result == {"sizes": [1], "durations": [1]}


def test_constant_autocorrelation_is_defined():
    result = calculate_autocorrelation(np.ones(8))
    assert np.array_equal(result, np.zeros(8))


def test_transfer_entropy_detects_delayed_source_and_stays_in_bounds():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2, 1000).astype(float)
    y = np.zeros_like(x)
    y[1:] = x[:-1]

    forward = calculate_transfer_entropy(x, y, lag=1, n_bins=2)
    reverse = calculate_transfer_entropy(y, x, lag=1, n_bins=2)
    assert forward > 0.95
    assert reverse < 0.05


def test_information_bottleneck_uses_t_y_relevance_term():
    p_xy = np.array([[0.5, 0.0], [0.0, 0.5]])
    p_xt = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert information_bottleneck(p_xy, p_xt, beta=1.0) == pytest.approx(0.0)

    compressed = np.array([[0.5], [0.5]])
    assert information_bottleneck(p_xy, compressed, beta=1.0) == pytest.approx(0.0)
