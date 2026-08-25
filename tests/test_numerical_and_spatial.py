import numpy as np
import pytest

from voidkit.clustering import spectral_clustering_with_temporal_kernel
from voidkit.numerical_integrate import numerical_integrate
from voidkit.numerical_ode_solver import numerical_ode_solver
from voidkit.sde import sde_solver
from voidkit.spatial import SpatialHashGrid


def test_numerical_integrate_supports_reversed_and_equal_bounds():
    forward, _ = numerical_integrate(lambda x: x * x, 0.0, 1.0)
    reverse, _ = numerical_integrate(lambda x: x * x, 1.0, 0.0)
    zero, _ = numerical_integrate(lambda x: x * x, 1.0, 1.0)
    assert forward == pytest.approx(1.0 / 3.0)
    assert reverse == pytest.approx(-1.0 / 3.0)
    assert zero == pytest.approx(0.0)


def test_ode_solver_supports_backward_integration():
    sol = numerical_ode_solver(
        lambda _t, y: -y,
        (1.0, 0.0),
        [np.exp(-1.0)],
        t_eval=[1.0, 0.5, 0.0],
        rtol=1e-9,
        atol=1e-12,
    )
    assert sol.success
    assert sol.y[0, -1] == pytest.approx(1.0, rel=1e-7)


def test_sde_grid_and_state_use_same_short_final_step():
    times, states = sde_solver(
        lambda x: np.ones_like(x),
        lambda x: np.zeros_like(x),
        np.array([0.0]),
        (0.0, 1.0),
        dt=0.3,
        rng=np.random.default_rng(0),
    )
    assert times == pytest.approx([0.0, 0.3, 0.6, 0.9, 1.0])
    assert states[:, 0] == pytest.approx(times)


def test_sde_supports_matrix_diffusion():
    rng = np.random.default_rng(4)
    times, states = sde_solver(
        lambda x: np.zeros_like(x),
        lambda x: np.eye(2),
        np.zeros(2),
        (0.0, 0.2),
        dt=0.1,
        rng=rng,
    )
    assert times.shape == (3,)
    assert states.shape == (3, 2)
    assert np.all(np.isfinite(states))


def test_spatial_hash_query_is_nd_and_exact_radius():
    grid = SpatialHashGrid(cell_size=1.0)
    grid.insert(np.array([0.0, 0.0, 0.0]), "origin")
    grid.insert(np.array([0.9, 0.9, 0.9]), "corner")
    grid.insert(np.array([2.0, 0.0, 0.0]), "far")
    assert grid.query(np.array([0.0, 0.0, 0.0]), radius=1.0) == ["origin"]
    assert set(grid.query(np.array([0.0, 0.0, 0.0]), radius=1.6)) == {"origin", "corner"}


def test_temporal_spectral_clustering_finds_two_separated_groups():
    rates = np.array([0.0, 0.05, 10.0, 10.05])
    times = np.array([0.0, 0.05, 10.0, 10.05])
    k, labels = spectral_clustering_with_temporal_kernel(
        rates,
        times,
        sigma=0.5,
        tau=0.5,
        max_clusters=3,
        random_state=0,
    )
    assert k == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
