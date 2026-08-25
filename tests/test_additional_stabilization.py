import numpy as np
import pytest

from voidkit.evolutionary import apply_mutation, apply_recombination
from voidkit.graph.graph_dynamics import calculate_path_score
from voidkit.structural_plasticity import calculate_advanced_growth_trigger, detect_bursts
from voidkit.tda.calculate_tda_metrics import calculate_tda_metrics


def test_burst_detection_handles_duplicate_times_without_index_lookup_bug():
    spikes = np.array([1.0, 1.0, 2.0, 20.0, 21.0, 22.0])
    bursts = detect_bursts(spikes, max_interspike_interval=2.0, min_spikes_in_burst=3)
    assert np.allclose(bursts, np.array([[1.0, 2.0], [20.0, 22.0]]))


def test_growth_trigger_is_numerically_stable_for_large_arguments():
    assert calculate_advanced_growth_trigger(-1e6, 0.0, 0.0) == pytest.approx(0.0)
    assert calculate_advanced_growth_trigger(1e6, 0.0, 0.0) == pytest.approx(1.0)


def test_mutation_and_recombination_are_rng_reproducible():
    weights = np.ones(8)
    first = apply_mutation(weights, mutation_rate=1.0, rng=np.random.default_rng(9))
    second = apply_mutation(weights, mutation_rate=1.0, rng=np.random.default_rng(9))
    assert first == pytest.approx(second)

    a = np.zeros(8)
    b = np.ones(8)
    c1 = apply_recombination(a, b, rng=np.random.default_rng(3))
    c2 = apply_recombination(a, b, rng=np.random.default_rng(3))
    assert np.array_equal(c1, c2)


def test_path_score_rejects_zero_decay_length():
    with pytest.raises(ValueError):
        calculate_path_score(np.ones(2), np.ones(2), np.ones(2), 0.0)


def test_tda_metrics_keep_essential_h1_out_of_finite_total():
    h0 = np.array([[0.0, 1.0], [0.0, np.inf]])
    h1 = np.array([[1.0, 3.0], [2.0, np.inf]])
    metrics = calculate_tda_metrics([h0, h1])
    assert metrics["component_count"] == 1
    assert metrics["total_b1_persistence"] == pytest.approx(2.0)
    assert metrics["essential_b1_count"] == 1
