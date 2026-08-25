import importlib

import numpy as np
import pytest

from voidkit.soc_analysis import fit_power_law


def test_power_law_fit_is_stable_on_small_positive_sample():
    alpha, r_squared = fit_power_law(np.array([1.0, 1.5, 2.0, 3.0, 5.0]))
    assert np.isfinite(alpha)
    assert alpha > 1.0
    assert np.isfinite(r_squared)


def test_symbolic_namespace_imports_when_sympy_is_installed():
    pytest.importorskip("sympy")
    module = importlib.import_module("voidkit.symbolic")
    assert hasattr(module, "manipulate_expression")


def test_graph_namespace_imports_when_networkx_is_installed():
    pytest.importorskip("networkx")
    module = importlib.import_module("voidkit.graph")
    assert hasattr(module, "calculate_pagerank")


def test_void_debt_module_uses_package_relative_equations_import():
    module = importlib.import_module("voidkit.void_dynamics.void_debt_modulation")
    constants = module.VoidDebtModulation().constants
    assert constants["ALPHA"] == pytest.approx(0.25)
    assert constants["BETA"] == pytest.approx(0.1)
