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


def test_void_debt_modulation_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("voidkit.vdm.void_debt_modulation")


def test_legacy_void_equations_are_quarantined():
    import inspect

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("voidkit.vdm.void_equations")

    equations = importlib.import_module("voidkit.vdm.easter_eggs.legacy_void_equations")
    for name in ("delta_re_vgsp", "delta_gdsp", "universal_void_dynamics"):
        assert "domain_modulation" not in inspect.signature(getattr(equations, name)).parameters


def test_old_void_dynamics_namespace_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("voidkit.void_dynamics")


def test_phase_calculus_namespace_imports():
    module = importlib.import_module("voidkit.phase_calculus")
    assert module.__name__ == "voidkit.phase_calculus"
