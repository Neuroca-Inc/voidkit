import json
import math
import pytest

from voidkit.advanced_math import descriptive_stats
from voidkit.advanced_math.calculate_descriptive_stats import main as stats_main


def test_descriptive_stats_basic():
    data = [1, 2, 3, 4]
    out = descriptive_stats(data, ddof=1)
    assert out["count"] == 4
    assert out["mean"] == pytest.approx(2.5)
    assert out["median"] == pytest.approx(2.5)
    assert out["var"] == pytest.approx(5.0 / 3.0, rel=1e-12)
    assert out["std"] == pytest.approx(math.sqrt(5.0 / 3.0), rel=1e-12)
    assert out["min"] == 1
    assert out["max"] == 4
    assert out["q1"] == pytest.approx(1.75)
    assert out["q3"] == pytest.approx(3.25)
    assert out["iqr"] == pytest.approx(1.5)


def test_stats_cli_json(capsys):
    rc = stats_main(["1", "2", "3", "4", "--json"])
    assert rc == 0
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["count"] == 4
    assert payload["mean"] == pytest.approx(2.5)


# ---------- Symbolic differentiation tests (skip if SymPy unavailable) ----------

sp = pytest.importorskip("sympy")
from voidkit.advanced_math import symbolic_diff  # noqa: E402
from voidkit.advanced_math.symbolic_differentiation import (  # noqa: E402
    main as diff_main,
)


def test_symbolic_diff_numeric_evaluation():
    expr = "sin(x)**2 + x**3"
    d = symbolic_diff(expr, var="x", order=1, simplify=True)
    x = sp.symbols("x")
    # d/dx = sin(2x) + 3x^2
    assert float(d.subs(x, 0).evalf()) == pytest.approx(0.0)
    assert float(d.subs(x, 1).evalf()) == pytest.approx(math.sin(2.0) + 3.0, rel=1e-12)


def test_symbolic_cli_json(capsys):
    rc = diff_main(["sin(x)**2 + x**3", "--eval", "0", "1", "--json"])
    assert rc == 0
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["variable"] == "x"
    evals = payload.get("evaluations") or []
    assert len(evals) == 2
    # Expect around 0 at x=0 and sin(2)+3 at x=1
    vals = {e["at"]: e["value"] for e in evals}
    assert vals.get(0.0) == pytest.approx(0.0)
    assert vals.get(1.0) == pytest.approx(math.sin(2.0) + 3.0, rel=1e-12)