from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidkit.wave.klein_gordon import energy, kg_energy


def test_kg_energy_source_alias_matches_formula() -> None:
    n = 64
    dx = 2.0 * np.pi / n
    x = np.arange(n) * dx
    phi = np.sin(x)
    momentum = 0.1 * np.cos(x)
    expected = 0.5 * float(np.sum(momentum * momentum) * dx)
    grad = np.cos(x)
    expected += 0.5 * float(np.sum(grad * grad) * dx)
    expected += 0.5 * (0.2**2) * float(np.sum(phi * phi) * dx)
    assert abs(energy(phi, momentum, dx, 1.0, 0.2) - expected) < 2e-12
    assert kg_energy is energy


def test_ore_namespace_reconciliation_does_not_create_duplicate_owners() -> None:
    root = Path(__file__).resolve().parents[1] / "voidkit"
    assert not (root / "information").exists()
    assert not (root / "timeseries").exists()
    assert not (root / "thermo").exists()
    assert not (root / "wave" / "kg.py").exists()
    assert (root / "info_theory" / "complexity.py").is_file()
    assert (root / "time_series" / "events.py").is_file()
    assert (root / "thermodynamics" / "lit.py").is_file()


def test_ore_reconciliation_receipt_covers_all_frozen_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "sources" / "ore_extraction_v0_1_0" / "SOURCES.json").read_text())
    reconciled = json.loads((root / "voidkit" / "provenance" / "ore_extraction_reconciliation_v0_1_0.json").read_text())
    assert len(frozen["sources"]) == 26
    assert len(reconciled["records"]) == 26
    assert {r["sha256"] for r in frozen["sources"]} == {r["sha256"] for r in reconciled["records"]}
