"""VoidKit numerical methods.

Legacy numerical adapters remain available through lazy proxy names while newer
research-extracted primitives live directly under this namespace.
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

__all__ = ["linear", "integration", "ode", "interval_roots", "spectral"]

_PROXY_MAP: Dict[str, str] = {
    "linear": "voidkit.linear_system_solver",
    "integration": "voidkit.numerical_integrate",
    "ode": "voidkit.numerical_ode_solver",
    "interval_roots": "voidkit.numerical.interval_roots",
    "spectral": "voidkit.numerical.spectral",
}


def __getattr__(name: str) -> ModuleType:
    target = _PROXY_MAP.get(name)
    if not target:
        raise AttributeError(f"module 'voidkit.numerical' has no attribute {name!r}")
    return importlib.import_module(target)
