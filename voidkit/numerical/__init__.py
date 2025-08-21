"""
VoidKit Numerical Methods (staging home)
- linear  -> proxies voidkit.linear_system_solver
- integration -> proxies voidkit.numerical_integrate
- ode -> proxies voidkit.numerical_ode_solver

This provides a stable namespace (voidkit.numerical.*) while legacy modules
remain at top-level. After verification, the code can be physically moved here.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

__all__ = ["linear", "integration", "ode"]

_PROXY_MAP: Dict[str, str] = {
    "linear": "voidkit.linear_system_solver",
    "integration": "voidkit.numerical_integrate",
    "ode": "voidkit.numerical_ode_solver",
}


def __getattr__(name: str) -> ModuleType:
    target = _PROXY_MAP.get(name)
    if not target:
        raise AttributeError(f"module 'voidkit.numerical' has no attribute {name!r}")
    return importlib.import_module(target)