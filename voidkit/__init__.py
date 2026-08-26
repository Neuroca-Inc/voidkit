"""VoidKit: research-derived mathematics and scientific computing for Neuroca.

The package contains general mathematical tools alongside research-owned
namespaces such as :mod:`voidkit.vdm` and :mod:`voidkit.phase_calculus`.
"""

from __future__ import annotations

from ._version import __version__

try:
    from .advanced_math import descriptive_stats
except Exception:  # pragma: no cover - optional dependency/environment boundary
    descriptive_stats = None  # type: ignore

try:
    from .advanced_math import symbolic_diff
except Exception:  # pragma: no cover - optional symbolic dependency
    symbolic_diff = None  # type: ignore

__all__ = ["__version__"]
if descriptive_stats is not None:
    __all__.append("descriptive_stats")
if symbolic_diff is not None:
    __all__.append("symbolic_diff")
