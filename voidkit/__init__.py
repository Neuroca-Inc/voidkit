"""
VoidKit: Unified Void Dynamics Model (VDM) core package.

This package aggregates multiple scientific and mathematical modules under the
'voidkit' namespace. The Advanced Math subpackage provides:
- descriptive_stats: basic descriptive statistics without external deps
- symbolic_diff: symbolic differentiation (requires SymPy)

Convenience imports are provided at the package root.
"""

from __future__ import annotations

# Convenience re-exports (Advanced Math)
try:
    from .advanced_math import descriptive_stats  # type: ignore
except Exception:  # pragma: no cover
    # Advanced Math may not be present in all builds
    pass

try:
    from .advanced_math import symbolic_diff  # type: ignore
except Exception:  # pragma: no cover
    # symbolic_diff requires SymPy; keep lazy import in submodule
    pass

__all__ = []
if "descriptive_stats" in globals():
    __all__.append("descriptive_stats")
if "symbolic_diff" in globals():
    __all__.append("symbolic_diff")

# Best-effort version exposure from installed distribution
try:
    from importlib.metadata import version as _pkg_version  # type: ignore
    try:
        __version__ = _pkg_version("voidkit")
    except Exception:  # pragma: no cover
        __version__ = "0.0.0"
except Exception:  # pragma: no cover
    __version__ = "0.0.0"