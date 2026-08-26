"""
VoidKit advanced mathematics package.

Exports:
- descriptive_stats: compute descriptive statistics for a 1-D numeric sequence
- symbolic_diff: symbolic differentiation (requires SymPy)

CLI entry points (see pyproject [project.scripts]):
- voidkit-stats
- voidkit-diff
"""

from __future__ import annotations

# Public API
from .calculate_descriptive_stats import descriptive_stats

# symbolic_diff may require SymPy; provide a graceful fallback if import fails
try:
    from .symbolic_differentiation import symbolic_diff  # type: ignore
except Exception as _e:  # pragma: no cover
    def symbolic_diff(*args, **kwargs):
        raise RuntimeError(
            "SymPy is required for symbolic differentiation. Install with: pip install sympy"
        ) from _e

__all__ = ["descriptive_stats", "symbolic_diff"]

