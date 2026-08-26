"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
"""

from sympy import expand, factor, simplify
from typing import Any, Dict

def manipulate_expression(
    expr: Any, 
    operation: str, 
    substitutions: Dict[Any, Any] = None
) -> Any:
    """
    Performs symbolic manipulation on a SymPy expression.

    Parameters
    ----------
    expr : Any
        A SymPy expression.
    operation : str
        The manipulation to perform. One of 'expand', 'factor', 'simplify', 'subs'.
    substitutions : Dict[Any, Any], optional
        A dictionary of substitutions to make, required for the 'subs' operation, 
        by default None.

    Returns
    -------
    Any
        The manipulated SymPy expression.
    """
    if operation == 'expand':
        return expand(expr)
    elif operation == 'factor':
        return factor(expr)
    elif operation == 'simplify':
        return simplify(expr)
    elif operation == 'subs':
        if substitutions is None:
            raise ValueError("Substitutions must be provided for the 'subs' operation.")
        return expr.subs(substitutions)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
