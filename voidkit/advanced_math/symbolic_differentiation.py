"""
Unified Void Dynamics Model (VDM) - Advanced Math: Symbolic differentiation.

CLI entry point: `voidkit-diff` (see pyproject [project.scripts]).

This tool prefers SymPy for symbolic parsing/differentiation. If SymPy is not
installed, it will report a clear error message.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None  # type: ignore


def symbolic_diff(
    expression: str,
    *,
    var: str = "x",
    order: int = 1,
    simplify: bool = True,
):
    """
    Differentiate 'expression' with respect to variable 'var' 'order' times.
    Returns a SymPy expression (if SymPy available), else raises RuntimeError.
    """
    if sp is None:
        raise RuntimeError(
            "SymPy is required for symbolic differentiation. Install with: pip install sympy"
        )
    sym_var = sp.symbols(var)
    expr = sp.sympify(expression, convert_xor=True)
    deriv = sp.diff(expr, sym_var, int(order))
    if simplify:
        deriv = sp.simplify(deriv)
    return deriv


def _fmt_output(
    deriv_expr,
    *,
    var: str,
    latex: bool,
    json_out: bool,
    eval_at: Optional[List[float]],
):
    if sp is None:
        raise RuntimeError("SymPy not available to format output.")

    result = {
        "variable": var,
        "derivative_str": str(deriv_expr),
        "derivative_latex": sp.latex(deriv_expr) if latex else None,
        "evaluations": None,
    }

    if eval_at:
        sym_var = sp.symbols(var)
        vals = []
        for v in eval_at:
            try:
                numeric = float(deriv_expr.subs(sym_var, v).evalf())
            except Exception:
                numeric = None
            vals.append({"at": v, "value": numeric})
        result["evaluations"] = vals

    if json_out:
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
        return

    lines = [f"d/d{var}: {result['derivative_str']}"]
    if latex and result["derivative_latex"]:
        lines.append(f"latex: {result['derivative_latex']}")
    if result["evaluations"]:
        for item in result["evaluations"]:
            lines.append(f"{var}={item['at']} -> {item['value']}")
    print("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="voidkit-diff",
        description="Symbolically differentiate an expression (VDM Advanced Math).",
    )
    ap.add_argument(
        "expression",
        help="Expression in SymPy syntax (e.g., 'sin(x)**2 + x**3'). Use ^ as XOR; use ** for power.",
    )
    ap.add_argument(
        "-v", "--var", default="x", help="Differentiation variable (default: x)."
    )
    ap.add_argument(
        "-n",
        "--order",
        type=int,
        default=1,
        help="Differentiation order (positive integer, default: 1).",
    )
    ap.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable SymPy simplification of the result.",
    )
    ap.add_argument(
        "--eval",
        nargs="*",
        type=float,
        help="Optionally evaluate the derivative at one or more numeric points. Example: --eval 0 1 3.14",
    )
    ap.add_argument(
        "--latex",
        action="store_true",
        help="Also output a LaTeX representation of the derivative.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    ns = ap.parse_args(argv)

    if ns.order < 1:
        ap.error("--order must be >= 1")

    try:
        deriv = symbolic_diff(
            ns.expression, var=ns.var, order=ns.order, simplify=not ns.no_simplify
        )
    except RuntimeError as e:
        ap.error(str(e))
        return 2
    except Exception as e:  # pragma: no cover
        ap.error(f"Failed to differentiate expression: {e}")
        return 2

    try:
        _fmt_output(
            deriv,
            var=ns.var,
            latex=ns.latex,
            json_out=ns.json,
            eval_at=ns.eval,
        )
    except RuntimeError as e:
        ap.error(str(e))
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())