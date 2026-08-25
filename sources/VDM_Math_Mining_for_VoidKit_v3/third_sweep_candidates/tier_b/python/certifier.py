"""Application service for retained inverse-word certification."""
from __future__ import annotations

import math
from typing import Any

import mpmath as mp

from pc_inverse_word_certifier.domain.fibonacci import corridor_state
from pc_inverse_word_certifier.domain.intervals import (
    bisect_real,
    lambertw_real_bracket,
    x_plus_sin_bracket,
)
from pc_inverse_word_certifier.domain.models import BranchMemory, InverseFiberState

mp.mp.dps = 80


def _fmt(value: Any) -> str:
    if isinstance(value, mp.mpc):
        return f"({mp.nstr(value.real, 50)} {mp.nstr(value.imag, 50)}j)"
    if isinstance(value, mp.mpf):
        return mp.nstr(value, 50)
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _abs_complex(value: Any) -> float:
    return float(abs(value))


class InverseWordCertifier:
    """Certifies inverse fibers without claiming finite elementary discharge."""

    def certify_lambertw(
        self,
        y: float | str,
        branch: int = 0,
        depth: int = 21,
        tolerance: float = 1e-30,
    ) -> InverseFiberState:
        y_mpf = mp.mpf(str(y))
        corridor = corridor_state(depth)
        x_mp = mp.lambertw(y_mpf, branch)
        residual = x_mp * mp.e**x_mp - y_mpf
        interval = None
        method = "mpmath_lambertw_terminal_projection"
        if branch in (0, -1):
            try:
                left, right = lambertw_real_bracket(float(y_mpf), branch)
                root, lft, rgt, steps = bisect_real(
                    lambda z: z * math.exp(z) - float(y_mpf),
                    left,
                    right,
                    tolerance=max(tolerance, 1e-15),
                    max_steps=300,
                )
                if abs(float(mp.im(x_mp))) < 1e-45:
                    interval = {
                        "left": _fmt(lft),
                        "right": _fmt(rgt),
                        "bisection_steps": str(steps),
                        "midpoint": _fmt(root),
                    }
                    method = "real_branch_bisection_plus_lambertw_projection"
            except ValueError:
                interval = None
        branch_memory = BranchMemory(
            sheet=f"W_{branch}",
            history=["InitFiber(x*exp(x)=y)", f"SelectSheet(W_{branch})", f"B^{depth}", "TerminalProject"],
            kappa=branch,
        )
        return InverseFiberState(
            equation="x*exp(x)=y",
            y=_fmt(y_mpf),
            x=_fmt(x_mp),
            residual_abs=_abs_complex(residual),
            tolerance=float(tolerance),
            branch_memory=branch_memory,
            corridor=corridor.__dict__,
            inverse_word=["I_f", f"sheet:W_{branch}", f"B^{depth}", "Pi_x", "Pi_f_residual"],
            interval=interval,
            metadata={
                "function": "LambertW",
                "branch_point": "-1/e",
                "method": method,
                "projection": "Pi_f(X)=x*exp(x)",
            },
        )

    def certify_x_plus_sin(
        self,
        y: float | str,
        depth: int = 34,
        tolerance: float = 1e-14,
    ) -> InverseFiberState:
        y_float = float(y)
        corridor = corridor_state(depth)
        left, right = x_plus_sin_bracket(y_float)
        root, lft, rgt, steps = bisect_real(
            lambda z: z + math.sin(z) - y_float,
            left,
            right,
            tolerance=tolerance,
            max_steps=500,
        )
        try:
            root_mp = mp.findroot(lambda z: z + mp.sin(z) - mp.mpf(str(y)), mp.mpf(str(root)))
        except Exception:
            root_mp = mp.mpf(str(root))
        residual_mp = root_mp + mp.sin(root_mp) - mp.mpf(str(y))
        branch_memory = BranchMemory(
            sheet="real_monotone_branch",
            history=["InitFiber(x+sin(x)=y)", "SelectRealMonotoneSheet", f"B^{depth}", "TerminalBracket"],
            kappa=0,
        )
        return InverseFiberState(
            equation="x+sin(x)=y",
            y=_fmt(y_float),
            x=_fmt(root_mp),
            residual_abs=_abs_complex(residual_mp),
            tolerance=float(tolerance),
            branch_memory=branch_memory,
            corridor=corridor.__dict__,
            inverse_word=["I_f", "sheet:real_monotone", f"B^{depth}", "BracketFiber", "Pi_f_residual"],
            interval={
                "left": _fmt(lft),
                "right": _fmt(rgt),
                "bisection_steps": str(steps),
                "midpoint": _fmt(root),
                "newton_refined": _fmt(root_mp),
            },
            metadata={
                "function": "x+sin(x)",
                "monotonicity": "f'(x)=1+cos(x)>=0; real fiber is singleton",
                "projection": "Pi_f(X)=x+sin(x)",
            },
        )

    def certify_exp_inverse(
        self,
        y: float | str,
        depth: int = 21,
        tolerance: float = 1e-30,
    ) -> InverseFiberState:
        y_mpf = mp.mpf(str(y))
        if y_mpf <= 0:
            raise ValueError("real exp inverse requires y > 0")
        corridor = corridor_state(depth)
        x_mp = mp.log(y_mpf)
        residual = mp.e**x_mp - y_mpf
        branch_memory = BranchMemory(
            sheet="log_real_branch",
            history=["InitFiber(exp(x)=y)", "SelectLogRealSheet", f"B^{depth}", "TerminalProject"],
            kappa=0,
        )
        return InverseFiberState(
            equation="exp(x)=y",
            y=_fmt(y_mpf),
            x=_fmt(x_mp),
            residual_abs=_abs_complex(residual),
            tolerance=float(tolerance),
            branch_memory=branch_memory,
            corridor=corridor.__dict__,
            inverse_word=["I_f", "sheet:log_real", f"B^{depth}", "Pi_x", "Pi_f_residual"],
            interval=None,
            metadata={"function": "log", "projection": "Pi_f(X)=exp(x)"},
        )
