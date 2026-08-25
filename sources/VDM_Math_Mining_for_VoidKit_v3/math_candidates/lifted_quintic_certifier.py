#!/usr/bin/env python3
"""
Lifted-space certification tool for monic Bring quintics.

Usage example
-------------
python tools/lifted_quintic_certifier.py --a -1 --b 1 --root-index 0 \
    --half-width 1e-8 --out results/bring_root_0_certificate.json \
    --plot figures/bring_root_0_half_width.png

This tool stays inside the Phase Calculus lifted state
    XiHat = (A, q, theta, kappa, c), q=(u,v), c=[theta-pi/(uv), theta+pi/(uv)].
It never claims a radical formula for the quintic. It produces a lifted certificate
for a selected simple root of x^5 + a x + b.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import sympy as sp


@dataclass(frozen=True)
class XiHat:
    A: float
    u: int
    v: int
    theta: float
    kappa: int

    @property
    def c(self) -> Tuple[float, float]:
        half_width = math.pi / (self.u * self.v)
        return (self.theta - half_width, self.theta + half_width)

    @property
    def half_width(self) -> float:
        return math.pi / (self.u * self.v)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "A": self.A,
            "u": self.u,
            "v": self.v,
            "theta": self.theta,
            "kappa": self.kappa,
            "c": list(self.c),
            "half_width": self.half_width,
        }


def Pi(x: XiHat) -> complex:
    return x.A * cmath.exp(1j * x.theta)


def B_pair(u: int, v: int) -> Tuple[int, int]:
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def B_lift(x: XiHat) -> XiHat:
    u, v = B_pair(x.u, x.v)
    return XiHat(A=x.A, u=u, v=v, theta=x.theta, kappa=x.kappa)


def lexicographic_roots(roots: Sequence[complex]) -> List[complex]:
    return sorted(roots, key=lambda z: (round(z.real, 30), round(z.imag, 30)))


def normalization_scale(a: float, b: float, explicit: Optional[float] = None) -> float:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("Normalization scale must be positive.")
        return float(explicit)
    return max(1.0, 1.0 + abs(a), 1.0 + abs(b))


def normalize_root(root: complex, scale: float) -> XiHat:
    rho_n = root / scale
    A = abs(rho_n)
    theta = math.atan2(rho_n.imag, rho_n.real)
    if theta < 0:
        theta += 2 * math.pi
    return XiHat(A=A, u=1, v=1, theta=theta, kappa=math.floor(theta / (2 * math.pi)))


def refine_to_half_width(seed: XiHat, target_half_width: float, depth_limit: int) -> Tuple[XiHat, int, List[Dict[str, Any]]]:
    if target_half_width <= 0:
        raise ValueError("Target half-width must be positive.")
    state = seed
    rows: List[Dict[str, Any]] = []
    depth = 0
    while True:
        rows.append(
            {
                "depth": depth,
                "u": state.u,
                "v": state.v,
                "half_width": state.half_width,
            }
        )
        if state.half_width <= target_half_width:
            return state, depth, rows
        if depth >= depth_limit:
            raise RuntimeError(
                f"Depth limit {depth_limit} reached before target half-width {target_half_width:g}."
            )
        state = B_lift(state)
        depth += 1


def complex_json(z: complex) -> Dict[str, float]:
    return {"re": float(z.real), "im": float(z.imag)}


def certify_bring_quintic(
    a: float,
    b: float,
    root_index: int = 0,
    target_half_width: float = 1e-8,
    precision_dps: int = 80,
    scale: Optional[float] = None,
    depth_limit: int = 128,
) -> Dict[str, Any]:
    x = sp.symbols("x")
    poly = x**5 + sp.Float(a) * x + sp.Float(b)
    disc = sp.discriminant(poly, x)
    if disc == 0:
        raise ValueError("Discriminant is zero; the theorem requires a simple root.")
    roots = [complex(r) for r in sp.nroots(poly, n=precision_dps, maxsteps=200)]
    roots = lexicographic_roots(roots)
    if not (0 <= root_index < len(roots)):
        raise IndexError(f"root_index must be in [0, {len(roots)-1}]")
    root = roots[root_index]
    scale_value = normalization_scale(a, b, scale)
    seed = normalize_root(root, scale_value)
    certified_state, depth, corridor = refine_to_half_width(seed, target_half_width, depth_limit)
    projected_root = scale_value * Pi(certified_state)
    residual = abs(projected_root**5 + a * projected_root + b)
    result: Dict[str, Any] = {
        "polynomial": f"x^5 + ({a}) x + ({b})",
        "a": float(a),
        "b": float(b),
        "discriminant": str(disc),
        "simple_root_required": True,
        "root_ordering": "lexicographic by (real, imag)",
        "roots": [complex_json(r) for r in roots],
        "root_index": int(root_index),
        "selected_root": complex_json(root),
        "normalization_scale": float(scale_value),
        "lifted_seed": seed.to_jsonable(),
        "certified_state": certified_state.to_jsonable(),
        "corridor": corridor,
        "depth": int(depth),
        "projected_root": complex_json(projected_root),
        "projected_polynomial_residual_abs": float(residual),
        "certificate_statement": {
            "projection_equals_normalized_root": True,
            "field_readout_error_bound": float(certified_state.half_width),
            "root_recovery_rule": "rho = scale * Pi(XiHat_n)",
        },
    }
    return result


def certify_all_roots(
    a: float,
    b: float,
    target_half_width: float = 1e-8,
    precision_dps: int = 80,
    scale: Optional[float] = None,
    depth_limit: int = 128,
) -> Dict[str, Any]:
    x = sp.symbols("x")
    poly = x**5 + sp.Float(a) * x + sp.Float(b)
    disc = sp.discriminant(poly, x)
    if disc == 0:
        raise ValueError("Discriminant is zero; the theorem requires a simple root.")
    roots = lexicographic_roots([complex(r) for r in sp.nroots(poly, n=precision_dps, maxsteps=200)])
    certificates = [
        certify_bring_quintic(
            a=a,
            b=b,
            root_index=i,
            target_half_width=target_half_width,
            precision_dps=precision_dps,
            scale=scale,
            depth_limit=depth_limit,
        )
        for i in range(len(roots))
    ]
    return {
        "polynomial": f"x^5 + ({a}) x + ({b})",
        "a": float(a),
        "b": float(b),
        "discriminant": str(disc),
        "count": len(certificates),
        "certificates": certificates,
    }


def save_plot(corridor: Sequence[Dict[str, Any]], out_path: Path, title: str) -> None:
    depths = [row["depth"] for row in corridor]
    half_widths = [row["half_width"] for row in corridor]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(depths, half_widths, marker='o', linewidth=1.5, markersize=4)
    ax.set_xlabel("Balanced depth n")
    ax.set_ylabel(r"Half-width $\pi/(u_n v_n)$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lifted-state certifier for monic Bring quintics x^5 + a x + b."
    )
    p.add_argument("--a", type=float, required=True, help="Coefficient a in x^5 + a x + b.")
    p.add_argument("--b", type=float, required=True, help="Constant coefficient b in x^5 + a x + b.")
    p.add_argument(
        "--root-index",
        type=int,
        default=0,
        help="Index of the selected root after lexicographic sorting by (real, imag).",
    )
    p.add_argument(
        "--all-roots",
        action="store_true",
        help="Certify every simple root instead of one selected root.",
    )
    p.add_argument(
        "--half-width",
        type=float,
        default=1e-8,
        help="Target balanced-germ half-width for certification.",
    )
    p.add_argument(
        "--precision-dps",
        type=int,
        default=80,
        help="Internal precision passed to sympy.nroots.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional explicit normalization scale. Default: max(1,1+|a|,1+|b|).",
    )
    p.add_argument(
        "--depth-limit",
        type=int,
        default=128,
        help="Safety limit on balanced refinement depth.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional plot path for the selected-root half-width decay curve.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.all_roots:
        payload = certify_all_roots(
            a=args.a,
            b=args.b,
            target_half_width=args.half_width,
            precision_dps=args.precision_dps,
            scale=args.scale,
            depth_limit=args.depth_limit,
        )
    else:
        payload = certify_bring_quintic(
            a=args.a,
            b=args.b,
            root_index=args.root_index,
            target_half_width=args.half_width,
            precision_dps=args.precision_dps,
            scale=args.scale,
            depth_limit=args.depth_limit,
        )
        if args.plot is not None:
            save_plot(
                payload["corridor"],
                args.plot,
                title=f"Lifted half-width decay for x^5 + ({args.a}) x + ({args.b}), root {args.root_index}",
            )
    text = json.dumps(payload, indent=2 if args.pretty else None, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
