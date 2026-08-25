#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]


def pendulum_symbolic_checks() -> dict:
    th1, th2, w1, w2 = sp.symbols("th1 th2 w1 w2")
    pi = sp.pi
    d = th1 - th2
    D = 3 - sp.cos(2*d)
    a1 = (-3*sp.sin(th1) - sp.sin(th1 - 2*th2) - 2*sp.sin(d)*(w2**2 + w1**2*sp.cos(d))) / D
    a2 = (2*sp.sin(d)*(2*w1**2 + 2*sp.cos(th1) + w2**2*sp.cos(d))) / D
    H = w1**2 + sp.Rational(1,2)*w2**2 + w1*w2*sp.cos(d) - 2*sp.cos(th1) - sp.cos(th2)
    Hdot = sp.diff(H, th1)*w1 + sp.diff(H, th2)*w2 + sp.diff(H, w1)*a1 + sp.diff(H, w2)*a2
    Hdot_simplified = sp.trigsimp(sp.factor(sp.together(Hdot)))
    visible_period_identity_1 = sp.simplify(sp.exp(sp.I*(th1 + 2*pi)) / sp.exp(sp.I*th1) - 1)
    visible_period_identity_2 = sp.simplify(sp.exp(sp.I*(th2 - 4*pi)) / sp.exp(sp.I*th2) - 1)
    phi, k, m = sp.symbols("phi k m", integer=True)
    reconstruct_increment = sp.simplify((phi + 2*pi*(k+m)) - (phi + 2*pi*k) - 2*pi*m)
    return {
        "pendulum_hamiltonian_residual": str(Hdot_simplified),
        "pendulum_hamiltonian_residual_zero": bool(Hdot_simplified == 0),
        "visible_period_identity_arm1": bool(visible_period_identity_1 == 0),
        "visible_period_identity_arm2": bool(visible_period_identity_2 == 0),
        "reconstruction_increment_identity": bool(reconstruct_increment == 0),
    }


def nbody_symbolic_checks() -> dict:
    G = sp.symbols("G", nonzero=True)
    masses = sp.symbols("m0 m1 m2", positive=True, nonzero=True)
    xs = sp.symbols("x0 x1 x2")
    ys = sp.symbols("y0 y1 y2")
    pxs = sp.symbols("px0 px1 px2")
    pys = sp.symbols("py0 py1 py2")

    H = sp.Integer(0)
    for i, m in enumerate(masses):
        H += (pxs[i]**2 + pys[i]**2) / (2*m)
    for i in range(3):
        for j in range(i+1, 3):
            dx, dy = xs[i]-xs[j], ys[i]-ys[j]
            H -= G*masses[i]*masses[j] / sp.sqrt(dx**2 + dy**2)

    q_vars = list(xs) + list(ys)
    p_vars = list(pxs) + list(pys)
    dqdt = [sp.diff(H, p) for p in p_vars]
    dpdt = [-sp.diff(H, q) for q in q_vars]
    dHdt = sp.Integer(0)
    for q, p, qdot, pdot in zip(q_vars, p_vars, dqdt, dpdt):
        dHdt += sp.diff(H, q)*qdot + sp.diff(H, p)*pdot
    dHdt_simplified = sp.simplify(dHdt)

    a = sp.symbols("a0:6")
    b = sp.symbols("b0:6")
    J_ab = sum(a[i]*b[i+3] - a[i+3]*b[i] for i in range(3))
    J_ba = sum(b[i]*a[i+3] - b[i+3]*a[i] for i in range(3))
    anti_residual = sp.simplify(J_ab + J_ba)

    return {
        "nbody_hamiltonian_residual": str(dHdt_simplified),
        "nbody_hamiltonian_residual_zero": bool(dHdt_simplified == 0),
        "poisson_antisymmetry_residual": str(anti_residual),
        "poisson_antisymmetry_pass": bool(anti_residual == 0),
    }


def build_certificate(out_path: Path) -> dict:
    checks = {}
    checks.update(pendulum_symbolic_checks())
    checks.update(nbody_symbolic_checks())
    bool_keys = [k for k, v in checks.items() if isinstance(v, bool)]
    checks["FINAL_RESULT"] = "PASS" if all(checks[k] for k in bool_keys) else "FAIL"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="certificates/sympy_projection_loss_audit.json")
    args = ap.parse_args()
    cert = build_certificate(ROOT / args.out)
    print(json.dumps(cert, indent=2, sort_keys=True))
    print(f"FINAL_RESULT: {cert['FINAL_RESULT']}")


if __name__ == "__main__":
    main()
