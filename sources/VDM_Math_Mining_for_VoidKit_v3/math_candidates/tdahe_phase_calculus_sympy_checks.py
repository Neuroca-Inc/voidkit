#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp

# Symbols
τ = sp.symbols('tau', real=True)
a, b, I, eps = sp.symbols('a b I eps', nonzero=True, real=True)
az, ay, hz, hy, chi = sp.symbols('a_z a_y h_z h_y chi')
alpha, beta = sp.symbols('alpha beta')

# 1. Through-thickness loop: identical planar projection, opposite retained moment.
r = sp.Matrix([a*sp.cos(τ), 0, eps*b*sp.sin(τ)])
drdt = sp.diff(r, τ)
cross = sp.simplify(sp.Matrix([
    r[1]*drdt[2] - r[2]*drdt[1],
    r[2]*drdt[0] - r[0]*drdt[2],
    r[0]*drdt[1] - r[1]*drdt[0],
]))
mu = sp.simplify(I/2 * sp.integrate(cross, (τ, 0, 2*sp.pi)))
expected_mu = sp.Matrix([0, -sp.pi*I*a*b*eps, 0])

proj_plus = sp.Matrix([a*sp.cos(τ), 0])
proj_minus = sp.Matrix([a*sp.cos(τ), 0])
projection_residual = sp.simplify(proj_plus - proj_minus)

# 2. Reduced Hall witnesses and transdimensional residual.
R_TD = az*hz + chi*ay*hy
R_2D = az*hz
residual = sp.simplify(R_TD - R_2D)
expected_residual = chi*ay*hy

# 3. Branch-history field-order commutator.
Uz = sp.Matrix([[1, 0], [alpha, 1]])
Uy = sp.Matrix([[1, beta], [0, 1]])
comm = sp.simplify(Uy*Uz - Uz*Uy)
expected_comm = alpha*beta*sp.Matrix([[1, 0], [0, -1]])
h = sp.Matrix([hz, hy])
order_signal = sp.simplify((sp.Matrix([[az, ay]]) * comm * h)[0])
expected_order = sp.simplify(alpha*beta*(az*hz - ay*hy))

# 4. Controls.
control_alpha0 = sp.simplify(order_signal.subs(alpha, 0))
control_beta0 = sp.simplify(order_signal.subs(beta, 0))
control_h0 = sp.simplify(order_signal.subs({hz: 0, hy: 0}))

results = {
    'loop_cross_product': [sp.sstr(x) for x in cross],
    'loop_moment': [sp.sstr(x) for x in mu],
    'loop_moment_expected': [sp.sstr(x) for x in expected_mu],
    'loop_moment_residual': [sp.sstr(sp.simplify(x)) for x in (mu - expected_mu)],
    'loop_moment_residual_zero': bool(sp.simplify(mu - expected_mu) == sp.zeros(3, 1)),
    'planar_projection_residual': [sp.sstr(x) for x in projection_residual],
    'planar_projection_residual_zero': bool(projection_residual == sp.zeros(2, 1)),
    'td_residual': sp.sstr(residual),
    'td_residual_expected': sp.sstr(expected_residual),
    'td_residual_zero_after_subtraction': bool(sp.simplify(residual - expected_residual) == 0),
    'field_order_commutator': [[sp.sstr(comm[i, j]) for j in range(2)] for i in range(2)],
    'field_order_commutator_expected': [[sp.sstr(expected_comm[i, j]) for j in range(2)] for i in range(2)],
    'field_order_commutator_residual_zero': bool(sp.simplify(comm - expected_comm) == sp.zeros(2, 2)),
    'order_signal': sp.sstr(order_signal),
    'order_signal_expected': sp.sstr(expected_order),
    'order_signal_residual_zero': bool(sp.simplify(order_signal - expected_order) == 0),
    'controls': {
        'alpha_zero': sp.sstr(control_alpha0),
        'beta_zero': sp.sstr(control_beta0),
        'h_zero': sp.sstr(control_h0),
        'all_zero': bool(control_alpha0 == 0 and control_beta0 == 0 and control_h0 == 0),
    },
}

out = Path(__file__).resolve().parents[1] / 'data' / 'tdahe_phase_calculus_sympy_results.json'
out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps(results, indent=2, sort_keys=True))
