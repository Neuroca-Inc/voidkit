#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

b, n, ell, B = sp.symbols('b n ell B', positive=True)
D = sp.symbols('D', real=True)

# Current-state cell obstruction:
ineq = sp.StrictLessThan(2 * b**(n - 1) * B, b**(-ell))

print("Cell obstruction inequality:")
print("  ", ineq)
print()

# Solve manually for B
bound_B = sp.simplify(sp.Rational(1, 2) * b**(-(n + ell - 1)))
print("Necessary condition solved for B:")
print("  B <", bound_B)
print()

# If B = 10^(-D), derive a decimal warm-up lower bound
rhs = sp.expand((n + ell - 1) * sp.log(b, 10) + sp.log(2, 10))
print("If B = 10^(-D), necessary decimal warm-up lower bound is:")
print("  D >", rhs)
print()

# Packet-depth monotonicity model
N, Dsym = sp.symbols('N D', positive=True)
K = (1 + sp.sqrt(1 + 12 * N * Dsym * sp.log(10) / sp.pi)) / 6
dK_dN = sp.simplify(sp.diff(K, N))
print("dK/dN for K_N(D) = (1 + sqrt(1 + 12*N*D*log(10)/pi))/6:")
print("  ", dK_dN)
print()
print("For N>0 and D>0 this derivative is positive, so deeper balanced packets are strictly worse in this family.")
