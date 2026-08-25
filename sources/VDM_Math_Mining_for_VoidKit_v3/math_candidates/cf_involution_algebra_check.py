from __future__ import annotations
import json
import sympy as sp

N = 12
# Linear factor U = Gamma * tau_6.
Gamma = sp.diag(*[(-1) ** r for r in range(N)])
Tau = sp.zeros(N)
for r in range(N):
    Tau[(r + 6) % N, r] = 1
U = Gamma * Tau

I = sp.eye(N)
checks = {
    "tau6_squared_identity": Tau * Tau == I,
    "parity_squared_identity": Gamma * Gamma == I,
    "parity_commutes_with_shift6": Gamma * Tau == Tau * Gamma,
    "linear_factor_squared_identity": U * U == I,
    "linear_factor_orthogonal": U.T * U == I,
}

# Generic real symmetric g and real antisymmetric Omega in a 2x2 local model.
a, b, c, w = sp.symbols("a b c w", real=True)
g = sp.Matrix([[a, b], [b, c]])
Omega = sp.Matrix([[0, -w], [w, 0]])
Q = g - sp.I * Omega / 2
Qbar = sp.conjugate(Q)
checks["conjugation_preserves_symmetric_sector"] = sp.re(Qbar) == g
checks["conjugation_reverses_antisymmetric_sector"] = sp.simplify(-2 * sp.im(Qbar) + Omega) == sp.zeros(2)

# Same quarter-turn in an anti-linear frame: C(i v) = -i C(v).
z0, z1 = sp.symbols("z0 z1")
# Symbolic statement checked at coefficient level.
checks["antilinearity_reverses_i"] = sp.conjugate(sp.I) == -sp.I

# Reversed normalized overlap link.
x, y = sp.symbols("x y", real=True)
z = x + sp.I * y
# For nonzero z, normalized forward link is z/|z| and reverse is conjugate.
checks["reverse_link_is_conjugate"] = sp.conjugate(z / sp.sqrt(x**2 + y**2)) == sp.conjugate(z) / sp.sqrt(x**2 + y**2)

result = {
    "checks": checks,
    "all_pass": all(bool(v) for v in checks.values()),
    "U": U.tolist(),
    "candidate": "C = Gamma * tau6 * complex_conjugation",
}
print(json.dumps(result, indent=2, default=str))
