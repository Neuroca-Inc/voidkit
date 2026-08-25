
import json
from pathlib import Path
import sympy as sp

# Symbols
DeltaU, Mz, My, Bperp, Bpar = sp.symbols('DeltaU Mz My Bperp Bpar', nonzero=True)
alpha, beta, az, ay, hz, hy = sp.symbols('alpha beta az ay hz hy')

# Vector switching surface: DeltaF = 0 gives B_parallel^c(B_perp)
DeltaF = DeltaU - 2*Mz*Bperp - 2*My*Bpar
Bpar_c = sp.solve(sp.Eq(DeltaF, 0), Bpar)[0]
Bpar_c0 = sp.simplify(Bpar_c.subs(Bperp, 0))
shift_law = sp.simplify(Bpar_c - (Bpar_c0 - (Mz/My)*Bperp))
slope_Bpar_vs_Bperp = sp.diff(Bpar_c, Bperp)

# Reciprocal switching surface: solve for B_perp^c(B_parallel)
Bperp_c = sp.solve(sp.Eq(DeltaF, 0), Bperp)[0]
Bperp_c0 = sp.simplify(Bperp_c.subs(Bpar, 0))
reciprocal_law = sp.simplify(Bperp_c - (Bperp_c0 - (My/Mz)*Bpar))
slope_Bperp_vs_Bpar = sp.diff(Bperp_c, Bpar)

# Field-order commutator for finite retained-memory register.
Uz = sp.Matrix([[1, 0], [alpha, 1]])
Uy = sp.Matrix([[1, beta], [0, 1]])
h = sp.Matrix([hz, hy])
comm_mat = sp.simplify(Uy*Uz - Uz*Uy)
comm_state = sp.simplify((Uy*Uz - Uz*Uy)*h)
R = sp.Matrix([[az, ay]])
order_signal = sp.simplify((R*comm_state)[0])
expected_order_signal = sp.simplify(alpha*beta*(az*hz - ay*hy))
order_signal_residual = sp.simplify(order_signal - expected_order_signal)

# Controls: remove either coupling or erase memory.
controls = {
    'alpha_zero': sp.simplify(order_signal.subs(alpha, 0)),
    'beta_zero': sp.simplify(order_signal.subs(beta, 0)),
    'memory_zero': sp.simplify(order_signal.subs({hz: 0, hy: 0})),
}

# Scalar quotient obstruction witness. Two states with same hz but different hy.
h0 = sp.Matrix([1, 0])
h1 = sp.Matrix([1, 1])
pz = sp.Matrix([[1, 0]])
scalar_same_projection = sp.simplify((pz*h0)[0] - (pz*h1)[0])
scalar_after_Uy_difference = sp.simplify((pz*Uy*h0)[0] - (pz*Uy*h1)[0])
scalar_after_Uz_difference = sp.simplify((pz*Uz*h0)[0] - (pz*Uz*h1)[0])

results = {
    'vector_switching': {
        'B_parallel_c': str(Bpar_c),
        'B_parallel_c0': str(Bpar_c0),
        'shift_law_residual': str(shift_law),
        'd_B_parallel_c_d_B_perp': str(slope_Bpar_vs_Bperp),
        'B_perp_c': str(Bperp_c),
        'B_perp_c0': str(Bperp_c0),
        'reciprocal_law_residual': str(reciprocal_law),
        'd_B_perp_c_d_B_parallel': str(slope_Bperp_vs_Bpar),
    },
    'field_order_commutator': {
        'commutator_matrix': str(comm_mat),
        'commutator_state': str(comm_state),
        'order_signal': str(order_signal),
        'expected_order_signal': str(expected_order_signal),
        'order_signal_residual': str(order_signal_residual),
        'controls': {k: str(v) for k, v in controls.items()},
    },
    'scalar_projection_obstruction': {
        'same_initial_pz_residual': str(scalar_same_projection),
        'after_Uy_pz_difference': str(scalar_after_Uy_difference),
        'after_Uz_pz_difference': str(scalar_after_Uz_difference),
    },
}

out = Path(__file__).resolve().parents[1] / 'data' / 'experimental_consequences_sympy_results.json'
out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
