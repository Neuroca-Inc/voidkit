from __future__ import annotations
import math, cmath, json
import numpy as np

N = 12
SUPPORT = {1: 1, 5: -1, 7: -1, 11: 1}

def chi12(r: int) -> int:
    return SUPPORT.get(r % 12, 0)

def S_matrix() -> np.ndarray:
    return np.array([
        [cmath.exp(-math.pi * 1j * r * s / 6) / math.sqrt(12) for s in range(N)]
        for r in range(N)
    ], dtype=complex)

def gates() -> dict:
    F = S_matrix()
    C = np.zeros((N, N), dtype=complex)
    for r in range(N):
        C[r, (-r) % N] = 1
    chi = np.array([chi12(r) for r in range(N)], dtype=complex)
    return {
        "T_support_square_seating": all((r*r) % 24 == 1 for r in SUPPORT),
        "S_unitary": float(np.max(np.abs(F.conjugate().T @ F - np.eye(N)))) < 1e-12,
        "S_square_orientation_reversal": float(np.max(np.abs(F @ F - C))) < 1e-12,
        "chi_is_S_eigenvector": float(np.max(np.abs(F @ chi - chi))) < 1e-12,
    }

if __name__ == "__main__":
    print(json.dumps(gates(), indent=2))
