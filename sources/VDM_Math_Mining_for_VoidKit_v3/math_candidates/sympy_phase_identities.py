#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp


def main() -> None:
    delta = sp.symbols("delta", integer=True, nonnegative=True)
    k = sp.symbols("k", integer=True, nonnegative=True)

    geom = sp.summation(3**k, (k, 0, delta - 1))
    closed = (3**delta - 1) / 2
    proof_1 = sp.simplify(geom - closed)

    print("Identity 1: sum_{k=0}^{delta-1} 3^k = (3^delta - 1)/2")
    print("difference =", proof_1)
    print()

    W = sp.symbols("W", integer=True, positive=True)
    coord = geom + 2 * 3**(W - 1)
    coord_closed = (3**delta - 1) / 2 + 2 * 3**(W - 1)
    proof_2 = sp.simplify(coord - coord_closed)

    print("Identity 2: E(R^delta S^(W-1-delta) T) = (3^delta - 1)/2 + 2*3^(W-1)")
    print("difference =", proof_2)
    print()

    # Numerical anchor check
    delta_num = 9
    W_num = 64
    lhs = sum(3**i for i in range(delta_num)) + 2 * 3**(W_num - 1)
    rhs = (3**delta_num - 1) // 2 + 2 * 3**(W_num - 1)
    print("Numerical anchor check:")
    print("lhs =", lhs)
    print("rhs =", rhs)
    print("match =", lhs == rhs)


if __name__ == "__main__":
    main()
