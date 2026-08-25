#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp


def main() -> None:
    delta, W = sp.symbols('delta W', integer=True, nonnegative=True)
    k = sp.symbols('k', integer=True, nonnegative=True)

    geom = sp.summation(3**k, (k, 0, delta - 1))
    closed = (3**delta - 1) / 2
    coord = geom + 2 * 3**(W - 1)
    norm = sp.simplify(coord / 3**W)
    target_norm = sp.Rational(2, 3) + (3**delta - 1) / (2 * 3**W)

    print('Identity 1: geometric prefix for R^delta')
    print(sp.simplify(geom - closed))
    print()

    print('Identity 2: execution coordinate of R^delta S^(W-1-delta) T')
    print(sp.simplify(coord - ((3**delta - 1) / 2 + 2 * 3**(W - 1))))
    print()

    print('Identity 3: normalized native address')
    print(sp.simplify(norm - target_norm))
    print()

    subs = {delta: 9, W: 64}
    coord_num = sp.simplify(coord.subs(subs))
    norm_num = sp.N(norm.subs(subs), 40)
    print('Example delta=9, W=64')
    print('coordinate =', coord_num)
    print('normalized =', norm_num)


if __name__ == '__main__':
    main()
