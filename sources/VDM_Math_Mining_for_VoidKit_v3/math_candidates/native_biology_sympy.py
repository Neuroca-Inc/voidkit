from __future__ import annotations

import sys
from sympy import Rational, simplify

E = Rational(1, 24)
R = 24 * E
w0 = Rational(1, 1)
w1 = Rational(1, 1) + R
w2 = Rational(1, 1)
w3 = Rational(1, 1)
W = simplify(w0 + w1 + w2 + w3)
checks = []

def check(name: str, expr, expected) -> None:
    ok = simplify(expr - expected) == 0
    checks.append((name, ok, expr, expected))

check('normalized_edge_bonus', R, 1)
check('total_weight_budget', W, 5)
check('triplet_word_count', 4**3, 64)
check('occurrences_per_base', 3 * 4**2, 48)
check('total_acceptors', 48 * W, 240)
check('centered_b0', 8 * W + 16 * w0, 56)
check('centered_b1', 8 * W + 16 * w1, 72)
check('centered_b2', 8 * W + 16 * w2, 56)
check('centered_b3', 8 * W + 16 * w3, 56)
check('partition_A0_positive', (8 * W + 16 * w1) + (8 * W + 16 * w2), 128)
check('partition_A0_negative', (8 * W + 16 * w0) + (8 * W + 16 * w3), 112)
check('partition_A1_positive', (8 * W + 16 * w0) + (8 * W + 16 * w1), 128)
check('partition_A1_negative', (8 * W + 16 * w2) + (8 * W + 16 * w3), 112)
check('partition_A2_positive', (8 * W + 16 * w1) + (8 * W + 16 * w3), 128)
check('partition_A2_negative', (8 * W + 16 * w0) + (8 * W + 16 * w2), 112)

failed = False
for name, ok, expr, expected in checks:
    print(f"{name}: {'PASS' if ok else 'FAIL'} | got={expr} | expected={expected}")
    failed = failed or (not ok)

if failed:
    print('FINAL_RESULT: FAIL')
    sys.exit(1)
print('FINAL_RESULT: PASS')
