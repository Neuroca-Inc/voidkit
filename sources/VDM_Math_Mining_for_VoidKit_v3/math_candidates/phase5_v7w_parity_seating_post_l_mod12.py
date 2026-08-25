from __future__ import annotations
from math import gcd

def chi12(n: int) -> int:
    r = n % 12
    if r in (1, 11): return 1
    if r in (5, 7): return -1
    return 0

def seat6(n: int) -> int:
    return n % 6

def parity_latch(n: int) -> int:
    return (n % 12) // 6

def seat12_from_latch(n: int) -> int:
    return seat6(n) + 6 * parity_latch(n)

def verify(limit: int = 48) -> dict:
    support_terms = [n for n in range(limit + 1) if gcd(n, 6) == 1]
    support_ok = all(chi12(n) == chi12(seat12_from_latch(n)) for n in support_terms)
    latch_ok = all(seat12_from_latch(r) == r for r in range(12))
    obstruction_ok = (seat6(1) == seat6(7) and chi12(1) != chi12(7)
                      and seat6(5) == seat6(11) and chi12(5) != chi12(11))
    return {
        'support_terms': len(support_terms),
        'support_ok': support_ok,
        'latch_ok': latch_ok,
        'mod6_obstruction_ok': obstruction_ok,
        'global_pass': support_ok and latch_ok and obstruction_ok,
    }

if __name__ == '__main__':
    result = verify()
    print(result)
    raise SystemExit(0 if result['global_pass'] else 1)
