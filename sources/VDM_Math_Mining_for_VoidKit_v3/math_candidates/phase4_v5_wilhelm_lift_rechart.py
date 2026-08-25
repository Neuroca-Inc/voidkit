#!/usr/bin/env python3
from itertools import product

def flip_line(binary: str, line: int) -> str:
    idx = 6 - line
    b = list(binary)
    b[idx] = '1' if b[idx] == '0' else '0'
    return ''.join(b)

def audit_line6():
    rows = []
    for n in range(64):
        s = format(n, '06b')
        t = flip_line(s, 6)
        rows.append((n + 1, s, int(t, 2) + 1, t, ((n ^ 32) + 1)))
    assert all(target == xor32 for _, _, target, _, xor32 in rows)
    assert all(0 <= target - 1 < 64 for _, _, target, _, _ in rows)
    assert all((source <= 32) != (target <= 32) for source, _, target, _, _ in rows)
    return rows

if __name__ == '__main__':
    rows = audit_line6()
    print('PASS line6 law: target_index = ((source_index - 1) xor 32) + 1')
    print('rows:', len(rows))
