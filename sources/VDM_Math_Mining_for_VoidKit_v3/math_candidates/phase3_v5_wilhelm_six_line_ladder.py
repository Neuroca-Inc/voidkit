#!/usr/bin/env python3
from itertools import product

def flip_line(binary: str, line: int) -> str:
    idx = 6 - line
    b = list(binary)
    b[idx] = '1' if b[idx] == '0' else '0'
    return ''.join(b)

def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

states = [''.join(bits) for bits in product('01', repeat=6)]
edges = []
for s in states:
    for line in range(1, 7):
        t = flip_line(s, line)
        edges.append((s, line, t, hamming(s, t), flip_line(t, line) == s))

assert len(edges) == 384
assert len({tuple(sorted((s, t))) for s, _, t, _, _ in edges}) == 192
assert all(h == 1 for _, _, _, h, _ in edges)
assert all(inv for *_, inv in edges)

print('PASS: Wilhelm Q6 line automaton closure')
print('states=', len(states), 'directed_edges=', len(edges), 'undirected_edges=', 192)
