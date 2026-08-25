from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from collections import defaultdict

@dataclass(frozen=True)
class Elem:
    scale: Fraction = Fraction(1, 1)
    phase: int = 0
    def __post_init__(self): object.__setattr__(self, 'phase', self.phase % 4)
    def __mul__(self, other): return Elem(self.scale * other.scale, self.phase + other.phase)
    def inv(self): return Elem(1 / self.scale, -self.phase)
    def div(self, other): return self * other.inv()
    def is_one(self): return self.scale == 1 and self.phase == 0
    def key(self): return f"{self.scale.numerator}/{self.scale.denominator}*i^{self.phase}"

@dataclass
class Axis:
    born: bool = False
    theta: int = 0
    u: int = 1
    v: int = 1
    def lens(self): return Elem(Fraction(1, self.u * self.v), self.theta)

@dataclass(frozen=True)
class Event:
    kind: str
    axis: int
    other: int | None = None
    def label(self): return f"{self.kind}{self.axis}" if self.other is None else f"{self.kind}{self.axis}_{self.other}"

class State:
    def __init__(self):
        self.axes = {0: Axis(True)}
        self.transitions = []
        self.edges = defaultdict(int)
        self.contact_clock = 0
        self.contact_z = 0
    def require(self, a):
        if a not in self.axes or not self.axes[a].born: raise RuntimeError(f"axis {a} not born")
    def apply(self, e: Event):
        if e.kind == 'Q':
            self.require(e.axis); ax = self.axes[e.axis]; before = ax.lens(); ax.theta = (ax.theta + 1) % 4
            self.transitions.append((e.label(), ax.lens().div(before).key()))
        elif e.kind == 'B':
            self.require(e.axis); ax = self.axes[e.axis]; before = ax.lens(); x, y = ax.v, ax.u + ax.v; ax.u, ax.v = (x, y) if x <= y else (y, x)
            self.transitions.append((e.label(), ax.lens().div(before).key()))
        elif e.kind == 'L':
            self.require(e.axis); ax = self.axes[e.axis]; before = ax.lens(); b = e.axis + 1; self.axes[b] = Axis(True)
            dc = (ax.theta + 1) * ax.u * ax.v; self.contact_clock += 1; self.contact_z += self.contact_clock * dc; self.edges[tuple(sorted((e.axis, b)))] += dc
            self.transitions.append((e.label(), self.axes[b].lens().div(before).key()))
        elif e.kind == 'O':
            a, b = sorted((e.axis, e.other)); self.require(a); self.require(b)
            self.transitions.append((e.label(), self.axes[b].lens().div(self.axes[a].lens()).key()))
        elif e.kind == 'R':
            self.require(e.axis); self.transitions.append((e.label(), Elem().key()))

def support(e: Event):
    a = e.axis
    if e.kind == 'Q': return {f'born:{a}', f'theta:{a}'}, {f'theta:{a}'}, {a}, set()
    if e.kind == 'B': return {f'born:{a}', f'q:{a}'}, {f'q:{a}'}, {a}, set()
    if e.kind == 'L': return {f'born:{a}', f'theta:{a}', f'q:{a}', 'contact_clock'}, {f'born:{a+1}', f'latch:{a}', f'edge:{a}:{a+1}', 'contact_clock', 'contact_z'}, {a}, {a+1}
    if e.kind == 'O':
        i, j = sorted((e.axis, e.other)); return {f'born:{i}', f'born:{j}', f'theta:{i}', f'theta:{j}', f'q:{i}', f'q:{j}'}, {f'edge:{i}:{j}', f'hol:{i}:{j}'}, {i, j}, set()
    if e.kind == 'R': return {f'born:{a}', f'theta:{a}', f'q:{a}'}, set(), {a}, set()

def independent(a, b):
    ar, aw, abr, abw = support(a); br, bw, bbr, bbw = support(b)
    return not (aw & (br | bw) or bw & (ar | aw) or abw & bbr or bbw & abr)

def run(events):
    s = State()
    for e in events: s.apply(e)
    return s

if __name__ == '__main__':
    cases = [
        [Event('Q',0), Event('B',0), Event('L',0)],
        [Event('L',0), Event('Q',0), Event('B',1), Event('O',0,1)],
        [Event('Q',0), Event('R',0), Event('B',0), Event('L',0)],
    ]
    for c in cases:
        st = run(c)
        print('PASS', [e.label() for e in c], st.transitions, dict(st.edges), st.contact_z)
