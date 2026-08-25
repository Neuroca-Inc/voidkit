#!/usr/bin/env python3
from dataclasses import dataclass
from math import prod
import json
import sympy as sp

@dataclass(frozen=True)
class State:
    A: int
    u: int
    v: int
    theta: int
    rank: int
    word: str


def B_pair(u, v):
    a, b = v, u + v
    return (a, b) if a <= b else (b, a)


def corrected_L(s: State) -> State:
    return State(A=s.A + 1, u=s.u, v=s.v, theta=s.theta + 1, rank=s.rank + 1, word=s.word + 'L')


def old_T_witness_layer(s: State) -> State:
    return State(A=s.A + 1, u=1, v=s.A + 1, theta=s.theta + 1, rank=s.rank + 1, word=s.word + 'T')


def visible_witness(s: State):
    return s.theta % 4


def ancient_yi_digits_lsd_first(n, places):
    out=[]
    for _ in range(places):
        out.append(n % 8)
        n//=8
    if n:
        out.append(n)
    return tuple(out)


def decimal_from_lsd_digits(digits):
    return sum(d * (8**i) for i,d in enumerate(digits))


def main():
    # corrected L versus old T conflict for carried q
    s = State(A=6, u=55, v=89, theta=30, rank=6, word='((BQ)^6L)^5(BQ)^6')
    c = corrected_L(s)
    t = old_T_witness_layer(s)
    old_T_carries_q = (t.u, t.v) == (s.u, s.v)
    corrected_L_carries_q = (c.u, c.v) == (s.u, s.v)

    # visible projection collision: same visible phase, different q/rank and different next B
    a = State(A=0,u=1,v=1,theta=0,rank=1,word='')
    b = State(A=5,u=55,v=89,theta=4,rank=6,word='W')
    same_visible = visible_witness(a) == visible_witness(b)
    different_state = (a.u,a.v,a.rank) != (b.u,b.v,b.rank)
    different_next_B = B_pair(a.u,a.v) != B_pair(b.u,b.v)

    # Ancient Yi base-8 carry: 63 as 77_lsd, then 64 as 001_lsd
    d63 = ancient_yi_digits_lsd_first(63,2)
    d64 = ancient_yi_digits_lsd_first(64,3)
    carry_ok = d63 == (7,7) and d64 == (0,0,1) and decimal_from_lsd_digits(d64) == 64

    # SymPy projection non-injectivity witness: P drops hidden coordinate h
    x,y,h = sp.symbols('x y h')
    P = sp.Matrix([[1,0,0],[0,1,0]])
    s1 = sp.Matrix([0,0,0])
    s2 = sp.Matrix([0,0,1])
    projection_collision = tuple(P*s1) == tuple(P*s2) and tuple(s1) != tuple(s2)

    gates = {
        'corrected_L_carries_q': corrected_L_carries_q,
        'old_T_witness_layer_does_not_carry_q_on_nonorigin_state': not old_T_carries_q,
        'visible_projection_collision_exists': same_visible and different_state,
        'visible_projection_not_state_complete_for_next_B': same_visible and different_next_B,
        'ancient_yi_lsd_first_63_to_64_carry_ok': carry_ok,
        'sympy_projection_noninjective_hidden_coordinate': projection_collision,
    }
    report = {
        'all_gates_pass': all(gates.values()),
        'gates': gates,
        'states': {
            'input': s.__dict__,
            'corrected_L': c.__dict__,
            'old_T_witness_layer': t.__dict__,
            'visible_collision_a': a.__dict__,
            'visible_collision_b': b.__dict__,
        },
        'ancient_yi': {'n63_lsd_digits': d63, 'n64_lsd_digits': d64},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['all_gates_pass'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
