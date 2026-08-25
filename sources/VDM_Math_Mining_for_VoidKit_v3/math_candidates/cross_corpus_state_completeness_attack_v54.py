#!/usr/bin/env python3
"""v54 cross-corpus state-completeness attack.
No external IO required for the claims; finite witnesses are constructed inline.
"""
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class Witness:
    name: str
    x: Any
    y: Any
    projection: Callable[[Any], Any]
    transition: Callable[[Any], Any]
    expected_collision: bool
    expected_transition_split: bool

def run(w: Witness):
    same_projection = w.projection(w.x) == w.projection(w.y)
    transition_split = w.transition(w.x) != w.transition(w.y)
    passed = same_projection == w.expected_collision and transition_split == w.expected_transition_split
    return w.name, passed, same_projection, transition_split, w.projection(w.x), w.projection(w.y), w.transition(w.x), w.transition(w.y)

# 1. Monodromy: visible sheet same, retained kappa/history differs after further action.
monodromy = Witness(
    name='monodromy_history_loss',
    x={'sheet':0,'kappa':0,'history':()},
    y={'sheet':0,'kappa':2,'history':('g','g')},
    projection=lambda s: s['sheet'],
    transition=lambda s: (s['sheet'] ^ 1, s['kappa'] + 1, s['history'] + ('g',)),
    expected_collision=True,
    expected_transition_split=True,
)

# 2. Branch grammar: same primitive word, different sheet action.
branch = Witness(
    name='branch_action_registry_loss',
    x={'word':'LQ','action':(0,1,2,3,4)},
    y={'word':'LQ','action':(1,0,2,3,4)},
    projection=lambda s: s['word'],
    transition=lambda s: s['action'],
    expected_collision=True,
    expected_transition_split=True,
)

# 3. NC phase: visible xy origin same, hidden z charge differs.
nc = Witness(
    name='noncommutative_hidden_order_loss',
    x={'x':0,'y':0,'z':0},
    y={'x':0,'y':0,'z':1},
    projection=lambda s: (s['x'],s['y']),
    transition=lambda s: (s['x'],s['y'],s['z']+1),
    expected_collision=True,
    expected_transition_split=True,
)

# 4. Pencil vector basis: same components, different chart basis means different physical vector.
pencil = Witness(
    name='chart_basis_loss',
    x={'components':(1,2,3),'basis':'yin'},
    y={'components':(1,2,3),'basis':'yang'},
    projection=lambda s: s['components'],
    transition=lambda s: (s['basis'], tuple(-v for v in s['components']) if s['basis']=='yang' else s['components']),
    expected_collision=True,
    expected_transition_split=True,
)

# 5. Wilhelm: same selected-line semantic flag class can have different retained transition targets.
wilhelm = Witness(
    name='wilhelm_semantic_readout_loss',
    x={'carrier':'000000','line':1,'semantic':'danger'},
    y={'carrier':'111111','line':1,'semantic':'danger'},
    projection=lambda s: s['semantic'],
    transition=lambda s: s['carrier'][:s['line']-1] + ('1' if s['carrier'][s['line']-1]=='0' else '0') + s['carrier'][s['line']:],
    expected_collision=True,
    expected_transition_split=True,
)

# 6. Ancient Yi: scalar without width/order convention is not enough for carrier-level operation.
yi = Witness(
    name='ancient_yi_width_convention_loss',
    x={'scalar':1,'width':2,'digits_lsd':'10'},
    y={'scalar':1,'width':3,'digits_lsd':'100'},
    projection=lambda s: s['scalar'],
    transition=lambda s: (s['width'], s['digits_lsd']),
    expected_collision=True,
    expected_transition_split=True,
)

witnesses=[monodromy,branch,nc,pencil,wilhelm,yi]
results=[run(w) for w in witnesses]
for name, passed, same_projection, transition_split, px, py, tx, ty in results:
    print(f'{name}: {"PASS" if passed else "FAIL"} | same_projection={same_projection} | transition_split={transition_split}')
assert all(r[1] for r in results)
print('SUMMARY: PASS all v54 state-completeness negative controls')
