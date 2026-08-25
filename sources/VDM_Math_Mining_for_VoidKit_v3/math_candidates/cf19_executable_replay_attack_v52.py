#!/usr/bin/env python3
from math import isclose

def B(pair):
    u,v=pair
    a,b=v,u+v
    return tuple(sorted((a,b)))

def iterB(pair,n):
    p=tuple(sorted(pair))
    for _ in range(n): p=B(p)
    return p

checks = []
checks.append(('origin_11_to_55_89', iterB((1,1),9)==(55,89)))
checks.append(('small_lift_12_to_55_89', iterB((1,2),8)==(55,89)))
checks.append(('origin_uv_4895', iterB((1,1),9)[0]*iterB((1,1),9)[1]==4895))
checks.append(('non_origin_13_not_origin_anchor', iterB((1,3),8)==(76,123)))
checks.append(('non_origin_2_13_not_origin_anchor', iterB((2,13),5)==(71,114)))
for name, ok in checks:
    print(f'{name}: {"PASS" if ok else "FAIL"}')
if not all(ok for _,ok in checks): raise SystemExit(1)
print('CF19 v52 state-completeness arithmetic attack: PASS')
