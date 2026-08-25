#!/usr/bin/env python3
import sympy as sp
x,y,z=sp.symbols('x y z')
A=sp.Matrix([[-1,0,0],[0,0,-1],[0,-1,0]])
v=sp.Matrix([x,y,z])
checks={
    'A_squared_identity': A*A == sp.eye(3),
    'orthogonal': A.T*A == sp.eye(3),
    'det_plus_one': A.det() == 1,
    'norm_preserved': sp.expand((A*v).dot(A*v)-v.dot(v)) == 0,
}
print('PENCIL_YINYANG_TRANSFORM_ATTACK_V48')
for k,val in checks.items(): print(k, 'PASS' if val else 'FAIL')
print('matrix=', A.tolist())
assert all(checks.values())
