#!/usr/bin/env python3
from pathlib import Path
import sympy as sp
import json

A=sp.symbols('A', integer=True, nonnegative=True)
F=[0,1]
for _ in range(2,80): F.append(F[-1]+F[-2])

def U(d,x):
    if d==0: return x
    return sp.expand((F[d]*x+F[d-1])*(F[d+1]*x+F[d]))

results={}
results['U7']=str(sp.expand(U(7,A)))
results['U8']=str(sp.expand(U(8,A)))
results['U9']=str(sp.expand(U(9,A)))
results['U10']=str(sp.expand(U(10,A)))
results['frontier_lock_lower_residual_U8_Aminus1_minus_U7_A']=str(sp.factor(U(8,A-1)-U(7,A)))
results['frontier_lock_upper_residual_U8_A_minus_U8_Aminus1']=str(sp.factor(U(8,A)-U(8,A-1)))
results['frontier_plus_lower_residual_U10_Aminus1_plus_A_minus_U9_A']=str(sp.factor(U(10,A-1)+A-U(9,A)))
results['frontier_plus_upper_residual_U10_A_minus_U10_Aminus1_plus_A']=str(sp.factor(U(10,A)-(U(10,A-1)+A)))
# finite checks
results['frontier_lock_verified_for_A_3_to_100000'] = all((U(7,n) < U(8,n-1) <= U(8,n)) for n in range(3,100001))
results['frontier_plus_depth10_verified_for_A_2_to_100000'] = all((U(9,n) < U(10,n-1)+n <= U(10,n)) for n in range(2,100001))
root=Path(__file__).resolve().parents[1]
(root/'results/sympy_proof_results.json').write_text(json.dumps(results,indent=2))
with (root/'results/sympy_proof_results.txt').open('w') as f:
    for k,v in results.items():
        f.write(f'{k}: {v}\n')
print(json.dumps(results,indent=2))
