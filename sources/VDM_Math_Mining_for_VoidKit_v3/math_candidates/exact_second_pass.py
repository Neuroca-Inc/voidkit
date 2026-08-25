#!/usr/bin/env python3
from __future__ import annotations
import json
from itertools import product
from pathlib import Path
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]

# VDM runtime involution on [0,1] and scaled [0,12].
phi, lam = sp.symbols('phi lam', real=True)
sigma=lambda x: 1-x
V=lambda x: lam*x**2*(1-x)**2
dV=lambda x: 2*lam*x*(1-x)*(1-2*x)
assert sp.simplify(sigma(sigma(phi))-phi)==0
assert sp.simplify(V(sigma(phi))-V(phi))==0
assert sp.simplify(dV(sigma(phi))+dV(phi))==0

# Telegraph update equivariance. The reflected source is -J and the reflected
# weighted Laplacian is -lap because constants are annihilated.
tau,D,lap,J,prev=sp.symbols('tau D lap J prev', real=True)
T=lambda x,xprev,L,S: (D*L-dV(x)+S+(2*tau+1)*x-tau*xprev)/(tau+1)
telegraph_res=sp.simplify(T(1-phi,1-prev,-lap,-J) - (1-T(phi,prev,lap,J)))
assert telegraph_res==0
x,y=sp.symbols('x y', real=True)
bond=lambda a,b: sp.Rational(1,2)*(b-a)**2
assert sp.simplify(bond(1-x,1-y)-bond(x,y))==0

# Z12 involutions.
N=12
H={r:(r+6)%N for r in range(N)}
S={r:(-r)%N for r in range(N)}
assert all(H[H[r]]==r for r in range(N))
assert all(S[S[r]]==r for r in range(N))
assert all(H[S[r]]==S[H[r]] for r in range(N))

# Recovered finite chart formula.
def f(r,c,s,p):
    return (-2*((r+s)%12)*((c+s)%12)+6*p)%24

def plus_cov(r): return r != 6
def minus_cov(r): return r != 0
chi={r:(12 if r%2 else 0) for r in range(12)} # phase 0 or pi in units pi/12

checks=0
residuals=[]
for s in range(6):
    for p in range(4):
        max_res=0
        for r in range(12):
            for c in range(12):
                if not (plus_cov(r) and plus_cov(c)): continue
                hr,hc=H[r],H[c]
                assert minus_cov(hr) and minus_cov(hc)
                # Omega^-_{H r,H c} chi_c = chi_r conjugate(Omega^+_{r,c})
                lhs=(-f(hr,hc,s,p)+chi[c])%24
                rhs=(-f(r,c,s,p)+chi[r])%24
                delta=(lhs-rhs)%24
                max_res=max(max_res, min(delta,24-delta))
                checks+=1
        residuals.append({'shift':s,'phase_mod4':p,'max_mod24_residual':max_res})
assert checks==24*11*11
assert all(row['max_mod24_residual']==0 for row in residuals)

# Enumerate normalized binary phase characters on the covered source seats.
# Seat 6 is absent from Omega+ and therefore not constrained by the chart equation.
covered=[r for r in range(12) if r!=6]
solutions=[]
for bits in product([0,12], repeat=len(covered)-1):
    d={covered[0]:0}
    for r,b in zip(covered[1:],bits): d[r]=b
    ok=True
    for s in range(6):
        for p in range(4):
            for r in covered:
                for c in covered:
                    lhs=(-f(H[r],H[c],s,p)+d[c])%24
                    rhs=(-f(r,c,s,p)+d[r])%24
                    if lhs!=rhs:
                        ok=False; break
                if not ok: break
            if not ok: break
        if not ok: break
    if ok: solutions.append(d)
assert len(solutions)==1
assert all(solutions[0][r]==chi[r] for r in covered)

# Transfer reflection support in recovered compiler.
transfer_rows=[]
for r in range(12):
    c=S[r]
    if plus_cov(r) and minus_cov(c): transfer_rows.append((r,c))
assert transfer_rows==[(r,(-r)%12) for r in range(1,6)]+[(r,(-r)%12) for r in range(7,12)]

out={
 'active_evidence_class':'CONDITIONAL DOWNSTREAM',
 'runtime_involution':{
   'active_evidence_class':'RECOVERED',
   'sigma':'1-phi', 'scaled_sigma':'12-x',
   'sigma_squared':'identity','potential_invariant':True,'force_anti_invariant':True,
   'telegraph_equivariance_with_source_reflection':True,
   'bond_source_invariant':True,
 },
 'z12':{
   'active_evidence_class':'PROVISIONAL',
   'half_turn_H':H,'valley_reflection_S':S,
   'H_squared':'identity','S_squared':'identity','commute':True,
   'S_fixed_points':[r for r in range(12) if S[r]==r],
 },
 'recovered_chart':{
   'active_evidence_class':'CONDITIONAL DOWNSTREAM',
   'exact_entry_checks':checks,
   'all_residuals_zero':True,
   'normalized_character_solutions_on_covered_seats':len(solutions),
   'character_phase':solutions[0],
   'excluded_source_seat':6,
   'excluded_target_seat':0,
   'transfer_reflection_support':transfer_rows,
 },
 'residuals':residuals,
}
(ROOT/'outputs/EXACT_SECOND_PASS_RESULTS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'pass':True,'entry_checks':checks,'character_solutions':len(solutions),'transfer_edges':len(transfer_rows)},sort_keys=True))
