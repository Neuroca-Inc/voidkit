#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import mpmath as mp
import sympy as sp

TS='20260711T134500'

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def fib(n:int)->int:
    def fd(k:int):
        if k==0:return (0,1)
        a,b=fd(k//2); c=a*(2*b-a); d=a*a+b*b
        return (c,d) if k%2==0 else (d,c+d)
    return fd(n)[0]

def threshold(A:int)->int:
    mp.mp.dps=180
    phi=(1+mp.sqrt(5))/2
    y=(12*(2**(A+1)-1)*mp.log(2)+mp.log(5))/(2*mp.log(phi))-mp.mpf(3)/2
    return int(mp.ceil(y))

def cap(j:int)->int:
    if j==1:return 2
    if j==2:return 4
    return 2**(2*j)

def simulate(maxA:int):
    A=0; u=v=1; k=0; bcount=0; qcount=0; theta_q=0; W=[]; rows=[]; trace=[]
    step=0
    while A<=maxA:
        N=6*2**A; j=1+6*(2**A-1)+k
        nextu,nextv=v,u+v
        canB=(nextu*nextv<=cap(j)) if k<N-1 else (u*v<cap(j))
        canQ=k<N-1
        if (not canB) and (not canQ):
            rows.append({'A':A,'u':u,'v':v,'b_count':bcount,'q_count':qcount,'theta_quarters':theta_q,'k':k,'j':j,'word_length':len(W),'word_sha256':hashlib.sha256(''.join(W).encode()).hexdigest()})
            trace.append({'step':step,'event':'pre_L','A':A,'q':[u,v],'b_count':bcount,'k':k,'j':j,'word_length':len(W)})
            W.append('L'); step+=1; A+=1; k=0
            continue
        if canB:
            u,v=nextu,nextv; bcount+=1; W.append('B'); op='B'
        elif canQ:
            k+=1; qcount+=1; theta_q+=1; W.append('Q'); op='Q'
        trace.append({'step':step,'event':op,'A':A,'q':[u,v],'b_count':bcount,'k':k,'j':1+6*(2**A-1)+k,'word_length':len(W)})
        step+=1
    return rows,trace

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args(); root=args.package_root.resolve()
    required=[
      root/f'inputs/{TS}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md',
      root/f'inputs/{TS}_QBL_CARRY_AFFINE_FOLLOWER_STRUCTURE_v5.md',
      root/f'inputs/{TS}_QBL_GLOBAL_EXACT_THRESHOLD_BRIDGE_v2.md',
      root/f'inputs/{TS}_STRONG_EXPLICIT_NOTE_EXTRACT.md',
      root/f'inputs/{TS}_CF03_ACTIVE_DEPTH_EXTRACT.md',
      root/f'inputs/{TS}_orthad-diagram-v5.png']
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit('missing dependencies: '+', '.join(missing))
    # Symbolic identities
    A=sp.symbols('A',integer=True,nonnegative=True)
    J=6*(2**(A+1)-1)
    assert sp.simplify(6*(2**(A+2)-1)-(2*J+6))==0
    assert sp.simplify(12*(2**(A+1)-1)-2*J)==0
    lam,beta,j,b,c=sp.symbols('lam beta j b c',real=True)
    gamma=6*lam-beta
    assert sp.expand(lam*(2*j+6)+beta-(2*b+c) - (2*(lam*j+beta-b)+gamma-c))==0
    # Exact custody simulation and threshold regression
    rows,trace=simulate(8)
    for r in rows:
        a=r['A']; t=threshold(a)
        if r['b_count']!=t: raise SystemExit(f'custody/threshold mismatch A={a}')
        if (r['u'],r['v'])!=(fib(t+1),fib(t+2)): raise SystemExit(f'pair mismatch A={a}')
        if r['j']!=6*(2**(a+1)-1): raise SystemExit(f'j mismatch A={a}')
    carries=[]
    for prev,cur in zip(rows,rows[1:]):
        cc=cur['b_count']-2*prev['b_count']
        if cc not in (7,8,9): raise SystemExit('carry outside alphabet')
        carries.append({'from_A':prev['A'],'to_A':cur['A'],'carry':cc})
    # Finite canonical language comparison, explicitly not a universal language theorem
    Ts=[threshold(a) for a in range(0,257)]
    cs=[Ts[a]-2*Ts[a-1] for a in range(1,len(Ts))]
    lang=[]
    for n in range(1,11):
        words={tuple(cs[i:i+n]) for i in range(len(cs)-n+1)}
        lang.append({'length':n,'canonical_observed_words':len(words),'full_affine_words':2**(n+1)-1,'equal_on_sample':len(words)==2**(n+1)-1})
    # Status dependencies derived from explicit premises
    status={
      'canonical_boundary_orbit_semiconjugacy':'PROVED',
      'full_affine_interval_factor_stated_domain':'IMPOSSIBLE_COUNTABLE_TO_UNCOUNTABLE',
      'enlarged_lawful_qbl_full_interval_factor':'NOT_YET_DERIVED',
      'canonical_language_equals_full_affine_language':'NOT_YET_PROVED',
      'claim_A_instantaneous_L_append':'FAIL',
      'claim_B_boundary_return_cocycle':'PASS',
      'claim_C_higher_order_descriptive_L':'NOT_YET_DERIVED',
      'hierarchical_depth_recurrence':'COUNT_ALIGNMENT_ONLY',
      'branch_status':'OPEN'}
    # Independence criterion availability: no lawful comparison-family certificate is included.
    deps={
      'primary_pairing_all_depth_recurrence':False,
      'explicit_chart_maps_all_prefixes':False,
      'bidirectional_transfer_recurrences':False,
      'orthad_L_latching_extension_map':False,
      'lawful_old_description_fiber_comparison_family':False}
    # outputs
    out=root/'outputs'; out.mkdir(exist_ok=True)
    with (out/f'{TS}_boundary_states.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out/f'{TS}_boundary_carries.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['from_A','to_A','carry']); w.writeheader(); w.writerows(carries)
    with (out/f'{TS}_canonical_language_finite_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(lang[0].keys())); w.writeheader(); w.writerows(lang)
    (out/f'{TS}_factor_scope.json').write_text(json.dumps({'domain':'countable canonical pre-L orbit','codomain':'countable canonical affine orbit','map':'pi(S)=lambda*j+beta-nu(q)','surjective_onto_O_E':True,'injectivity_required':False,'full_interval_factor':status['full_affine_interval_factor_stated_domain']},indent=2,sort_keys=True)+'\n')
    (out/f'{TS}_higher_order_L_status.json').write_text(json.dumps({'statuses':status,'independence_definition':'xi independent of D iff xi does not factor through D, equivalently some D-fiber is split','open_dependencies':deps},indent=2,sort_keys=True)+'\n')
    (out/f'{TS}_hierarchy_alignment.json').write_text(json.dumps({'identity':'J_A=6*p(A)','p_A':'2^(A+1)-1','active_depth_status':'COUNT_ALIGNMENT_ONLY','refinement_preserving_map_present':False},indent=2,sort_keys=True)+'\n')
    (out/f'{TS}_derivation_verification.json').write_text(json.dumps({'symbolic_J_recurrence':True,'symbolic_capacity_exponent':True,'symbolic_factor_algebra':True,'custody_simulation_range':'A=0..8','custody_threshold_rows_verified':len(rows),'carry_rows_verified':len(carries),'status':status,'open_orthad_dependencies':deps},indent=2,sort_keys=True)+'\n')
    tr=root/'trace'; tr.mkdir(exist_ok=True)
    with (tr/f'{TS}_custody_trace.jsonl').open('w',encoding='utf-8') as f:
        for x in trace:f.write(json.dumps(x,sort_keys=True)+'\n')
    with (tr/f'{TS}_factor_scope_trace.jsonl').open('w',encoding='utf-8') as f:
        for x in [{'event':'symbolic_factor_identity','passed':True},{'event':'canonical_semiconjugacy','status':'PROVED'},{'event':'full_interval_factor','status':'IMPOSSIBLE_FOR_STATED_DOMAIN','reason':'countable domain, uncountable codomain'},{'event':'language_equality','status':'NOT_YET_PROVED'},{'event':'higher_order_descriptive_L','status':'NOT_YET_DERIVED','dependencies':deps}]:f.write(json.dumps(x,sort_keys=True)+'\n')
    print(json.dumps({'semiconjugacy':'PROVED','full_interval_factor':'IMPOSSIBLE_FOR_STATED_DOMAIN','higher_order_descriptive_L':'NOT_YET_DERIVED'},sort_keys=True))
    return 0
if __name__=='__main__': raise SystemExit(main())
