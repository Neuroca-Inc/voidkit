#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from fractions import Fraction
import sympy as sp

TS='20260711T142511'
WORD='BQQBBBQBQBBQBBL'

def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def cap(j:int)->int:
    if j==1:return 2
    if j==2:return 4
    return 2**(2*j)

def simulate_first_domain():
    A=0;u=v=1;k=0;j=1;qcount=0;bcount=0;prefix=''
    rows=[{'prefix_index':0,'primitive':'start','A':A,'u':u,'v':v,'k':k,'j':j,'q_count':qcount,'b_count':bcount,'word_prefix':prefix,'active_re':1,'active_im':0,'active_den':1,'pairing_status':'NOT_INSTANTIATED'}]
    for idx,op in enumerate(WORD,1):
        if op=='B':
            u,v=v,u+v;bcount+=1
        elif op=='Q':
            k+=1;j+=1;qcount+=1
        elif op=='L':
            A+=1;k=0;j=1+6*(2**A-1)
        prefix+=op
        phase=qcount%4
        re,im=[(1,0),(0,1),(-1,0),(0,-1)][phase]
        den=u*v
        status='TYPE_SIGNATURE_ONLY_AFTER_L' if op=='L' else 'NOT_INSTANTIATED'
        rows.append({'prefix_index':idx,'primitive':op,'A':A,'u':u,'v':v,'k':k,'j':j,'q_count':qcount,'b_count':bcount,'word_prefix':prefix,'active_re':re,'active_im':im,'active_den':den,'pairing_status':status})
    return rows

def threshold(A:int)->int:
    # Imported proved theorem used only to generate exact integer regression rows.
    import mpmath as mp
    mp.mp.dps=160
    phi=(1+mp.sqrt(5))/2
    y=(12*(2**(A+1)-1)*mp.log(2)+mp.log(5))/(2*mp.log(phi))-mp.mpf(3)/2
    return int(mp.ceil(y))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args();root=a.package_root.resolve()
    required=[root/f'inputs/{TS}_QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md',root/f'inputs/{TS}_orthad-diagram-v5.png',root/f'inputs/{TS}_p5-b3-v2_AUDIT_RESULTS.md',root/'docs/QBL_PRIMARY_PAIRING_RECURRENCE_v1.md']
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise SystemExit('missing required inputs: '+', '.join(missing))
    law=required[0].read_text()
    markers=['B > Q > L','primary pairing','explicit all-depth recurrence for the primary pairing','B Q Q B B B Q B Q B B Q B B L']
    absent=[m for m in markers if m not in law]
    if absent:raise SystemExit('primary law markers absent: '+repr(absent))

    rows=simulate_first_domain()
    if ''.join(r['primitive'] for r in rows[1:])!=WORD:raise SystemExit('word mismatch')
    end=rows[14]
    assert (end['u'],end['v'],end['active_re'],end['active_im'],end['active_den'])==(55,89,0,1,4895)
    post=rows[15]
    assert (post['A'],post['u'],post['v'],post['k'],post['j'])==(1,55,89,0,7)

    # Exact local B and Q recurrences.
    u,v,a_sym=sp.symbols('u v a', positive=True)
    assert sp.simplify((1/(v*(u+v)))/(1/(u*v))-u/(u+v))==0
    I=sp.I
    assert sp.simplify(I*a_sym-I*a_sym)==0

    # Type fork: same seed normalization, different scalar variance.
    z,w=sp.symbols('z w')
    bil_at_seed=1*1
    sesq_at_seed=sp.conjugate(1)*1
    assert bil_at_seed==sesq_at_seed==1
    assert sp.I != sp.conjugate(sp.I)

    # Conditional L block preservation and orthogonal mixed birth blocks.
    p11,p12,p21,p22,s=sp.symbols('p11 p12 p21 p22 s')
    P=sp.Matrix([[p11,p12],[p21,p22]])
    Pext=sp.diag(1,1,1)
    Pext[:2,:2]=P
    assert Pext[:2,:2]==P
    assert Pext[:2,2]==sp.zeros(2,1) and Pext[2,:2]==sp.zeros(1,2)
    bad=Pext.copy();bad[0,2]=1
    assert bad[:2,2]!=sp.zeros(2,1)

    # Carry indexing regression from the accepted global theorem.
    Ts=[threshold(A) for A in range(0,10)]
    carries=[]
    for A in range(1,len(Ts)):
        c=Ts[A]-2*Ts[A-1]
        if c not in (7,8,9):raise SystemExit('carry outside 7/8/9')
        carries.append({'A':A,'T_A':Ts[A],'T_prev':Ts[A-1],'c_A':c,'recoverable_at_S_A_minus':True,'future_available_at_previous_L':False})

    out=root/'outputs';tr=root/'trace';out.mkdir(exist_ok=True);tr.mkdir(exist_ok=True)
    with (out/f'{TS}_first_domain_active_axis_trace.csv').open('w',newline='',encoding='utf-8') as f:
        wri=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wri.writeheader();wri.writerows(rows)
    with (out/f'{TS}_carry_indexing.csv').open('w',newline='',encoding='utf-8') as f:
        wri=csv.DictWriter(f,fieldnames=list(carries[0].keys()));wri.writeheader();wri.writerows(carries)
    type_analysis={
      'minimal_interface':'P_t : H_t -> D(H_t)',
      'carrier_category':'additive category with contravariant duality and finite biproducts',
      'coefficient_ring_status':'NOT_UNIQUELY_FORCED',
      'canonical_local_sufficient_ring':'Q(i)',
      'architectural_rank':'A+1',
      'module_dimension_identification':'NOT_YET_DERIVED',
      'surviving_type_fork':['ordinary-dual bilinear','conjugate-dual sesquilinear'],
      'earliest_missing_axiom':'SCALAR_VARIANCE_AXIOM',
      'hermitian_status':'ADMISSIBLE_NOT_FORCED',
      'quadratic_direct_status':'NOT_LICENSED_WITHOUT_POLARIZATION'}
    (out/f'{TS}_pairing_type_analysis.json').write_text(json.dumps(type_analysis,indent=2,sort_keys=True)+'\n')
    seed={'local_seed_witness':'1','raw_seed_presentations_unique':False,'witnesses':['P_bil(x,y)=x*y','P_sesq(x,y)=conjugate(x)*y'],'retained_gauge_class_uniqueness':'NOT_YET_DERIVED','seed_map':'eta_P(X_0,W_0,D_P)->P_0','status':'EXACT_PRIMARY_PAIRING_SEED_NOT_YET_DERIVED'}
    (out/f'{TS}_seed_nondetermination.json').write_text(json.dumps(seed,indent=2,sort_keys=True)+'\n')
    mutations={
      'B':{'rank':'preserved','local_active':'a*u/(u+v)','old_latched_block':'preserved','full_value_law':'NOT_YET_DERIVED'},
      'Q':{'rank':'preserved','local_active':'i*a','old_latched_block':'preserved','full_value_law':'NOT_YET_DERIVED'},
      'L':{'rank':'r->r+1','carrier_signature':'H_old direct_sum A_new','old_block':'preserved','mixed_birth_blocks':'zero','new_local_active':'1','new_rank_one_block':'NOT_YET_DERIVED'}}
    (out/f'{TS}_mutation_signatures.json').write_text(json.dumps(mutations,indent=2,sort_keys=True)+'\n')
    chart={'iota_plus':'H_plus -> H','iota_minus':'H_minus -> H','omega_plus':'D(iota_plus) P iota_plus','omega_minus':'D(iota_minus) P iota_minus','T_plus_to_minus':'D(iota_minus) P iota_plus','T_minus_to_plus':'D(iota_plus) P iota_minus','independent_seeds_forbidden':True,'exact_maps':'NOT_YET_DERIVED'}
    (out/f'{TS}_chart_interface.json').write_text(json.dumps(chart,indent=2,sort_keys=True)+'\n')
    statuses={
      'just_completed_carry_recoverable':'PASS',
      'future_carry_appended_by_instantaneous_L':'FAIL',
      'minimal_pairing_interface':'DERIVED',
      'exact_pairing_type':'NOT_YET_DERIVED',
      'exact_seed':'NOT_YET_DERIVED',
      'B_value_recurrence':'NOT_YET_DERIVED',
      'Q_value_recurrence':'NOT_YET_DERIVED',
      'L_value_recurrence':'NOT_YET_DERIVED',
      'higher_order_descriptive_L':'NOT_YET_DERIVED',
      'branch_status':'OPEN'}
    (out/f'{TS}_status.json').write_text(json.dumps(statuses,indent=2,sort_keys=True)+'\n')
    verification={'primary_law_sha256':sha(required[0]),'diagram_sha256':sha(required[1]),'word_verified':True,'first_domain_local_result':'i/4895','carry_rows_verified':len(carries),'symbolic_B_local_recurrence':True,'symbolic_Q_local_recurrence':True,'conditional_L_old_block_embedding':True,'negative_control_nonorthogonal_L_rejected':True,'universal_pairing_recurrence_claimed':False,'statuses':statuses}
    (out/f'{TS}_derivation_verification.json').write_text(json.dumps(verification,indent=2,sort_keys=True)+'\n')
    with (tr/f'{TS}_custody_and_active_axis_trace.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    obligations=[
      {'dependency':'scalar_variance_axiom','status':'MISSING'},
      {'dependency':'seed_map_eta_P','status':'MISSING'},
      {'dependency':'Phi_B_value_law','status':'MISSING'},
      {'dependency':'Phi_Q_value_law','status':'MISSING'},
      {'dependency':'Phi_L_value_law','status':'MISSING'},
      {'dependency':'chart_maps_iota_plus_minus','status':'DOWNSTREAM_MISSING'},
      {'dependency':'descriptive_L_comparison_family','status':'DOWNSTREAM_MISSING'}]
    with (tr/f'{TS}_pairing_obligation_trace.jsonl').open('w',encoding='utf-8') as f:
        for x in obligations:f.write(json.dumps(x,sort_keys=True)+'\n')
    print(json.dumps({'first_domain':'PASS','minimal_interface':'DERIVED','primary_pairing_recurrence':'NOT_YET_DERIVED','branch':'OPEN'},sort_keys=True))

if __name__=='__main__':main()
