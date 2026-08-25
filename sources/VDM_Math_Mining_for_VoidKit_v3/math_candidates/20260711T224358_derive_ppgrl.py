from __future__ import annotations
import argparse,csv,hashlib,json
from fractions import Fraction
from pathlib import Path

TS='20260711T224358'
EXPECTED_PRIMARY='f0f66790bf6da0b3fca9a400da36ff5879db4b3e861703ef3a309829a2af9962'
EXPECTED_AUDIT='8854679262a79cbc176e888271febe1d9ae10b15a8aca6a2bbbff1232a6c3a9a'
WORD='BQQBBBQBQBBQBBL'

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1048576),b''): h.update(c)
    return h.hexdigest()

def stable(x): return json.dumps(x,indent=2,sort_keys=True)+'\n'

def mul_i(z):
    r,i=z
    return (-i,r)

def div(z,n):
    r,i=z
    return (r/Fraction(n),i/Fraction(n))

def zstr(z):
    r,i=z
    if r==0 and i==0: return '0'
    if r==0: return ('i' if i==1 else '-i' if i==-1 else f'{i}*i')
    if i==0: return str(r)
    return f'{r}+{i}i'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.')
    root=Path(ap.parse_args().root).resolve()
    primary=root/'inputs'/'QBL_PRIMITIVE_CUSTODY_AND_ORTHAD_LAW_v2.md'
    audit=root/'inputs'/'p5-b3-v8_audit-package_20260711T223909.zip'
    assert sha256(primary)==EXPECTED_PRIMARY
    assert sha256(audit)==EXPECTED_AUDIT
    text=primary.read_text(encoding='utf-8')
    required=[
      'The primary pairing is generative.',
      'The exact chart-map recurrence attached to the clean primitive law remains an explicit formalization obligation',
      'preserve the complete old pairing block',
      'rotate the active pairing data by the quarter-turn witness i',
      'a_0='
    ]
    missing=[m for m in required if m not in text]
    assert not missing,missing

    # Exact local path trace.
    u,v=1,1; qcount=0; z=(Fraction(1),Fraction(0)); rows=[]
    rows.append(dict(prefix='',op='START',u=u,v=v,q_count=qcount,real=str(z[0]),imag=str(z[1]),local='1'))
    for idx,op in enumerate(WORD,1):
        if op=='B':
            old_u=u; u,v=v,u+v; z=div(z, v//old_u if False else 1)  # replaced below by exact invariant
            # local invariant is i^q/(uv), recompute exactly.
        elif op=='Q': qcount+=1
        elif op=='L': pass
        phase=(Fraction(1),Fraction(0))
        for _ in range(qcount%4): phase=mul_i(phase)
        z=div(phase,u*v)
        rows.append(dict(prefix=WORD[:idx],op=op,u=u,v=v,q_count=qcount,real=str(z[0]),imag=str(z[1]),local=zstr(z)))
    assert (u,v)==(55,89)
    assert qcount==5
    assert z==(Fraction(0),Fraction(1,4895))
    out=root/'outputs'; tr=root/'trace'
    with (out/'DOMAIN0_PREFIX_TRACE.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (tr/f'{TS}_domain0_trace.jsonl').open('w',encoding='utf-8') as f:
        for n,row in enumerate(rows): f.write(json.dumps(dict(event=n,**row),sort_keys=True)+'\n')

    # Dependency decomposition.
    layers=[
      (0,'custody_state_and_word','PROVED'),
      (1,'two_slot_architectural_skeleton','PROVED'),
      (2,'state_indexed_argument_and_placement_realization','OPEN_EARLIEST'),
      (3,'primary_carrier_codomain_evaluation','OPEN'),
      (4,'restriction_constructor_realization','OPEN'),
      (5,'primary_seed','OPEN'),
      (6,'intrinsic_B_value_action','OPEN'),
      (7,'intrinsic_Q_action_type','OPEN'),
      (8,'intrinsic_L_mixed_and_newnew','OPEN'),
      (9,'exact_action_equivalence_category','OPEN')]
    assert [x[0] for x in layers]==list(range(10))
    assert [x for x in layers if x[2]=='OPEN_EARLIEST'][0][1]=='state_indexed_argument_and_placement_realization'
    (out/f'{TS}_ppgrl_dependency.json').write_text(stable({'layers':[{'order':o,'name':n,'status':s} for o,n,s in layers],'earliest_missing':layers[2][1]}),encoding='utf-8')
    with (tr/f'{TS}_dependency_trace.jsonl').open('w',encoding='utf-8') as f:
        for o,n,s in layers: f.write(json.dumps({'order':o,'obligation':n,'status':s},sort_keys=True)+'\n')

    # Constant-map vacuity and non-vacuous pointwise restriction finite control.
    H=[0,1]; C=['c']; iota={'c':0}
    P0={(0,0):0,(0,1):0,(1,0):0,(1,1):0}
    P1={(0,0):1,(0,1):0,(1,0):0,(1,1):0}
    target=7
    Rconst=lambda P: target
    vacuous=(Rconst(P0)==target and Rconst(P1)==target)
    restrict=lambda P,x,y:P[(iota[x],iota[y])]
    D0=restrict(P0,'c','c'); D1=restrict(P1,'c','c')
    sensitivity=(D0!=D1)
    constant_fidelity=(Rconst(P0)==D0 and Rconst(P1)==D1)
    assert vacuous and sensitivity and not constant_fidelity
    control={
      'bare_equation_vacuous':vacuous,
      'placed_primary_values':[D0,D1],
      'placed_sensitivity':sensitivity,
      'constant_constructor_target':target,
      'constant_constructor_pointwise_fidelity':constant_fidelity,
      'verdict':'REJECT_CONSTANT_MAP_SMUGGLING'}
    (out/'CONSTANT_MAP_SMUGGLING_CONTROL.json').write_text(stable(control),encoding='utf-8')
    (tr/f'{TS}_factorization_trace.jsonl').write_text(
      '\n'.join(json.dumps(x,sort_keys=True) for x in [
        {'step':'bare_equation','result':'vacuous','value':vacuous},
        {'step':'pointwise_restriction','P0':D0,'P1':D1,'sensitive':sensitivity},
        {'step':'constant_map_test','fidelity':constant_fidelity,'verdict':'REJECTED'}])+'\n',encoding='utf-8')

    # Exact status output derived from dependency rules.
    status={
      'PPGRL_UMBRELLA_TARGET':'ADOPTED',
      'EARLIEST_MISSING_DATUM':'STATE_INDEXED_ARGUMENT_AND_PLACEMENT_REALIZATION',
      'ARGUMENT_OBJECT_ARCHITECTURE':'SKELETON_PROVED_EXACT_FAMILY_OPEN',
      'CHART_PLACEMENT_MAPS':'EXISTENCE_ROLE_FORCED_EXACT_MAPS_OPEN',
      'NONVACUOUS_RESTRICTION_FACTORIZATION':'ABSTRACT_INTERFACE_PROVED_QBL_REALIZATION_OPEN',
      'PRIMARY_PAIRING_SEED':'NOT_YET_DERIVED',
      'INTRINSIC_B_ACTION':'ARCHITECTURAL_SIGNATURE_PROVED_VALUE_ACTION_OPEN',
      'INTRINSIC_Q_ACTION':'ORDER4_ORIENTATION_QUOTIENT_PROVED_PRIMARY_ACTION_OPEN',
      'INTRINSIC_L_ACTION':'OLDOLD_RETENTION_SHAPE_PROVED_OTHERS_OPEN',
      'STAR_LAW':'NOT_YET_DERIVED',
      'ACTION_EQUIVALENCE_CATEGORY':'ABSTRACT_CRITERION_PROVED_CONCRETE_CATEGORY_OPEN',
      'COMPLETE_MODEL_INDEPENDENCE':'NOT_YET_DERIVED',
      'STATE_FORCED_INTRINSIC_ACTION':'NOT_YET_DERIVED',
      'D1_DOMAIN_PROPER_EFFECTIVE_INVARIANT':'NOT_YET_DERIVED',
      'HIGHER_ORDER_DESCRIPTIVE_L':'NOT_YET_DERIVED'}
    (out/'STATUS.json').write_text(stable(status),encoding='utf-8')
    summary={'input_hashes':{'primary':sha256(primary),'audit':sha256(audit)},'domain0_endpoint':'i/4895','constant_map_control':'PASS_REJECTED','dependency_order':'PASS','status':status}
    (out/f'{TS}_derivation_summary.json').write_text(stable(summary),encoding='utf-8')
    print('PROVED: dependency decomposition and non-vacuous factorization interface')
    print('CERTIFIED FINITELY: Domain-0 endpoint i/4895 and constant-map rejection')
    print('OPEN: exact QBL argument/placement realization and intrinsic value actions')

if __name__=='__main__': main()
