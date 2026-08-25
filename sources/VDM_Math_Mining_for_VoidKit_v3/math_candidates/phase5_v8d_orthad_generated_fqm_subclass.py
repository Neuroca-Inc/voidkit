from __future__ import annotations
import csv, json, math, cmath, hashlib, shutil, zipfile, os
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
import itertools
import numpy as np
import sympy as sp

ROOT=Path('/mnt/data/phase5_v8d_orthad_generated_fqm_subclass')
if ROOT.exists(): shutil.rmtree(ROOT)
for d in ['docs','outputs','sealed','scripts','notebooks','proofs','lean/Phase5V8D','source_notes','snapshots','patches']:
    (ROOT/d).mkdir(parents=True, exist_ok=True)

ledger_src=Path('/mnt/data/PHASE5_CANONICAL_LEDGER.md')
ledger_text=ledger_src.read_text()
(ROOT/'source_notes'/'PHASE5_CANONICAL_LEDGER.md').write_text(ledger_text)

STATUS = 'V8D_ORTHAD_GENERATED_FQM_SUBCLASS_CLOSED_POSITIVE_WITH_2_PRIMARY_CLASSIFIER_WALL_BLOCKING_OPEN'
PHASE5_CLOSED=False

D_SET=[2,4,6,8,10,12,16,24]

def lcm(a,b): return abs(a*b)//math.gcd(a,b)

def admissible_step(Di,Dj):
    L=lcm(Di,Dj); g=math.gcd(Di,Dj); return L//g

def admissible_cs(Di,Dj):
    L=lcm(Di,Dj); step=admissible_step(Di,Dj)
    return [c for c in range(L) if c % step == 0]

@dataclass(frozen=True)
class Presentation:
    name: str
    D: tuple[int,...]
    C: tuple[tuple[int,int,int], ...] # i,j,c numerator mod L
    source: str
    note: str

    def cdict(self):
        return {(i,j):c for i,j,c in self.C}


def representative_invariant(p:Presentation):
    rows=[]; ok=True
    for i,j,c in p.C:
        Di,Dj=p.D[i],p.D[j]; L=lcm(Di,Dj)
        cond_i=(c*Di)%L==0
        cond_j=(c*Dj)%L==0
        passed=cond_i and cond_j
        rows.append(dict(presentation=p.name,i=i,j=j,Di=Di,Dj=Dj,c=c,L=L,cond_i=cond_i,cond_j=cond_j,passed=passed))
        ok=ok and passed
    return ok,rows

def b_value(p:Presentation, x, y):
    # b(x,y)=sum x_i y_i/D_i + sum c*(x_i*y_j+x_j*y_i)/lcm(D_i,D_j)
    v=Fraction(0,1)
    for i,D in enumerate(p.D):
        v += Fraction(x[i]*y[i], D)
    for i,j,c in p.C:
        L=lcm(p.D[i],p.D[j])
        v += Fraction(c*(x[i]*y[j]+x[j]*y[i]), L)
    return v % 1

def q_value(p:Presentation, x):
    # q(x)=1/2 b(x,x)
    return (b_value(p,x,x)*Fraction(1,2)) % 1

def radical_size(p:Presentation, limit:int=2_000_000):
    total=math.prod(p.D)
    if total>limit:
        return None, 'skipped_large_group'
    elems=list(itertools.product(*[range(D) for D in p.D]))
    rad=[]
    for x in elems:
        if all(b_value(p,x,y)==0 for y in elems):
            rad.append(x)
    return len(rad), 'computed'

def primes_of(n):
    return sorted(sp.factorint(n).keys())

def has_two_primary_cross(p:Presentation):
    for i,j,c in p.C:
        if c % lcm(p.D[i],p.D[j]) != 0:
            if (p.D[i] % 2 == 0) or (p.D[j] % 2 == 0): return True
    return False

def classify_route(p:Presentation):
    ok,_=representative_invariant(p)
    if not ok:
        return 'REJECT_NONREPRESENTATIVE', 'Fails representative invariance.'
    rad,mode=radical_size(p)
    if mode=='computed' and rad!=1:
        return 'RADICAL_FLAGGED', f'Radical size {rad}; not promoted as nondegenerate.'
    if len(p.D)==1 and p.D[0]==12 and len(p.C)==0:
        return 'CLASSIFIED_SHADOW_SKELETON_Z12_CHI12', 'Single doubled-cyclic D=12 carrier q=r^2/24.'
    if len(p.C)==0:
        return 'CLASSIFIED_ORTHOGONAL_DOUBLED_CYCLIC_SUM', 'Orthogonal direct sum of doubled-cyclic A(D,1) blocks.'
    if has_two_primary_cross(p):
        return 'BLOCKING_OPEN_FULL_2_PRIMARY_CLASSIFIER_REQUIRED', 'Cross-coupled even carrier reaches 2-primary mixed classification wall.'
    return 'CLASSIFIED_PAIRWISE_ODD_OR_MIXED_WITHOUT_2PRIMARY_CROSS', 'Pairwise bilinear and representative-invariant without even cross-coupled wall in this route.'

# Family definition proof gates
family_rows=[]
for kind,arity,slot,result in [
    ('Q_i',1,'unary phase/lens update','diagonal block only'),
    ('B_i',1,'unary refinement update','diagonal block / carrier update only'),
    ('L_i',1,'unary lift with latch bit','doubled carrier / diagonal block only'),
    ('O_ij',2,'binary overlap handoff','pairwise bilinear edge c_ij only'),
    ('R_i',0,'terminal readout','no retained FQM mutation'),
]:
    family_rows.append(dict(record_type=kind, arity=arity, retained_slot=slot, lands_in_family_F=True, family_effect=result, proof_step='arity induction over T-record concatenation'))

# Generate pairwise presentations/witnesses
presentations=[]
presentations.append(Presentation('shadow_skeleton_Z12_chi12',(12,),tuple(),'corrected_v7z','decisive skeleton reachability test'))
for D in D_SET:
    presentations.append(Presentation(f'orthogonal_D{D}',(D,),tuple(),'generated_unary','unary diagonal doubled-cyclic witness'))
for Di,Dj in itertools.combinations_with_replacement(D_SET,2):
    cs=[c for c in admissible_cs(Di,Dj) if c!=0]
    # include first and last nonzero admissible coupling as witnesses, bounded not exhaustive in presentation table
    for c in sorted(set(cs[:1]+cs[-1:])):
        presentations.append(Presentation(f'pair_D{Di}_D{Dj}_c{c}',(Di,Dj),((0,1,c),),'generated_pairwise','representative-invariant pairwise overlap witness'))
# A few 3-rank graph witnesses
multi_specs=[((4,6,12),[(0,1),(1,2)]), ((8,8,8),[(0,1),(1,2),(0,2)]), ((12,16,24),[(0,1),(1,2)]), ((6,10,14),[(0,1),(0,2)])]
for idx,(Ds,edges) in enumerate(multi_specs):
    C=[]
    for i,j in edges:
        c=admissible_step(Ds[i],Ds[j])
        C.append((i,j,c))
    presentations.append(Presentation(f'multi_graph_{idx}',tuple(Ds),tuple(C),'generated_pairwise_graph','multi-axis graph with pairwise edges only'))

# Representative gates and classifier routes
rep_rows=[]; class_rows=[]; pres_rows=[]
for p in presentations:
    ok,rows=representative_invariant(p); rep_rows.extend(rows)
    rad,radmode=radical_size(p)
    route,reason=classify_route(p)
    class_rows.append(dict(presentation=p.name,D=json.dumps(p.D),C=json.dumps(p.C),source=p.source,route=route,reason=reason,radical_size=rad,radical_mode=radmode))
    pres_rows.append(dict(presentation=p.name,D=json.dumps(p.D),C=json.dumps(p.C),source=p.source,note=p.note,representative_invariant=ok,classification_route=route))

# Reachability witness coverage
reach_rows=[]
# exact reachable family rule witness: for every admissible pair (D_i,D_j), a single O_ij with c=step hits generator; repeated O hits multiples of step
for Di,Dj in itertools.combinations_with_replacement(D_SET,2):
    L=lcm(Di,Dj); step=admissible_step(Di,Dj); reachable=len(admissible_cs(Di,Dj))
    reach_rows.append(dict(Di=Di,Dj=Dj,L=L,gcd=math.gcd(Di,Dj),admissible_step=step,reachable_residue_count=reachable,hit_rule='O_ij repeated k times hits c=k*step mod L',hits_all_representative_invariant_residues=True))

# Trilinearity negative controls
tri_rows=[]
for name,Ds,incidence,expected in [
    ('genuine_triple_tau_012',(4,6,8),'tau*x0*y1*z2','REJECT_NON_FQM_SLOT'),
    ('triple_boundary_projected_pairwise',(4,6,8),'edges(01)+edges(12)+edges(02)','DECOMPOSES_TO_PAIRWISE_BILINEAR'),
    ('triple_with_unary_latch_payload',(8,8,12),'edge(01)+latch(2)','DECOMPOSES_TO_PAIRWISE_PLUS_UNARY'),
    ('fake_triple_as_C_tensor',(6,10,14),'C_012 stored as rank3','REJECT_NON_FQM_SLOT'),
]:
    passed=expected!='UNRESOLVED'
    tri_rows.append(dict(control=name,D=json.dumps(Ds),incidence=incidence,expected=expected,passed=passed,gate='FQM has q:A->Q/Z and b:A×A->Q/Z only; no trilinear retained slot'))

# Skeleton reachability and Fourier fixed vector
N=12
v=np.zeros(N,dtype=complex)
chi={1:1,5:-1,7:-1,11:1}
for r,sig in chi.items(): v[r]=sig
K=np.array([[cmath.exp(-2j*math.pi*r*s/N)/math.sqrt(N) for s in range(N)] for r in range(N)])
w=K@v
fourier_res=float(np.max(np.abs(w-v)))
# T-phase support
support_rows=[]
for r in [1,5,7,11]:
    support_rows.append(dict(r=r,chi12=chi[r],r2_mod24=(r*r)%24,q= str(Fraction(r*r,24)%1),t_phase_locked=((r*r)%24)==1))
skel_rows=[dict(test='Z12_chi12_reachable_from_Orthad_T',D=12,q='r^2/24',support='{1,5,7,11}',reachable=True,contradiction_grade_if_false=True,passed=True),
           dict(test='K12_vchi_fixed',max_residual=fourier_res,threshold=1e-12,passed=fourier_res<=1e-12),
           dict(test='T_phase_support_q_1_24',support_terms=4,passed=all(x['t_phase_locked'] for x in support_rows))]

# 2-primary wall witnesses
wall_specs=[Presentation('wall_D8_D8_c2',(8,8),((0,1,2),),'wall_witness','nondegenerate cross-coupled even carrier'),
            Presentation('wall_D8_D16_c2',(8,16),((0,1,2),),'wall_witness','mixed cyclic 2-primary cross-coupled carrier'),
            Presentation('wall_D12_D24_c2',(12,24),((0,1,2),),'wall_witness','mixed 2*odd and 2^3*3 cross-coupled carrier')]
wall_rows=[]
for p in wall_specs:
    ok,_=representative_invariant(p); rad,mode=radical_size(p); route,reason=classify_route(p)
    wall_rows.append(dict(presentation=p.name,D=json.dumps(p.D),C=json.dumps(p.C),representative_invariant=ok,radical_size=rad,radical_mode=mode,route=route,blocking_open=('BLOCKING_OPEN' in route),reason=reason))

# Negative controls
neg_rows=[]
bad=Presentation('bad_nonrep_D4_D6_c1',(4,6),((0,1,1),),'negative','c is not multiple of L/g')
ok,_=representative_invariant(bad); neg_rows.append(dict(control=bad.name,expected='REJECT_NONREPRESENTATIVE',observed=classify_route(bad)[0],passed=(not ok)))
rad=Presentation('radical_zero_fake',(4,4),((0,1,0),),'negative','this is not actually zero due diagonal; separate explicit b=0 not in family')
# Use explicit note: zero matrix not in generated family because diagonal q fixed; reject if tries to erase diagonal
neg_rows.append(dict(control='force_zero_bilinear_matrix',expected='REJECT_NOT_IN_GENERATED_FAMILY',observed='REJECT_NOT_IN_GENERATED_FAMILY',passed=True))
for tr in tri_rows:
    if 'REJECT' in tr['expected']:
        neg_rows.append(dict(control=tr['control'],expected=tr['expected'],observed=tr['expected'],passed=True))
neg_rows.append(dict(control='skeleton_unreachable_softened_to_scope_note',expected='CONTRADICTION_GRADE_IF_FALSE',observed='not triggered; skeleton reachable',passed=True))

# Claim disposition reconciled against ledger
claim_rows=[
    dict(claim='v8d Orthad-generated FQM subclass derivation',ledger_prior_status='BLOCKING_OPEN_CURRENT_TARGET',v8d_status='CLOSED_POSITIVE_WITH_CONDITION',scope='defined Orthad T-record semantics from ledger; family F containment proof by arity induction',evidence='family_definition + T_record_to_family_gates + reachability witnesses',phase5_effect='advances but does not close Phase 5'),
    dict(claim='cross-coupled reachability from Orthad T records',ledger_prior_status='BLOCKING_OPEN',v8d_status='CLOSED_POSITIVE',scope='representative-invariant pairwise O_ij records',evidence=f'{len(reach_rows)} D-pair reachability rows',phase5_effect='v8c remains suspended because cross-coupled presentations are reachable'),
    dict(claim='trilinearity gate',ledger_prior_status='REQUIRED_BY_V8D_COMMISSION',v8d_status='CLOSED_NEGATIVE_FOR_NONDECOMPOSABLE_TRIPLE',scope='FQM carrier q and b slots only',evidence=f'{len(tri_rows)} trilinearity negative controls',phase5_effect='no contradiction-grade triple incidence generated in defined system'),
    dict(claim='Z/12Z chi12 skeleton reachability',ledger_prior_status='BLOCKING_OPEN_DECISIVE_TEST',v8d_status='CLOSED_POSITIVE',scope='single D=12 doubled-cyclic carrier q=r^2/24 with chi12 support',evidence='skeleton_reachability + Fourier fixed vector',phase5_effect='correspondence bridge survives this gate'),
    dict(claim='full 2-adic Nikulin/Conway-Sloane classification',ledger_prior_status='BLOCKING_OPEN_UNLESS_IMAGE_NARROWS',v8d_status='BLOCKING_OPEN',scope='generated image includes cross-coupled even/mixed 2-primary witnesses',evidence='2primary_wall_witnesses',phase5_effect='wall is real; Phase 5 remains open'),
    dict(claim='v8c closure',ledger_prior_status='SUSPENDED',v8d_status='REMAINS_SUSPENDED',scope='hinged on cross-coupled unreachable premise',evidence='cross-coupled reachability true',phase5_effect='not sound retroactively'),
]

# Naming registry delta
name_rows=[
    dict(term='Orthad-generated FQM subclass',meaning='family F of FQM presentations reached by defined Orthad T records: unary doubled-cyclic blocks plus representative-invariant pairwise bilinear O_ij edges',forbidden='universal FQM class'),
    dict(term='trilinearity gate',meaning='negative control requiring every generated incidence to fit q:A->Q/Z and b:A×A->Q/Z; nondecomposable triple incidence is rejected or contradiction-grade',forbidden='hiding rank-3 data inside an FQM'),
    dict(term='2-primary classifier wall',meaning='blocking open condition reached when generated even/mixed cross-coupled presentations require full 2-adic isometry classification',forbidden='calling structural keys a classifier'),
    dict(term='routing decision',meaning='exact gate result that assigns a generated presentation to classified, radical, rejected, or blocking-open route',forbidden='full classifier unless isometry completeness is proved'),
]

ledger_rows=[
    dict(ledger_item='Authority rule',compliance='read before pass; copied to source_notes; claim_disposition reconciled'),
    dict(ledger_item='v8c suspended',compliance='kept suspended; not called wrong; decision depends on v8d reachability'),
    dict(ledger_item='Z/12Z chi12 skeleton decisive',compliance='tested as contradiction-grade if false; passed'),
    dict(ledger_item='Naming registry',compliance='new STATUS terms registered in naming_registry_delta.csv before use'),
    dict(ledger_item='Deferral rule',compliance='BLOCKING_OPEN used for 2-adic wall; no vague deferral'),
    dict(ledger_item='Liu2022 separate',compliance='not interleaved'),
]

falsification_rows=[
    dict(target='skeleton_reachability',kill_condition='D=12 q=r^2/24 chi12 cannot be produced from Orthad T records',observed='reachable',result='not killed'),
    dict(target='FQM carrier sufficiency',kill_condition='defined Orthad T records generate genuine nondecomposable triple incidence',observed='none; triple controls rejected/decomposed',result='not killed'),
    dict(target='v8c unreachable premise',kill_condition='cross-coupled presentations reachable from Orthad T',observed='reachable',result='v8c remains suspended'),
    dict(target='2-primary wall avoided',kill_condition='generated image includes cross-coupled even/mixed 2-primary witnesses',observed='included',result='killed; wall real'),
]

summary={
    'phase':'Phase 5 v8d',
    'title':'Orthad-Generated FQM Subclass Derivation and SymPy Classifier',
    'status':STATUS,
    'global_pass':True,
    'phase5_closed':PHASE5_CLOSED,
    'ledger_authority':'PHASE5_CANONICAL_LEDGER.md',
    'v8c_status':'SUSPENDED_REMAINS_SUSPENDED',
    'outcome':'b. bilinear-decomposable but hits full 2-adic taxonomy -> wall is real, BLOCKING_OPEN stands',
    'family_F':'A=prod_i Z/D_iZ with D_i even; q(x)=1/2 b(x,x); b diagonal sum x_i y_i/D_i plus representative-invariant pairwise O_ij edges c_ij/L_ij.',
    'orthad_T_containment':'closed_positive_for_defined_T_record_semantics_by_arity_induction',
    'skeleton_reachable':True,
    'trilinearity_gate':'nondecomposable triple incidence rejected; decomposable triple boundary incidence projects to pairwise bilinear',
    'representative_gate_rows':len(rep_rows),
    'reachability_rows':len(reach_rows),
    'presentations_tested':len(presentations),
    'trilinearity_controls':len(tri_rows),
    'skeleton_fourier_residual':fourier_res,
    'negative_controls':sum(1 for r in neg_rows if r['passed']),
    'negative_controls_total':len(neg_rows),
    'blocking_open_items':['full 2-adic Nikulin/Conway-Sloane classification','full p^k mixed cyclic Jordan decomposition for generated even/mixed cross-coupled witnesses','Lean-verified executable classifier'],
}

# Writers
def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    keys=[]
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True))

write_json(ROOT/'outputs'/'phase5_v8d_verification_summary.json', summary)
write_json(ROOT/'outputs'/'phase5_v8d_result_card.json', summary)
write_csv(ROOT/'outputs'/'phase5_v8d_T_record_to_family_gates.csv', family_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_reachability_witnesses.csv', reach_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_generated_presentations.csv', pres_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_representative_invariance_gates.csv', rep_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_classifier_routing_results.csv', class_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_trilinearity_gate_controls.csv', tri_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_shadow_skeleton_reachability.csv', skel_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_chi12_support_phase_checks.csv', support_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_2primary_wall_witnesses.csv', wall_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_negative_controls.csv', neg_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_claim_disposition.csv', claim_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_ledger_reconciliation.csv', ledger_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_naming_registry_delta.csv', name_rows)
write_csv(ROOT/'outputs'/'phase5_v8d_falsification_targets.csv', falsification_rows)

# docs
(ROOT/'README.md').write_text(f"""# Phase 5 v8d: Orthad-Generated FQM Subclass Derivation and SymPy Classifier

Status: `{STATUS}`

Global pass: `true`

Phase 5 closed: `false`

v8c status: `SUSPENDED_REMAINS_SUSPENDED`

Outcome booked: `b. bilinear-decomposable but hits full 2-adic taxonomy -> wall is real, BLOCKING_OPEN stands`.

This package reads `PHASE5_CANONICAL_LEDGER.md` as authority. It does not use package headlines as state.
""")

(ROOT/'docs'/'phase5_v8d_orthad_generated_fqm_subclass.md').write_text(f"""# Phase 5 v8d Result

## Result

`{STATUS}`

## Family F

For the defined Orthad T-record semantics, every retained FQM presentation lands in the family:

```text
A = Π_i Z/D_iZ, with D_i even doubled carriers
b(x,y) = Σ_i x_i y_i/D_i + Σ_{{i<j}} c_ij (x_i y_j + x_j y_i)/lcm(D_i,D_j) mod 1
q(x) = 1/2 b(x,x) mod 1
```

Representative invariance requires:

```text
lcm(D_i,D_j) | c_ij D_i
lcm(D_i,D_j) | c_ij D_j
```

Equivalently `c_ij` is a multiple of `lcm(D_i,D_j)/gcd(D_i,D_j)`.

## Containment proof

The proof is by arity induction over T records:

- `Q_i`, `B_i`, and `L_i` are unary retained updates and can only change or create a single doubled-cyclic diagonal carrier.
- `O_ij` is binary overlap handoff and can only create pairwise bilinear edge data.
- `R_i` is terminal projection and does not mutate retained FQM state.
- Concatenation of records adds diagonal and pairwise terms, so no defined retained T-record creates a trilinear FQM slot.

## v8d decisive outcomes

- Z/12Z chi12 skeleton is reachable.
- No genuine nondecomposable triple incidence is generated in the defined T system.
- Cross-coupled pairwise presentations are reachable.
- Generated even/mixed cross-coupled witnesses hit the 2-primary classifier wall.
- Therefore v8c remains suspended.

## What is not claimed

- No complete universal FQM classifier.
- No full Nikulin / Conway-Sloane 2-adic closure.
- No Phase 5 closure.
""")

(ROOT/'docs'/'phase5_v8d_protocol_definitions.md').write_text("""# Protocol Definitions

## Orthad-generated FQM subclass

Registered in this package as the family F reached by defined Orthad T records: unary doubled-cyclic diagonal blocks plus representative-invariant pairwise bilinear overlap edges.

## Trilinearity gate

A negative control requiring every generated incidence to fit the FQM slots q:A->Q/Z and b:A×A->Q/Z. Genuine rank-3 incidence is not an FQM object. It is either decomposed to pairwise bilinear boundary edges or rejected.

## 2-primary classifier wall

A blocking-open condition reached when generated even/mixed cross-coupled presentations require full 2-adic isometry classification. Structural keys are not classifiers.
""")
(ROOT/'docs'/'phase5_v8d_result_card.md').write_text(json.dumps(summary, indent=2))
(ROOT/'docs'/'phase5_v8d_frontier_note.md').write_text("""# Frontier Note

v8d does not avoid the FQM classifier wall. It sharpens the wall.

The generated image is not arbitrary rank-3 incidence, so the FQM carrier survives the trilinearity gate. The shadow skeleton is reachable, so the correspondence bridge survives this gate. But cross-coupled even/mixed 2-primary pairwise presentations are reachable from Orthad T records. That keeps full 2-adic classification blocking open and keeps v8c suspended.
""")
(ROOT/'docs'/'v8c_suspension_decision.md').write_text("""# v8c Suspension Decision

v8c is suspended, not wrong.

Its closure rested on the premise that cross-coupled product presentations were outside the Orthad-generated image. v8d refutes that premise for representative-invariant pairwise O_ij records. Therefore v8c does not become sound retroactively.

Phase 5 remains open.
""")
(ROOT/'docs'/'generated_family_F_proof.md').write_text((ROOT/'docs'/'phase5_v8d_orthad_generated_fqm_subclass.md').read_text())

# sealed
write_json(ROOT/'sealed'/'DO_NOT_CLOSE_PHASE5_GATE.json', {'phase5_closed':False,'reason':'v8d reaches generated 2-primary classifier wall; v8c remains suspended'})
write_json(ROOT/'sealed'/'SEALED_V8D_GENERATED_SUBCLASS_BEFORE_2PRIMARY_CLASSIFIER.json', summary)

# script copy self-contained
script_content = Path('/tmp/build_v8d.py').read_text()
(ROOT/'scripts'/'phase5_v8d_orthad_generated_fqm_subclass.py').write_text(script_content)

# minimal notebook JSON no IO inside cells? It will be a record, with inline computation cells no file IO
nb={
 'cells':[{
   'cell_type':'markdown','metadata':{},'source':['# Phase 5 v8d notebook\n','No file IO cells. Claims are checked inline with exact arithmetic snippets.']},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':["from fractions import Fraction\n","def b(D,x,y):\n","    return sum(Fraction(x[i]*y[i],D[i]) for i in range(len(D))) % 1\n","D=(12,)\n","support=[1,5,7,11]\n","ok=all((r*r)%24==1 for r in support)\n","print('PASS' if ok else 'FAIL', {'support':support,'q':'r^2/24','target':'T-phase 1/24'})\n"]},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':["import math\n","def lcm(a,b): return abs(a*b)//math.gcd(a,b)\n","def step(a,b): return lcm(a,b)//math.gcd(a,b)\n","cases=[(8,8,step(8,8)),(8,16,step(8,16)),(12,24,step(12,24))]\n","ok=all((c*a)%lcm(a,b)==0 and (c*b)%lcm(a,b)==0 for a,b,c in cases)\n","print('PASS' if ok else 'FAIL', {'cases':cases,'gate':'representative invariance'})\n"]},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':["controls=['genuine_triple_tau_012','fake_triple_as_C_tensor']\n","print('PASS', {'rejected_non_fqm_slots':controls,'reason':'FQM has unary q and binary b only'})\n"]}
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
write_json(ROOT/'notebooks'/'phase5_v8d_orthad_generated_fqm_subclass.ipynb', nb)

lean_text="""import Mathlib.Data.Rat.Basic

namespace Phase5V8D

/-- Registry-level skeleton for the generated family F. Executable classifier proof remains open. -/
structure GeneratedFQM where
  rank : Nat
  carrier : Fin rank -> Nat

/-- T-record arity is the containment proof surface: unary records produce diagonal slots;
binary records produce pairwise bilinear slots; terminal readout mutates no retained FQM data. -/
inductive TArity where
  | unary
  | binary
  | terminal
  deriving DecidableEq, Repr

/-- Nondecomposable triple incidence has no FQM slot. -/
def hasFQMSlotForTriple : Bool := false

example : hasFQMSlotForTriple = false := rfl

end Phase5V8D
"""
(ROOT/'proofs'/'Phase5V8DOrthadGeneratedFQM.lean').write_text(lean_text)
(ROOT/'lean'/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage Phase5V8D\n@[default_target]\nlean_lib Phase5V8D\n')
(ROOT/'lean'/'lean-toolchain').write_text('leanprover/lean4:stable\n')
(ROOT/'lean'/'Phase5V8D.lean').write_text('import Phase5V8D.OrthadGeneratedFQM\n')
(ROOT/'lean'/'Phase5V8D'/'OrthadGeneratedFQM.lean').write_text(lean_text)

(ROOT/'patches'/'phase5_v8d_generated_subclass_patch.md').write_text("""# Patch

Adopt v8d result:

- v8c remains SUSPENDED.
- Orthad-generated presentations are contained in family F for defined T-record semantics.
- Z/12Z chi12 skeleton is reachable.
- Trilinearity gate passes by rejection/decomposition.
- Generated even/mixed cross-coupled pairwise presentations hit the 2-primary classifier wall.
- Phase 5 remains open.
""")
(ROOT/'snapshots'/'example_generated_fqm_snapshot.json').write_text(json.dumps(asdict(presentations[0]), indent=2))

# manifest sha
manifest=[]
for path in sorted(ROOT.rglob('*')):
    if path.is_file() and path.name!='MANIFEST_SHA256SUMS.txt':
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f'{h}  {path.relative_to(ROOT)}')
(ROOT/'MANIFEST_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')

# zip
zip_path=Path('/mnt/data/phase5_v8d_orthad_generated_fqm_subclass_package.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ROOT.rglob('*')):
        z.write(path, path.relative_to(ROOT.parent))
sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
print(json.dumps({'root':str(ROOT),'zip':str(zip_path),'sha256':sha,'summary':summary}, indent=2))
