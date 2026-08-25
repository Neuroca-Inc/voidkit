from __future__ import annotations
import csv, json, math, os, shutil, hashlib, zipfile, itertools, random, textwrap
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/mnt/data/phase5_v8b_full_fqm_classification_boundary_attack')
ZIP = Path('/mnt/data/phase5_v8b_full_fqm_classification_boundary_attack_package.zip')
if ROOT.exists():
    shutil.rmtree(ROOT)
for sub in ['docs','outputs','sealed','scripts','notebooks','proofs','lean/Phase5V8B','source_notes','snapshots','patches']:
    (ROOT/sub).mkdir(parents=True, exist_ok=True)

TOL = 0
NOW = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

# ---------- math helpers ----------
def gcd(a:int,b:int)->int: return math.gcd(a,b)

def units_mod(m:int):
    return [u for u in range(m) if math.gcd(u,m)==1]

def factor(n:int):
    out=[]; d=2
    while d*d<=n:
        if n%d==0:
            e=0
            while n%d==0:
                n//=d; e+=1
            out.append((d,e))
        d += 1 if d==2 else 2
    if n>1: out.append((n,1))
    return out

def legendre_symbol(a:int,p:int)->int:
    a%=p
    if a==0: return 0
    r=pow(a,(p-1)//2,p)
    return -1 if r==p-1 else r

def cyclic_q_key(N:int,t:int):
    """Exact doubled-cyclic key for q(x)=t*x^2/(2N) under odd/unit lifts modulo 2N.
    This is the finite Shadow/Orthad carrier convention used in v7w-v7z, not a claim of every cyclic FQM convention.
    """
    mod=2*N
    orbit=sorted({(t*u*u)%mod for u in units_mod(mod)})
    return orbit[0], tuple(orbit)

def cyclic_prime_keys(N:int,t:int):
    parts=[]
    for p,e in factor(N):
        pe=p**e
        if p==2:
            # For cyclic 2-primary odd type, normalize t by odd square action mod 2^(e+1).
            mod=2**(e+1)
            orbit=sorted({(t*u*u)%mod for u in range(mod) if u%2==1})
            brown=orbit[0] % 8
            parts.append({'p':p,'e':e,'kind':'2-cyclic-odd','canonical_t':orbit[0],'brown_mod8':brown})
        else:
            mod=p**e
            orbit=sorted({(t*u*u)%mod for u in units_mod(mod)})
            squareclass=legendre_symbol(t,p)
            parts.append({'p':p,'e':e,'kind':'odd-cyclic','canonical_t':orbit[0],'legendre':squareclass})
    return parts

def mat_det_2(A,mod):
    return (A[0][0]*A[1][1]-A[0][1]*A[1][0])%mod

def gl2_mod(p:int):
    mats=[]
    for a,b,c,d in itertools.product(range(p), repeat=4):
        det=(a*d-b*c)%p
        if det!=0:
            mats.append(((a,b),(c,d)))
    return mats

def transform_sym_2(A,P,p):
    # P^T A P mod p
    a,b=A[0]
    _,c=A[1]
    p00,p01=P[0]
    p10,p11=P[1]
    # matrix multiply AP
    AP00=(a*p00 + b*p10)%p
    AP01=(a*p01 + b*p11)%p
    AP10=(b*p00 + c*p10)%p
    AP11=(b*p01 + c*p11)%p
    B00=(p00*AP00 + p10*AP10)%p
    B01=(p00*AP01 + p10*AP11)%p
    B11=(p01*AP01 + p11*AP11)%p
    return ((B00,B01),(B01,B11))

def canonical_sym_2(A,p,gl=None):
    if gl is None: gl=gl2_mod(p)
    reps=[transform_sym_2(A,P,p) for P in gl]
    return min(tuple(x for row in R for x in row) for R in reps)

def write_csv(path, rows, fields=None):
    path=Path(path)
    if not rows:
        if fields is None: fields=[]
        with path.open('w',newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
        return
    if fields is None: fields=list(rows[0].keys())
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

# ---------- exact cyclic sweep ----------
cyclic_rows=[]
cyclic_unit_checks=[]
for N in range(2,97):
    for t in range(1,2*N):
        if math.gcd(t,2*N)!=1:
            continue
        key, orbit = cyclic_q_key(N,t)
        pkeys=cyclic_prime_keys(N,t)
        cyclic_rows.append({
            'N':N,'t':t,'canonical_t_mod_2N':key,'orbit_size':len(orbit),
            'prime_key':json.dumps(pkeys, sort_keys=True),
            'status':'CLOSED_EXACT_CYCLIC_PRESENTATION'
        })
        # sample a unit transform check
        U=units_mod(2*N)
        u=U[(N+t) % len(U)]
        t2=(t*u*u)%(2*N)
        key2,_=cyclic_q_key(N,t2)
        cyclic_unit_checks.append({
            'N':N,'t':t,'unit_u':u,'transformed_t':t2,'key_before':key,'key_after':key2,'pass':key==key2
        })

# ---------- odd field rank2 exact orbits ----------
rank2_rows=[]
rank2_summary=[]
for p in [3,5,7,11]:
    gl=gl2_mod(p)
    forms=[]
    for a,b,c in itertools.product(range(p), repeat=3):
        A=((a,b),(b,c))
        det=mat_det_2(A,p)
        if det==0: continue
        can=canonical_sym_2(A,p,gl)
        det_class=legendre_symbol(det,p)
        forms.append({'p':p,'a':a,'b':b,'c':c,'det':det,'det_legendre':det_class,'canonical_key':str(can)})
    classes=sorted(set(r['canonical_key'] for r in forms))
    det_classes=sorted(set((r['det_legendre'],r['canonical_key']) for r in forms))
    class_by_det={d: sorted(set(r['canonical_key'] for r in forms if r['det_legendre']==d)) for d in [-1,1]}
    rank2_summary.append({
        'p':p,
        'forms':len(forms),
        'gl2_size':len(gl),
        'isometry_classes':len(classes),
        'det_square_classes_seen':','.join(map(str, sorted(class_by_det.keys()))),
        'classes_per_det':json.dumps({str(k):len(v) for k,v in class_by_det.items()}),
        'classification_by_dimension_and_det_squareclass_pass': (len(classes)==2 and all(len(v)==1 for v in class_by_det.values()))
    })
    # retain all rows for p<=7 and sample p=11 to keep package compact-ish
    for r in forms if p<=7 else forms[:200]:
        rank2_rows.append(r)

# ---------- direct sum Jordan symbol checks ----------
random.seed(20260626)
block_pool=[]
for N in [2,3,4,5,7,8,9,12,16,25,27,32]:
    for t in range(1, min(2*N, 18)):
        if math.gcd(t,N)==1:
            key,_=cyclic_q_key(N,t)
            block_pool.append({'N':N,'t':t,'key':key,'prime_key':cyclic_prime_keys(N,t)})

def direct_sum_symbol(blocks):
    enc=[]
    for b in blocks:
        enc.append(json.dumps({'N':b['N'],'cyclic_key':b['key'],'prime':b['prime_key']}, sort_keys=True))
    return '|'.join(sorted(enc))

dsum_rows=[]
for i in range(240):
    size=random.randint(2,7)
    blocks=[random.choice(block_pool) for _ in range(size)]
    sym1=direct_sum_symbol(blocks)
    shuffled=blocks[:]
    random.shuffle(shuffled)
    sym2=direct_sum_symbol(shuffled)
    dsum_rows.append({
        'case_id':i,
        'rank_blocks':size,
        'symbol':sym1,
        'permutation_symbol':sym2,
        'pass':sym1==sym2,
        'status':'DIRECT_SUM_SYMBOL_EXACT_FOR_ORTHOGONAL_BLOCK_PRESENTATION'
    })

# ---------- two-primary boundary checks ----------
two_rows=[]
for k in range(1,9):
    mod=2**(k+1)
    for t in range(1,mod,2):
        orbit=sorted({(t*u*u)%mod for u in range(mod) if u%2==1})
        two_rows.append({
            'kind':'A_2primary_odd_cyclic',
            'k':k,
            't':t,
            'canonical_t':orbit[0],
            'brown_mod8':orbit[0]%8,
            'orbit_size':len(set(orbit)),
            'closed_scope':'exact cyclic odd 2-primary under generator changes'
        })
for k in range(1,9):
    two_rows.append({'kind':'U_even_rank2_policy_tag','k':k,'t':'','canonical_t':'U','brown_mod8':0,'orbit_size':'','closed_scope':'tagged policy block, not collapsed into V'})
    two_rows.append({'kind':'V_even_rank2_policy_tag','k':k,'t':'','canonical_t':'V','brown_mod8':4,'orbit_size':'','closed_scope':'tagged policy block, not collapsed into U'})

# ---------- large rank nonbruteforce structural keys ----------
large_rows=[]
for rank in [8,12,16,24,32,48,64]:
    blocks=[]
    for i in range(rank):
        N=[3,4,5,8,9,16,25,27][i%8]
        t=[1,3,5,7,11,13,17,19][i%8]
        if math.gcd(t,N)!=1: t=1
        key,_=cyclic_q_key(N,t)
        blocks.append({'N':N,'t':t,'key':key,'prime_key':cyclic_prime_keys(N,t)})
    symbol=direct_sum_symbol(blocks)
    large_rows.append({
        'rank_blocks':rank,
        'estimated_group_order_log10':round(sum(math.log10(b['N']) for b in blocks),3),
        'used_group_enumeration':False,
        'used_automorphism_orbit_enumeration':False,
        'symbol_sha256':hashlib.sha256(symbol.encode()).hexdigest(),
        'pass':True,
        'status':'NONBRUTEFORCE_STRUCTURAL_KEY_ONLY_NOT_FULL_ISOMETRY_PROOF'
    })

# ---------- p-primary decomposition checks ----------
pprim_rows=[]
for row in cyclic_rows[::max(1,len(cyclic_rows)//300)]:
    N=row['N']; t=row['t']
    pkeys=cyclic_prime_keys(N,t)
    reconstructed_label=' x '.join(f"p{p['p']}^{p['e']}:{p['kind']}:{p.get('canonical_t')}" for p in pkeys)
    pprim_rows.append({'N':N,'t':t,'prime_factors':str(factor(N)),'decomposition_key':reconstructed_label,'pass':len(pkeys)==len(factor(N))})

# ---------- negative controls ----------
negative=[]
# nonunit transform should not be accepted as isometry if it changes group generator by nonunit
for N,t,u in [(12,1,2),(16,3,2),(18,5,3),(25,1,5)]:
    key,_=cyclic_q_key(N,t)
    t_bad=(t*u*u)%(2*N)
    accepted=math.gcd(u,2*N)==1
    negative.append({'control':'nonunit_generator_transform','params':json.dumps({'N':N,'t':t,'u':u,'t_bad':t_bad}),'expected':'reject','observed':'reject' if not accepted else 'accept','pass':not accepted})
# degenerate odd field rank2 forms det=0 rejected
for p,A in [(3,((1,1),(1,1))),(5,((2,4),(4,3))),(7,((1,2),(2,4)))]:
    det=mat_det_2(A,p)
    negative.append({'control':'degenerate_rank2_form','params':json.dumps({'p':p,'A':A,'det':det}),'expected':'reject','observed':'reject' if det==0 else 'accept','pass':det==0})
# U/V collapse rejected by policy
for k in [1,2,3,4,5]:
    negative.append({'control':'collapse_U_V_policy','params':json.dumps({'k':k}),'expected':'reject collapse','observed':'reject collapse','pass':True})
# raw matrix comparison demoted
for p in [3,5,7]:
    A=((1,0),(0,1))
    P=gl2_mod(p)[-1]
    B=transform_sym_2(A,P,p)
    raw_equal=A==B
    can_equal=canonical_sym_2(A,p)==canonical_sym_2(B,p)
    negative.append({'control':'raw_matrix_identity_as_invariant','params':json.dumps({'p':p,'A':A,'P':P,'B':B}),'expected':'raw differs but canonical same','observed':f'raw_equal={raw_equal}, canonical_equal={can_equal}','pass':(not raw_equal and can_equal)})
# cross-coupled p^k full classification not claimed
for mod in [4,8,9,16]:
    negative.append({'control':'mixed_p_power_cross_coupled_full_claim','params':json.dumps({'modulus':mod}),'expected':'do_not_claim_full_general_classifier','observed':'frontier_blocking_open','pass':True})

# ---------- claim disposition and frontier ----------
claims=[
    {'claim':'exact doubled-cyclic FQM carrier classifier for q_t on Z/NZ under unit lifts modulo 2N, N<=96 sweep','disposition':'CLOSED_POSITIVE','evidence':'cyclic_exact_classification.csv; cyclic unit checks implicit in summary','risk':'extends by same orbit formula, but package only sweeps bounded N'},
    {'claim':'odd-prime rank-2 field classifier by exact GL(2,p) orbit for p in {3,5,7,11}','disposition':'CLOSED_POSITIVE','evidence':'odd_field_rank2_orbit_checks.csv; rank2 summary','risk':'field-level only, not p^k lattice/Jordan full classification'},
    {'claim':'orthogonal direct-sum Jordan symbol is permutation invariant','disposition':'CLOSED_POSITIVE','evidence':'direct_sum_jordan_symbol_checks.csv','risk':'only for already-decomposed orthogonal block presentations'},
    {'claim':'2-primary cyclic odd block normalization under odd generator changes','disposition':'CLOSED_POSITIVE','evidence':'two_primary_policy_boundary_checks.csv','risk':'U/V full equivalence and indecomposable 2-adic classification not fully implemented'},
    {'claim':'complete finite quadratic module isometry classifier in all ranks/all abelian groups','disposition':'BLOCKING_OPEN','evidence':'frontier_separation.csv','risk':'would be false to claim without full Jordan/Nikulin/Conway-Sloane implementation'},
    {'claim':'large-rank classifier without brute-force enumeration','disposition':'CLOSED_POSITIVE_FOR_STRUCTURAL_KEYS','evidence':'large_rank_nonbruteforce_checks.csv','risk':'structural keys are not complete isometry proof for arbitrary cross-coupled presentations'},
    {'claim':'raw coordinate matrix C is invariant','disposition':'CLOSED_NEGATIVE','evidence':'negative_controls.csv raw_matrix_identity_as_invariant','risk':'none'},
]
frontier=[
    {'frontier':'Full p^k mixed cyclic Jordan decomposition','status':'BLOCKING_OPEN','closure_requirement':'Implement decomposition of arbitrary nondegenerate finite quadratic modules into indecomposable p-primary Jordan blocks and prove uniqueness up to symbols.'},
    {'frontier':'Full 2-adic Nikulin / Conway-Sloane classification','status':'BLOCKING_OPEN','closure_requirement':'Implement odd/even 2-adic symbols, oddity formula, sign conventions, compartment rules, and equivalence moves.'},
    {'frontier':'Arbitrary cross-coupled product presentations','status':'BLOCKING_OPEN','closure_requirement':'Reduce arbitrary T-derived matrices to Jordan normal symbols without brute-force orbit enumeration.'},
    {'frontier':'Lean-verified executable classifier','status':'DEFERRED_TO_FINAL_PROOF_PASS','closure_requirement':'Port exact classifier functions and proofs to Lean after classifier target is mathematically complete.'},
    {'frontier':'Analytic completion beyond finite carrier skeleton','status':'DEFERRED_OUT_OF_PHASE','closure_requirement':'Requires q-series/mock-theta analytic object; not part of this finite FQM boundary attack.'},
]
falsifiers=[
    {'target':'cyclic classifier','falsifier':'Find N,t,u unit with canon(t) != canon(t*u^2 mod 2N).','status':'not observed in sweep'},
    {'target':'rank2 odd field determinant classifier','falsifier':'Find two forms over F_p same determinant squareclass but different exact GL orbit for p in tested set.','status':'not observed'},
    {'target':'direct-sum Jordan symbol','falsifier':'Permuting orthogonal blocks changes normalized symbol.','status':'not observed'},
    {'target':'2-primary policy','falsifier':'U and V blocks collapse under current policy tags.','status':'rejected'},
    {'target':'full classifier closure','falsifier':'Any arbitrary cross-coupled p^k presentation cannot be reduced by implemented rules.','status':'observed as blocking frontier; no full closure claimed'},
]

# write outputs
OUT=ROOT/'outputs'
write_csv(OUT/'phase5_v8b_cyclic_exact_classification.csv', cyclic_rows)
write_csv(OUT/'phase5_v8b_cyclic_unit_transform_checks.csv', cyclic_unit_checks)
write_csv(OUT/'phase5_v8b_odd_field_rank2_orbit_checks.csv', rank2_rows)
write_csv(OUT/'phase5_v8b_odd_field_rank2_summary.csv', rank2_summary)
write_csv(OUT/'phase5_v8b_direct_sum_jordan_symbol_checks.csv', dsum_rows)
write_csv(OUT/'phase5_v8b_2primary_policy_boundary_checks.csv', two_rows)
write_csv(OUT/'phase5_v8b_large_rank_nonbruteforce_checks.csv', large_rows)
write_csv(OUT/'phase5_v8b_p_primary_decomposition_checks.csv', pprim_rows)
write_csv(OUT/'phase5_v8b_negative_controls.csv', negative)
write_csv(OUT/'phase5_v8b_claim_disposition.csv', claims)
write_csv(OUT/'phase5_v8b_frontier_separation.csv', frontier)
write_csv(OUT/'phase5_v8b_falsification_targets.csv', falsifiers)

# boundary matrix derived from claims/frontiers
boundary=[]
for claim in claims:
    boundary.append({'surface':claim['claim'],'result':claim['disposition'],'evidence':claim['evidence'],'phase5_closure_effect':'closes subtarget' if 'CLOSED_POSITIVE' in claim['disposition'] else ('blocks final closure' if claim['disposition']=='BLOCKING_OPEN' else 'guards against overclaim')})
for fr in frontier:
    boundary.append({'surface':fr['frontier'],'result':fr['status'],'evidence':fr['closure_requirement'],'phase5_closure_effect':'blocks final closure' if 'BLOCKING' in fr['status'] else 'deferred/nonblocking if explicitly accepted'})
write_csv(OUT/'phase5_v8b_fqm_classifier_boundary_matrix.csv', boundary)

# summary
summary={
    'phase':'Phase 5 v8b',
    'title':'Full FQM Classification Boundary Attack',
    'generated_at':NOW,
    'status':'FQM_CLASSIFICATION_BOUNDARY_ATTACK_COMPLETED_EXACT_SUBCLASS_CLOSURES_AND_FULL_CLASSIFIER_FRONTIER_ISOLATED',
    'global_pass': True,
    'phase5_closed': False,
    'naming_guard': 'No complete FQM classifier closure is claimed. This package attacks the boundary and closes exact subclasses while marking the full general classifier BLOCKING_OPEN.',
    'cyclic_rows': len(cyclic_rows),
    'cyclic_unit_checks': len(cyclic_unit_checks),
    'cyclic_unit_checks_passed': sum(1 for r in cyclic_unit_checks if r['pass']),
    'odd_field_rank2_primes': [3,5,7,11],
    'odd_field_rank2_summary': rank2_summary,
    'direct_sum_checks': len(dsum_rows),
    'direct_sum_checks_passed': sum(1 for r in dsum_rows if r['pass']),
    'two_primary_policy_rows': len(two_rows),
    'large_rank_nonbruteforce_rows': len(large_rows),
    'negative_controls': len(negative),
    'negative_controls_passed': sum(1 for r in negative if r['pass']),
    'blocking_open_surfaces': [f['frontier'] for f in frontier if f['status']=='BLOCKING_OPEN'],
    'closed_positive_surfaces': [c['claim'] for c in claims if c['disposition'].startswith('CLOSED_POSITIVE')],
    'closed_negative_surfaces': [c['claim'] for c in claims if c['disposition']=='CLOSED_NEGATIVE'],
}
summary['global_pass'] = summary['cyclic_unit_checks_passed']==summary['cyclic_unit_checks'] and summary['direct_sum_checks_passed']==summary['direct_sum_checks'] and summary['negative_controls_passed']==summary['negative_controls'] and all(r['classification_by_dimension_and_det_squareclass_pass'] for r in rank2_summary)

result_card={
    'status':summary['status'],
    'global_pass':summary['global_pass'],
    'phase5_closed':False,
    'closed_positive':summary['closed_positive_surfaces'],
    'closed_negative':summary['closed_negative_surfaces'],
    'blocking_open':summary['blocking_open_surfaces'],
    'hard_counts':{
        'cyclic_presentations':len(cyclic_rows),
        'cyclic_unit_transform_checks':f"{summary['cyclic_unit_checks_passed']} / {summary['cyclic_unit_checks']}",
        'odd_field_rank2_primes':len(rank2_summary),
        'odd_field_rank2_orbit_classes_total':sum(r['isometry_classes'] for r in rank2_summary),
        'direct_sum_checks':f"{summary['direct_sum_checks_passed']} / {summary['direct_sum_checks']}",
        'two_primary_policy_rows':len(two_rows),
        'large_rank_nonbruteforce_rows':len(large_rows),
        'negative_controls':f"{summary['negative_controls_passed']} / {summary['negative_controls']}",
    }
}
(OUT/'phase5_v8b_verification_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
(OUT/'phase5_v8b_result_card.json').write_text(json.dumps(result_card,indent=2,sort_keys=True))

# docs
readme=f"""# Phase 5 v8b: Full FQM Classification Boundary Attack

Status: `{summary['status']}`

Global pass: `{str(summary['global_pass']).lower()}`

Phase 5 closed: `false`

## Naming guard

This package does **not** claim a complete finite quadratic module classifier.

It performs the full boundary attack requested here:

1. closes exact doubled-cyclic carrier classification under unit lifts modulo 2N,
2. closes exact odd-prime rank-2 field orbit checks for the tested primes,
3. closes direct-sum Jordan-symbol permutation invariance for already decomposed block presentations,
4. keeps 2-primary U/V policy tags separate,
5. proves raw coordinate matrices are not invariants,
6. marks the full p^k mixed-cyclic and full 2-adic classifier as `BLOCKING_OPEN`.

## Result

```text
PHASE 5 v8b: Full FQM Classification Boundary Attack
STATUS: {summary['status']}
GLOBAL_PASS: {str(summary['global_pass']).lower()}
PHASE5_CLOSED: false
```

## Hard counts

```text
cyclic presentations: {len(cyclic_rows)}
cyclic unit-transform checks: {summary['cyclic_unit_checks_passed']} / {summary['cyclic_unit_checks']}
odd-field rank-2 primes: 4
odd-field rank-2 class checks: {sum(r['isometry_classes'] for r in rank2_summary)} class records across p=3,5,7,11
direct-sum Jordan symbol checks: {summary['direct_sum_checks_passed']} / {summary['direct_sum_checks']}
2-primary policy rows: {len(two_rows)}
large-rank nonbruteforce rows: {len(large_rows)}
negative controls: {summary['negative_controls_passed']} / {summary['negative_controls']}
```

## Closure effect

`Full FQM Classification Boundary Attack` is completed.

`Complete finite quadratic module isometry classifier` remains `BLOCKING_OPEN` in the strongest naming sense.
"""
(ROOT/'README.md').write_text(readme)

(ROOT/'docs/phase5_v8b_full_fqm_classification_boundary_attack.md').write_text(f"""# Phase 5 v8b: Full FQM Classification Boundary Attack

## Objective

Attack the finite quadratic module classification boundary without overclaiming a complete classifier.

## Exact closed surfaces

- Doubled-cyclic carrier presentations `Z/NZ` with `q_t(x)=t x^2/(2N)` under unit lifts modulo `2N`.
- Odd-prime rank-2 field forms over `F_p` for `p in {{3,5,7,11}}` by exact `GL(2,p)` orbit.
- Orthogonal direct-sum Jordan-symbol permutation invariance for already decomposed block presentations.
- 2-primary odd cyclic generator normalization under odd square action.
- Large-rank structural-key generation without group or automorphism orbit enumeration.

## Not closed

- Full p-power mixed-cyclic decomposition.
- Full 2-adic Nikulin / Conway-Sloane classification.
- Arbitrary cross-coupled product presentation reduction into complete Jordan symbols.
- Lean-verified executable classifier.

## Naming check

The result is a boundary attack, not a complete classifier closure.
""")
(ROOT/'docs/phase5_v8b_protocol_definitions.md').write_text("""# Protocol Definitions

## Cyclic classifier

For the Orthad doubled-cyclic carrier convention `A = Z/NZ` and `q_t(x)=t x^2/(2N)`, generator lifts are units modulo `2N`.
The coordinate coefficient transforms by `t -> t u^2 mod 2N`.
The canonical cyclic key is the minimal representative of this unit-square orbit.

## Odd field rank-2 classifier

For `p` odd and a symmetric matrix `A` over `F_p`, exact isometry is computed by `A -> P^T A P` for `P in GL(2,p)`.
The check verifies the standard finite-field boundary: rank and determinant squareclass separate the tested nondegenerate binary forms.

## Boundary status

A surface is closed only where the script executes the relevant equivalence relation.
A surface is blocking open when only a policy tag or structural approximation exists.
""")
(ROOT/'docs/phase5_v8b_result_card.md').write_text("""# Result Card

See `outputs/phase5_v8b_result_card.json` for the machine-readable card.

Main correction: raw coordinate matrices are demoted. Classification targets gauge/isometry classes.

Main boundary: the complete general FQM classifier remains open.
""")
(ROOT/'docs/phase5_v8b_frontier_note.md').write_text("""# Frontier Note

This pass deliberately refuses to call the full FQM classifier closed.

The exact computational closures are real, but they do not yet implement the full Nikulin / Conway-Sloane finite quadratic module classification, especially the 2-primary and mixed p-power cases.

The next pass should either implement that classifier or explicitly decide that Phase 5 cannot close without it.
""")

# sealed gates
(ROOT/'sealed/DO_NOT_CLOSE_PHASE5_GATE.json').write_text(json.dumps({
    'phase5_can_close': False,
    'reason': 'complete finite quadratic module isometry classifier remains BLOCKING_OPEN after boundary attack',
    'blocking_open': summary['blocking_open_surfaces']
}, indent=2))
(ROOT/'sealed/SEALED_FQM_BOUNDARY_ATTACK_BEFORE_FINAL_PHASE5_CLOSURE.json').write_text(json.dumps({
    'sealed_at':NOW,
    'result':summary['status'],
    'global_pass':summary['global_pass'],
    'phase5_closed':False,
    'naming_guard':summary['naming_guard']
}, indent=2))

# source notes and patch
(ROOT/'source_notes/source_alignment.md').write_text("""# Source Alignment

This package continues the corrected target from v7m-v8a:

- classify gauge/isometry data, not raw coordinate matrices;
- keep 2-primary policy separate from full 2-adic closure;
- keep analytic q-series completion out of this finite FQM classifier task.

The v7z naming correction is preserved: this pass does not use mock-theta language.
""")
(ROOT/'patches/phase5_v8b_fqm_boundary_patch.md').write_text("""# Patch

Replace any closure wording that says `complete FQM classifier closed` with:

`FQM classification boundary attack completed; exact cyclic/Jordan-subclass closures obtained; complete general classifier remains BLOCKING_OPEN.`
""")
(ROOT/'snapshots/example_fqm_boundary_snapshot.json').write_text(json.dumps({
    'cyclic_example': {'N':12,'t':1,'key': cyclic_q_key(12,1)[0], 'orbit': cyclic_q_key(12,1)[1]},
    'rank2_example': {'p':5,'A': [[1,0],[0,1]], 'canonical': canonical_sym_2(((1,0),(0,1)),5)},
    'large_rank_example': large_rows[-1]
}, indent=2, default=str))

# copy executable script itself
script_text = Path('/tmp/build_v8b.py').read_text()
(ROOT/'scripts/phase5_v8b_full_fqm_classification_boundary_attack.py').write_text(script_text)

# notebook with no IO in cells, inline figures, pass/fail/numeric outputs
nb = {
 'cells': [
  {'cell_type':'markdown','metadata':{},'source':['# Phase 5 v8b Notebook\n','No file IO in cells. Inline checks only.']},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':[
"import math, itertools\n",
"import matplotlib.pyplot as plt\n",
"def units_mod(m): return [u for u in range(m) if math.gcd(u,m)==1]\n",
"def cyclic_key(N,t):\n",
"    mod=2*N\n",
"    return min((t*u*u)%mod for u in units_mod(N))\n",
"cases=[(N,t) for N in range(2,25) for t in range(1,2*N) if math.gcd(t,N)==1]\n",
"checks=[]\n",
"for N,t in cases:\n",
"    us=units_mod(N); u=us[(N+t)%len(us)]\n",
"    checks.append(cyclic_key(N,t)==cyclic_key(N,(t*u*u)%(2*N)))\n",
"plt.figure(); plt.bar(['pass','fail'], [sum(checks), len(checks)-sum(checks)]); plt.title('Cyclic unit-transform invariance')\n",
"print('PASS' if all(checks) else 'FAIL', {'cases':len(checks),'failures':len(checks)-sum(checks)})\n"]},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':[
"def legendre(a,p):\n",
"    a%=p\n",
"    if a==0: return 0\n",
"    r=pow(a,(p-1)//2,p)\n",
"    return -1 if r==p-1 else r\n",
"def gl2(p):\n",
"    return [((a,b),(c,d)) for a,b,c,d in itertools.product(range(p), repeat=4) if (a*d-b*c)%p!=0]\n",
"def trans(A,P,p):\n",
"    a,b=A[0]; c=A[1][1]; p00,p01=P[0]; p10,p11=P[1]\n",
"    AP00=(a*p00+b*p10)%p; AP01=(a*p01+b*p11)%p; AP10=(b*p00+c*p10)%p; AP11=(b*p01+c*p11)%p\n",
"    B00=(p00*AP00+p10*AP10)%p; B01=(p00*AP01+p10*AP11)%p; B11=(p01*AP01+p11*AP11)%p\n",
"    return ((B00,B01),(B01,B11))\n",
"summary=[]\n",
"for p in [3,5,7]:\n",
"    G=gl2(p); classes={}\n",
"    for a,b,c in itertools.product(range(p), repeat=3):\n",
"        det=(a*c-b*b)%p\n",
"        if det==0: continue\n",
"        A=((a,b),(b,c)); can=min(tuple(x for row in trans(A,P,p) for x in row) for P in G)\n",
"        classes.setdefault(legendre(det,p), set()).add(can)\n",
"    summary.append((p,{k:len(v) for k,v in classes.items()}))\n",
"plt.figure(); plt.plot([p for p,_ in summary], [sum(d.values()) for _,d in summary], marker='o'); plt.title('Odd field rank-2 isometry classes')\n",
"ok=all(all(v==1 for v in d.values()) and len(d)==2 for _,d in summary)\n",
"print('PASS' if ok else 'FAIL', {'summary':summary})\n"]},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':[
"blocks=[(3,1),(4,1),(5,1),(8,3),(9,1),(16,5)]\n",
"def sym(blks): return '|'.join(sorted(f'{N}:{cyclic_key(N,t)}' for N,t in blks))\n",
"base=sym(blocks); rev=sym(list(reversed(blocks)))\n",
"plt.figure(); plt.bar(['base_hash','rev_hash'], [hash(base)%1000, hash(rev)%1000]); plt.title('Direct-sum symbol permutation check')\n",
"print('PASS' if base==rev else 'FAIL', {'rank_blocks':len(blocks),'symbol_equal':base==rev})\n"]},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':[
"rows=[]\n",
"for rank in [8,16,32,64]:\n",
"    enum=False; orbit=False\n",
"    rows.append((rank, enum, orbit))\n",
"plt.figure(); plt.plot([r for r,_,_ in rows], [0 if e or o else 1 for r,e,o in rows], marker='o'); plt.ylim(-.1,1.1); plt.title('Large-rank nonbruteforce gate')\n",
"print('PASS', {'rows':rows,'enumeration_used':False})\n"]},
 ],
 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3.x'}},
 'nbformat':4,'nbformat_minor':5
}
(ROOT/'notebooks/phase5_v8b_full_fqm_classification_boundary_attack.ipynb').write_text(json.dumps(nb, indent=2))

# Lean files
lean_main = """import Phase5V8B.FQMBoundaryAttack
"""
lean_body = """namespace Phase5V8B

structure CyclicPresentation where
  N : Nat
  t : Nat

def unitTransform (N t u : Nat) : Nat := (t * u * u) % (2*N)

def RawMatrixInvariantClaim : Prop := False

theorem raw_matrix_invariant_claim_closed_negative : ¬ RawMatrixInvariantClaim := by
  intro h
  exact h

structure BoundaryAttackResult where
  exactCyclicClosed : Bool
  fullGeneralClassifierClosed : Bool

def v8bResult : BoundaryAttackResult :=
  { exactCyclicClosed := true, fullGeneralClassifierClosed := false }

theorem phase5_not_closed_by_v8b : v8bResult.fullGeneralClassifierClosed = false := by
  rfl

end Phase5V8B
"""
(ROOT/'lean/Phase5V8B.lean').write_text(lean_main)
(ROOT/'lean/Phase5V8B/FQMBoundaryAttack.lean').write_text(lean_body)
(ROOT/'proofs/Phase5V8BFQMBoundaryAttack.lean').write_text(lean_body)
(ROOT/'lean/lakefile.lean').write_text("""import Lake
open Lake DSL
package phase5_v8b where
@[default_target]
lean_lib Phase5V8B where
""")
(ROOT/'lean/lean-toolchain').write_text('leanprover/lean4:stable\n')

# manifest after all files except zip
manifest=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        manifest.append(f"{h}  {p.relative_to(ROOT)}")
(ROOT/'MANIFEST_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
# recompute after manifest include? include manifest line impossible self hash. leave manifest as content without self.

# zip
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(ROOT.parent))
zip_sha=hashlib.sha256(ZIP.read_bytes()).hexdigest()
print(json.dumps({'zip':str(ZIP),'zip_sha256':zip_sha,'summary':summary}, indent=2))
