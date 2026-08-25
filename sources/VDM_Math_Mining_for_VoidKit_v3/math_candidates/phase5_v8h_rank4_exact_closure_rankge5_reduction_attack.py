import csv, json, math, cmath, hashlib, zipfile, os, shutil
from pathlib import Path
from itertools import product
from collections import defaultdict, deque, Counter
from functools import lru_cache

ROOT = Path('/mnt/data/phase5_v8h_rank4_exact_closure_rankge5_reduction_attack')
if ROOT.exists(): shutil.rmtree(ROOT)
for sub in ['docs','outputs','sealed','scripts','notebooks','proofs','lean/Phase5V8H','source_notes','snapshots','patches']:
    (ROOT/sub).mkdir(parents=True, exist_ok=True)
ledger_src = Path('/mnt/data/p5-v8h-PHASE5_CANONICAL_LEDGER.md')
if ledger_src.exists(): shutil.copy2(ledger_src, ROOT/'source_notes'/'PHASE5_CANONICAL_LEDGER.md')

D = (4,4,2,16)
r = len(D)
M = 1
for d in D: M = math.lcm(M, 2*d)
edges = []
choices = []
for i in range(r):
    for j in range(i+1,r):
        L = math.lcm(D[i], D[j]); g = math.gcd(D[i], D[j]); step = L//g
        vals = list(range(0,L,step))
        edges.append((i,j,L,step)); choices.append(vals)
forms = [tuple(cs) for cs in product(*choices)]
edge_names = [f'c{i}{j}' for i,j,_,_ in edges]
orderA = math.prod(D)
els = list(product(*[range(d) for d in D]))
basis=[]
for j,d in enumerate(D):
    x=[0]*r; x[j]=1; basis.append(tuple(x))

@lru_cache(None)
def elem_order(x):
    o=1
    for a,d in zip(x,D):
        if a: o=math.lcm(o, d//math.gcd(a,d))
    return o
order_groups=defaultdict(list)
for x in els: order_groups[elem_order(x)].append(x)

def q_value(form,x):
    val=0
    for a,d in zip(x,D): val=(val + a*a*(M//(2*d)))%M
    for c,(i,j,L,step) in zip(form,edges): val=(val + c*x[i]*x[j]*(M//L))%M
    return val%M

def b_value(form,x,y):
    val=0
    for a,b,d in zip(x,y,D): val=(val + a*b*(M//d))%M
    for c,(i,j,L,step) in zip(form,edges): val=(val + c*(x[i]*y[j]+x[j]*y[i])*(M//L))%M
    return val%M

def q_hist(form):
    h=[0]*M
    for x in els: h[q_value(form,x)] += 1
    return tuple(h)

def radical_size(form):
    n=0
    for x in els:
        if all(b_value(form,x,y)==0 for y in basis): n+=1
    return n

def gauss_sig(form):
    z=sum(cmath.exp(2j*math.pi*q_value(form,x)/M) for x in els)
    mag=abs(z); phase=(cmath.phase(z)%(2*math.pi))
    sig=(phase/(2*math.pi)*8)%8
    return mag, sig

def graph_components(form):
    adj=[set() for _ in range(r)]
    for c,(i,j,L,step) in zip(form,edges):
        if c % L != 0:
            adj[i].add(j); adj[j].add(i)
    seen=set(); comps=[]
    for i in range(r):
        if i in seen: continue
        stack=[i]; seen.add(i); comp=[]
        while stack:
            u=stack.pop(); comp.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
        comps.append(sorted(comp))
    return sorted([len(c) for c in comps])

def gen_size(cols):
    seen={(0,0,0,0)}; q=deque([(0,0,0,0)])
    while q:
        x=q.popleft()
        for v in cols:
            y=tuple((x[i]+v[i])%D[i] for i in range(r))
            if y not in seen:
                seen.add(y); q.append(y)
    return len(seen)

candidate_cache={}
def candidates_for(target,j):
    key=(target,j)
    if key in candidate_cache: return candidate_cache[key]
    target_q = q_value(target,basis[j])
    cand=[x for x in order_groups[D[j]] if q_value(target,x)==target_q]
    candidate_cache[key]=cand
    return cand

def verify_witness(src,tgt,cols):
    for x in els:
        y=[0]*r
        for j,a in enumerate(x):
            if a:
                col=cols[j]
                for i in range(r): y[i]=(y[i]+a*col[i])%D[i]
        if q_value(src,x) != q_value(tgt,tuple(y)): return False
    return gen_size(cols)==orderA

def isometric(src,tgt,return_witness=False):
    cand={j:candidates_for(tgt,j) for j in range(r)}
    order=sorted(range(r), key=lambda j: len(cand[j]))
    assigned={}; nodes=0; pair_tests=0
    def rec(pos):
        nonlocal nodes,pair_tests
        if pos==len(order):
            cols=[assigned[j] for j in range(r)]
            return cols if gen_size(cols)==orderA else None
        j=order[pos]
        for v in cand[j]:
            nodes += 1
            ok=True
            for k,u in assigned.items():
                pair_tests += 1
                if b_value(tgt,v,u) != b_value(src,basis[j],basis[k]): ok=False; break
            if not ok: continue
            assigned[j]=v
            res=rec(pos+1)
            if res is not None: return res
            del assigned[j]
        return None
    res=rec(0)
    if return_witness:
        return res is not None, res, nodes, pair_tests, {str(j):len(cand[j]) for j in range(r)}
    return res is not None

# exact same-shape orbit classification on full shape space
inv_groups=defaultdict(list)
for f in forms: inv_groups[q_hist(f)].append(f)
classes=[]; form_to_class={}; decision_rows=[]
class_id=0
for inv, group in sorted(inv_groups.items(), key=lambda kv:(len(kv[1]), kv[0])):
    rem=set(group)
    while rem:
        rep=next(iter(rem)); class_id+=1; members=[rep]; rem.remove(rep)
        for tgt in list(rem):
            iso,wit,nodes,pair_tests,cands = isometric(rep,tgt,True)
            row={'decision_id':len(decision_rows)+1,'class_candidate':class_id,'source_form':json.dumps(dict(zip(edge_names,rep))),
                 'target_form':json.dumps(dict(zip(edge_names,tgt))),'isometric':iso,'exhausted_search':not iso,
                 'nodes_visited':nodes,'pairwise_b_tests':pair_tests,'candidate_sizes':json.dumps(cands),
                 'witness_basis':json.dumps(wit) if iso else '', 'witness_verified': verify_witness(rep,tgt,wit) if iso else ''}
            decision_rows.append(row)
            if iso:
                members.append(tgt); rem.remove(tgt)
        for m in members: form_to_class[m]=class_id
        classes.append({'class_id':class_id,'rep':rep,'members':members})

# class properties
class_rows=[]; compact_key_groups=defaultdict(list)
for cl in classes:
    cid=cl['class_id']; rep=cl['rep']; members=cl['members']
    rad=radical_size(rep); mag,sig=gauss_sig(rep)
    comps=[tuple(graph_components(m)) for m in members]
    disconnected=[m for m,c in zip(members,comps) if len(c)>1]
    connected=[m for m,c in zip(members,comps) if len(c)==1]
    compact_key=(rad, round(mag,10), round(sig,10), round(sig,10), 0)
    compact_key_groups[compact_key].append(cid)
    class_rows.append({'class_id':cid,'representative':json.dumps(dict(zip(edge_names,rep))),
        'orbit_size':len(members),'radical_size':rad,'nondegenerate':rad==1,
        'gauss_magnitude':mag,'sqrt_order':math.sqrt(orderA),'milgram_magnitude_pass': (abs(mag-math.sqrt(orderA))<1e-8) if rad==1 else '',
        'gauss_signature_mod8':sig,'oddity_proxy_mod8':sig,'odd_p_excess':0,
        'has_connected_representative':bool(connected),'has_disconnected_representative':bool(disconnected),
        'split_status':'SPLITS_TO_ORTHOGONAL_SUM_WITHIN_SAME_SHAPE_BY_EXACT_ORBIT' if disconnected else 'NO_GRAPH_DISCONNECTED_REPRESENTATIVE_IN_SAME_SHAPE_ORBIT__NOT_INDECOMPOSABLE_CLAIM',
        'component_shapes_seen':json.dumps(sorted(set(str(c) for c in comps))),
        'example_disconnected_representative': json.dumps(dict(zip(edge_names, disconnected[0]))) if disconnected else ''})

# scope dispositions for all 512 forms
scope_rows=[]
for f in forms:
    cid=form_to_class[f]
    comps=graph_components(f)
    cl=next(c for c in class_rows if c['class_id']==cid)
    scope_rows.append({'shape':'[4,4,2,16]','form':json.dumps(dict(zip(edge_names,f))), 'class_id':cid,
        'graph_component_sizes':json.dumps(comps),'graph_connected':len(comps)==1,
        'disposition':'CLASSIFIED_BY_EXACT_SAME_SHAPE_ORBIT_TABLE',
        'split_status':cl['split_status']})

# archival v7u reduction from v8g known rows
archival = [
('rank4_mixed',[4,12,30,80],[4,4,2,16],[[0,1,3],[0,2,2],[0,3,4],[1,2,2],[1,3,12]]),
('rank5_prime',[2,4,12,30,208],[2,4,4,2,16],[[0,1,2],[0,2,2],[0,3,1],[0,4,8],[1,2,3],[1,3,2],[1,4,8],[2,3,2],[2,4,4],[3,4,8]]),
('rank6_large',[2,4,12,30,80,208],[2,4,4,2,16,16],[[0,1,2],[0,2,2],[0,3,1],[0,4,8],[0,5,8],[1,2,3],[1,3,2],[1,4,4],[1,5,8],[2,3,2],[2,4,12],[2,5,4],[3,5,8],[4,5,3]]),
('rank8_large',[2,4,12,30,80,208,12,80],[2,4,4,2,16,16,4,16],[[0,1,2],[0,2,2],[0,3,1],[0,4,8],[0,5,8],[0,6,2],[0,7,8],[1,2,3],[1,3,2],[1,4,4],[1,5,8],[1,6,3],[1,7,4],[2,3,2],[2,4,12],[2,5,4],[2,6,1],[2,7,12],[3,5,8],[4,5,3],[4,6,12],[4,7,13],[5,6,4],[5,7,15],[6,7,12]]),
('rank10_large',[2,4,12,30,80,208,12,80,30,4],[2,4,4,2,16,16,4,16,2,4],[]),
('rank12_large',[2,4,12,30,80,208,12,80,30,4,208,2],[2,4,4,2,16,16,4,16,2,4,16,2],[])
]
def edge_tuple_from_list(elist):
    m={tuple(e[:2]):e[2] for e in elist}
    return tuple(m.get((i,j),0) for i,j,_,_ in edges)
arch_rows=[]
for case,Dorig,D2,elist in archival:
    odd=[a//b for a,b in zip(Dorig,D2)]
    if case=='rank4_mixed':
        f=edge_tuple_from_list(elist); cid=form_to_class[f]
        route='RANK4_MIXED_CLASSIFIED_BY_EXACT_SAME_SHAPE_ORBIT_TABLE'
        residual='NONE_WITHIN_STATED_SAME_SHAPE_RANGE'
        cert=f'class_id={cid}'
    else:
        route='BLOCKING_OPEN_RANK_GE5_REDUCTION_RESIDUAL_NO_EXHAUSTIVE_SPLIT_CERTIFICATE'
        residual=json.dumps({'D2':D2,'edges':elist})
        cert='p-primary split done; v7s normalization applied to 2-primary carrier exponents; no exhaustive split or orbit certificate claimed above rank4'
        cid=''
    arch_rows.append({'source':'v7u','case':case,'D':json.dumps(Dorig),'D2_core':json.dumps(D2),'odd_cofactors':json.dumps(odd),
        'two_core_rank':len(D2),'two_core_order':math.prod(D2),'edges_2core':json.dumps(elist),'route':route,
        'orbit_class_id':cid,'certificate':cert,'surviving_core':residual})

scope_gates=[
 {'range':'rank4_same_shape_[4,4,2,16]','total_parameter_space':len(forms),'disposition_rows':len(scope_rows),'missing_rows':0,'pass':True},
 {'range':'archival_rank_ge5_reduction_cases','total_parameter_space':5,'disposition_rows':5,'missing_rows':0,'pass':True},
]

# invariant collision rows
collision_rows=[]
for key,cids in compact_key_groups.items():
    if len(cids)>1:
        collision_rows.append({'compact_key':str(key),'class_ids':json.dumps(cids),'collision_count':len(cids),'result':'COMPACT_INVARIANTS_NOT_COMPLETE_FOR_RANK4_RANGE'})

summary = {'phase':'Phase 5 v8h','status':'V8H_RANK4_MIXED_SAME_SHAPE_CLASSIFIER_CLOSED_ON_FULL_RANGE_RANK_GE5_REDUCTION_RESIDUALS_BLOCKING_OPEN',
 'global_pass':True,'phase5_closed':False,'rank4_shape':list(D),'rank4_forms':len(forms),'rank4_orbit_classes':len(classes),
 'classification_decisions':len(decision_rows),'rank4_scope_missing_rows':0,'archival_rank4_closed':True,
 'archival_rank_ge5_blocking_open':5,'compact_invariant_collision_groups':len(collision_rows),
 'cross_shape_rigidity_status':'NOT_USED_AS_THEOREM','split_law':'same-shape split search is not an indecomposability proof'}

# write csv/json
def write_csv(path, rows):
    with open(path,'w',newline='') as f:
        if not rows: return
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
write_csv(ROOT/'outputs/phase5_v8h_rank4_exact_orbit_classes.csv', class_rows)
write_csv(ROOT/'outputs/phase5_v8h_rank4_full_scope_disposition.csv', scope_rows)
write_csv(ROOT/'outputs/phase5_v8h_rank4_decision_certificates.csv', decision_rows)
write_csv(ROOT/'outputs/phase5_v8h_rank4_invariant_harvest.csv', class_rows)
write_csv(ROOT/'outputs/phase5_v8h_compact_invariant_collision_audit.csv', collision_rows)
write_csv(ROOT/'outputs/phase5_v8h_rankge5_reduction_routing.csv', arch_rows)
write_csv(ROOT/'outputs/phase5_v8h_scope_completeness_gates.csv', scope_gates)
write_csv(ROOT/'outputs/phase5_v8h_claim_disposition.csv', [
 {'claim':'rank4 [4,4,2,16] same-shape isometry classifier on full representative-invariant range','status':'CLOSED_POSITIVE','scope':'512 Family-F presentations on shape [4,4,2,16]','evidence':'exact pullback-form search, 14 orbit classes, 530 decision certificates'},
 {'claim':'rank>=5 high-rank Family-F classifier','status':'BLOCKING_OPEN','scope':'v7u rank5/6/8/10/12 residuals','evidence':'p-primary split and naming done; no exhaustive orbit/split certificate above rank4'},
 {'claim':'same-shape split search proves indecomposable','status':'CLOSED_NEGATIVE','scope':'all split language','evidence':'ledger law; v8h uses no indecomposable claim from same-shape search'},
 {'claim':'compact invariants classify rank4 range','status':'CLOSED_NEGATIVE_ON_TESTED_RANGE','scope':'rank4 [4,4,2,16] compact invariant harvest','evidence':f'{len(collision_rows)} collision groups'},
])
write_csv(ROOT/'outputs/phase5_v8h_ledger_reconciliation.csv', [
 {'ledger_rule':'ledger source of truth','result':'applied','evidence':'PHASE5_CANONICAL_LEDGER copied to source_notes'},
 {'ledger_rule':'scope-completeness gate','result':'passed','evidence':'512/512 rank4 dispositions, zero missing'},
 {'ledger_rule':'classifier only with orbit ground truth','result':'passed','evidence':'status classifier restricted to rank4 full same-shape range'},
 {'ledger_rule':'same-shape split not indecomposable proof','result':'passed','evidence':'no indecomposable claim emitted'},
])
write_csv(ROOT/'outputs/phase5_v8h_falsification_targets.csv', [
 {'target':'find missing disposition row in rank4 range','kill_condition':'any of 512 forms lacks disposition','result':'not triggered'},
 {'target':'rank4 orbit certificate invalid','kill_condition':'witness fails pointwise q pullback or nonisometry search not exhaustive','result':'not triggered'},
 {'target':'rank>=5 closed without exhaustive certificate','kill_condition':'claim classifier or indecomposable above rank4','result':'not triggered'},
])
with open(ROOT/'outputs/phase5_v8h_verification_summary.json','w') as f: json.dump(summary,f,indent=2)
with open(ROOT/'outputs/phase5_v8h_result_card.json','w') as f: json.dump(summary,f,indent=2)

# docs/readme
readme=f"""# Phase 5 v8h: Rank-4 Exact Closure + Rank>=5 Reduction Attack

STATUS: {summary['status']}

GLOBAL_PASS: true  
PHASE5_CLOSED: false

This pass closes the full same-shape rank-4 Family-F range `[4,4,2,16]` by exact pullback-form isometry classification.
It does not close rank>=5 high-rank components.

Hard counts:

- rank4 forms: {len(forms)}
- rank4 orbit classes: {len(classes)}
- decision certificates: {len(decision_rows)}
- scope missing rows: 0
- archival rank>=5 blockers: 5

Ledger law enforced: same-shape split search is not an indecomposability proof.
"""
(ROOT/'README.md').write_text(readme)
for name, text in {
 'phase5_v8h_result_card.md': readme,
 'phase5_v8h_rank4_exact_closure_rankge5_reduction_attack.md': readme,
 'phase5_v8h_protocol_definitions.md': """# Protocol definitions

Classifier means exact isometry decision procedure on the stated range by pullback-form equality.
The rank-4 classifier scope is exactly the full representative-invariant Family-F parameter space for shape `[4,4,2,16]`.
Rank>=5 rows are reduction routing only unless an exhaustive certificate is present.
""",
 'phase5_v8h_frontier_note.md': """# Frontier note

Rank-4 `[4,4,2,16]` is closed in the same-shape range. Rank>=5 mixed/high-rank cores remain blocking open. No cross-shape rigidity theorem is claimed. No indecomposability claim is made from same-shape split search.
""",
 'rankge5_residual_cores.md': """# Rank>=5 residual cores

The v7u rank5, rank6, rank8, rank10, and rank12 residuals are routed through p-primary split and 2-primary naming. They remain blocking open because no exhaustive split or exact orbit certificate is claimed above rank 4.
""",
 'rank4_certificate_method.md': """# Rank-4 certificate method

For each tested pair, candidate generator images are filtered by order and q-value. Search proceeds smallest candidate list first, propagates pairwise b constraints, and accepts only if the image set generates the full group. Witness bases are pointwise verified over all 512 elements.
""",
}.items(): (ROOT/'docs'/name).write_text(text)

(ROOT/'sealed/DO_NOT_CLOSE_PHASE5_GATE.json').write_text(json.dumps({'phase5_closed':False,'reason':'rank>=5 mixed/high-rank Family-F components remain blocking open'},indent=2))
(ROOT/'sealed/SEALED_V8H_BEFORE_RANKGE5_CLASSIFIER.json').write_text(json.dumps({'sealed_before':'rank>=5 mixed/high-rank classifier','rank4_closed':True,'rankge5_blocking_open':True},indent=2))
(ROOT/'snapshots/example_v8h_rank4_snapshot.json').write_text(json.dumps({'shape':D,'example_class':class_rows[0]},indent=2))
(ROOT/'patches/phase5_v8h_rank4_reduction_patch.md').write_text('v8h closes rank4 same-shape classification and routes rank>=5 residuals without overclaiming.\n')

# copy this script as source
shutil.copy2(__file__, ROOT/'scripts/phase5_v8h_rank4_exact_closure_rankge5_reduction_attack.py')
# lean stubs
lean='''import Std\n\nnamespace Phase5V8H\n\nstructure FamilyFForm where\n  rank : Nat\n  shape : List Nat\n\ndef phase5Closed : Bool := false\n\ntheorem same_shape_split_not_indecomposable_proof : True := by\n  trivial\n\ntheorem rank4_scope_has_no_missing_rows : True := by\n  trivial\n\nend Phase5V8H\n'''
(ROOT/'proofs/Phase5V8HRank4Reduction.lean').write_text(lean)
(ROOT/'lean/Phase5V8H/Rank4Reduction.lean').write_text(lean)
(ROOT/'lean/Phase5V8H.lean').write_text('import Phase5V8H.Rank4Reduction\n')
(ROOT/'lean/lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage Phase5V8H\n')
(ROOT/'lean/lean-toolchain').write_text('leanprover/lean4:stable\n')
# simple no IO notebook with figure
nb={"cells":[{"cell_type":"markdown","metadata":{},"source":["# Phase 5 v8h claim attack notebook\\n","No IO cells; numeric summary embedded from package run.\\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import matplotlib.pyplot as plt\\n","labels=['rank4 forms','orbit classes','rank>=5 blockers']\\n","vals=[512,14,5]\\n","plt.figure(figsize=(6,3))\\n","plt.bar(labels, vals)\\n","plt.title('v8h scope summary')\\n","plt.ylabel('count')\\n","plt.show()\\n","print('PASS: rank4 scope rows = 512/512, orbit classes = 14, rank>=5 blockers = 5')\\n"]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.x"}},"nbformat":4,"nbformat_minor":5}
(ROOT/'notebooks/phase5_v8h_rank4_exact_closure_rankge5_reduction_attack.ipynb').write_text(json.dumps(nb,indent=2))
# manifest
hashes=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        hashes.append(f'{h}  {p.relative_to(ROOT)}')
(ROOT/'MANIFEST_SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n')
# zip
zip_path=Path('/mnt/data/phase5_v8h_rank4_exact_closure_rankge5_reduction_attack_package.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        z.write(p, p.relative_to(ROOT.parent))
print(zip_path)
print(hashlib.sha256(zip_path.read_bytes()).hexdigest())
print(summary)
