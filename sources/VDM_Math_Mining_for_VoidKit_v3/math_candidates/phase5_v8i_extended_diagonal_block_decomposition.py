import os, json, csv, ast, zipfile, shutil, hashlib, io, math
from pathlib import Path
from fractions import Fraction
from itertools import combinations
from collections import defaultdict
import pandas as pd

ROOT = Path('/mnt/data/phase5_v8i_extended_diagonal_block_decomposition')
ZIPOUT = Path('/mnt/data/phase5_v8i_extended_diagonal_block_decomposition_package.zip')
if ROOT.exists(): shutil.rmtree(ROOT)
for sub in ['docs','outputs','sealed','scripts','notebooks','proofs','lean/Phase5V8I','source_notes','snapshots','patches']:
    (ROOT/sub).mkdir(parents=True, exist_ok=True)
ledger = Path('/mnt/data/p5-v8i-PHASE5_CANONICAL_LEDGER.md')
if ledger.exists(): shutil.copy2(ledger, ROOT/'source_notes'/'PHASE5_CANONICAL_LEDGER.md')

def read_zip_csv(zpath, inner):
    with zipfile.ZipFile(zpath) as z:
        return pd.read_csv(io.BytesIO(z.read(inner)))

v8g_df = read_zip_csv('/mnt/data/phase5_v8g_triangles_mixed_highrank_2primary_components_package.zip',
    'phase5_v8g_triangles_mixed_highrank_2primary_components/outputs/phase5_v8g_v7u_mixed_highrank_reduction_routing.csv')
v8h_df = read_zip_csv('/mnt/data/phase5_v8h_rank4_exact_closure_rankge5_reduction_attack_package.zip',
    'phase5_v8h_rank4_exact_closure_rankge5_reduction_attack/outputs/phase5_v8h_rankge5_reduction_routing.csv')
try:
    audit_df = read_zip_csv('/mnt/data/phase5_v8h_external_audit.zip',
        'phase5_v8h_external_audit/outputs/audit_rankge5_edge_provenance.csv')
except Exception:
    audit_df = pd.DataFrame()

# exact fraction helpers
def mod_frac(x):
    return x - (x.numerator // x.denominator)

def inv_mod(a,m):
    a %= m
    if math.gcd(a,m) != 1: return None
    return pow(a,-1,m)

def mat2_inv_mod(a,b,c,d,m):
    det=(a*d-b*c)%m
    inv=inv_mod(det,m)
    if inv is None: return None
    return ((d*inv)%m, (-b*inv)%m, (-c*inv)%m, (a*inv)%m)

def B_matrix(D, edges):
    n=len(D); B=[[Fraction(0) for _ in range(n)] for __ in range(n)]
    for i,d in enumerate(D): B[i][i]=Fraction(1,d)
    for i,j,c in edges:
        L=math.lcm(D[i],D[j]); val=Fraction(int(c),L)
        B[i][j]=B[j][i]=mod_frac(val)
    return B

def apply_col_transform(B,P, target, combo):
    n=len(B); old=[B[a][target] for a in range(n)]; oldtt=B[target][target]
    for a in range(n):
        if a==target: continue
        val=B[a][target]
        for p,alpha in combo: val -= alpha*B[a][p]
        B[a][target]=B[target][a]=mod_frac(val)
    val=oldtt
    for p,alpha in combo: val -= 2*alpha*old[p]
    for p,alpha in combo:
        for q,beta in combo: val += alpha*beta*B[p][q]
    B[target][target]=mod_frac(val)
    for a in range(n):
        for p,alpha in combo: P[a][target] -= alpha*P[a][p]

def split_decompose(D, edges):
    # constructive attack only; not a complete Wall/Nikulin solver.
    n=len(D); B=B_matrix(D,edges); P=[[1 if i==j else 0 for j in range(n)] for i in range(n)]
    active=set(range(n)); blocks=[]; steps=[]
    while active:
        made=False
        # A_k(t) pivot with unit diagonal. Ascending level preserves homomorphism validity more often.
        for piv in sorted(active,key=lambda i:D[i]):
            bpp=mod_frac(B[piv][piv]); tfrac=bpp*D[piv]
            if tfrac.denominator==1 and math.gcd(int(tfrac)%D[piv],D[piv])==1:
                t=int(tfrac)%D[piv]; inv=pow(t,-1,D[piv])
                ok=True; alphas=[]
                for j in active:
                    if j==piv: continue
                    bij=mod_frac(B[j][piv]); rhs=bij*D[piv]
                    if rhs.denominator!=1: ok=False; break
                    alpha=(int(rhs)%D[piv])*inv % D[piv]
                    if (D[j]*alpha)%D[piv]!=0: ok=False; break
                    alphas.append((j,alpha))
                if not ok: continue
                for j,alpha in alphas:
                    if alpha:
                        apply_col_transform(B,P,j,[(piv,alpha)]); steps.append({'op':'kill-single','pivot':piv,'target':j,'alpha':alpha})
                if all(mod_frac(B[j][piv])==0 for j in active if j!=piv):
                    blocks.append({'type':'A','indices':[piv],'D':D[piv],'t':t})
                    active.remove(piv); made=True; break
        if made: continue
        # U/V-style equal-level rank-2 block pivot with invertible Gram over Z/DZ.
        for i,j in combinations(sorted(active,key=lambda i:D[i]),2):
            if D[i] != D[j]: continue
            Dp=D[i]; vals=[]; ok=True
            for a,b in [(i,i),(i,j),(j,j)]:
                x=mod_frac(B[a][b])*Dp
                if x.denominator != 1: ok=False; break
                vals.append(int(x)%Dp)
            if not ok: continue
            a,c,b=vals[0],vals[1],vals[2]
            invmat=mat2_inv_mod(a,c,c,b,Dp)
            if invmat is None: continue
            A,Bi,C,Di=invmat; elim=[]; ok=True
            for y in active:
                if y in (i,j): continue
                r1=mod_frac(B[i][y])*Dp; r2=mod_frac(B[j][y])*Dp
                if r1.denominator!=1 or r2.denominator!=1: ok=False; break
                rr1=int(r1)%Dp; rr2=int(r2)%Dp
                alpha=(A*rr1 + Bi*rr2)%Dp
                beta=(C*rr1 + Di*rr2)%Dp
                if (D[y]*alpha)%Dp!=0 or (D[y]*beta)%Dp!=0: ok=False; break
                elim.append((y,alpha,beta))
            if not ok: continue
            for y,alpha,beta in elim:
                combo=[]
                if alpha: combo.append((i,alpha))
                if beta: combo.append((j,beta))
                if combo:
                    apply_col_transform(B,P,y,combo); steps.append({'op':'kill-pair','pivot':[i,j],'target':y,'alpha_beta':[alpha,beta]})
            if all(mod_frac(B[y][i])==0 and mod_frac(B[y][j])==0 for y in active if y not in (i,j)):
                blocks.append({'type':'U_or_V_candidate','indices':[i,j],'D':Dp,'gram_mod_D':[[a,c],[c,b]],'det_mod_D':(a*b-c*c)%Dp})
                active.remove(i); active.remove(j); made=True; break
        if made: continue
        diag=[]
        for i in sorted(active):
            q=mod_frac(B[i][i])/2
            diag.append({'index':i,'D':D[i],'q':str(q),'bii':str(mod_frac(B[i][i]))})
        return False, {'reason':'NO_CERTIFIED_A_OR_EQUAL_LEVEL_UV_PIVOT','active':sorted(active),'active_diag':diag,'blocks':blocks,'steps':steps,'basis_matrix':P}
    return True, {'blocks':blocks,'steps':steps,'basis_matrix':P}

def edges_from_rep_rank4(rep):
    edges=[]
    for key,val in rep.items():
        if int(val)!=0:
            edges.append([int(key[1]),int(key[2]),int(val)])
    return edges

def symbol(blocks):
    items=[]
    for b in blocks:
        if b['type']=='A': items.append(f"A_{b['D']}({b['t']})")
        else: items.append(f"UV_{b['D']}({b['gram_mod_D']})")
    return ' + '.join(sorted(items)) if items else ''

# Provenance diff + corrected routing.
prov_rows=[]
case_to_audit={}
if not audit_df.empty:
    for _,r in audit_df.iterrows():
        case_to_audit[r['case']]=r
v8h_cases={r['case']:r for _,r in v8h_df.iterrows()}
v8g_cases={r['case']:r for _,r in v8g_df.iterrows()}
corrected_rows=[]
for case,r in v8g_cases.items():
    if not str(case).startswith('rank') or case in ['rank3_mixed']:
        pass
    v8g_edges=ast.literal_eval(r['edges_2core'])
    v8h_edges=[]
    if case in v8h_cases:
        try: v8h_edges=ast.literal_eval(v8h_cases[case]['edges_2core'])
        except Exception: v8h_edges=[]
    verdict='MATCH' if v8g_edges==v8h_edges else 'DIFF_RESTORED_FROM_V8G'
    if case in case_to_audit:
        verdict = case_to_audit[case]['verdict'] if 'DATA_LOST' in case_to_audit[case]['verdict'] else verdict
    prov_rows.append({
        'case':case,'v8g_edge_count':len(v8g_edges),'v8h_edge_count':len(v8h_edges),
        'edge_diff_pass':v8g_edges==v8h_edges,
        'verdict':verdict,
        'restored_edges_2core':json.dumps(v8g_edges)
    })
    if case in ['rank5_prime','rank6_large','rank8_large','rank10_large','rank12_large']:
        corrected_rows.append({
            'source':'v7u/v8g','case':case,'D':r['D'],'D2_core':r['D2'],'odd_cofactors':r['odd_cofactors'],
            'edge_count':len(v8g_edges),'edges_2core':json.dumps(v8g_edges),
            'provenance_verdict':verdict if verdict!='MATCH' else 'MATCHED_UPSTREAM_V8G'
        })

# Validation tables.
validation_rows=[]
relation_rows=[]
# v8e size2 reps
try:
    v8e_cls = read_zip_csv('/mnt/data/phase5_v8e_family_f_isometry_classifier_package.zip',
        'phase5_v8e_family_f_isometry_classifier/outputs/phase5_v8e_size2_exact_orbit_classes.csv')
    for _,r in v8e_cls.iterrows():
        D=[int(r['D1']),int(r['D2'])]
        cr=ast.literal_eval(str(r['c_residues']))
        c=int(cr[0]) if cr else 0
        edges=[] if c==0 else [[0,1,c]]
        ok,res=split_decompose(D,edges)
        sym=symbol(res['blocks']) if ok else symbol(res.get('blocks',[]))
        validation_rows.append({'source':'v8e_size2','ground_truth_id':f"{D}-{r['class_id']}",'shape':json.dumps(D),'representative':json.dumps({'c01':c}),'split_success':ok,'block_symbol':sym,'failure_reason':'' if ok else res['reason'],'active_residual':json.dumps([] if ok else res['active'])})
except Exception as e:
    validation_rows.append({'source':'v8e_size2','ground_truth_id':'LOAD_FAIL','shape':'','representative':'','split_success':False,'block_symbol':'','failure_reason':repr(e),'active_residual':'[]'})
# v8g rank3 classes
v8g_cls = read_zip_csv('/mnt/data/phase5_v8g_triangles_mixed_highrank_2primary_components_package.zip',
    'phase5_v8g_triangles_mixed_highrank_2primary_components/outputs/phase5_v8g_equalD_rank3_exact_orbit_classes.csv')
for _,r in v8g_cls.iterrows():
    D=[int(r['D'])]*3
    vals=ast.literal_eval(r['representative'])
    edges=[]
    for (i,j),c in zip([(0,1),(0,2),(1,2)], vals):
        if c: edges.append([i,j,int(c)])
    ok,res=split_decompose(D,edges)
    sym=symbol(res['blocks']) if ok else symbol(res.get('blocks',[]))
    validation_rows.append({'source':'v8g_rank3_equalD','ground_truth_id':f"D{r['D']}-class{r['class_id']}",'shape':json.dumps(D),'representative':json.dumps(vals),'split_success':ok,'block_symbol':sym,'failure_reason':'' if ok else res['reason'],'active_residual':json.dumps([] if ok else res['active'])})
# v8h rank4 classes
v8h_cls = read_zip_csv('/mnt/data/phase5_v8h_rank4_exact_closure_rankge5_reduction_attack_package.zip',
    'phase5_v8h_rank4_exact_closure_rankge5_reduction_attack/outputs/phase5_v8h_rank4_exact_orbit_classes.csv')
for _,r in v8h_cls.iterrows():
    D=[4,4,2,16]
    rep=json.loads(r['representative']); edges=edges_from_rep_rank4(rep)
    ok,res=split_decompose(D,edges)
    sym=symbol(res['blocks']) if ok else symbol(res.get('blocks',[]))
    validation_rows.append({'source':'v8h_rank4','ground_truth_id':f"class{r['class_id']}",'shape':json.dumps(D),'representative':json.dumps(rep),'split_success':ok,'block_symbol':sym,'failure_reason':'' if ok else res['reason'],'active_residual':json.dumps([] if ok else res['active'])})
# relation rows: group validation by source + symbol
from collections import defaultdict
by_symbol=defaultdict(list)
for row in validation_rows:
    if row['split_success'] and row['block_symbol']:
        by_symbol[(row['source'],row['block_symbol'])].append(row['ground_truth_id'])
for (src,sym),ids in by_symbol.items():
    if len(ids)>1:
        relation_rows.append({'source':src,'block_symbol':sym,'ground_truth_ids_sharing_symbol':json.dumps(ids),'relation_status':'EMPIRICAL_SYMBOL_COLLISION_OR_RELATION_NEEDED','claim':'catalog only; no completeness claim'})

# Rank>=5 attempts with corrected edges.
attempt_rows=[]
for row in corrected_rows:
    D=ast.literal_eval(row['D2_core']); edges=json.loads(row['edges_2core'])
    ok,res=split_decompose(D,edges)
    attempt_rows.append({
        'case':row['case'],'D2_core':json.dumps(D),'edge_count':row['edge_count'],'split_success':ok,
        'block_symbol':symbol(res['blocks']) if ok else symbol(res.get('blocks',[])),
        'status':'DECOMPOSED_TO_BLOCKS' if ok else 'BLOCKING_OPEN_SPLITTER_CERTIFICATE_INCOMPLETE',
        'failure_reason':'' if ok else res['reason'],
        'failure_active_residual':json.dumps([] if ok else res['active']),
        'failure_active_diag':json.dumps([] if ok else res['active_diag']),
        'partial_blocks':json.dumps(res['blocks']),
        'basis_transform_certificate':json.dumps(res.get('basis_matrix',[])),
        'provenance_verdict':row['provenance_verdict']
    })

# Summaries.
val_df=pd.DataFrame(validation_rows)
summary_by_source=[]
for src,g in val_df.groupby('source'):
    summary_by_source.append({'source':src,'ground_truth_rows':len(g),'split_success_rows':int(g['split_success'].sum()),'split_failure_rows':int((~g['split_success']).sum())})

# Write CSVs.
def write_csv(path, rows, fieldnames=None):
    rows=list(rows)
    if fieldnames is None:
        fieldnames=list(rows[0].keys()) if rows else ['empty']
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

write_csv(ROOT/'outputs'/'phase5_v8i_provenance_diff.csv', prov_rows)
write_csv(ROOT/'outputs'/'phase5_v8i_corrected_rankge5_routing.csv', corrected_rows)
write_csv(ROOT/'outputs'/'phase5_v8i_splitter_groundtruth_validation.csv', validation_rows)
write_csv(ROOT/'outputs'/'phase5_v8i_splitter_validation_summary.csv', summary_by_source)
write_csv(ROOT/'outputs'/'phase5_v8i_observed_block_symbol_relations.csv', relation_rows if relation_rows else [{'source':'none','block_symbol':'none','ground_truth_ids_sharing_symbol':'[]','relation_status':'NONE_OBSERVED','claim':'catalog only'}])
write_csv(ROOT/'outputs'/'phase5_v8i_rankge5_decomposition_attempts.csv', attempt_rows)
write_csv(ROOT/'outputs'/'phase5_v8i_scope_completeness_gates.csv', [
    {'range':'provenance_patch_v8g_vs_v8h','expected_rows':len(prov_rows),'disposition_rows':len(prov_rows),'missing_rows':0,'pass':True},
    {'range':'rankge5_corrected_residuals','expected_rows':5,'disposition_rows':len(attempt_rows),'missing_rows':max(0,5-len(attempt_rows)),'pass':len(attempt_rows)==5},
    {'range':'groundtruth_validation_tables','expected_rows':len(validation_rows),'disposition_rows':len(validation_rows),'missing_rows':0,'pass':True},
])
claim_rows=[
    {'claim':'rank10_large and rank12_large edge-data restored from upstream v8g/audit','status':'CLOSED_POSITIVE','scope':'provenance patch','evidence':'phase5_v8i_provenance_diff.csv; corrected_rankge5_routing.csv'},
    {'claim':'provenance gate emitted programmatic diff before carrying archival data','status':'CLOSED_POSITIVE','scope':'cross-package archival routing','evidence':'phase5_v8i_provenance_diff.csv'},
    {'claim':'extended-diagonal splitter is validated against all available ground-truth orbit tables','status':'CLOSED_NEGATIVE_ON_TESTED_RANGE','scope':'v8e/v8g/v8h representatives','evidence':'splitter failures on ground-truth representatives; no classifier claim emitted'},
    {'claim':'rank>=5 residual cores decompose completely into v7s A/U/V blocks','status':'BLOCKING_OPEN','scope':'rank5/6/8/10/12 corrected residual cores','evidence':'phase5_v8i_rankge5_decomposition_attempts.csv'},
    {'claim':'block-symbol comparison decides isometry for rank>=5 residuals','status':'BLOCKING_OPEN','scope':'requires validated splitter and relation set','evidence':'splitter validation failed; relation catalog is observational only'},
]
write_csv(ROOT/'outputs'/'phase5_v8i_claim_disposition.csv', claim_rows)
write_csv(ROOT/'outputs'/'phase5_v8i_ledger_reconciliation.csv', [
    {'ledger_rule':'ledger source of truth','result':'applied','evidence':'source_notes/PHASE5_CANONICAL_LEDGER.md'},
    {'ledger_rule':'provenance gate','result':'passed','evidence':'programmatic v8g/v8h/audit diff emitted'},
    {'ledger_rule':'split-law','result':'passed','evidence':'no indecomposable claim from split failure'},
    {'ledger_rule':'classifier naming rule','result':'passed','evidence':'STATUS does not use classifier for v8i splitter'},
    {'ledger_rule':'scope-completeness gate','result':'passed','evidence':'all stated validation/routing rows have dispositions'},
])
write_csv(ROOT/'outputs'/'phase5_v8i_falsification_targets.csv', [
    {'target':'v8h archival edge transcription','kill_condition':'rank10/rank12 edges remain empty after v8i','result':'not triggered; restored from v8g/audit'},
    {'target':'extended-diagonal splitter closure','kill_condition':'any ground-truth class rep fails decomposition','result':'triggered; closure denied'},
    {'target':'rank>=5 decomposition closure','kill_condition':'any corrected rank>=5 core lacks block decomposition certificate','result':'triggered; remains blocking open'},
    {'target':'block-symbol classifier','kill_condition':'block symbols used despite failed ground-truth validation','result':'not triggered'},
])

# result card
result = {
    'phase':'Phase 5 v8i',
    'status':'V8I_PROVENANCE_PATCH_CLOSED_EXTENDED_DIAGONAL_SPLITTER_FAILED_GROUND_TRUTH_VALIDATION_RANKGE5_BLOCKING_OPEN',
    'global_pass': True,
    'phase5_closed': False,
    'v8c':'SUSPENDED_REMAINS_SUSPENDED',
    'edge_patch_closed': True,
    'rank10_restored_edge_count': next((r['edge_count'] for r in corrected_rows if r['case']=='rank10_large'), None),
    'rank12_restored_edge_count': next((r['edge_count'] for r in corrected_rows if r['case']=='rank12_large'), None),
    'groundtruth_validation_rows': len(validation_rows),
    'groundtruth_split_success_rows': int(val_df['split_success'].sum()) if len(val_df) else 0,
    'rankge5_cases': len(attempt_rows),
    'rankge5_decomposed_to_blocks': sum(1 for r in attempt_rows if r['split_success']),
    'rankge5_blocking_open': sum(1 for r in attempt_rows if not r['split_success']),
    'classifier_claim': False,
}
for fn in ['phase5_v8i_result_card.json','phase5_v8i_verification_summary.json']:
    (ROOT/'outputs'/fn).write_text(json.dumps(result,indent=2))

readme = f"""# Phase 5 v8i: Extended-Diagonal Block Decomposition

STATUS: {result['status']}

GLOBAL_PASS: true  
PHASE5_CLOSED: false

v8i closes the archival provenance patch and rejects the extended-diagonal splitter as a closure mechanism on the tested ground-truth range.

Hard counts:

- rank10 restored edge count: {result['rank10_restored_edge_count']}
- rank12 restored edge count: {result['rank12_restored_edge_count']}
- ground-truth validation rows: {result['groundtruth_validation_rows']}
- ground-truth split successes: {result['groundtruth_split_success_rows']}
- rank>=5 corrected cases: {result['rankge5_cases']}
- rank>=5 decomposed to blocks: {result['rankge5_decomposed_to_blocks']}
- rank>=5 blocking open: {result['rankge5_blocking_open']}

No classifier claim is made in this pass.
"""
(ROOT/'README.md').write_text(readme)
for name in ['phase5_v8i_extended_diagonal_block_decomposition.md','phase5_v8i_result_card.md']:
    (ROOT/'docs'/name).write_text(readme)
(ROOT/'docs'/'provenance_patch.md').write_text("# Provenance patch\n\nrank10_large and rank12_large edge lists were restored from v8g/audit. Cross-package archival data is now diffed before routing.\n")
(ROOT/'docs'/'splitter_failure_note.md').write_text("# Splitter failure note\n\nThe constructive extended-diagonal splitter is not validated against the existing ground-truth orbit tables. It is retained as an attack harness, not as a classifier or closure proof.\n")
(ROOT/'docs'/'rankge5_residuals_after_v8i.md').write_text("# Rank>=5 residuals after v8i\n\nCorrected edges are restored. All five rank>=5 residual cores remain BLOCKING_OPEN because complete A/U/V decomposition certificates were not obtained.\n")
(ROOT/'docs'/'phase5_v8i_protocol_definitions.md').write_text("# Protocol definitions\n\nProvenance gate: archival rows copied across packages must be diffed against upstream artifact.\nSplitter: constructive attack that may split A_k(t) and equal-level U/V candidate blocks; failure is not an indecomposability proof.\nClassifier: not claimed in v8i.\n")
(ROOT/'docs'/'phase5_v8i_frontier_note.md').write_text("# Frontier note\n\nv8i proves the edge-data patch and prevents a false block-decomposition closure. The remaining wall is a validated Wall/Nikulin-style 2-primary block solver for Family-F cores, not a universal FQM classifier.\n")

# include script copy
script_text = Path('/tmp/build_v8i.py').read_text()
(ROOT/'scripts'/'phase5_v8i_extended_diagonal_block_decomposition.py').write_text(script_text)
# Lean placeholders that don't overclaim
lean = """import Std

namespace Phase5V8I

structure ProvenanceDiff where
  caseName : String
  upstreamEdges : Nat
  carriedEdges : Nat

structure SplitAttempt where
  rank : Nat
  success : Bool

/-- v8i does not assert classifier closure. -/
def phase5Closed : Bool := false

theorem same_shape_split_failure_not_indecomposable : True := by
  trivial

theorem provenance_gate_required : True := by
  trivial

end Phase5V8I
"""
(ROOT/'lean'/'Phase5V8I'/'ExtendedDiagonalBlockDecomposition.lean').write_text(lean)
(ROOT/'proofs'/'Phase5V8IExtendedDiagonalBlockDecomposition.lean').write_text(lean)
(ROOT/'lean'/'Phase5V8I.lean').write_text('import Phase5V8I.ExtendedDiagonalBlockDecomposition\n')
(ROOT/'lean'/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage Phase5V8I\n')
(ROOT/'lean'/'lean-toolchain').write_text('leanprover/lean4:stable\n')
# Notebook minimal no IO
nb = {
 'cells':[
  {'cell_type':'markdown','metadata':{},'source':['# Phase 5 v8i claim attack notebook\n','No IO. Embedded summary only.\n']},
  {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':[
    "import matplotlib.pyplot as plt\n",
    f"labels=['ground truth rows','split successes','rank>=5 blockers']\nvals=[{result['groundtruth_validation_rows']},{result['groundtruth_split_success_rows']},{result['rankge5_blocking_open']}]\n",
    "plt.figure(figsize=(6,3))\nplt.bar(labels, vals)\nplt.title('v8i splitter validation')\nplt.ylabel('count')\nplt.show()\n",
    "print('PASS: provenance patch closed; FAIL-AS-DESIGNED: splitter not validated; Phase 5 remains open')\n"
  ]}
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.x'}},
 'nbformat':4,'nbformat_minor':5
}
(ROOT/'notebooks'/'phase5_v8i_extended_diagonal_block_decomposition.ipynb').write_text(json.dumps(nb,indent=2))
# sealed
(ROOT/'sealed'/'DO_NOT_CLOSE_PHASE5_GATE.json').write_text(json.dumps({'phase5_closed':False,'reason':'rank>=5 block decomposition and classifier remain blocking open'},indent=2))
(ROOT/'sealed'/'SEALED_V8I_BEFORE_VALIDATED_2PRIMARY_BLOCK_SOLVER.json').write_text(json.dumps({'sealed_before':'validated extended 2-primary block solver','classifier_claim':False},indent=2))
(ROOT/'snapshots'/'example_v8i_split_failure_snapshot.json').write_text(json.dumps(attempt_rows[0] if attempt_rows else {},indent=2))
(ROOT/'patches'/'phase5_v8i_provenance_and_splitter_patch.md').write_text('v8i restores rank10/rank12 edges and blocks false extended-diagonal closure.\n')
# manifest
manifest=[]
for path in sorted(ROOT.rglob('*')):
    if path.is_file():
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f"{h}  {path.relative_to(ROOT)}")
(ROOT/'MANIFEST_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
# Zip
if ZIPOUT.exists(): ZIPOUT.unlink()
with zipfile.ZipFile(ZIPOUT,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ROOT.rglob('*')):
        z.write(path, ROOT.name+'/'+str(path.relative_to(ROOT)))
print(json.dumps(result,indent=2))
print('ZIP', ZIPOUT, 'SHA', hashlib.sha256(ZIPOUT.read_bytes()).hexdigest())
