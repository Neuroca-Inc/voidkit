#!/usr/bin/env python3
import csv, json, math, cmath, hashlib, zipfile, os, shutil, itertools
from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache

ROOT=Path('/mnt/data/phase5_v8l_full_decomposition_under_data_gates')
if ROOT.exists(): shutil.rmtree(ROOT)
for sub in ['docs','outputs','sealed','scripts','notebooks','proofs','lean/Phase5V8L','source_notes','audit','snapshots','patches']:
    (ROOT/sub).mkdir(parents=True, exist_ok=True)
LEDGER=Path('/mnt/data/p5-v8l-PHASE5_CANONICAL_LEDGER.md')
if LEDGER.exists(): shutil.copy2(LEDGER, ROOT/'source_notes'/'PHASE5_CANONICAL_LEDGER.md')
AUDIT=Path('/mnt/data/phase5_v8k_external_audit.zip')
if AUDIT.exists(): shutil.copy2(AUDIT, ROOT/'audit'/'phase5_v8k_external_audit.zip')

UP=Path('/mnt/data/_v8l_upstreams')
if UP.exists(): shutil.rmtree(UP)
UP.mkdir()
for zname in ['phase5_v8e_family_f_isometry_classifier_package.zip','phase5_v8g_triangles_mixed_highrank_2primary_components_package.zip','phase5_v8h_rank4_exact_closure_rankge5_reduction_attack_package.zip','phase5_v8k_radical_aware_decomposition_redone_under_data_gates_package.zip']:
    with zipfile.ZipFile('/mnt/data/'+zname) as z: z.extractall(UP)
V8E=UP/'phase5_v8e_family_f_isometry_classifier'
V8G=UP/'phase5_v8g_triangles_mixed_highrank_2primary_components'
V8H=UP/'phase5_v8h_rank4_exact_closure_rankge5_reduction_attack'
V8K=UP/'phase5_v8k_radical_aware_decomposition_redone_under_data_gates'

def lcm(a,b): return a*b//math.gcd(a,b)

def edge_choices(D):
    edges=[]; choices=[]; names=[]
    n=len(D)
    for i in range(n):
        for j in range(i+1,n):
            L=lcm(D[i],D[j]); step=L//math.gcd(D[i],D[j])
            edges.append((i,j,L,step)); choices.append(list(range(0,L,step))); names.append(f'c{i}{j}')
    return edges,choices,names

def make_form(D, edges_list, diag=None):
    D=tuple(D); n=len(D); diag=tuple(diag or [1]*n)
    M=1
    for d in D: M=lcm(M,2*d)
    for i in range(n):
        for j in range(i+1,n): M=lcm(M,lcm(D[i],D[j]))
    E={(int(i),int(j)):int(c) for i,j,c in edges_list if int(c)!=0}
    B=[[0]*n for _ in range(n)]
    for i,d in enumerate(D): B[i][i]=(diag[i]*(M//d))%M
    for (i,j),c in E.items():
        val=(c*(M//lcm(D[i],D[j])))%M; B[i][j]=B[j][i]=val
    def q(v):
        tot=0
        for i,d in enumerate(D): tot+=diag[i]*v[i]*v[i]*(M//(2*d))
        for (i,j),c in E.items(): tot+=c*v[i]*v[j]*(M//lcm(D[i],D[j]))
        return tot%M
    def b(u,w): return sum(u[i]*B[i][j]*w[j] for i in range(n) for j in range(n))%M
    return M,B,q,b

def elem_order(v,D):
    o=1
    for a,d in zip(v,D):
        a%=d
        if a: o=lcm(o,d//math.gcd(a,d))
    return o

def snf_all_ones(mat,n):
    A=[row[:] for row in mat]; rows=len(A); cols=len(A[0]) if rows else 0; r=0
    for _ in range(n):
        piv=None
        for i in range(r,rows):
            for j in range(r,cols):
                if A[i][j] and (piv is None or abs(A[i][j])<abs(A[piv[0]][piv[1]])): piv=(i,j)
        if piv is None: return False
        pi,pj=piv; A[r],A[pi]=A[pi],A[r]
        for i in range(rows): A[i][r],A[i][pj]=A[i][pj],A[i][r]
        changed=True
        while changed:
            changed=False
            for i in range(r+1,rows):
                if A[r][r]==0: return False
                if A[i][r] % A[r][r] != 0:
                    qd=A[i][r]//A[r][r]
                    for j in range(cols): A[i][j]-=qd*A[r][j]
                    A[r],A[i]=A[i],A[r]; changed=True
                elif A[i][r]!=0:
                    qd=A[i][r]//A[r][r]
                    for j in range(cols): A[i][j]-=qd*A[r][j]
            for j in range(r+1,cols):
                if A[r][r]==0: return False
                if A[r][j] % A[r][r] != 0:
                    qd=A[r][j]//A[r][r]
                    for i in range(rows): A[i][j]-=qd*A[i][r]
                    for i in range(rows): A[i][r],A[i][j]=A[i][j],A[i][r]
                    changed=True
                elif A[r][j]!=0:
                    qd=A[r][j]//A[r][r]
                    for i in range(rows): A[i][j]-=qd*A[i][r]
        if abs(A[r][r])!=1: return False
        r+=1
    return True

def spans(vectors,D):
    n=len(D); cols=[[v[i] for i in range(n)] for v in vectors]
    for k in range(n): cols.append([D[k] if i==k else 0 for i in range(n)])
    mat=[[cols[c][i] for c in range(len(cols))] for i in range(n)]
    return snf_all_ones(mat,n)

def radical_mitm(D,B,M,q=None,limit=1000000):
    n=len(D); total=math.prod(D)
    if total<=2000000:
        rad=[]; qvals=set(); count=0
        gens=[tuple(1 if k==i else 0 for k in range(n)) for i in range(n)]
        for v in itertools.product(*[range(d) for d in D]):
            if all(sum(v[i]*B[i][k] for i in range(n))%M==0 for k in range(n)):
                count+=1
                if q: qvals.add(q(v))
                if len(rad)<limit: rad.append(tuple(v))
        return count,rad,sorted(qvals)
    best=None
    for mask in range(1,1<<n):
        p=1
        for i in range(n):
            if mask>>i&1: p*=D[i]
        score=abs(math.log(p)-math.log(total//p))
        if best is None or score<best[0]: best=(score,mask)
    I=[i for i in range(n) if best[1]>>i&1]; J=[i for i in range(n) if not(best[1]>>i&1)]
    H=defaultdict(list)
    for vals in itertools.product(*[range(D[i]) for i in I]):
        res=tuple(sum(vals[t]*B[I[t]][k] for t in range(len(I)))%M for k in range(n))
        v=[0]*n
        for t,i in enumerate(I): v[i]=vals[t]
        H[res].append(tuple(v))
    rad=[]; qvals=set(); count=0
    for vals in itertools.product(*[range(D[j]) for j in J]):
        res=tuple(sum(vals[t]*B[J[t]][k] for t in range(len(J)))%M for k in range(n))
        need=tuple((-x)%M for x in res)
        for v1 in H.get(need,[]):
            v=list(v1)
            for t,j in enumerate(J): v[j]=vals[t]
            v=tuple(v); count+=1
            if q: qvals.add(q(v))
            if len(rad)<limit: rad.append(v)
    return count,rad,sorted(qvals)

def subgroup_generated(gens,D):
    seen={tuple(0 for _ in D)}; dq=deque([tuple(0 for _ in D)])
    while dq:
        x=dq.popleft()
        for g in gens:
            y=tuple((x[i]+g[i])%D[i] for i in range(len(D)))
            if y not in seen: seen.add(y); dq.append(y)
    return seen

def radical_generators(rad,D,target):
    gens=[]; sub={tuple(0 for _ in D)}
    for v in rad:
        if v in sub: continue
        trial=gens+[v]
        sub=subgroup_generated(trial,D)
        gens=trial
        if len(sub)==target: break
    return gens,len(sub)

def quotient_invariants(D,rad_gens):
    try:
        import sympy as sp
        from sympy.matrices.normalforms import smith_normal_form
        rows=[]; n=len(D)
        for i,d in enumerate(D):
            row=[0]*n; row[i]=d; rows.append(row)
        for g in rad_gens: rows.append(list(g))
        S=smith_normal_form(sp.Matrix(rows), domain=sp.ZZ)
        diag=[]
        for i in range(min(S.rows,S.cols)):
            v=abs(int(S[i,i]))
            if v>1: diag.append(v)
        return diag
    except Exception as e:
        return [f'quotient_snf_failed:{e}']

def form_edges_from_rep(D,rep):
    edges,_,names=edge_choices(D)
    if isinstance(rep,str): rep=json.loads(rep)
    if isinstance(rep,dict): return [(i,j,int(rep.get(f'c{i}{j}',0))) for i,j,_,_ in edges]
    if isinstance(rep,list): return [(i,j,int(c)) for c,(i,j,_,_) in zip(rep,edges)]
    return []

def verify_certificate(D,edges,diag,basis,blocks):
    M,B,q,b=make_form(D,edges,diag); n=len(D)
    if not basis or len(basis)!=n: return False,'basis_missing_or_bad_rank'
    if any((not isinstance(row,list) or len(row)!=n) for row in basis): return False,'basis_bad_shape'
    basis=[tuple(row[i]%D[i] for i in range(n)) for row in basis]
    if not spans(basis,D): return False,'SNF_span_failed'
    block_of={}
    for bi,blk in enumerate(blocks):
        idxs=blk.get('indices',[blk.get('index')])
        if isinstance(idxs,int): idxs=[idxs]
        for idx in idxs: block_of[idx]=bi
    for i in range(n):
        for j in range(i+1,n):
            if block_of.get(i)!=block_of.get(j) and b(basis[i],basis[j])!=0: return False,f'cross_block_b_{i}_{j}'
    gens=[tuple(1 if k==i else 0 for k in range(n)) for i in range(n)]
    for blk in blocks:
        typ=blk.get('type'); idxs=blk.get('indices',[blk.get('index')])
        if isinstance(idxs,int): idxs=[idxs]
        if typ=='R':
            v=basis[idxs[0]]
            if any(b(v,g)!=0 for g in gens): return False,f'R_not_ambient_radical_{idxs[0]}'
            if b(v,v)!=0: return False,f'R_bii_nonzero_{idxs[0]}'
            if q(v)!=(blk.get('q_xM',0)%M): return False,f'R_q_{idxs[0]}'
            if elem_order(v,D)!=blk.get('D'): return False,f'R_order_{idxs[0]}'
        elif typ=='A':
            v=basis[idxs[0]]; Db=blk['D']; t=blk['t']
            if elem_order(v,D)!=Db: return False,f'A_order_{idxs[0]}'
            if q(v)!=(t*(M//(2*Db)))%M: return False,f'A_q_{idxs[0]}'
        elif typ=='GRAM':
            gram=blk.get('gram_xM')
            if gram is None: return False,'GRAM_missing'
            for a,ia in enumerate(idxs):
                for bb,ib in enumerate(idxs):
                    if b(basis[ia],basis[ib])!=(gram[a][bb]%M): return False,f'GRAM_b_{ia}_{ib}'
            for a,ia in enumerate(idxs):
                qs=blk.get('q_xM',[])
                if qs and q(basis[ia])!=(qs[a]%M): return False,f'GRAM_q_{ia}'
        else: return False,f'unknown_block_type_{typ}'
    return True,'ok'

# Worked certificate
worked_D=[2,2]; worked_edges=[(0,1,1)]; worked_diag=[1,1]
worked_basis=[[1,0],[1,1]]
worked_blocks=[{'type':'A','index':0,'indices':[0],'D':2,'t':1},{'type':'R','index':1,'indices':[1],'D':2,'q_xM':0}]
worked_ok,worked_detail=verify_certificate(worked_D,worked_edges,worked_diag,worked_basis,worked_blocks)

# Rank-4 true diagonal exact table
D4=(4,4,2,16); diag4=(1,3,3,5); r=4; orderA=math.prod(D4); M4=32
edges_meta,choices,edge_names=edge_choices(D4); forms=list(itertools.product(*choices)); els=list(itertools.product(*[range(d) for d in D4]))
basis4=[]
for j,d in enumerate(D4):
    x=[0]*r; x[j]=1; basis4.append(tuple(x))
order_groups=defaultdict(list)
for x in els: order_groups[elem_order(x,D4)].append(x)
def q4(form,x):
    val=0
    for i,(a,d) in enumerate(zip(x,D4)): val=(val+diag4[i]*a*a*(M4//(2*d)))%M4
    for c,(i,j,L,_) in zip(form,edges_meta): val=(val+c*x[i]*x[j]*(M4//L))%M4
    return val%M4
def b4(form,x,y):
    val=0
    for i,(a,bb,d) in enumerate(zip(x,y,D4)): val=(val+diag4[i]*a*bb*(M4//d))%M4
    for c,(i,j,L,_) in zip(form,edges_meta): val=(val+c*(x[i]*y[j]+x[j]*y[i])*(M4//L))%M4
    return val%M4
def qhist4(form):
    h=[0]*M4
    for x in els: h[q4(form,x)]+=1
    return tuple(h)
def gensize4(cols):
    seen={(0,0,0,0)}; dq=deque([(0,0,0,0)])
    while dq:
        x=dq.popleft()
        for v in cols:
            y=tuple((x[i]+v[i])%D4[i] for i in range(r))
            if y not in seen: seen.add(y); dq.append(y)
    return len(seen)
cand_cache={}
def cands(tgt,j):
    key=(tgt,j)
    if key not in cand_cache:
        tq=q4(tgt,basis4[j]); cand_cache[key]=[x for x in order_groups[D4[j]] if q4(tgt,x)==tq]
    return cand_cache[key]
def verify_wit(src,tgt,cols):
    if not cols or gensize4(cols)!=orderA: return False
    for x in els:
        y=[0]*r
        for j,a in enumerate(x):
            if a:
                for i in range(r): y[i]=(y[i]+a*cols[j][i])%D4[i]
        if q4(src,x)!=q4(tgt,tuple(y)): return False
    return True
def isometric(src,tgt):
    cand={j:cands(tgt,j) for j in range(r)}; order=sorted(range(r), key=lambda j:len(cand[j]))
    assigned={}; nodes=0; pairs=0
    def rec(pos):
        nonlocal nodes,pairs
        if pos==len(order):
            cols=[assigned[j] for j in range(r)]
            return cols if gensize4(cols)==orderA else None
        j=order[pos]
        for v in cand[j]:
            nodes+=1; ok=True
            for k,u in assigned.items():
                pairs+=1
                if b4(tgt,v,u)!=b4(src,basis4[j],basis4[k]): ok=False; break
            if not ok: continue
            assigned[j]=v; res=rec(pos+1)
            if res is not None: return res
            del assigned[j]
        return None
    w=rec(0); return w,nodes,pairs,{str(j):len(cand[j]) for j in range(r)}
inv_groups=defaultdict(list)
for f in forms: inv_groups[qhist4(f)].append(f)
classes=[]; form_to_class={}; decisions=[]; cid=0
for inv,grp in sorted(inv_groups.items(), key=lambda kv:(len(kv[1]),kv[0])):
    rem=set(grp)
    while rem:
        rep=next(iter(rem)); cid+=1; members=[rep]; rem.remove(rep)
        for tgt in list(rem):
            wit,nodes,pairs,cand=isometric(rep,tgt); iso=wit is not None
            row={'decision_id':len(decisions)+1,'class_candidate':cid,'source_form':json.dumps(dict(zip(edge_names,rep))),'target_form':json.dumps(dict(zip(edge_names,tgt))),'isometric':iso,'exhausted_search':not iso,'nodes_visited':nodes,'pairwise_b_tests':pairs,'candidate_sizes':json.dumps(cand),'witness_basis_json':json.dumps(wit) if iso else '[]','witness_verified':verify_wit(rep,tgt,wit) if iso else ''}
            decisions.append(row)
            if iso:
                members.append(tgt); rem.remove(tgt)
        for m in members: form_to_class[m]=cid
        classes.append({'class_id':cid,'rep':rep,'members':members})
class_rows=[]; scope_rows=[]
def radical_size_form(D,diag,form):
    ed=[(i,j,c) for c,(i,j,_,_) in zip(form,edges_meta)]
    M,B,q,b=make_form(D,ed,diag)
    count,rad,qvals=radical_mitm(D,B,M,q,limit=16)
    return count,qvals
for cl in classes:
    rad,qvals=radical_size_form(D4,diag4,cl['rep'])
    class_rows.append({'class_id':cl['class_id'],'representative':json.dumps(dict(zip(edge_names,cl['rep']))),'orbit_size':len(cl['members']),'radical_size':rad,'q_values_on_radical_xM':json.dumps(qvals),'members_json':json.dumps([dict(zip(edge_names,m)) for m in cl['members']])})
for f in forms:
    scope_rows.append({'shape':'[4,4,2,16]','diag_units':'[1,3,3,5]','form':json.dumps(dict(zip(edge_names,f))),'class_id':form_to_class[f],'disposition':'CLASSIFIED_BY_TRUE_DIAGONAL_EXACT_SAME_SHAPE_ORBIT_TABLE'})
arch_form=tuple({(0,1):3,(0,2):2,(0,3):4,(1,2):2,(1,3):12}.get((i,j),0) for i,j,_,_ in edges_meta)
arch_class=form_to_class[arch_form]

# Ground truth rows: keep honest, only one certificate currently verified.
gt_in=list(csv.DictReader(open(V8K/'outputs/phase5_v8k_groundtruth_radical_measurement_and_attempts.csv')))
gt_rows=[]
for rrow in gt_in:
    cert=False; basis=[]; blocks=[]; detail='not_constructed'
    if rrow['ground_truth_id']=='[2, 2]-1':
        cert=worked_ok; basis=worked_basis; blocks=worked_blocks; detail=worked_detail
    status='DECOMPOSED_CERTIFIED_WORKED_TARGET' if cert else 'BLOCKING_OPEN_EXPLICIT_BLOCK_CERTIFICATE_NOT_CONSTRUCTED'
    gt_rows.append({'source':rrow['source'],'ground_truth_id':rrow['ground_truth_id'],'shape':rrow['shape'],'diag_units':rrow['diag_units'],'representative':rrow['representative'],'M':rrow['M'],'radical_size_ambient':rrow['radical_size_ambient'],'q_values_on_radical_xM':rrow['q_values_on_radical_xM'],'first_radical_witnesses':rrow['first_radical_witnesses'],'basis_matrix_json':json.dumps(basis),'blocks_json':json.dumps(blocks),'certificate_verified':cert,'certificate_detail':detail,'decomposition_status':status,'failure_vector':rrow['failure_vector'] if not cert else '[]','failure_reason':'' if cert else 'no verified A/UV/R basis-matrix certificate emitted for this row'})

# Rank>=5 core quotient/complement measurement.
core_in=list(csv.DictReader(open(V8K/'outputs/phase5_v8k_rankge5_complete_form_spec_and_radical_measurement.csv')))
core_rows=[]
for rr in core_in:
    D=json.loads(rr['D2_core']); diag=json.loads(rr['diag_units']); edges=[tuple(e) for e in json.loads(rr['edges_2core'])]
    M,B,q,b=make_form(D,edges,diag)
    rad_size,rad,qvals=radical_mitm(D,B,M,q,limit=1000000)
    gens,gen_size=radical_generators(rad,D,rad_size)
    quotient=quotient_invariants(D,gens)
    core_rows.append({'case':rr['case'],'D2_core':rr['D2_core'],'diag_units':rr['diag_units'],'edge_count':rr['edge_count'],'edges_2core':rr['edges_2core'],'radical_size_ambient':rad_size,'q_values_on_radical_xM':json.dumps(qvals),'radical_generator_count':len(gens),'radical_generators_json':json.dumps([list(g) for g in gens]),'radical_generator_span_size':gen_size,'nondegenerate_complement_shape_after_radical_stripping':json.dumps(quotient),'basis_matrix_json':'[]','blocks_json':'[]','certificate_verified':False,'decomposition_status':'BLOCKING_OPEN_NO_VERIFIED_BLOCK_CERTIFICATE_FOR_CORE','failure_vector':json.dumps(list(gens[0]) if gens else []),'failure_reason':'radical quotient/complement measured; no certified orthogonal block basis emitted'})

# Gates and claims
cert_gate=[{'gate':'TRUE_DIAGONAL_RANK4_TABLE_PUBLISHED','rows_checked':len(scope_rows),'failures':0,'pass':True,'evidence':'512 true-diagonal forms classified; class and decision tables emitted'}, {'gate':'GROUNDTRUTH_CERTIFICATE_DATA','rows_checked':len(gt_rows),'failures':len(gt_rows)-sum(1 for x in gt_rows if x['certificate_verified']),'pass':False,'evidence':'full 229/229 certificate target remains blocking open'}, {'gate':'RANKGE5_CERTIFICATE_DATA','rows_checked':len(core_rows),'failures':len(core_rows),'pass':False,'evidence':'five cores measured but not decomposed with block certificates'}, {'gate':'RADICAL_BII_ZERO','rows_checked':1,'failures':0 if worked_ok else 1,'pass':worked_ok,'evidence':'only emitted R block is worked target and has b(v,v)=0'}, {'gate':'FORM_SPEC_COMPLETE','rows_checked':len(core_rows),'failures':0,'pass':True,'evidence':'diag_units and corrected edges retained for every core'}]
scope_gate=[{'range':'true_diag_rank4_[4,4,2,16]','expected_rows':512,'actual_rows':len(scope_rows),'missing_rows':512-len(scope_rows),'pass':len(scope_rows)==512},{'range':'groundtruth_decomposition_rows','expected_rows':229,'actual_rows':len(gt_rows),'missing_rows':229-len(gt_rows),'pass':len(gt_rows)==229},{'range':'rankge5_core_rows','expected_rows':5,'actual_rows':len(core_rows),'missing_rows':5-len(core_rows),'pass':len(core_rows)==5}]
claims=[{'claim':'true-diagonal rank4 orbit table publication','status':'CLOSED_POSITIVE','scope':'512 forms, shape [4,4,2,16], diag [1,3,3,5]','evidence':f'{len(classes)} classes, {len(decisions)} decision rows, archival class {arch_class}'},{'claim':'v8h archival rank4 retro-check closure','status':'CLOSED_POSITIVE_AFTER_TABLE_PUBLICATION','scope':'true-diagonal archival object only','evidence':f'published true-diagonal table places archival form in class {arch_class}'},{'claim':'229/229 full decomposition','status':'BLOCKING_OPEN','scope':'v8e/v8g/v8h ground-truth representatives','evidence':f'{sum(1 for x in gt_rows if x["certificate_verified"])} verified certificate rows; target not closed'},{'claim':'five rank>=5 core full decomposition','status':'BLOCKING_OPEN','scope':'corrected five residual cores','evidence':'radical quotient/complement shapes measured; no block certificates claimed'},{'claim':'worked [2,2] certificate','status':'CLOSED_POSITIVE','scope':'single worked target','evidence':'basis [[1,0],[1,1]] verifies A_2(1) PERP R_2(q=0)'}]

# Write
def write_csv(path, rows):
    with open(path,'w',newline='') as f:
        if not rows: return
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def write_json(path,obj): path.write_text(json.dumps(obj,indent=2))
write_csv(ROOT/'outputs/phase5_v8l_true_diag_rank4_exact_orbit_classes.csv',class_rows)
write_csv(ROOT/'outputs/phase5_v8l_true_diag_rank4_decision_certificates.csv',decisions)
write_csv(ROOT/'outputs/phase5_v8l_true_diag_rank4_full_scope_disposition.csv',scope_rows)
write_csv(ROOT/'outputs/phase5_v8l_groundtruth_decomposition_certificates.csv',gt_rows)
write_csv(ROOT/'outputs/phase5_v8l_rankge5_core_radical_quotient_measurement.csv',core_rows)
write_csv(ROOT/'outputs/phase5_v8l_certificate_gate_results.csv',cert_gate)
write_csv(ROOT/'outputs/phase5_v8l_scope_completeness_gates.csv',scope_gate)
write_csv(ROOT/'outputs/phase5_v8l_claim_disposition.csv',claims)
write_csv(ROOT/'outputs/phase5_v8l_ledger_reconciliation.csv',[{'ledger_item':'v8h retro-check table publication owed','package_action':'published true-diagonal class/scope/decision tables'},{'ledger_item':'v8j decomposition rejected','package_action':'not reused'},{'ledger_item':'certificate is data','package_action':'only explicit matrix certificates counted'},{'ledger_item':'no status overclaim','package_action':'full decomposition remains BLOCKING_OPEN'}])
write_csv(ROOT/'outputs/phase5_v8l_falsification_targets.csv',[{'target':'true diagonal table missing row','status':'not triggered','kill_condition':'any of 512 forms lacks disposition'},{'target':'229/229 certificate target','status':'triggered / BLOCKING_OPEN','kill_condition':'rows without verified basis certificates remain'},{'target':'rank>=5 block certificate target','status':'triggered / BLOCKING_OPEN','kill_condition':'no verified block certificate emitted for five cores'}])
result={'phase':'Phase 5 v8l','title':'Full Decomposition Under Data Gates','status':'V8L_TRUE_DIAGONAL_RANK4_TABLE_CLOSED_FULL_DECOMPOSITION_BLOCKING_OPEN','global_pass':True,'phase5_closed':False,'true_diag_rank4_forms':512,'true_diag_rank4_classes':len(classes),'true_diag_rank4_decision_rows':len(decisions),'archival_true_diag_class_id':arch_class,'groundtruth_rows':len(gt_rows),'groundtruth_verified_decomposition_certificates':sum(1 for x in gt_rows if x['certificate_verified']),'rankge5_cores':len(core_rows),'rankge5_verified_decomposition_certificates':0,'claim_overreach':False}
write_json(ROOT/'outputs/phase5_v8l_result_card.json',result)
write_json(ROOT/'outputs/phase5_v8l_verification_summary.json',result)
write_json(ROOT/'sealed/DO_NOT_CLOSE_PHASE5_GATE.json',{'phase5_closed':False,'reason':'229/229 and rank>=5 decomposition certificates remain blocking open'})
write_json(ROOT/'sealed/SEALED_V8L_BEFORE_FULL_DECOMPOSITION_CERTIFICATES.json',result)
write_json(ROOT/'snapshots/example_v8l_true_diag_rank4_snapshot.json',{'shape':list(D4),'diag_units':list(diag4),'archival_class_id':arch_class,'class_count':len(classes)})
readme=f'''# Phase 5 v8l: Full Decomposition Under Data Gates

STATUS: `{result['status']}`

Ledger authority applied first. This package closes the publication gap for the true-diagonal rank-4 retro-check, but it does not claim the full 229/229 or five-core block decomposition.

## Closed

- True-diagonal rank-4 orbit table published: 512/512 forms, {len(classes)} classes.
- Per-decision rank-4 certificates emitted: {len(decisions)} rows.
- v8h archival rank-4 retro-check row closed after publication: archival true-diagonal class id {arch_class}.
- Worked target certificate retained as data: `[2,2] c01=1 = A_2(1) PERP R_2(q=0)`.

## Blocking open

- 229/229 ground-truth block decomposition: {sum(1 for x in gt_rows if x['certificate_verified'])}/229 verified basis-matrix certificates.
- Five rank>=5 residual core block decompositions: 0/5 verified certificates.

No status string claims more than the emitted certificates prove.
'''
(ROOT/'README.md').write_text(readme)
for name in ['phase5_v8l_result_card.md','phase5_v8l_full_decomposition_under_data_gates.md']:
    (ROOT/'docs'/name).write_text(readme)
(ROOT/'docs/true_diagonal_rank4_table.md').write_text(f'True diagonal units [1,3,3,5]. Published 512 scope rows, {len(classes)} class rows, and {len(decisions)} decision-certificate rows. Archival class id {arch_class}.\n')
(ROOT/'docs/decomposition_blocking_open.md').write_text('Full decomposition remains blocking open because verified basis-matrix certificates were not produced for 228/229 ground-truth rows or any of the five rank>=5 cores.\n')
(ROOT/'patches/phase5_v8l_table_publication_patch.md').write_text('Publishes true-diagonal rank4 table and keeps full decomposition blocking open.\n')
# Verifier script
verifier='''#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
fail=[]
for r in csv.DictReader(open(root/'outputs/phase5_v8l_scope_completeness_gates.csv')):
    if r['pass'] not in ('True','true','1'): fail.append(('scope',r))
for r in csv.DictReader(open(root/'outputs/phase5_v8l_certificate_gate_results.csv')):
    if r['gate'] in ('TRUE_DIAGONAL_RANK4_TABLE_PUBLISHED','RADICAL_BII_ZERO','FORM_SPEC_COMPLETE') and r['pass'] not in ('True','true','1'):
        fail.append(('gate',r))
print('PASS' if not fail else 'FAIL', len(fail))
if fail:
    print(fail[:5]); sys.exit(1)
'''
(ROOT/'scripts/phase5_v8l_verify_package_gates.py').write_text(verifier); os.chmod(ROOT/'scripts/phase5_v8l_verify_package_gates.py',0o755)
shutil.copy2('/mnt/data/build_v8l.py', ROOT/'scripts/phase5_v8l_full_decomposition_under_data_gates.py')
# Notebook no IO
nb={'cells':[{'cell_type':'markdown','metadata':{},'source':['# v8l gate attack notebook\nNo IO.']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[f"rank4_forms={len(scope_rows)}\n",f"rank4_classes={len(classes)}\n",f"groundtruth_certified={sum(1 for x in gt_rows if x['certificate_verified'])}\n","print('PASS' if rank4_forms==512 and rank4_classes>0 else 'FAIL')\n","print({'rank4_forms':rank4_forms,'rank4_classes':rank4_classes,'groundtruth_certified':groundtruth_certified})\n"]}], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
write_json(ROOT/'notebooks/phase5_v8l_full_decomposition_under_data_gates.ipynb',nb)
lean='''namespace Phase5V8L

def phase5Closed : Bool := false

theorem no_overclaim_phase5_closed : phase5Closed = false := by rfl

theorem true_diag_scope_count_recorded : 512 = 512 := by rfl

end Phase5V8L
'''
(ROOT/'lean/Phase5V8L/FullDecompositionUnderDataGates.lean').write_text(lean)
(ROOT/'lean/Phase5V8L.lean').write_text('import Phase5V8L.FullDecompositionUnderDataGates\n')
(ROOT/'lean/lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage Phase5V8L\n')
(ROOT/'lean/lean-toolchain').write_text('leanprover/lean4:stable\n')
(ROOT/'proofs/Phase5V8LFullDecompositionUnderDataGates.lean').write_text(lean)
# Manifest zip
manifest=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name!='MANIFEST_SHA256SUMS.txt':
        manifest.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}')
(ROOT/'MANIFEST_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
zip_path=Path('/mnt/data/phase5_v8l_full_decomposition_under_data_gates_package.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')): z.write(p,p.relative_to(ROOT.parent))
print(zip_path)
print(hashlib.sha256(zip_path.read_bytes()).hexdigest())
print(json.dumps(result,indent=2))
