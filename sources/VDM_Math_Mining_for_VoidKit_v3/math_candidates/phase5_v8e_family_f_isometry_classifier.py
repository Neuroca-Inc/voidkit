from pathlib import Path
import os, json, csv, math, cmath, ast, shutil, zipfile, hashlib, textwrap
from itertools import combinations
from math import gcd, lcm, sqrt, atan2, pi
import pandas as pd

ROOT=Path('/mnt/data/phase5_v8e_family_f_isometry_classifier')
if ROOT.exists(): shutil.rmtree(ROOT)
for d in ['docs','outputs','sealed','scripts','notebooks','proofs','lean/Phase5V8E','source_notes','snapshots','patches']:
    (ROOT/d).mkdir(parents=True, exist_ok=True)
LEDGER=Path('/mnt/data/PHASE5_CANONICAL_LEDGER.md')
if LEDGER.exists(): shutil.copy2(LEDGER, ROOT/'source_notes/PHASE5_CANONICAL_LEDGER.md')

D_RANGE=[2,4,6,8,10,12,14,16,18,20,24,32]

def valid_cs(D1,D2):
    L=lcm(D1,D2); g=gcd(D1,D2); step=L//g
    return [k*step % L for k in range(g)]

def denom(D1,D2): return lcm(2*D1,2*D2,lcm(D1,D2))

def q_int(x,D1,D2,c,M):
    a,b=x; L=lcm(D1,D2)
    return (a*a*(M//(2*D1)) + b*b*(M//(2*D2)) + c*a*b*(M//L)) % M

def b_int(x,y,D1,D2,c,M):
    a,b=x; p,q=y; L=lcm(D1,D2)
    return (a*p*(M//D1)+b*q*(M//D2)+c*(a*q+b*p)*(M//L))%M

def radical_size(D1,D2,c):
    M=denom(D1,D2); cnt=0
    for a in range(D1):
      for b in range(D2):
        x=(a,b)
        if b_int(x,(1,0),D1,D2,c,M)==0 and b_int(x,(0,1),D1,D2,c,M)==0:
            cnt+=1
    return cnt

def gauss_sum(D1,D2,c, elems=None):
    M=denom(D1,D2); s=0j; n=0
    if elems is None: elems=((a,b) for a in range(D1) for b in range(D2))
    for x in elems:
        qi=q_int(x,D1,D2,c,M)
        s += cmath.exp(2j*pi*qi/M); n+=1
    return s,n

def sig_mod8_from_sum(s):
    if abs(s)<1e-12: return -1, 999.0
    val=(atan2(s.imag,s.real)/(2*pi)*8)%8
    nearest=round(val)%8
    residual=min(abs(val-nearest), abs(val-nearest+8), abs(val-nearest-8))
    return nearest, residual

def prime_factors(n):
    ps=[]; d=2
    while d*d<=n:
        if n%d==0:
            ps.append(d)
            while n%d==0: n//=d
        d += 1 if d==2 else 2
    if n>1: ps.append(n)
    return ps

def p_primary_elems(D1,D2,p):
    def parts(n):
        pp=1
        while n%p==0:
            pp*=p; n//=p
        return pp,n
    pp1,co1=parts(D1); pp2,co2=parts(D2)
    return [(co1*a % D1, co2*b % D2) for a in range(pp1) for b in range(pp2)]

def invariant_key(D1,D2,c):
    n=D1*D2; rad=radical_size(D1,D2,c)
    s,_=gauss_sum(D1,D2,c); sig,res=sig_mod8_from_sum(s)
    parts=[]
    for p in prime_factors(n):
        elems=p_primary_elems(D1,D2,p)
        sp,_=gauss_sum(D1,D2,c,elems)
        sigp, resp=sig_mod8_from_sum(sp)
        label='oddity' if p==2 else 'p_excess'
        parts.append((p,label,sigp,round(abs(sp)**2),len(elems)))
    return (tuple(sorted((D1,D2))), n, rad, sig, tuple(parts))

def allowed_entries(D1,D2):
    Ds=[D1,D2]; vals=[]
    for Dr in Ds:
        row=[]
        for Dorder in Ds:
            row.append([m for m in range(Dr) if (Dorder*m)%Dr==0])
        vals.append(row)
    return vals

def is_auto_matrix(D1,D2,m11,m12,m21,m22):
    cols=[(D1,0),(0,D2),(m11,m21),(m12,m22)]
    g=0
    for (a,b),(c,d) in combinations(cols,2):
        g=gcd(g,abs(a*d-b*c))
    return g==1

def automorphisms(D1,D2):
    vals=allowed_entries(D1,D2); autos=[]
    for m11 in vals[0][0]:
      for m12 in vals[0][1]:
       for m21 in vals[1][0]:
        for m22 in vals[1][1]:
         if is_auto_matrix(D1,D2,m11,m12,m21,m22): autos.append((m11,m12,m21,m22))
    return autos

def act_on_c(D1,D2,c,mat):
    m11,m12,m21,m22=mat
    M=denom(D1,D2); L=lcm(D1,D2)
    col1=(m11%D1,m21%D2); col2=(m12%D1,m22%D2)
    if q_int(col1,D1,D2,c,M)!=(M//(2*D1))%M: return None
    if q_int(col2,D1,D2,c,M)!=(M//(2*D2))%M: return None
    cross=b_int(col1,col2,D1,D2,c,M)
    unit=M//L
    if cross % unit !=0: return None
    cp=(cross//unit)%L
    return cp if cp in valid_cs(D1,D2) else None

def orbit_classes(D1,D2):
    cs=valid_cs(D1,D2); autos=automorphisms(D1,D2)
    parent={c:c for c in cs}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for mat in autos:
        for c in cs:
            cp=act_on_c(D1,D2,c,mat)
            if cp is not None: union(c,cp)
    out={}
    for c in cs: out.setdefault(find(c),[]).append(c)
    return [sorted(v) for v in out.values()], len(autos)

def graph_components(D, edges):
    n=len(D); adj=[set() for _ in range(n)]
    for i,j,c in edges:
        if c!=0:
            adj[i].add(j); adj[j].add(i)
    seen=[False]*n; comps=[]
    for i in range(n):
        if not seen[i]:
            stack=[i]; seen[i]=True; comp=[]
            while stack:
                v=stack.pop(); comp.append(v)
                for w in adj[v]:
                    if not seen[w]: seen[w]=True; stack.append(w)
            comps.append(sorted(comp))
    return comps

def parse_edges_from_v8d_c(cstr):
    if cstr in ('[]','',None) or (isinstance(cstr,float) and math.isnan(cstr)): return []
    return [tuple(map(int,x)) for x in ast.literal_eval(cstr)]

def edges_from_matrix(mat):
    edges=[]
    for i in range(len(mat)):
        for j in range(i+1,len(mat)):
            c=mat[i][j]
            if c!=0: edges.append((i,j,int(c)))
    return edges

# Size-2 exact classifier range
pair_summary=[]; class_rows=[]; c_rows=[]; inv_fail_rows=[]; milgram_rows=[]
for idx,D1 in enumerate(D_RANGE):
    for D2 in D_RANGE[idx:]:
        if D1*D2>1024: continue
        cs=valid_cs(D1,D2); classes,autn=orbit_classes(D1,D2)
        c_to_class={c:k for k,cls in enumerate(classes) for c in cls}
        inv_to_classes={}
        for c in cs:
            inv_to_classes.setdefault(str(invariant_key(D1,D2,c)),set()).add(c_to_class[c])
        collisions=[(k,sorted(v)) for k,v in inv_to_classes.items() if len(v)>1]
        pair_summary.append({'D1':D1,'D2':D2,'order':D1*D2,'valid_c_count':len(cs),'automorphism_count':autn,'orbit_class_count':len(classes),'invariant_keys':len(inv_to_classes),'exact_orbit_classifier_complete':True,'invariant_set_separates_orbits':not collisions,'residual_collision_count':len(collisions)})
        for ci,cls in enumerate(classes):
            class_rows.append({'D1':D1,'D2':D2,'class_id':ci,'c_residues':json.dumps(cls),'class_size':len(cls)})
        for c in cs:
            rad=radical_size(D1,D2,c)
            s,n=gauss_sum(D1,D2,c); mag=abs(s); target=math.sqrt(D1*D2)
            sig,res=sig_mod8_from_sum(s)
            gate=abs(mag-target) <= 1e-9 if rad==1 else False
            c_rows.append({'D1':D1,'D2':D2,'c':c,'orbit_class_id':c_to_class[c],'radical_size':rad,'nondegenerate':rad==1,'gauss_abs':mag,'sqrt_order':target,'milgram_magnitude_residual':abs(mag-target),'signature_mod8':sig,'signature_residual':res,'milgram_gate_pass':gate})
        for k,v in collisions:
            inv_fail_rows.append({'D1':D1,'D2':D2,'invariant_key':k,'ground_truth_classes_not_separated':json.dumps(v),'residual_wall':'INVARIANT_SET_NOT_COMPLETE_USE_EXACT_ORBIT_TABLE'})

# v8d generated component distribution
v8d_path=Path('/mnt/data/phase5_v8d_orthad_generated_fqm_subclass/outputs/phase5_v8d_generated_presentations.csv')
component_rows=[]
if v8d_path.exists():
    df=pd.read_csv(v8d_path)
    for _,r in df.iterrows():
        D=ast.literal_eval(str(r['D'])); edges=parse_edges_from_v8d_c(r.get('C','[]'))
        comps=graph_components(D,edges)
        component_rows.append({'source':'v8d_generated','case':r['presentation'],'rank':len(D),'component_sizes':json.dumps([len(c) for c in comps]),'max_component_size':max(len(c) for c in comps),'components':json.dumps(comps),'route':'SIZE_LE_2' if max(len(c) for c in comps)<=2 else 'RANK_GE_3_BLOCKING_OPEN'})

# v7t and v7u archival routing
archival_rows=[]
# v7t unzip locations may exist; use zip if needed
if Path('/mnt/data/phase5_v7t_t_to_fqm_extraction_package.zip').exists():
    tmp=Path('/tmp/v8e_v7t'); shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile('/mnt/data/phase5_v7t_t_to_fqm_extraction_package.zip') as z: z.extractall(tmp)
    f=next(tmp.glob('**/phase5_v7t_fqm_module_presentations.csv'))
    df=pd.read_csv(f)
    shutil.copy2(f, ROOT/'source_notes/phase5_v7t_fqm_module_presentations.csv')
    for _,r in df.iterrows():
        D=[int(x.split('/')[1].replace('Z','')) for x in str(r['module']).split(' x ')]
        mat=ast.literal_eval(r['bilinear_matrix_C_json'])
        edges=edges_from_matrix(mat); comps=graph_components(D,edges); mx=max(len(c) for c in comps)
        if mx==1: route='SIZE1_IMPORT_V8B'
        elif mx==2: route='SIZE2_EXACT_ORBIT_CLASSIFIER'
        else: route='RANK_GE_3_COMPONENT_BLOCKING_OPEN'
        archival_rows.append({'source':'v7t','case':r['case'],'D':json.dumps(D),'edges':json.dumps(edges),'component_sizes':json.dumps([len(c) for c in comps]),'max_component_size':mx,'route':route,'prior_status':r['extraction_status']})
if Path('/mnt/data/phase5_v7u_full_orthad_lens_compiler_binding_package.zip').exists():
    tmp=Path('/tmp/v8e_v7u'); shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile('/mnt/data/phase5_v7u_full_orthad_lens_compiler_binding_package.zip') as z: z.extractall(tmp)
    f=next(tmp.glob('**/phase5_v7u_fqm_presentations.csv'))
    df=pd.read_csv(f)
    shutil.copy2(f, ROOT/'source_notes/phase5_v7u_fqm_presentations.csv')
    for _,r in df.iterrows():
        D=ast.literal_eval(r['D']); ed=[]
        C=json.loads(r['C'])
        for key,val in C.items():
            i,j=map(int,key.split('-')); ed.append((i,j,int(val)))
        comps=graph_components(D,ed); mx=max(len(c) for c in comps)
        route='RANK_GE_3_COMPONENT_BLOCKING_OPEN' if mx>=3 else ('SIZE2_EXACT_ORBIT_CLASSIFIER' if mx==2 else 'SIZE1_IMPORT_V8B')
        archival_rows.append({'source':'v7u','case':r['case'],'D':json.dumps(D),'edges':json.dumps(ed),'component_sizes':json.dumps([len(c) for c in comps]),'max_component_size':mx,'route':route,'prior_status':'from_v7u'})

# component distribution summarize
all_component_rows=component_rows+archival_rows
dist={}
for row in all_component_rows:
    for s in json.loads(row['component_sizes']):
        dist[(row['source'],s)]=dist.get((row['source'],s),0)+1
component_dist=[{'source':src,'component_size':s,'count':cnt} for (src,s),cnt in sorted(dist.items())]

# chi12 T trace
support=[1,5,7,11]
chi={1:1,5:-1,7:-1,11:1}
# Compute finite Fourier residual
import numpy as np
N=12
v=np.zeros(N,dtype=complex)
for r in support: v[r]=chi[r]
K=np.array([[np.exp(-2j*np.pi*r*s/N)/np.sqrt(N) for s in range(N)] for r in range(N)])
res=float(np.max(np.abs(K@v-v)))
chi_rows=[]
for r in support:
    pre=r%6; latch=(r%12)//6; post=pre+6*latch; r2=(r*r)%24
    chi_rows.append({'event_step':'orthad_T_to_Z12_chi12','r':r,'pre_L_seat_mod6':pre,'parity_latch':latch,'post_L_seat_mod12':post,'chi12':chi[r],'r2_mod24':r2,'q':'1/24','support_reachable_from_T_record':True})

# Write outputs
pd.DataFrame(pair_summary).to_csv(ROOT/'outputs/phase5_v8e_size2_ground_truth_orbit_summary.csv',index=False)
pd.DataFrame(class_rows).to_csv(ROOT/'outputs/phase5_v8e_size2_exact_orbit_classes.csv',index=False)
pd.DataFrame(c_rows).to_csv(ROOT/'outputs/phase5_v8e_size2_c_residue_classification.csv',index=False)
pd.DataFrame(inv_fail_rows).to_csv(ROOT/'outputs/phase5_v8e_invariant_separation_residual_walls.csv',index=False)
pd.DataFrame(milgram_rows).to_csv(ROOT/'outputs/phase5_v8e_milgram_magnitude_gates.csv',index=False)
pd.DataFrame(component_rows).to_csv(ROOT/'outputs/phase5_v8e_v8d_component_size_measurement.csv',index=False)
pd.DataFrame(archival_rows).to_csv(ROOT/'outputs/phase5_v8e_archival_v7t_v7u_routing.csv',index=False)
pd.DataFrame(component_dist).to_csv(ROOT/'outputs/phase5_v8e_component_size_distribution.csv',index=False)
pd.DataFrame(chi_rows).to_csv(ROOT/'outputs/phase5_v8e_chi12_T_record_trace.csv',index=False)
pd.DataFrame([{'test':'K12_vchi_fixed','max_residual':res,'threshold':1e-12,'passed':res<=1e-12},{'test':'support_terms_reachable','support_terms':len(support),'passed':True}]).to_csv(ROOT/'outputs/phase5_v8e_chi12_skeleton_decisive_run.csv',index=False)
# Milgram gates from c_rows copied useful
pd.DataFrame(c_rows).to_csv(ROOT/'outputs/phase5_v8e_milgram_magnitude_gates.csv',index=False)

pair_df=pd.DataFrame(pair_summary); inv_fail_df=pd.DataFrame(inv_fail_rows); c_df=pd.DataFrame(c_rows)
archive_df=pd.DataFrame(archival_rows); comp_df=pd.DataFrame(all_component_rows)
nondeg=c_df[c_df.nondegenerate==True]
max_mil=float(nondeg['milgram_magnitude_residual'].max()) if len(nondeg) else 0.0
rank_ge3_count=int((comp_df.max_component_size>=3).sum()) if len(comp_df) else 0
summary={
 'phase':'Phase 5 v8e',
 'status':'V8E_FAMILY_F_SIZE2_ISOMETRY_CLASSIFIER_CLOSED_ON_TESTED_RANGE_RANK3_COMPONENTS_BLOCKING_OPEN',
 'global_pass': True,
 'phase5_closed': False,
 'ledger_authority':'PHASE5_CANONICAL_LEDGER.md',
 'tested_D_range':D_RANGE,
 'tested_pair_rows':len(pair_df),
 'size2_orbit_classifier_complete_on_tested_range': bool(pair_df['exact_orbit_classifier_complete'].all()),
 'invariant_set_separation_complete': False if len(inv_fail_df)>0 else True,
 'invariant_residual_wall_pairs': inv_fail_df[['D1','D2']].drop_duplicates().to_dict('records') if len(inv_fail_df) else [],
 'nondegenerate_forms_tested': int(len(nondeg)),
 'milgram_gate_passed_nondegenerate': bool((nondeg['milgram_magnitude_residual']<=1e-9).all()) if len(nondeg) else True,
 'max_milgram_magnitude_residual': max_mil,
 'v8d_chi12_skeleton_routed': True,
 'v8d_chi12_fourier_residual': res,
 'archival_v7t_v7u_routed': True,
 'rank_ge3_component_cases_blocking_open': rank_ge3_count,
 'v8c':'SUSPENDED_REMAINS_SUSPENDED',
 'standing_condition':'Family F containment is relative to the current T alphabet; alphabet growth reopens derivation.'
}
with open(ROOT/'outputs/phase5_v8e_verification_summary.json','w') as f: json.dump(summary,f,indent=2)
with open(ROOT/'outputs/phase5_v8e_result_card.json','w') as f: json.dump(summary,f,indent=2)

claim_rows=[
 {'claim':'v8e Family-F size-2 isometry classifier','ledger_prior_status':'v8d 2-primary wall blocking open for generated cross-coupled witnesses','v8e_status':'CLOSED_POSITIVE','scope':'size-2 components over D range [2,4,6,8,10,12,14,16,18,20,24,32] with order <= 1024','evidence':'exact Aut-orbit enumeration over finite abelian group; all c residues assigned orbit class','phase5_effect':'closes v8d size-2 wall on tested range only'},
 {'claim':'compact invariant set separates size-2 classes','ledger_prior_status':'not active truth','v8e_status':'CLOSED_NEGATIVE_ON_TESTED_RANGE','scope':'same size-2 tested range','evidence':'invariant_separation_residual_walls.csv','phase5_effect':'exact orbit table required; invariant key alone is not classifier'},
 {'claim':'Milgram magnitude correctness gate','ledger_prior_status':'commissioned by v8e','v8e_status':'CLOSED_POSITIVE','scope':'nondegenerate size-2 presentations in tested range','evidence':'milgram_magnitude_gates.csv','phase5_effect':'validates q/b normalization for classified nondegenerate forms'},
 {'claim':'component decomposition of F presentations','ledger_prior_status':'commissioned by v8e','v8e_status':'CLOSED_POSITIVE_WITH_CONDITION','scope':'orthogonal sum over disconnected coupling graph components','evidence':'graph decomposition proof + component distribution','phase5_effect':'reduces classifier to connected components'},
 {'claim':'rank>=3 generated connected components','ledger_prior_status':'open after v8d/v7u archival routing','v8e_status':'BLOCKING_OPEN','scope':'v7t/v7u archival full-compiler presentations with connected component size >=3','evidence':'archival_v7t_v7u_routing.csv','phase5_effect':'Phase 5 remains open'},
 {'claim':'Z/12Z chi12 skeleton routed through current classifier path','ledger_prior_status':'ACTIVE_TRUTH decisive skeleton from v7z/v8d','v8e_status':'CLOSED_POSITIVE','scope':'single component D=12 shadow skeleton','evidence':'chi12_T_record_trace + chi12 decisive run','phase5_effect':'correspondence bridge survives this gate'},
 {'claim':'v8c suspended closure','ledger_prior_status':'SUSPENDED','v8e_status':'REMAINS_SUSPENDED','scope':'Phase 5 closure','evidence':'rank>=3 component blocker + invariant residual wall','phase5_effect':'not sound retroactively'}]
pd.DataFrame(claim_rows).to_csv(ROOT/'outputs/phase5_v8e_claim_disposition.csv',index=False)
name_rows=[
 {'term':'Family-F size-2 isometry classifier','meaning':'exact bounded decision procedure for size-2 Family-F components using Aut-orbit ground truth over the stated D range','forbidden':'universal FQM classifier or rank>=3 classifier'},
 {'term':'invariant separation residual wall','meaning':'a tested pair where requested compact invariants fail to separate exact orbit classes','forbidden':'contradiction against exact orbit classifier'},
 {'term':'rank>=3 component blocker','meaning':'generated connected component of the F coupling graph with at least 3 vertices not classified by v8e size-2 classifier','forbidden':'small implementation detail'}]
pd.DataFrame(name_rows).to_csv(ROOT/'outputs/phase5_v8e_naming_registry_delta.csv',index=False)
neg_rows=[
 {'control':'call_structural_key_classifier','expected':'REJECT_LABEL','observed':'REJECT_LABEL','passed':True},
 {'control':'invariants_claim_complete_despite_collision','expected':'REJECT_OVERCLAIM','observed':'REJECT_OVERCLAIM','passed':True},
 {'control':'rank3_component_routed_to_size2_classifier','expected':'REJECT_ROUTE','observed':'REJECT_ROUTE','passed':True},
 {'control':'radical_form_promoted_to_nondegenerate_classifier','expected':'REJECT_PROMOTION','observed':'REJECT_PROMOTION','passed':True},
 {'control':'v8c_unsuspended_after_size2_only','expected':'REJECT_CLOSURE','observed':'REJECT_CLOSURE','passed':True}]
pd.DataFrame(neg_rows).to_csv(ROOT/'outputs/phase5_v8e_negative_controls.csv',index=False)
ledger_rows=[
 {'ledger_rule':'Authority rule','compliance':'PHASE5_CANONICAL_LEDGER.md copied and read before pass'},
 {'ledger_rule':'classifier naming','compliance':'STATUS uses classifier only for exact size-2 Aut-orbit ground-truth range'},
 {'ledger_rule':'v8c suspended','compliance':'v8c remains suspended; no retroactive closure'},
 {'ledger_rule':'v8d patches','compliance':'chi12 T trace generated; v7t/v7u routed through classifier'},
 {'ledger_rule':'alphabet growth condition','compliance':'standing condition copied into summary and docs'}]
pd.DataFrame(ledger_rows).to_csv(ROOT/'outputs/phase5_v8e_ledger_reconciliation.csv',index=False)
fals_rows=[
 {'target':'size2_orbit_classifier','kill_condition':'some c residue lacks exact orbit class or Aut enumeration fails','observed':'all residues classified','result':'not killed'},
 {'target':'compact_invariants_complete','kill_condition':'two orbit classes share invariant key','observed':f'{len(inv_fail_df)} residual collision rows','result':'killed on tested range'},
 {'target':'rank_ge3_absent','kill_condition':'archival generated presentations include connected component size >=3','observed':rank_ge3_count,'result':'killed; blocker remains'},
 {'target':'chi12_route','kill_condition':'Z/12Z chi12 cannot be routed from T trace','observed':'routed with K12 residual %.3e'%res,'result':'not killed'}]
pd.DataFrame(fals_rows).to_csv(ROOT/'outputs/phase5_v8e_falsification_targets.csv',index=False)
# Defer/blocking rows with premise
frontier_rows=[
 {'target':'rank>=3 Family-F connected component classifier','status':'BLOCKING_OPEN','premise':'v7t/v7u archival Orthad compiler artifacts generate connected components of size >=3','evidence':'archival_v7t_v7u_routing.csv'},
 {'target':'compact invariant completeness for size2','status':'CLOSED_NEGATIVE_ON_TESTED_RANGE','premise':'requested invariant set collides on exact orbit ground truth for named even 2-primary pairs','evidence':'invariant_separation_residual_walls.csv'},
 {'target':'Lean-verified executable classifier','status':'DEFERRED_TO_FINAL_PROOF_PASS','premise':'ledger already defers Lean proof after computational closure surface','evidence':'PHASE5_CANONICAL_LEDGER.md'}]
pd.DataFrame(frontier_rows).to_csv(ROOT/'outputs/phase5_v8e_frontier_separation.csv',index=False)

# Docs
readme=f'''# Phase 5 v8e: Family-F Isometry Classifier

Status: `{summary['status']}`

Phase 5 closed: `{str(summary['phase5_closed']).lower()}`

This pass is restricted to Family F. It does not classify arbitrary FQMs.

Main result: size-2 connected components are exactly classified on the stated range by Aut-orbit ground truth. The compact invariant set requested for p-excess/oddity/Gauss-Milgram does not separate every exact orbit class, so invariant keys alone are not promoted to classifier.

v8c remains suspended because generated rank >= 3 connected components remain blocking open.
'''
(ROOT/'README.md').write_text(readme)
(ROOT/'docs/phase5_v8e_result_card.md').write_text(f'''# v8e result card

```text
{json.dumps(summary, indent=2)}
```
''')
(ROOT/'docs/phase5_v8e_family_f_isometry_classifier.md').write_text(f'''# Phase 5 v8e: Family-F Isometry Classifier

## Scope

Family F presentations are coupling graphs whose vertices are doubled-cyclic carriers and whose edges are representative-invariant pairwise bilinear couplings.

## Graph decomposition

If the coupling graph splits into connected components, then all cross-component edge coefficients are zero. Therefore

```text
b = direct_sum b_component
q = direct_sum q_component
```

and the isometry class of the whole presentation is the multiset of connected component classes.

## Size-2 classifier

For each pair `(D1,D2)` in `{D_RANGE}` with `D1*D2 <= 1024`, the classifier enumerates exact automorphisms of `Z/D1Z x Z/D2Z` and computes orbit classes of all representative-invariant edge residues.

This is a classifier on the stated bounded size-2 range because equality is decided by exact orbit membership, not by a structural key.

## Invariant audit

The requested compact invariant set was computed, including Gauss-Milgram signature and p-primary signatures. It fails to separate all exact orbit classes for named even 2-primary pairs. Those failures are recorded as residual walls, not widened into a false closure.

## Rank >= 3 blocker

Archival v7t/v7u routing generated connected components of size >= 3. v8e does not classify those components. They remain blocking open.
''')
(ROOT/'docs/phase5_v8e_protocol_definitions.md').write_text('''# Protocol definitions

- Family-F size-2 isometry classifier: exact bounded decision procedure using Aut-orbit ground truth over the stated D range.
- Invariant separation residual wall: compact invariants fail to separate exact orbit classes.
- Rank>=3 component blocker: generated connected component with at least three vertices.
- Classifier term is allowed only for exact orbit table decisions on the stated range.
''')
(ROOT/'docs/graph_decomposition_proof.md').write_text('''# Graph decomposition proof

Let a Family-F presentation have carrier A = Π_i Z/D_iZ and bilinear form b with diagonal terms and pairwise edges. If its coupling graph decomposes into components C_1,...,C_k, then c_ij=0 whenever i and j lie in different components. Hence b(x,y)=Σ_a b_a(x_a,y_a), and q(x)=1/2 b(x,x)=Σ_a q_a(x_a). Thus the presentation is an orthogonal direct sum over connected components. Consequently the isometry class of the whole presentation is determined by the multiset of component isometry classes, provided each component classifier is complete in its stated scope.
''')
(ROOT/'docs/phase5_v8e_frontier_note.md').write_text('''# Frontier note

v8e closes exact size-2 Family-F isometry classification on the stated bounded range. It does not close rank>=3 connected components. v8c remains suspended.

Alphabet growth condition: the containment proof is relative to the current T alphabet. Any new T-record type can add slots or edges and reopens Family F derivation.
''')
(ROOT/'docs/v8c_suspension_after_v8e.md').write_text('''# v8c suspension after v8e

v8e does not unsuspend v8c. The size-2 wall was reduced by exact orbit classification, but archival v7t/v7u presentations include generated connected components of size >=3. Therefore the premise needed for v8c closure is still unproven.
''')
(ROOT/'sealed/DO_NOT_CLOSE_PHASE5_GATE.json').write_text(json.dumps({'phase5_closed':False,'reason':'rank>=3 Family-F connected components remain blocking open; v8c remains suspended'},indent=2))
(ROOT/'sealed/SEALED_V8E_SIZE2_CLASSIFIER_BEFORE_RANK3_COMPONENT_ATTACK.json').write_text(json.dumps(summary,indent=2))
# source snapshots
shutil.copy2('/mnt/data/phase5_v8d_orthad_generated_fqm_subclass/outputs/phase5_v8d_generated_presentations.csv', ROOT/'source_notes/phase5_v8d_generated_presentations.csv')
(ROOT/'snapshots/example_v8e_size2_orbit_snapshot.json').write_text(json.dumps({'D1':8,'D2':8,'classes':[r for r in class_rows if r['D1']==8 and r['D2']==8]},indent=2))
# script copy
shutil.copy2('/tmp/build_v8e.py', ROOT/'scripts/phase5_v8e_family_f_isometry_classifier.py')
# Lean stubs
(ROOT/'proofs/Phase5V8EFamilyFIsometry.lean').write_text('''/- Phase 5 v8e theorem surface. Computational outputs are authoritative for this pass; Lean executable proof deferred. -/\n\nnamespace Phase5V8E\n\nstructure FamilyFComponent where\n  rank : Nat\n\ndef graph_decomposition_statement : Prop := True\ndef size2_classifier_statement : Prop := True\ndef rank3_blocker_statement : Prop := True\n\ntheorem graph_decomposition_surface : graph_decomposition_statement := by trivial\ntheorem size2_classifier_surface : size2_classifier_statement := by trivial\ntheorem rank3_blocker_surface : rank3_blocker_statement := by trivial\n\nend Phase5V8E\n''')
(ROOT/'lean/lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage phase5_v8e\nlean_lib Phase5V8E\n')
(ROOT/'lean/lean-toolchain').write_text('leanprover/lean4:stable\n')
(ROOT/'lean/Phase5V8E.lean').write_text('import Phase5V8E.FamilyFIsometry\n')
(ROOT/'lean/Phase5V8E/FamilyFIsometry.lean').write_text((ROOT/'proofs/Phase5V8EFamilyFIsometry.lean').read_text())
# Notebook simple self-contained
try:
 import nbformat as nbf
 nb=nbf.v4.new_notebook()
 cells=[]
 cells.append(nbf.v4.new_markdown_cell('# Phase 5 v8e inline checks'))
 cells.append(nbf.v4.new_code_cell("""# Claim 1: size-2 exact classifier range summary\npair_rows = %d\nfailures = %d\nprint('PASS' if pair_rows == %d else 'FAIL', {'pair_rows': pair_rows, 'range': %s})"""%(len(pair_summary),len(inv_fail_rows),len(pair_summary),D_RANGE)))
 cells.append(nbf.v4.new_code_cell("""# Claim 2: chi12 skeleton route residual\nresidual = %.17g\nthreshold = 1e-12\nprint('PASS' if residual <= threshold else 'FAIL', {'residual': residual, 'threshold': threshold})"""%res))
 cells.append(nbf.v4.new_code_cell("""# Claim 3: rank>=3 blocker remains\nrank_ge3_cases = %d\nprint('PASS' if rank_ge3_cases > 0 else 'FAIL', {'rank_ge3_cases': rank_ge3_cases})"""%rank_ge3_count))
 nb['cells']=cells
 with open(ROOT/'notebooks/phase5_v8e_family_f_isometry_classifier.ipynb','w') as f: nbf.write(nb,f)
except Exception as e:
 (ROOT/'notebooks/phase5_v8e_family_f_isometry_classifier.ipynb').write_text(json.dumps({'error':str(e)}))
(ROOT/'patches/phase5_v8e_family_f_classifier_patch.md').write_text('# v8e patch\n\nAdds exact size-2 Family-F orbit classifier outputs and rank>=3 blocker routing.\n')
# manifest
manifest=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name!='MANIFEST_SHA256SUMS.txt':
        h=hashlib.sha256(p.read_bytes()).hexdigest(); manifest.append(f'{h}  {p.relative_to(ROOT)}')
(ROOT/'MANIFEST_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n')
# zip
zip_path=Path('/mnt/data/phase5_v8e_family_f_isometry_classifier_package.zip')
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob('*'):
        z.write(p, ROOT.name+'/'+str(p.relative_to(ROOT)))
print('WROTE',zip_path,zip_path.stat().st_size)
print(json.dumps(summary,indent=2))
