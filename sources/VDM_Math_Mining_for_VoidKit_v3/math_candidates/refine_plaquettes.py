#!/usr/bin/env python3
import gzip, json, hashlib, csv
from pathlib import Path
import mpmath as mp

mp.mp.dps = 80
root=Path(__file__).resolve().parents[1]
q=[]
with gzip.open(root/'results/q_closures_exact_hex.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        for k in ('product','u','v'):
            r[k]=int(r[k],16)
        q.append(r)
by={(int(r['domain']),int(r['k'])):r for r in q}
rows=[]
def ib(v):
    return v.to_bytes(max(1,(v.bit_length()+7)//8),'big')
def hpair(a,b):
    h=hashlib.sha256()
    for v in (a,b):
        x=ib(v);h.update(len(x).to_bytes(8,'big'));h.update(x)
    return h.hexdigest()
for d in range(10):
    n=6*(2**d)
    for k in range(1,n-1):
        keys=[(d,k),(d,k+1),(d+1,2*k),(d+1,2*k+2)]
        if not all(x in by for x in keys):continue
        c0,c1,f0,f1=[by[x] for x in keys]
        num=f1['product']*c0['product'];den=f0['product']*c1['product']
        val=mp.log(mp.mpf(num))-mp.log(mp.mpf(den))
        cchg=int(c1['b_load'])-int(c0['b_load'])
        fchg=int(f1['b_load'])-int(f0['b_load'])
        rows.append({
            'coarse_domain':d,'coarse_k':k,'fine_domain':d+1,'fine_k0':2*k,
            'factor_sha256':hpair(num,den),'factor_is_one':num==den,
            'log_defect_mp':mp.nstr(val,50),
            'log_defect':float(val),'abs_log_defect':float(abs(val)),
            'coarse_b_load_change':cchg,'fine_b_load_change':fchg,
            'any_cadence_change':(cchg!=0 or fchg!=0),
        })
with (root/'results/burden_plaquette_census.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'nonunit':sum(not r['factor_is_one'] for r in rows),'min_abs':min(r['abs_log_defect'] for r in rows),'max_abs':max(r['abs_log_defect'] for r in rows)},indent=2))
