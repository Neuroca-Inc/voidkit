#!/usr/bin/env python3
"""QBL B-count dyadic carry investigation.

Uses only the clean Q/B/L domain-count laws. No R/S/T scheduler, fixed
window, or terminal projection is used.
"""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
import mpmath as mp
from sympy import isprime


def constants(dps:int):
    mp.mp.dps=dps
    phi=(1+mp.sqrt(5))/2
    log_phi=mp.log(phi)
    lam=6*mp.log(2)/log_phi
    gamma=lam+mp.mpf('1.5')-mp.log(5)/(2*log_phi)
    return phi,log_phi,lam,gamma


def binet_log_correction(phi,m:int):
    s1=-1 if (m+1)%2 else 1
    s2=-1 if (m+2)%2 else 1
    return mp.log1p(-s1*phi**(-2*m-2))+mp.log1p(-s2*phi**(-2*m-4))


def scan(max_a:int,prime_max_a:int,dps:int):
    phi,log_phi,lam,gamma=constants(dps)
    y=2*lam-gamma
    rows=[];prev_t=prev_b=None
    for A in range(max_a+1):
        t=int(mp.ceil(y));r=mp.mpf(t)-y
        b=t if prev_t is None else t-prev_t
        carry=None if prev_t is None else t-2*prev_t
        defect=None if prev_b is None else b-2*prev_b
        row={'A':A,'T':t,'B':b,'carry':carry,'defect':defect,'r':mp.nstr(r,40),
              'B_odd':bool(b&1),'prime_candidate':defect in (-1,1) if defect is not None else False}
        if A<=prime_max_a:
            q=6*(1<<A)-1
            row.update(Q=q,Q_prime=bool(isprime(q)),B_prime=bool(isprime(b)))
        rows.append(row);prev_t,prev_b=t,b;y=2*y+gamma
    min_r=min((mp.mpf(r['r']),r['A']) for r in rows)
    min_1r=min((1-mp.mpf(r['r']),r['A']) for r in rows)
    right=2*log_phi*min_r[0]-abs(binet_log_correction(phi,9))
    left=2*log_phi*min_1r[0]-abs(binet_log_correction(phi,8))
    certificate={'pass':bool(right>0 and left>0),'min_r':mp.nstr(min_r[0],30),'min_r_A':min_r[1],
                 'min_one_minus_r':mp.nstr(min_1r[0],30),'min_one_minus_r_A':min_1r[1],
                 'terminal_margin':mp.nstr(right,30),'preterminal_margin':mp.nstr(left,30)}
    return rows,certificate,{'phi':mp.nstr(phi,80),'lambda':mp.nstr(lam,80),'gamma':mp.nstr(gamma,80)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--max-a',type=int,default=10000)
    p.add_argument('--prime-max-a',type=int,default=1000);p.add_argument('--dps',type=int,default=3500)
    p.add_argument('--out',type=Path);args=p.parse_args()
    rows,cert,const=scan(args.max_a,min(args.prime_max_a,args.max_a),args.dps)
    dyn=rows[2:]
    payload={'constants':const,'threshold_bridge':cert,
      'carry_alphabet':sorted({r['carry'] for r in rows[1:]}),
      'defect_alphabet':sorted({r['defect'] for r in dyn}),
      'defect_counts':Counter(r['defect'] for r in dyn),
      'B_prime_domains':[r['A'] for r in rows if r.get('B_prime')],
      'Q_prime_domains':[r['A'] for r in rows if r.get('Q_prime')],
      'both_prime_domains':[r['A'] for r in rows if r.get('B_prime') and r.get('Q_prime')]}
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'summary.json').write_text(json.dumps(payload,indent=2,default=int))
        fields=sorted({k for r in rows for k in r})
        with (args.out/'scan.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    else: print(json.dumps(payload,indent=2,default=int))

if __name__=='__main__': main()
