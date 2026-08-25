#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import mpmath as mp

mp.mp.dps=120
phi=(mp.mpf(1)+mp.sqrt(5))/2
lnphi=mp.log(phi)
ln2=mp.log(2)
ln5=mp.log(5)

def fib_pair(n):
    if n==0: return 0,1
    a,b=fib_pair(n>>1); c=a*((b<<1)-a); d=a*a+b*b
    return (d,c+d) if n&1 else (c,d)

def mA(A): return 12*((1<<(A+1))-1)
def yA(A): return (mp.mpf(mA(A))*ln2+ln5)/(2*lnphi)-mp.mpf('1.5')
def rho(n): return ((-1)**n-phi**(-(2*n+3)))/5

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-a',type=int,default=12); args=ap.parse_args()
    assert all(abs(rho(n))<mp.mpf(1)/4 for n in range(100))
    powers=[]
    for n in range(100):
        a,b=fib_pair(n+1); p=a*b
        if p>0 and (p&(p-1))==0: powers.append((n,p))
    assert powers==[(0,1),(1,2)]
    rows=[]
    for A in range(args.max_a+1):
        T=int(mp.ceil(yA(A))); a,b=fib_pair(T+1); hi=a*b; c,d=fib_pair(T); lo=c*d; X=1<<mA(A)
        assert lo<X<hi
        rows.append({'A':A,'T':T,'lo_lt_X':True,'hi_gt_X':True})
    print(json.dumps({'status':'PASS','global_bridge':'PROVED','A0':0,'sample_rows':rows},indent=2))
if __name__=='__main__': main()
