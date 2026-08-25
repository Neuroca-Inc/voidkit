from __future__ import annotations
import math, cmath, hashlib, csv, json
from pathlib import Path

def budget(r:int)->int:
    return 4**r

def f(r:int)->int:
    return sum(budget(j) for j in range(r))

def self_entry(r:int)->complex:
    return cmath.exp(1j*math.pi*f(r)/2)

def pair_entry(r:int,s:int,N:int)->complex:
    p=f((r+s)%N)-f(r)-f(s)
    return cmath.exp(1j*math.pi*p/2)

def weil_g(r:int,N:int)->complex:
    return cmath.exp(1j*math.pi*r*r/N)

def weil_f(r:int,s:int,N:int)->complex:
    return (1/math.sqrt(N))*cmath.exp(-2j*math.pi*r*s/N)

def main(out='outputs'):
    out=Path(out); out.mkdir(exist_ok=True)
    for N in (12,8):
        native_self=[{'N':N,'r':r,'f':f(r),'entry':self_entry(r)} for r in range(N)]
        native_pair=[{'N':N,'r':r,'s':s,'entry':pair_entry(r,s,N)} for r in range(N) for s in range(N)]
        h=hashlib.sha256(str(native_self+native_pair).encode()).hexdigest()
        res_g=max(abs(row['entry']-weil_g(row['r'],N)) for row in native_self)
        res_f=max(abs(row['entry']-weil_f(row['r'],row['s'],N)) for row in native_pair)
        print(N,h,res_g,res_f)

if __name__=='__main__':
    main()
