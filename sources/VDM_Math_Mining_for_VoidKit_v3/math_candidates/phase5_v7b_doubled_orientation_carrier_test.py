#!/usr/bin/env python3
import math, cmath, csv, json
import numpy as np

def K(D, sign=-1):
    r=np.arange(D).reshape((D,1)); s=np.arange(D).reshape((1,D))
    return (1/math.sqrt(D))*np.exp(sign*2j*math.pi*r*s/D)

def G(D):
    r=np.arange(D)
    return np.exp(1j*math.pi*r*r/D)

def reversal(D):
    R=np.zeros((D,D), dtype=complex)
    for r in range(D): R[r,(-r)%D]=1
    return R

def max_abs(A): return float(np.max(np.abs(A)))

def audit(base_ns):
    rows=[]
    for n in base_ns:
        D=2*n; Km=K(D,-1); g=G(D); I=np.eye(D); R=reversal(D)
        pol=0.0
        for a in range(D):
            for b in range(D):
                pol=max(pol, abs(g[(a+b)%D]/(g[a]*g[b])-cmath.exp(2j*math.pi*a*b/D)))
        rows.append({
            'base_modulus_n': n,
            'D': D,
            'unitarity': max_abs(Km@Km.conj().T-I),
            'reversal': max_abs(Km@Km-R),
            'K4': max_abs(np.linalg.matrix_power(Km,4)-I),
            'polarization': pol,
        })
    return rows

if __name__ == '__main__':
    rows=audit(range(1,61))
    print(json.dumps({'rows':len(rows), 'max_unitarity':max(r['unitarity'] for r in rows), 'max_polarization':max(r['polarization'] for r in rows)}, indent=2))
