#!/usr/bin/env python3
import math, cmath, hashlib, csv, json
import numpy as np
from pathlib import Path

POS_N = [2,4,6,8,10,12,14,16,18,20,24,26,28,30,32,36,40,48,52,60]

def K(N, sign=-1):
    r=np.arange(N).reshape(N,1); s=np.arange(N).reshape(1,N)
    return np.exp(sign*2j*np.pi*r*s/N)/math.sqrt(N)

def G(N):
    r=np.arange(N)
    return np.exp(1j*np.pi*r*r/N)

def reversal(N):
    R=np.zeros((N,N), dtype=complex)
    for r in range(N): R[r, (-r)%N] = 1
    return R

def maxabs(M): return float(np.max(np.abs(M)))

def run():
    rows=[]
    for N in POS_N:
        Km=K(N,-1); I=np.eye(N); R=reversal(N); g=G(N)
        pol=0.0
        for r in range(N):
            for s in range(N):
                pol=max(pol, abs(g[(r+s)%N]/(g[r]*g[s]) - np.exp(2j*np.pi*r*s/N)))
        rows.append({
            'N':N,
            'unitarity':maxabs(Km@Km.conj().T-I),
            'K2_reversal':maxabs(Km@Km-R),
            'K4_identity':maxabs(np.linalg.matrix_power(Km,4)-I),
            'polarization':pol,
        })
    print(json.dumps(rows, indent=2))

if __name__ == '__main__': run()
