#!/usr/bin/env python3
import math, cmath, json
import numpy as np

def K(N, sign=-1):
    r=np.arange(N).reshape((N,1)); s=np.arange(N).reshape((1,N))
    return (1/np.sqrt(N))*np.exp(sign*2j*np.pi*r*s/N)

def G(N):
    r=np.arange(N); return np.exp(1j*np.pi*r*r/N)

def R(N):
    M=np.zeros((N,N), complex)
    for i in range(N): M[i,(-i)%N]=1
    return M

def maxabs(A): return float(np.max(np.abs(A)))

def pol(N):
    r=np.arange(N).reshape((N,1)); s=np.arange(N).reshape((1,N)); g=G(N)
    return maxabs(g[(r+s)%N]/(g[r]*g[s])-np.exp(2j*np.pi*r*s/N))

Ns=sorted(set(list(range(2,122,2))+[144,168,180,240,360]))
mx=0
for N in Ns:
    k=K(N); I=np.eye(N,dtype=complex); k2=k@k
    mx=max(mx,maxabs(k@k.conj().T-I),maxabs(k2-R(N)),maxabs(k2@k2-I),pol(N))
print(json.dumps({'STATUS':'PASS' if mx<1e-10 else 'FAIL','max_residual':mx,'N_count':len(Ns)},indent=2))
