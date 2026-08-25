#!/usr/bin/env python3
"""Phase 5 v7c doubled-carrier reverification script.
Regenerates core numeric gates for D=2N carriers.
"""
import numpy as np, math

def K(D):
    a=np.arange(D)[:,None]; b=np.arange(D)[None,:]
    return np.exp(-2j*np.pi*a*b/D)/np.sqrt(D)

def G(D):
    a=np.arange(D)
    return np.diag(np.exp(1j*np.pi*a*a/D))

def R(D):
    M=np.zeros((D,D), complex)
    for i in range(D): M[i,(-i)%D]=1
    return M

def maxabs(M): return float(np.max(np.abs(M)))

def check(N):
    D=2*N
    k=K(D); g=G(D); I=np.eye(D); rev=R(D)
    pol=0.0
    for a in range(D):
        for b in range(D):
            lhs=g[(a+b)%D,(a+b)%D]/(g[a,a]*g[b,b])
            rhs=np.exp(2j*np.pi*a*b/D)
            pol=max(pol, abs(lhs-rhs))
    return {
        'N': N,
        'D': D,
        'unitarity': maxabs(k@k.conj().T-I),
        'reversal': maxabs(k@k-rev),
        'fourth': maxabs(np.linalg.matrix_power(k,4)-I),
        'polarization': pol,
    }

if __name__ == '__main__':
    for N in [1,2,3,4,5,6,8,10,12,13,20,26,60,180]:
        print(check(N))
