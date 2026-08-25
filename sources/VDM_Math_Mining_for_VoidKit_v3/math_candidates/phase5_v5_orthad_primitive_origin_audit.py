#!/usr/bin/env python3
import numpy as np

def cyclic_shift(N):
    P=np.zeros((N,N),dtype=complex)
    for r in range(N):
        P[(r+1)%N,r]=1
    return P

def Kminus(N):
    return np.array([[np.exp(-2j*np.pi*r*s/N)/np.sqrt(N) for r in range(N)] for s in range(N)])

def G(N):
    return np.diag([np.exp(1j*np.pi*r*r/N) for r in range(N)])

def reversal(N):
    R=np.zeros((N,N),dtype=complex)
    for r in range(N): R[(-r)%N,r]=1
    return R

def check(N):
    P=cyclic_shift(N); K=Kminus(N); R=reversal(N)
    D=np.diag([np.exp(-2j*np.pi*s/N) for s in range(N)])
    return {
        'N':N,
        'eigen':float(np.max(np.abs(K@P-D@K))),
        'unitarity':float(np.max(np.abs(K@K.conj().T-np.eye(N)))),
        'reversal':float(np.max(np.abs(K@K-R))),
        'fourth':float(np.max(np.abs(np.linalg.matrix_power(K,4)-np.eye(N)))),
    }

if __name__ == '__main__':
    for N in [2,4,6,8,10,12,14,16,18,20,24,26,28,30,32,36,40,48,52,60]:
        print(check(N))
