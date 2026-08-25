#!/usr/bin/env python3
import math, cmath

def K(N, sign=-1):
    return [[(1/math.sqrt(N))*cmath.exp(sign*2j*math.pi*r*s/N) for s in range(N)] for r in range(N)]

def G(N):
    return [[(cmath.exp(1j*math.pi*r*r/N) if r==s else 0j) for s in range(N)] for r in range(N)]

def matmul(A,B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def conjT(A):
    return [[A[i][j].conjugate() for i in range(len(A))] for j in range(len(A[0]))]

def identity(N): return [[1 if i==j else 0 for j in range(N)] for i in range(N)]
def reversal(N): return [[1 if (i+j)%N==0 else 0 for j in range(N)] for i in range(N)]
def maxdiff(A,B): return max(abs(A[i][j]-B[i][j]) for i in range(len(A)) for j in range(len(A[0])))

for N in [8,12]:
    Km=K(N,-1); Kp=K(N,+1); Gn=G(N)
    print('N',N)
    print('unitarity', maxdiff(matmul(Km,conjT(Km)), identity(N)))
    print('K^2 reversal', maxdiff(matmul(Km,Km), reversal(N)))
    print('K^4 identity', maxdiff(matmul(matmul(Km,Km),matmul(Km,Km)), identity(N)))
    print('Weil_F residual', maxdiff(Km, K(N,-1)))
    print('Weil_G residual', maxdiff(Gn, G(N)))
