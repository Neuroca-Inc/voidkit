#!/usr/bin/env python3
from __future__ import annotations
import math, cmath, csv, json
from pathlib import Path

# Reproduce core Phase 5 v6 regression checks.
def K(N:int, sign:int=-1):
    return [[(1/math.sqrt(N))*cmath.exp(sign*2j*math.pi*r*s/N) for s in range(N)] for r in range(N)]
def G(N:int):
    return [cmath.exp(1j*math.pi*r*r/N) for r in range(N)]
def I(N:int): return [[1 if r==s else 0 for s in range(N)] for r in range(N)]
def maxdiff(A,B):
    if isinstance(A[0], list):
        return max(abs(A[i][j]-B[i][j]) for i in range(len(A)) for j in range(len(A[0])))
    return max(abs(A[i]-B[i]) for i in range(len(A)))
def matmul(A,B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def conjT(A):
    return [[A[j][i].conjugate() for j in range(len(A))] for i in range(len(A[0]))]
def reversal(N:int):
    return [[1 if j == (-i)%N else 0 for j in range(N)] for i in range(N)]

def run():
    rows=[]
    for N in (8,12):
        KN=K(N,-1); GN=G(N)
        rows.append((N,'correct transfer unitarity',maxdiff(matmul(KN,conjT(KN)), I(N))))
        rows.append((N,'correct transfer reversal',maxdiff(matmul(KN,KN), reversal(N))))
        rows.append((N,'remove dual chart fails Fourier',maxdiff(I(N), KN)))
        rows.append((N,'remove self twist fails Gauss',max(abs(1-g) for g in GN)))
    return rows

if __name__ == '__main__':
    for row in run():
        print(row)
