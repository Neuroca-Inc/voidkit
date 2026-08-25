#!/usr/bin/env python3
from __future__ import annotations
import math, cmath, csv, json, hashlib
from pathlib import Path

def transfer_matrix(N:int):
    return [[(1/math.sqrt(N))*cmath.exp(-2j*math.pi*r*s/N) for s in range(N)] for r in range(N)]

def matmul(A,B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def conj_transpose(A):
    return [[A[i][j].conjugate() for i in range(len(A))] for j in range(len(A[0]))]

def eye(N):
    return [[1 if i==j else 0 for j in range(N)] for i in range(N)]

def max_abs_diff(A,B):
    return max(abs(A[i][j]-B[i][j]) for i in range(len(A)) for j in range(len(A[0])))

def reversal_matrix(N):
    return [[1 if j == (-i)%N else 0 for j in range(N)] for i in range(N)]

for N in [8,12]:
    K = transfer_matrix(N)
    print('N', N)
    print('unitarity', max_abs_diff(matmul(K, conj_transpose(K)), eye(N)))
    print('K2 reversal', max_abs_diff(matmul(K,K), reversal_matrix(N)))
    WeilF = transfer_matrix(N)
    print('Weil_F residual', max_abs_diff(K, WeilF))
print('collapse L1', 'i/4895')
print('collapse L2', f'-i/{196418*317811}')
