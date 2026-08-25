from __future__ import annotations
import math, cmath

def weil_S(N):
    return [[cmath.exp(-2j*math.pi*r*s/N)/math.sqrt(N) for s in range(N)] for r in range(N)]

def weil_T(N):
    return [[(cmath.exp(1j*math.pi*r*r/N) if r==s else 0j) for s in range(N)] for r in range(N)]

def identity(N, scalar):
    return [[(scalar if r==s else 0j) for s in range(N)] for r in range(N)]

def perm_shift(N):
    M = [[0j for _ in range(N)] for __ in range(N)]
    for r in range(N):
        M[(r+1)%N][r] = 1+0j
    return M

def max_abs_diff(A,B):
    return max(abs(A[i][j]-B[i][j]) for i in range(len(A)) for j in range(len(A[0])))

for N in (12, 8):
    S = weil_S(N)
    T = weil_T(N)
    Q = identity(N, 1j)
    Q2 = identity(N, -1)
    C = perm_shift(N)
    print('N', N)
    print('Q vs T', max_abs_diff(Q,T))
    print('Q2 vs T', max_abs_diff(Q2,T))
    print('carry vs S', max_abs_diff(C,S))
