#!/usr/bin/env python3
import cmath, math, json
import numpy as np

def finite_F(N):
    return np.array([[cmath.exp(-2j*math.pi*r*s/N)/math.sqrt(N) for s in range(N)] for r in range(N)], dtype=complex)

def T_diag(m):
    N = 2*m
    return np.array([cmath.exp(1j*math.pi*r*r/(2*m)) for r in range(N)], dtype=complex)

def chi8():
    v=np.zeros(8,dtype=complex)
    for r,s in {1:1,3:-1,5:-1,7:1}.items():
        v[r]=s
    return v

def odd8():
    v=np.zeros(8,dtype=complex)
    for r in [1,3,5,7]:
        v[r]=1
    return v

def max_abs(x):
    return float(np.max(np.abs(x)))

F=finite_F(8)
T=T_diag(4)
c=chi8()
o=odd8()
alpha=cmath.exp(1j*math.pi/8)

summary={
    "F_unitary": max_abs(F.conj().T@F-np.eye(8)),
    "F_square_reversal": max_abs(F@F - np.array([[1 if (r+s)%8==0 else 0 for s in range(8)] for r in range(8)], dtype=complex)),
    "S_chi8_equals_chi8": max_abs(F@c-c),
    "T_chi8_equals_alpha_odd": max_abs(T*c-alpha*o),
    "T_odd_equals_alpha_chi8": max_abs(T*o-alpha*c),
}
print(json.dumps(summary, indent=2))
