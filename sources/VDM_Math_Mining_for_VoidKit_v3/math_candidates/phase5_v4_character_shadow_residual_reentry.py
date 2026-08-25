#!/usr/bin/env python3
import math, cmath, csv, json
import numpy as np

def chi12(r):
    r %= 12
    if math.gcd(r,6)!=1: return 0
    return 1 if r in (1,11) else -1

def chi8(r):
    r %= 8
    if r%2==0: return 0
    return 1 if r in (1,7) else -1

def legendre(p,r):
    r %= p
    if r == 0: return 0
    return 1 if pow(r,(p-1)//2,p)==1 else -1

def K(N):
    return np.array([[np.exp(-2j*np.pi*r*s/N)/np.sqrt(N) for s in range(N)] for r in range(N)], dtype=complex)

def G(N):
    return np.diag([np.exp(1j*np.pi*r*r/N) for r in range(N)])

def main():
    cases = [("chi12",12),("chi8",8),("legendre5",10),("legendre13",26)]
    for name,N in cases:
        if name == "chi12":
            a = np.array([chi12(r) for r in range(N)], complex)
        elif name == "chi8":
            a = np.array([chi8(r) for r in range(N)], complex)
        elif name == "legendre5":
            a = np.array([legendre(5,r) for r in range(N)], complex)
        else:
            a = np.array([legendre(13,r) for r in range(N)], complex)
        Ka = K(N) @ a
        alpha = np.vdot(a, Ka)/np.vdot(a,a)
        residual = np.max(np.abs(Ka-alpha*a))
        print(name, N, "residual", residual, "alpha", alpha)

if __name__ == "__main__":
    main()
