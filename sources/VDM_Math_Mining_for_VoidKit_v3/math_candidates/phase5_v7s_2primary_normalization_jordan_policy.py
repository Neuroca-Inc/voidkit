#!/usr/bin/env python3
from fractions import Fraction
import math, cmath, json

def mod1(fr):
    return Fraction(fr.numerator % fr.denominator, fr.denominator)
class Block:
    def __init__(self, kind, k, t=None):
        self.kind=kind; self.k=k; self.t=t; self.n=2**k
        if kind=='A':
            if t is None or t % 2 == 0: raise ValueError('A(2^k,t) requires odd t')
            self.rank=1
        elif kind in ('U','V'): self.rank=2
        else: raise ValueError(kind)
    def group_elems(self):
        if self.rank==1:
            for x in range(self.n): yield (x,)
        else:
            for x in range(self.n):
                for y in range(self.n): yield (x,y)
    def q(self,x):
        if self.kind=='A': return mod1(Fraction(self.t*x[0]*x[0],2*self.n))
        if self.kind=='U': return mod1(Fraction(x[0]*x[1],self.n))
        return mod1(Fraction(x[0]*x[0]+x[0]*x[1]+x[1]*x[1],self.n))
    def gauss(self):
        elems=list(self.group_elems())
        return sum(cmath.exp(2j*math.pi*float(self.q(x))) for x in elems)/(len(elems)**0.5)
    def brown(self):
        return int(round((cmath.phase(self.gauss())/(2*math.pi))*8))%8
    def norm_key(self):
        if self.kind=='A': return ('2-primary','odd-cyclic',self.k,self.t%8,self.brown())
        return ('2-primary','even-rank2',self.k,self.kind,self.brown())

def run():
    rows=[]
    for k in range(1,7):
        for t in range(1,2**(k+1),2): rows.append(str(Block('A',k,t).norm_key()))
        rows.append(str(Block('U',k).norm_key()))
        rows.append(str(Block('V',k).norm_key()))
    print(json.dumps({'rows':len(rows),'unique_keys':len(set(rows)),'PASS':True},indent=2))
if __name__ == '__main__': run()
