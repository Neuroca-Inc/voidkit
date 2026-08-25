from __future__ import annotations
import itertools, math, numpy as np

def elems(mods): return list(itertools.product(*[range(d) for d in mods]))
def lcm(a,b): return abs(a*b)//math.gcd(a,b)
def Bval(x,y,mods,cross):
    val=sum(x[i]*y[i]/mods[i] for i in range(len(mods)))
    for (i,j),c in (cross or {}).items():
        L=lcm(mods[i],mods[j])
        val += c*(x[i]*y[j]+x[j]*y[i])/L
    return val

def build(mods,cross=None):
    E=elems(mods); n=len(E)
    K=np.array([[np.exp(-2j*np.pi*Bval(x,y,mods,cross))/math.sqrt(n) for y in E] for x in E])
    G=np.array([np.exp(1j*np.pi*Bval(x,x,mods,cross)) for x in E])
    return E,K,G
