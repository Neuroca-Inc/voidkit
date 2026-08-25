from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Tuple, Iterable
import cmath, math

PHASE_MOD = 24
BASE_CARRIER = 12
BASE_PHASE_POSITIONS = 6

@dataclass(frozen=True)
class Coeff:
    support: bool
    magnitude: Fraction = Fraction(0,1)
    phase: int = 0
    def __post_init__(self):
        if not self.support:
            object.__setattr__(self, 'magnitude', Fraction(0,1))
            object.__setattr__(self, 'phase', 0)
        else:
            object.__setattr__(self, 'phase', self.phase % PHASE_MOD)
            if self.magnitude <= 0:
                raise ValueError('nonzero coefficient magnitude must be positive')
    @staticmethod
    def zero() -> 'Coeff': return Coeff(False)
    @staticmethod
    def one() -> 'Coeff': return Coeff(True, Fraction(1,1), 0)
    def star(self) -> 'Coeff':
        return self if not self.support else Coeff(True, self.magnitude, -self.phase)
    def scale(self, q: Fraction) -> 'Coeff':
        if not self.support: return self
        if q <= 0: raise ValueError
        return Coeff(True, self.magnitude*q, self.phase)
    def phase_shift(self, e: int) -> 'Coeff':
        return self if not self.support else Coeff(True, self.magnitude, self.phase+e)
    def mul(self, other:'Coeff')->'Coeff':
        if not self.support or not other.support: return Coeff.zero()
        return Coeff(True, self.magnitude*other.magnitude, self.phase+other.phase)
    def to_complex(self)->complex:
        if not self.support: return 0j
        return float(self.magnitude)*cmath.exp(2j*math.pi*self.phase/PHASE_MOD)

Point = Tuple[int,int]  # (base residue mod 12, binary L-history word as int)
Matrix = Dict[Tuple[Point,Point],Coeff]


def bit_count_for_A(A:int)->int:
    return 1<<A

def carrier(A:int)->Tuple[Point,...]:
    return tuple((r,b) for b in range(1<<A) for r in range(BASE_CARRIER))

def walsh_phase(a:int,b:int)->int:
    return 12*((a & b).bit_count() & 1)

def phase_quarters(theta_quarters:int)->int:
    return theta_quarters % 4

def shift(u:int,v:int)->int:
    return (u+v)%BASE_PHASE_POSITIONS

def primary_closed(A:int,u:int,v:int,theta_quarters:int)->Matrix:
    s=shift(u,v); p=phase_quarters(theta_quarters); amp=Fraction(1,u*v)
    out={}
    for x in carrier(A):
        r,a=x
        for y in carrier(A):
            c,b=y
            e=-2*((r+s)%12)*((c+s)%12)+6*p+walsh_phase(a,b)
            out[(x,y)]=Coeff(True,amp,e)
    return out

def seed_primary(A:int,u:int,v:int,theta_quarters:int)->Matrix:
    if A!=0: raise ValueError('birth seed is defined at A=0; higher A arises by L')
    return primary_closed(A,u,v,theta_quarters)

def translate_point(x:Point,delta:int)->Point:
    r,b=x
    return ((r+delta)%12,b)

def transport_B(P:Matrix,A:int,u:int,v:int)->Matrix:
    old_s=shift(u,v)
    new_s=shift(v,u+v)
    delta=(new_s-old_s)%12
    q=Fraction(u,u+v)
    out={}
    for x in carrier(A):
        tx=translate_point(x,delta)
        for y in carrier(A):
            ty=translate_point(y,delta)
            out[(x,y)]=P[(tx,ty)].scale(q)
    return out

def transport_Q(P:Matrix)->Matrix:
    return {k:v.phase_shift(6) for k,v in P.items()}

def transport_L(P:Matrix,A:int)->Matrix:
    old_bits=1<<A
    out={}
    for eps in (0,1):
        for eta in (0,1):
            factor=12*(eps*eta)
            for a in range(old_bits):
                for b in range(old_bits):
                    na=a+eps*old_bits; nb=b+eta*old_bits
                    for r in range(12):
                        for c in range(12):
                            out[((r,na),(c,nb))]=P[((r,a),(c,b))].phase_shift(factor)
    return out

def extract_primary_coordinates(P:Matrix,A:int):
    z=(0,0); o=(1,0)
    c00=P[(z,z)]; c01=P[(z,o)]
    if not c00.support or not c01.support: raise ValueError('primary relation must be full support')
    if c00.magnitude!=c01.magnitude: raise ValueError('nonuniform amplitude at seed probes')
    d=(c01.phase-c00.phase)%24
    candidates=[s for s in range(6) if (-2*s)%24==d]
    if len(candidates)!=1: raise ValueError(f'shift not uniquely extractable: d={d}, cands={candidates}')
    s=candidates[0]
    p_res=(c00.phase+2*s*s)%24
    pc=[p for p in range(4) if 6*p%24==p_res]
    if len(pc)!=1: raise ValueError(f'quarter phase not extractable {p_res}')
    return c00.magnitude,s,pc[0]

def validate_primary(P:Matrix,A:int)->bool:
    try: amp,s,p=extract_primary_coordinates(P,A)
    except Exception: return False
    for x in carrier(A):
        r,a=x
        for y in carrier(A):
            c,b=y
            expected=Coeff(True,amp,-2*((r+s)%12)*((c+s)%12)+6*p+walsh_phase(a,b))
            if P.get((x,y))!=expected:
                return False
    return True

def chart_plus(P:Matrix,A:int)->Matrix:
    out={}
    for x in carrier(A):
        r,_=x
        for y in carrier(A):
            c,_=y
            out[(x,y)]=P[(x,y)] if r!=6 and c!=6 else Coeff.zero()
    return out

def chart_minus(P:Matrix,A:int)->Matrix:
    out={}
    for x in carrier(A):
        r,_=x
        for y in carrier(A):
            c,_=y
            out[(x,y)]=P[(x,y)].star() if r!=0 and c!=0 else Coeff.zero()
    return out

def transfer_forward(P:Matrix,A:int)->Matrix:
    amp,s,p=extract_primary_coordinates(P,A)
    out={}
    for x in carrier(A):
        r,a=x
        for y in carrier(A):
            c,b=y
            support=(r not in (0,6) and c==(-r)%12)
            if support:
                e=4*((r+s)%6)+6*p+walsh_phase(a,b)
                out[(x,y)]=Coeff(True,amp,e)
            else:
                out[(x,y)]=Coeff.zero()
    return out

def transfer_forward_from_derivative(P:Matrix,A:int)->Matrix:
    amp,s,p=extract_primary_coordinates(P,A)
    out={}
    for x in carrier(A):
        r,a=x
        for y in carrier(A):
            c,b=y
            support=(r not in (0,6) and c==(-r)%12)
            if not support:
                out[(x,y)]=Coeff.zero(); continue
            # discrete logarithmic derivative of P along the second base coordinate
            p0=P[(x,(0,b))]; p1=P[(x,(1,b))]
            deriv=(p1.phase-p0.phase)%24
            # inverse square of derivative + retained Q phase + L-Walsh factor
            e=(-2*deriv)+6*p+walsh_phase(a,b)
            out[(x,y)]=Coeff(True,amp,e)
    return out

def transfer_reverse(P:Matrix,A:int)->Matrix:
    return {k:v.star() for k,v in transfer_forward(P,A).items()}

def descendants(P:Matrix,A:int):
    return chart_plus(P,A),chart_minus(P,A),transfer_forward(P,A),transfer_reverse(P,A)

def clean_reference(A:int,s:int,p:int,amp=Fraction(1,1))->Matrix:
    # for exact finite-state comparisons independent of q representatives
    out={}
    for x in carrier(A):
        r,a=x
        for y in carrier(A):
            c,b=y
            out[(x,y)]=Coeff(True,amp,-2*((r+s)%12)*((c+s)%12)+6*p+walsh_phase(a,b))
    return out

def monomial_C(A:int):
    pts=carrier(A); idx={x:i for i,x in enumerate(pts)}
    n=len(pts); C=[[0j]*n for _ in range(n)]
    for x in pts:
        r,b=x; y=((r+6)%12,b)
        C[idx[y]][idx[x]]=(-1)**r
    return C,pts,idx

def dense(M:Matrix,pts):
    return [[M[(x,y)].to_complex() for y in pts] for x in pts]

def matmul(A,B):
    n=len(A); m=len(B[0]); k=len(B)
    return [[sum(A[i][t]*B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]
def conjM(A): return [[z.conjugate() for z in row] for row in A]
def maxdiff(A,B): return max(abs(A[i][j]-B[i][j]) for i in range(len(A)) for j in range(len(A[0])))

def verify_phi(P:Matrix,A:int)->Tuple[float,float]:
    op,om,tf,tr=descendants(P,A)
    C,pts,_=monomial_C(A)
    OP=dense(op,pts); OM=dense(om,pts); TF=dense(tf,pts); TR=dense(tr,pts)
    chart_err=maxdiff(matmul(OM,C),matmul(C,conjM(OP)))
    transfer_err=maxdiff(TR,matmul(matmul(C,conjM(TF)),C))
    return chart_err,transfer_err

@dataclass
class Custody:
    A:int=0; u:int=1; v:int=1; theta_quarters:int=0; k:int=0; W:str=''
    def N(self): return 6*(2**self.A)
    def j(self): return 1+6*((2**self.A)-1)+self.k
    def capacity(self):
        j=self.j()
        if j==1:return 2
        if j==2:return 4
        return 2**(2*j)
    def can_q(self): return self.k<self.N()-1
    def can_b(self):
        if self.k<self.N()-1:
            nu,nv=self.v,self.u+self.v
            return nu*nv<=self.capacity()
        return self.u*self.v<self.capacity()
    def select(self):
        if self.can_b(): return 'B'
        if self.can_q(): return 'Q'
        return 'L'

@dataclass
class Lifted:
    X:Custody
    P:Matrix
    def step(self):
        U=self.X.select()
        oldA,oldu,oldv=self.X.A,self.X.u,self.X.v
        if U=='B':
            self.P=transport_B(self.P,oldA,oldu,oldv)
            self.X.u,self.X.v=oldv,oldu+oldv
        elif U=='Q':
            self.P=transport_Q(self.P)
            self.X.theta_quarters+=1; self.X.k+=1
        else:
            self.P=transport_L(self.P,oldA)
            self.X.A+=1; self.X.k=0
        self.X.W+=U
        return U

def build_initial()->Lifted:
    X=Custody(); return Lifted(X,seed_primary(0,1,1,0))

if __name__=='__main__':
    # smoke
    L=build_initial()
    for _ in range(15): L.step()
    print(L.X,L.P[((0,0),(0,0))])
