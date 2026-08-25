#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import mpmath as mp
import numpy as np
import sympy as sp

STAMP = "20260711T103100"
STATES = (7, 8, 9)
CHARS = ("7", "8", "9")
DEFECTS = (-2, -1, 0, 1, 2)
M_INT = ((0, 1, 1), (1, 1, 1), (1, 1, 0))


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, mp.mpf): return mp.nstr(x, 100)
    if isinstance(x, sp.MatrixBase):
        return [[str(sp.simplify(x[i, j])) for j in range(x.cols)] for i in range(x.rows)]
    if isinstance(x, sp.Basic): return str(sp.simplify(x))
    if isinstance(x, Fraction): return f"{x.numerator}/{x.denominator}"
    return x


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def constants(dps: int = 180) -> dict[str, mp.mpf]:
    mp.mp.dps = dps
    phi = (1 + mp.sqrt(5)) / 2
    log_phi = mp.log(phi)
    lam = 6 * mp.log(2) / log_phi
    gamma = lam + mp.mpf("1.5") - mp.log(5) / (2 * log_phi)
    a = (gamma - 8) / 2
    p = 2 * a
    y0 = 2 * lam - gamma
    return {"phi": phi, "lambda": lam, "gamma": gamma, "a": a, "p": p, "y0": y0}


def outward_a_bounds(dps: int = 180) -> tuple[Fraction, Fraction, dict[str, str]]:
    mp.iv.dps = dps
    phi = (mp.iv.mpf(1) + mp.iv.sqrt(5)) / 2
    log_phi = mp.iv.log(phi)
    lam = 6 * mp.iv.log(2) / log_phi
    gamma = lam + mp.iv.mpf("1.5") - mp.iv.log(5) / (2 * log_phi)
    a = (gamma - 8) / 2
    s = str(a)
    inside = s[s.find("[")+1:s.rfind("]")]
    lo_s, hi_s = [x.strip() for x in inside.split(",", 1)]
    lo = Fraction(Decimal(lo_s)); hi = Fraction(Decimal(hi_s))
    return lo, hi, {"lower": lo_s, "upper": hi_s, "method": "mpmath.iv outward-rounded interval evaluation", "dps": str(dps)}


@dataclass(frozen=True)
class Affine:
    qa: Fraction
    c: Fraction

    def __add__(self, other: "Affine") -> "Affine": return Affine(self.qa + other.qa, self.c + other.c)
    def __sub__(self, other: "Affine") -> "Affine": return Affine(self.qa - other.qa, self.c - other.c)
    def scale(self, k: Fraction | int) -> "Affine": return Affine(self.qa * k, self.c * k)
    def divide(self, k: Fraction | int) -> "Affine": return Affine(self.qa / k, self.c / k)
    def eval_fraction(self, a: Fraction) -> Fraction: return self.qa * a + self.c
    def plain(self) -> str:
        pieces=[]
        if self.qa:
            if self.qa == 1: pieces.append("a")
            elif self.qa == -1: pieces.append("-a")
            else: pieces.append(f"{self.qa}*a")
        if self.c:
            if pieces and self.c > 0: pieces.append(f"+ {self.c}")
            else: pieces.append(str(self.c))
        return " ".join(pieces) if pieces else "0"


PARTITIONS = {
    "7": (Affine(Fraction(0), Fraction(-1)), Affine(Fraction(-1), Fraction(-1, 2))),
    "8": (Affine(Fraction(-1), Fraction(-1, 2)), Affine(Fraction(-1), Fraction(0))),
    "9": (Affine(Fraction(-1), Fraction(0)), Affine(Fraction(0), Fraction(0))),
}


def affine_range(x: Affine, a_lo: Fraction, a_hi: Fraction) -> tuple[Fraction, Fraction]:
    v1=x.eval_fraction(a_lo); v2=x.eval_fraction(a_hi)
    return (min(v1,v2), max(v1,v2))


def compare_affine(x: Affine, y: Affine, a_lo: Fraction, a_hi: Fraction) -> int:
    lo, hi = affine_range(x-y, a_lo, a_hi)
    if hi < 0: return -1
    if lo > 0: return 1
    if lo == 0 and hi == 0: return 0
    raise RuntimeError(f"unresolved affine comparison {x.plain()} vs {y.plain()} over a enclosure")


def max_aff(x: Affine, y: Affine, a_lo: Fraction, a_hi: Fraction) -> Affine:
    return y if compare_affine(x,y,a_lo,a_hi)<0 else x


def min_aff(x: Affine, y: Affine, a_lo: Fraction, a_hi: Fraction) -> Affine:
    return x if compare_affine(x,y,a_lo,a_hi)<0 else y


def cylinder_initial(word: str, a_lo: Fraction, a_hi: Fraction) -> tuple[Affine, Affine, bool]:
    lo, hi = PARTITIONS[word[0]]
    slope=Fraction(1); off=Affine(Fraction(0),Fraction(0))
    for idx in range(1,len(word)):
        c=int(word[idx-1])
        slope*=2
        off=off.scale(2)+Affine(Fraction(2),Fraction(8-c))
        tlo,thi=PARTITIONS[word[idx]]
        prelo=(tlo-off).divide(slope); prehi=(thi-off).divide(slope)
        lo=max_aff(lo,prelo,a_lo,a_hi); hi=min_aff(hi,prehi,a_lo,a_hi)
    return lo,hi,compare_affine(lo,hi,a_lo,a_hi)<0


def edge_envelope(word: str) -> bool:
    idx={"7":0,"8":1,"9":2}
    return all(M_INT[idx[x]][idx[y]] for x,y in zip(word,word[1:]))


def generate_levels(max_depth: int, a_lo: Fraction, a_hi: Fraction) -> list[dict[str, tuple[Affine,Affine]]]:
    levels=[]
    first={}
    for s in CHARS:
        lo,hi,ok=cylinder_initial(s,a_lo,a_hi)
        if ok:first[s]=(lo,hi)
    levels.append(first)
    for n in range(2,max_depth+1):
        cur={}
        for w in levels[-1]:
            for s in CHARS:
                ww=w+s
                lo,hi,ok=cylinder_initial(ww,a_lo,a_hi)
                if ok:cur[ww]=(lo,hi)
        levels.append(cur)
    return levels


def fraction_decimal(x: Fraction, digits: int=60) -> str:
    getcontext().prec=digits+10
    return format(Decimal(x.numerator)/Decimal(x.denominator), f".{digits}g")


def interval_payload(lo: Affine, hi: Affine, a_lo: Fraction, a_hi: Fraction) -> dict[str, str]:
    llo,lhi=affine_range(lo,a_lo,a_hi); hlo,hhi=affine_range(hi,a_lo,a_hi)
    width_lo=min((hi-lo).eval_fraction(a_lo),(hi-lo).eval_fraction(a_hi))
    return {
        "lo_affine": lo.plain(), "hi_affine": hi.plain(),
        "lo_lower": fraction_decimal(llo), "lo_upper": fraction_decimal(lhi),
        "hi_lower": fraction_decimal(hlo), "hi_upper": fraction_decimal(hhi),
        "width_lower": fraction_decimal(width_lo),
    }


def exact_markov_witnesses(levels: list[dict[str,tuple[Affine,Affine]]], a_lo: Fraction, a_hi: Fraction, max_k: int=10) -> list[dict[str,Any]]:
    out=[]
    for k in range(1,max_k+1):
        found=None
        for n in range(k+1,len(levels)+1):
            words=levels[n-1]; next_words=levels[n] if n < len(levels) else {}
            groups=defaultdict(list)
            for w in words:
                ext=tuple(s for s in CHARS if w+s in next_words)
                groups[w[-k:]].append((w,ext))
            for suffix,vals in sorted(groups.items()):
                by={}
                for w,e in vals: by.setdefault(e,w)
                if len(by)>1:
                    pairs=list(sorted(by.items(),key=lambda z:(z[0],z[1])))[:2]
                    w1,w2=pairs[0][1],pairs[1][1]
                    p1=interval_payload(*words[w1],a_lo,a_hi); p2=interval_payload(*words[w2],a_lo,a_hi)
                    found={"tested_order":k,"witness_length":n,"shared_suffix":suffix,
                           "word_1":w1,"extensions_1":"".join(pairs[0][0]),
                           "word_2":w2,"extensions_2":"".join(pairs[1][0]),
                           "word_1_interval":f"({p1['lo_affine']},{p1['hi_affine']}]",
                           "word_2_interval":f"({p2['lo_affine']},{p2['hi_affine']}]",
                           "certificate":"rational-affine endpoint comparison over outward a enclosure"}
                    break
            if found:break
        if not found: raise RuntimeError(f"no exact witness for order {k}")
        out.append(found)
    return out


def load_trace(root: Path) -> list[dict[str,str]]:
    files=sorted((root/"inputs").glob("*_PRIOR_CARRY_DEFECT_A0_A10000.csv"))
    if len(files)!=1: raise RuntimeError("expected one prior trace")
    with files[0].open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if len(rows)!=10001: raise RuntimeError("trace must have 10001 rows")
    by={int(r["A"]):r for r in rows}
    if set(by)!=set(range(10001)): raise RuntimeError("trace A coverage invalid")
    rows=[by[i] for i in range(10001)]
    if rows[0]["carry"].strip(): raise RuntimeError("A0 carry must be blank")
    for A in range(1,10001):
        if int(rows[A]["carry"]) not in STATES: raise RuntimeError(f"bad carry A={A}")
    return rows


def load_prior_counts(root: Path) -> np.ndarray:
    files=sorted((root/"inputs").glob("*_PRIOR_TRANSITION_COUNTS.csv"))
    if len(files)!=1: raise RuntimeError("expected one prior count table")
    idx={7:0,8:1,9:2}; arr=np.zeros((3,3),dtype=np.int64); seen=set()
    with files[0].open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key=(int(r["from_state"]),int(r["to_state"]));
            if key in seen: raise RuntimeError("duplicate transition count")
            seen.add(key);arr[idx[key[0]],idx[key[1]]]=int(r["count"])
    if len(seen)!=9:return (_ for _ in ()).throw(RuntimeError("incomplete transition counts"))
    return arr


def empirical(rows:list[dict[str,str]], prior:np.ndarray)->dict[str,Any]:
    carries={A:int(rows[A]["carry"]) for A in range(1,10001)}
    tc=Counter((carries[A-1],carries[A]) for A in range(2,10001))
    dc=Counter(carries[A]-carries[A-1] for A in range(2,10001))
    sc=Counter(carries.values())
    arr=np.array([[tc[(i,j)] for j in STATES] for i in STATES],dtype=np.int64)
    if arr.sum()!=9999:raise RuntimeError("transition total not 9999")
    if not np.array_equal(arr,prior):raise RuntimeError("prior transition counts disagree")
    return {"counts":arr,"joint":arr/arr.sum(),"state_counts":[sc[s] for s in STATES],"defect_counts":[dc[d] for d in DEFECTS],"carries":carries}


def symbolic_core()->dict[str,Any]:
    a=sp.symbols("a",real=True)
    J=sp.Matrix([[0,(1-3*a)/2,a/2],[(1-2*a)/4,sp.Rational(1,4),a/2],[(1-2*a)/4,3*a/2-sp.Rational(1,4),0]])
    pi=sp.Matrix([[sp.Rational(1,2)-a,sp.Rational(1,2),a]])
    P=sp.Matrix([[0,(1-3*a)/(1-2*a),a/(1-2*a)],[(1-2*a)/2,sp.Rational(1,2),a],[(1-2*a)/(4*a),(6*a-1)/(4*a),0]])
    M=sp.Matrix(M_INT); M2=M**2; rho=1+sp.sqrt(2); r=sp.Matrix([1,sp.sqrt(2),1])
    K=sp.Matrix([[0,sp.sqrt(2),1],[sp.sqrt(2),2,sp.sqrt(2)],[1,sp.sqrt(2),0]])/(4*rho)
    defect={-2:sp.simplify(J[2,0]),-1:sp.simplify(J[1,0]+J[2,1]),0:sp.simplify(J[1,1]),1:sp.simplify(J[0,1]+J[1,2]),2:sp.simplify(J[0,2])}
    checks={
      "J_rows":[sp.simplify(sum(J[i,j] for j in range(3))) for i in range(3)],
      "J_cols":[sp.simplify(sum(J[i,j] for i in range(3))) for j in range(3)],
      "J_total":sp.simplify(sum(J)),
      "P_rows":[sp.simplify(sum(P[i,j] for j in range(3))) for i in range(3)],
      "stationarity":[sp.simplify(x) for x in list(pi*P-pi)],
      "M2":M2,"perron":[sp.simplify(x) for x in list(M*r-rho*r)],
      "K_total":sp.simplify(sum(K))}
    expected={"J_rows":[sp.Rational(1,2)-a,sp.Rational(1,2),a],"J_cols":[sp.Rational(1,2)-a,sp.Rational(1,2),a],"J_total":1,"P_rows":[1,1,1],"stationarity":[0,0,0],"M2":sp.Matrix([[2,2,1],[2,3,2],[1,2,2]]),"perron":[0,0,0],"K_total":1}
    if checks!=expected:raise RuntimeError(f"symbolic core failure {checks}")
    return {"a":a,"J":J,"pi":pi,"P":P,"M":M,"M2":M2,"rho":rho,"r":r,"K":K,"defect":defect,"checks":checks}


def hp_metrics(counts:np.ndarray,target:list[list[mp.mpf]])->dict[str,str]:
    mp.mp.dps=120; total=mp.mpf(int(counts.sum())); diffs=[]
    for i in range(3):
        for j in range(3): diffs.append(abs(mp.mpf(int(counts[i,j]))/total-target[i][j]))
    l1=sum(diffs)
    return {"max_absolute_error":mp.nstr(max(diffs),80),"L1_error":mp.nstr(l1,80),"total_variation":mp.nstr(l1/2,80)}


def boundary_interval_certificate(rows:list[dict[str,str]],dps:int=3300)->dict[str,Any]:
    mp.mp.dps=dps;mp.iv.dps=dps
    phi=(mp.iv.mpf(1)+mp.iv.sqrt(5))/2;lp=mp.iv.log(phi)
    lam=6*mp.iv.log(2)/lp;gamma=lam+mp.iv.mpf("1.5")-mp.iv.log(5)/(2*lp);a=(gamma-8)/2;y0=2*lam-gamma
    lo=lambda x:mp.mpf(x.a);hi=lambda x:mp.mpf(x.b)
    def payload(x):return {"lower":mp.nstr(lo(x),120),"upper":mp.nstr(hi(x),120),"width":mp.nstr(hi(x)-lo(x),120)}
    if not(lo(y0)>8 and hi(y0)<9):raise RuntimeError("y0 ceil not certified")
    E=y0-9;b7=-mp.iv.mpf("0.5")-a;b8=-a
    bounds=[("-1",mp.iv.mpf(-1)),("b7",b7),("b8",b8),("0",mp.iv.mpf(0))]
    minm=mp.inf;minrec=None;maxw=mp.mpf(0);steps=0
    for A in range(10001):
        maxw=max(maxw,hi(E)-lo(E))
        for name,b in bounds:
            if hi(E)<lo(b):dist=lo(b)-hi(E)
            elif lo(E)>hi(b):dist=lo(E)-hi(b)
            else:raise RuntimeError(f"boundary overlap A={A} {name}")
            if dist<minm:
                minm=dist;minrec={"A":A,"boundary":name,"distance_lower_bound":mp.nstr(dist,120),"E_interval":payload(E),"boundary_interval":payload(b)}
        if A==10000:break
        c=int(rows[A+1]["carry"])
        ok=(lo(E)>-1 and hi(E)<=lo(b7)) if c==7 else ((lo(E)>hi(b7) and hi(E)<=lo(b8)) if c==8 else (lo(E)>hi(b8) and hi(E)<=0))
        if not ok:raise RuntimeError(f"partition failure A={A}")
        z=2*E+gamma
        if not(lo(z)>c-1 and hi(z)<=c):raise RuntimeError(f"ceil failure A={A+1}")
        E=z-c;steps+=1
    return {"method":"mpmath.iv outward-rounded interval arithmetic","mpmath_version":mp.__version__,"decimal_digits":dps,"gamma_interval":payload(gamma),"a_interval":payload(a),"y0_interval":payload(y0),"E0_interval":payload(y0-9),"all_imported_carries_certified":steps==10000,"certified_carry_steps":steps,"all_E_A_boundary_disjoint_for_A_0_10000":True,"minimum_boundary_distance_lower_bound":minrec,"maximum_orbit_interval_width":mp.nstr(maxw,120),"final_E10000_interval":payload(E),"scope":"finite affine orbit A=0..10000 only"}


def mod1_affine(x:Affine,a_lo:Fraction,a_hi:Fraction)->Affine:
    lo,hi=affine_range(x,a_lo,a_hi)
    flo=math.floor(lo);fhi=math.floor(hi)
    if flo!=fhi:raise RuntimeError(f"mod1 floor unresolved {x.plain()}")
    return Affine(x.qa,x.c-Fraction(flo))



def image_after_word(word: str, lo: Affine, hi: Affine) -> tuple[Affine, Affine]:
    """Exact image of the half-open cylinder (lo,hi] after its full word."""
    slope = Fraction(1)
    off = Affine(Fraction(0), Fraction(0))
    for ch in word:
        c = int(ch)
        slope *= 2
        off = off.scale(2) + Affine(Fraction(2), Fraction(8 - c))
    return lo.scale(slope) + off, hi.scale(slope) + off


def follower_region_geometry(
    levels: list[dict[str, tuple[Affine, Affine]]],
    a_lo: Fraction,
    a_hi: Fraction,
    max_n: int = 12,
) -> list[dict[str, Any]]:
    """Finite exact geometry for the two cylinders adjacent to the cut 0=-1.

    The left cylinder has the half-open form (ell,0] and the right cylinder
    has the form (-1,r].  Their full word images are half-open follower
    regions (alpha,q] and (q,beta].  The shared q is the oriented handoff
    from minus to plus.  The other outer endpoints may coincide on the
    circle, so the proof uses oriented adjacency rather than an unqualified
    unique-common-boundary assertion.
    """
    zero = Affine(Fraction(0), Fraction(0))
    minus_one = Affine(Fraction(0), Fraction(-1))
    one = Affine(Fraction(0), Fraction(1))
    rows: list[dict[str, Any]] = []
    for n in range(1, min(max_n, len(levels)) + 1):
        lv = levels[n - 1]
        left_candidates = [(w, iv) for w, iv in lv.items() if compare_affine(iv[1], zero, a_lo, a_hi) == 0]
        right_candidates = [(w, iv) for w, iv in lv.items() if compare_affine(iv[0], minus_one, a_lo, a_hi) == 0]
        if len(left_candidates) != 1 or len(right_candidates) != 1:
            raise RuntimeError(f"expected one boundary-adjacent pair at n={n}")
        lw, (clo, chi) = left_candidates[0]
        rw, (dlo, dhi) = right_candidates[0]
        hml, hmh = image_after_word(lw, clo, chi)
        hpl, hph = image_after_word(rw, dlo, dhi)
        if compare_affine(hmh, hpl, a_lo, a_hi) != 0:
            raise RuntimeError(f"follower handoff mismatch at n={n}")
        minus_width = hmh - hml
        plus_width = hph - hpl
        source_minus_width = chi - clo
        source_plus_width = dhi - dlo
        scale_bound = Affine(Fraction(0), Fraction(1, 2**n))
        if compare_affine(source_minus_width, scale_bound, a_lo, a_hi) >= 0:
            raise RuntimeError(f"minus source cylinder not strictly below 2^-n at n={n}")
        if compare_affine(source_plus_width, scale_bound, a_lo, a_hi) >= 0:
            raise RuntimeError(f"plus source cylinder not strictly below 2^-n at n={n}")
        outer_delta = hph - hml
        outer_same_circle = outer_delta.qa == 0 and outer_delta.c.denominator == 1
        qlo, qhi = affine_range(hmh, a_lo, a_hi)
        mlo0, mlo1 = affine_range(hml, a_lo, a_hi)
        phi0, phi1 = affine_range(hph, a_lo, a_hi)
        mw0, mw1 = affine_range(minus_width, a_lo, a_hi)
        pw0, pw1 = affine_range(plus_width, a_lo, a_hi)
        rows.append({
            "n": n,
            "minus_word": lw,
            "plus_word": rw,
            "source_minus": f"({clo.plain()},{chi.plain()}]",
            "source_plus": f"({dlo.plain()},{dhi.plain()}]",
            "H_minus": f"({hml.plain()},{hmh.plain()}]",
            "H_plus": f"({hpl.plain()},{hph.plain()}]",
            "H_minus_lo_affine": hml.plain(),
            "handoff_affine": hmh.plain(),
            "H_plus_hi_affine": hph.plain(),
            "H_minus_lo_lower": fraction_decimal(mlo0),
            "H_minus_lo_upper": fraction_decimal(mlo1),
            "handoff_lower": fraction_decimal(qlo),
            "handoff_upper": fraction_decimal(qhi),
            "H_plus_hi_lower": fraction_decimal(phi0),
            "H_plus_hi_upper": fraction_decimal(phi1),
            "H_minus_width_lower": fraction_decimal(mw0),
            "H_minus_width_upper": fraction_decimal(mw1),
            "H_plus_width_lower": fraction_decimal(pw0),
            "H_plus_width_upper": fraction_decimal(pw1),
            "source_minus_strict_below_2^-n": True,
            "source_plus_strict_below_2^-n": True,
            "outer_endpoints_coincide_on_circle": outer_same_circle,
            "oriented_handoff": "minus interior ends; plus interior begins",
            "endpoint_scope": "finite exact rational-affine geometry over outward a enclosure",
        })
    return rows

def boundary_follower_pairs(levels:list[dict[str,tuple[Affine,Affine]]],a_lo:Fraction,a_hi:Fraction,max_n:int=20)->list[dict[str,Any]]:
    out=[]; prev_p=None
    minus1=Affine(Fraction(0),Fraction(-1));zero=Affine(Fraction(0),Fraction(0))
    for n in range(1,max_n+1):
        lv=levels[n-1] if n<=len(levels) else None
        left=right=""
        if lv is not None:
            for w,(lo,hi) in lv.items():
                if compare_affine(hi,zero,a_lo,a_hi)==0:left=w
                if compare_affine(lo,minus1,a_lo,a_hi)==0:right=w
        pn=mod1_affine(Affine(Fraction(2**(n+1)),Fraction(0)),a_lo,a_hi)
        plo,phi=affine_range(pn,a_lo,a_hi)
        out.append({"n":n,"left_adjacent_word":left,"right_adjacent_word":right,"D_n_p_affine":pn.plain(),"D_n_p_lower":fraction_decimal(plo),"D_n_p_upper":fraction_decimal(phi),"distinct_from_previous":prev_p is None or pn!=prev_p,"role":"common boundary of the two one-sided follower arcs"})
        prev_p=pn
    return out


def follower_signature_counts(levels:list[dict[str,tuple[Affine,Affine]]],max_n:int=9,horizon:int=4)->list[dict[str,Any]]:
    rows=[]
    for n in range(1,max_n+1):
        words=levels[n-1]; hs=min(horizon,len(levels)-n)
        sigs={}
        descendant_maps=[]
        for k in range(1,hs+1):
            d={w:[] for w in words}
            for x in levels[n+k-1]: d[x[:n]].append(x[n:])
            descendant_maps.append(d)
        for w in words:
            sig=tuple(tuple(dm[w]) for dm in descendant_maps);sigs.setdefault(sig,[]).append(w)
        rows.append({"word_length":n,"words":len(words),"distinct_finite_horizon_follower_signatures":len(sigs),"future_horizon":hs,"status":"finite exact evidence; non-soficity is proved separately"})
    return rows


def run(root:Path)->dict[str,Any]:
    outputs=root/"outputs";trace_dir=root/"trace";outputs.mkdir(exist_ok=True);trace_dir.mkdir(exist_ok=True)
    c=constants();a_lo,a_hi,a_bounds=outward_a_bounds()
    if not(Fraction(1,6)<a_lo<a_hi<Fraction(1,4)):raise RuntimeError("a enclosure outside assumptions")
    if not(a_lo>Fraction(3,14)):raise RuntimeError("a>3/14 not certified")
    sym=symbolic_core();levels=generate_levels(12,a_lo,a_hi)
    for n,lv in enumerate(levels,1):
        expected=2**(n+1)-1
        if len(lv)!=expected:raise RuntimeError(f"complexity direct failure n={n}: {len(lv)} != {expected}")
    # exact cylinders and trace
    cyl_rows=[]
    with (trace_dir/f"{STAMP}_symbolic_cylinder_trace.jsonl").open("w",encoding="utf-8") as tf:
        for n,lv in enumerate(levels,1):
            for w,(lo,hi) in sorted(lv.items()):
                pld=interval_payload(lo,hi,a_lo,a_hi)
                row={"length":n,"word":w,**pld}
                cyl_rows.append(row);tf.write(json.dumps(row,sort_keys=True)+"\n")
    write_csv(outputs/f"{STAMP}_exact_cylinders_depth12.csv",list(cyl_rows[0].keys()),cyl_rows)
    # length 3 full table
    l3=[]
    for i in CHARS:
      for j in CHARS:
       for k in CHARS:
        w=i+j+k;lo,hi,ok=cylinder_initial(w,a_lo,a_hi)
        reason="nonempty exact affine cylinder" if ok else ("forbidden 989 interval" if w=="989" else ("forbidden 787 at current a" if w=="787" else ("zero pair edge" if "77" in w or "99" in w else "empty exact cylinder")))
        l3.append({"word":w,"edge_envelope":edge_envelope(w),"realizable":ok,"cylinder":f"({lo.plain()},{hi.plain()}]" if ok else "","reason":reason})
    if sum(r["realizable"] for r in l3)!=15:raise RuntimeError("length3 count")
    write_csv(outputs/f"{STAMP}_realizable_length3_words.csv",list(l3[0].keys()),l3)
    # complexity
    comp=[]
    for n in range(1,21):
        env=sum(np.linalg.matrix_power(np.array(M_INT,dtype=object),n-1).flatten()) if n>1 else 3
        comp.append({"length":n,"actual_affine_complexity":2**(n+1)-1,"direct_exact_cylinders":len(levels[n-1]) if n<=12 else "","edge_envelope_paths":int(env),"proof":"refinement-boundary/cylinder bijection"})
    write_csv(outputs/f"{STAMP}_word_complexity.csv",list(comp[0].keys()),comp)
    # markov exact
    mw=exact_markov_witnesses(levels,a_lo,a_hi,10)
    write_csv(outputs/f"{STAMP}_exact_markov_order_counterexamples.csv",list(mw[0].keys()),mw)
    # follower pairs/signatures
    fp=boundary_follower_pairs(levels,a_lo,a_hi,20)
    write_csv(outputs/f"{STAMP}_boundary_adjacent_follower_pairs.csv",list(fp[0].keys()),fp)
    with (trace_dir/f"{STAMP}_follower_pair_trace.jsonl").open("w",encoding="utf-8") as f:
        for r in fp:f.write(json.dumps(r,sort_keys=True)+"\n")
    fr=follower_region_geometry(levels,a_lo,a_hi,12)
    write_csv(outputs/f"{STAMP}_half_open_follower_regions.csv",list(fr[0].keys()),fr)
    with (trace_dir/f"{STAMP}_half_open_follower_region_trace.jsonl").open("w",encoding="utf-8") as f:
        for r in fr:f.write(json.dumps(r,sort_keys=True)+"\n")
    fs=follower_signature_counts(levels,9,4)
    write_csv(outputs/f"{STAMP}_follower_signature_counts.csv",list(fs[0].keys()),fs)
    # empirical
    rows=load_trace(root);prior=load_prior_counts(root);emp=empirical(rows,prior)
    idx={7:0,8:1,9:2}
    trans_rows=[]
    for i in STATES:
      for j in STATES:
        cnt=int(emp["counts"][idx[i],idx[j]])
        trans_rows.append({"from_state":i,"to_state":j,"count":cnt,"frequency":cnt/9999})
    write_csv(outputs/f"{STAMP}_empirical_joint_transition.csv",list(trans_rows[0].keys()),trans_rows)
    state_rows=[{"state":s,"count":emp["state_counts"][i],"frequency":emp["state_counts"][i]/10000} for i,s in enumerate(STATES)]
    defect_rows=[{"defect":d,"count":emp["defect_counts"][i],"frequency":emp["defect_counts"][i]/9999} for i,d in enumerate(DEFECTS)]
    write_csv(outputs/f"{STAMP}_state_frequencies.csv",list(state_rows[0].keys()),state_rows)
    write_csv(outputs/f"{STAMP}_defect_frequencies.csv",list(defect_rows[0].keys()),defect_rows)
    with (trace_dir/f"{STAMP}_finite_transition_trace.jsonl").open("w",encoding="utf-8") as f:
        for A in range(2,10001):
            rec={"A":A,"from":emp["carries"][A-1],"to":emp["carries"][A],"defect":emp["carries"][A]-emp["carries"][A-1]};f.write(json.dumps(rec,sort_keys=True)+"\n")
    # numerical matrices and metrics
    a=c["a"]
    J=[[mp.mpf(0),(1-3*a)/2,a/2],[(1-2*a)/4,mp.mpf(1)/4,a/2],[(1-2*a)/4,3*a/2-mp.mpf(1)/4,mp.mpf(0)]]
    rt=mp.sqrt(2);rho=1+rt
    K=[[mp.mpf(0),rt/(4*rho),1/(4*rho)],[rt/(4*rho),2/(4*rho),rt/(4*rho)],[1/(4*rho),rt/(4*rho),mp.mpf(0)]]
    comparison={"J":[[mp.nstr(x,80) for x in row] for row in J],"K_edge_envelope":[[mp.nstr(x,80) for x in row] for row in K],"empirical_joint":emp["joint"].tolist(),"metrics_empirical_vs_J":hp_metrics(emp["counts"],J),"metrics_empirical_vs_K_edge_envelope":hp_metrics(emp["counts"],K),"K_scope":"Parry joint measure of pairwise envelope only"}
    write_json(outputs/f"{STAMP}_edge_envelope_comparison.json",comparison)
    # finite boundary
    boundary=boundary_interval_certificate(rows)
    write_json(outputs/f"{STAMP}_finite_boundary_certificate.json",boundary)
    # abstract statuses
    structure={
      "complexity_theorem":"p(n)=2^(n+1)-1 for every n>=1",
      "entropy":"log(2)",
      "soficity":"PROVED NON-SOFIC",
      "follower_bridge":"PROVED FOR STANDARD HALF-OPEN WORD FOLLOWERS",
      "follower_identity":"Fol(w)={v : H_w intersect C(v) is nonempty}, H_w=D^|w|(C(w)); exact including all included endpoints",
      "soficity_argument":[
        "standard follower sets equal exact intersection languages of half-open follower regions",
        "equality of those languages forces equality of region interiors by the shrinking cylinder basis",
        "boundary-adjacent cylinders lie in one D^-n(p) fundamental gap and D^n preserves their order",
        "the ordered interior pair has oriented minus-to-plus handoff D^n(p), even when the outer endpoints also coincide",
        "repetition of an ordered follower-set pair forces D^n(p)=D^m(p)",
        "irrational p excludes eventual periodicity"
      ],
      "mixing":"PROVED TOPOLOGICALLY MIXING; independent of soficity proof",
      "mixing_argument":"every nonempty cylinder has open image; doubling sends every nonempty open arc onto the circle after finitely many iterates",
      "finite_markov_order":"NONE; finite Markov order implies SFT implies sofic; exact order 1..10 witnesses remain finite certificates",
      "holds":["SPECIFIC-ORBIT EQUIDISTRIBUTION: NOT PROVED","GLOBAL T_A=ceil(y_A) BRIDGE: NOT YET PROVED","GAUGE/FQM MAP FROM d_A=±1: NOT YET DERIVED"]
    }
    write_json(outputs/f"{STAMP}_language_structure_status.json",structure)
    write_json(outputs/f"{STAMP}_symbolic_core.json",sym)
    numerical={"constants":{k:mp.nstr(v,120) for k,v in c.items()},"a_outward_bounds":a_bounds,"assumptions":["1/6<a<1/4","a>3/14 at current constant"],"direct_depth":12}
    write_json(outputs/f"{STAMP}_numerical_core.json",numerical)
    summary={"step":"p5-b1-v7","direct_cylinders_depth12":sum(len(x) for x in levels),"length3_realizable":15,"complexity_n20":2**21-1,"standard_follower_bridge":"PROVED","soficity":"PROVED NON-SOFIC","finite_markov_order":"NONE","mixing":"PROVED INDEPENDENTLY","branch_status":"CLOSED","boundary_min":boundary["minimum_boundary_distance_lower_bound"],"empirical_vs_J":comparison["metrics_empirical_vs_J"],"empirical_vs_K":comparison["metrics_empirical_vs_K_edge_envelope"]}
    write_json(outputs/f"{STAMP}_run_summary.json",summary)
    return summary


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    print(json.dumps(run(args.root),indent=2,sort_keys=True))

if __name__=="__main__":main()
