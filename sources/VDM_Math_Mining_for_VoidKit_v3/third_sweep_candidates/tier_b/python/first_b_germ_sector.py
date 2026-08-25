from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class Germ:
    center: sp.Expr
    lower: sp.Expr
    upper: sp.Expr


def lawful_germ(theta: sp.Expr, u: int, v: int) -> Germ:
    a = sp.pi / (u * v)
    return Germ(theta, sp.simplify(theta - a), sp.simplify(theta + a))


def b_update_germ(germ: Germ, u: int, v: int) -> Germ:
    lam = sp.Rational(u, u + v)
    return Germ(
        germ.center,
        sp.simplify(germ.center + lam * (germ.lower - germ.center)),
        sp.simplify(germ.center + lam * (germ.upper - germ.center)),
    )


def components(germ: Germ) -> dict[str, sp.Expr]:
    op = sp.simplify(germ.upper - germ.center)
    om = sp.simplify(germ.lower - germ.center)
    return {
        'omega_plus': op,
        'omega_minus': om,
        'tau_plus_to_minus': sp.simplify(germ.lower - germ.upper),
        'tau_minus_to_plus': sp.simplify(germ.upper - germ.lower),
    }


def reflect(germ: Germ, x: sp.Expr) -> sp.Expr:
    return sp.simplify(2 * germ.center - x)


def recovered_rj_control() -> dict:
    zeta = np.exp(2j * np.pi / 24)
    splus = [0,1,2,3,4,5,7,8,9,10,11]
    sminus = [1,2,3,4,5,6,7,8,9,10,11]

    def f(s,p,r,c):
        return (-2*((r+s)%12)*((c+s)%12)+6*p)%24

    def op(s,p,plus=True):
        seats=splus if plus else sminus
        sign=1 if plus else -1
        return np.array([[zeta**((sign*f(s,p,r,c))%24) for c in seats] for r in seats],complex)

    C=np.zeros((11,11),complex)
    ri={s:i for i,s in enumerate(sminus)}
    ci={s:i for i,s in enumerate(splus)}
    for r in splus:
        C[ri[(r+6)%12],ci[r]]=(-1)**r
    D=np.conjugate(np.linalg.inv(C))
    Z=np.zeros_like(C)
    R=np.block([[Z,D],[C,Z]])
    J=np.block([[Z,-D],[C,Z]])

    def transfer(s,p,forward=True):
        rows=sminus if forward else splus
        cols=splus if forward else sminus
        M=np.zeros((11,11),complex)
        ri2={x:i for i,x in enumerate(rows)}
        ci2={x:i for i,x in enumerate(cols)}
        for r in range(12):
            c=(-r)%12
            if r in (0,6):
                continue
            if r in rows and c in cols:
                ph=(4*((r+s)%6)+6*p)%24
                if not forward:
                    ph=(-ph)%24
                M[ri2[r],ci2[c]]=zeta**ph
        return M

    def blockO(s,p):
        return np.block([[op(s,p,True),transfer(s,p,False)],
                         [transfer(s,p,True),op(s,p,False)]])

    rr=[]; jj=[]
    for s in range(6):
        for p in range(4):
            O=blockO(s,p)
            rr.append(float(np.linalg.norm(O@R-R@np.conjugate(O))))
            jj.append(float(np.linalg.norm(O@J-J@np.conjugate(O))**2))
    return {
        'states':24,
        'R_max_residual':max(rr),
        'J_squared_residual_min':min(jj),
        'J_squared_residual_max':max(jj),
        'R_pass_count':sum(v < 1e-10 for v in rr),
        'J_80_count':sum(abs(v-80.0) < 1e-9 for v in jj),
    }


def calculate() -> dict:
    theta=sp.Symbol('theta', real=True)
    u0=v0=1
    g0=lawful_germ(theta,u0,v0)
    g1=b_update_germ(g0,u0,v0)
    direct1=lawful_germ(theta,1,2)
    c0=components(g0); c1=components(g1)
    lam=sp.Rational(1,2)

    # Source-to-active B compatibility.
    source_active_commutes = (
        sp.simplify(g1.lower-direct1.lower)==0 and
        sp.simplify(g1.upper-direct1.upper)==0
    )

    # R reflection and B commutation, tested on a symbolic point x.
    x=sp.Symbol('x', real=True)
    rx=reflect(g0,x)
    b_rx=sp.simplify(theta+lam*(rx-theta))
    r_bx=sp.simplify(2*theta-(theta+lam*(x-theta)))
    r_commutes=sp.simplify(b_rx-r_bx)==0
    r_squared=sp.simplify(reflect(g0,reflect(g0,x))-x)==0
    r_exchanges=(sp.simplify(reflect(g0,g0.lower)-g0.upper)==0 and
                 sp.simplify(reflect(g0,g0.upper)-g0.lower)==0)

    # J type control: nonzero real germ displacement becomes imaginary.
    delta=sp.pi/2
    j_delta=sp.I*delta
    j_preserves_real = sp.im(j_delta)==0

    # Corrupted incoming relation with same primitive coordinates.
    corrupt=Germ(theta,theta-sp.Rational(3,2)*sp.pi,theta+sp.pi/4)
    corrupt_next=b_update_germ(corrupt,1,1)
    intrinsic_difference=sp.simplify(
        (corrupt_next.lower-g1.lower)**2+(corrupt_next.upper-g1.upper)**2
    )

    # Prior-state control: same selected B, different lawful center.
    theta2=theta+sp.pi/3
    prior2=lawful_germ(theta2,1,1)
    next2=b_update_germ(prior2,1,1)
    prior_state_difference=sp.simplify(
        (next2.lower-g1.lower)**2+(next2.upper-g1.upper)**2
    )

    # Fake metadata regeneration ignores incoming germ.
    fake_lawful=direct1
    fake_corrupt=direct1
    fake_collision=(fake_lawful.lower==fake_corrupt.lower and fake_lawful.upper==fake_corrupt.upper)
    intrinsic_separates=intrinsic_difference!=0

    recovered=recovered_rj_control()

    checks={
        'source_active_B_commutes':source_active_commutes,
        'first_B_half_width':sp.simplify((g1.upper-theta)-sp.pi/2)==0,
        'chart_counter_orientation':sp.simplify(c1['omega_plus']+c1['omega_minus'])==0,
        'plus_to_minus_transfer':sp.simplify(c1['tau_plus_to_minus']+sp.pi)==0,
        'minus_to_plus_transfer':sp.simplify(c1['tau_minus_to_plus']-sp.pi)==0,
        'all_four_scale_by_half':all(sp.simplify(c1[k]-lam*c0[k])==0 for k in c0),
        'R_squared_identity':r_squared,
        'R_exchanges_boundaries':r_exchanges,
        'R_commutes_with_B':r_commutes,
        'J_rejected_by_real_type':not bool(j_preserves_real),
        'intrinsic_update_separates_corruption':intrinsic_separates,
        'metadata_regeneration_collision_detected':fake_collision,
        'prior_state_changes_output':prior_state_difference!=0,
        'recovered_R_control':recovered['R_pass_count']==24,
        'recovered_J_control':recovered['J_80_count']==24,
    }

    def s(e): return str(sp.simplify(e))
    return {
        'scope':'completion-germ restriction of first complete B articulation',
        'inputs':{'q0':[1,1],'selected_primitive':'B','theta':'symbolic real'},
        'source_to_active':{
            'hatXi_identical_to_X_t':False,
            'B_compatible_extension':source_active_commutes,
        },
        'prior':{
            'a0':'pi',
            'germ0':['theta - pi','theta + pi'],
            **{k:s(v) for k,v in c0.items()},
        },
        'update':{
            'lambda':'1/2',
            'boundary_law':'x -> theta + (1/2)*(x-theta)',
        },
        'next':{
            'q1':[1,2],
            'a1':'pi/2',
            'germ1':['theta - pi/2','theta + pi/2'],
            **{k:s(v) for k,v in c1.items()},
        },
        'R_control':{
            'formula':'R_theta(x)=2*theta-x',
            'squared_identity':r_squared,
            'exchanges_boundaries':r_exchanges,
            'commutes_with_B':r_commutes,
        },
        'J_type_control':{
            'input_displacement':'pi/2',
            'output':'I*pi/2',
            'preserves_real_germ':bool(j_preserves_real),
        },
        'intrinsic_mutation_control':{
            'corrupt_prior':['theta - 3*pi/2','theta + pi/4'],
            'corrupt_next':[s(corrupt_next.lower),s(corrupt_next.upper)],
            'squared_difference_from_lawful':s(intrinsic_difference),
        },
        'prior_state_control':{
            'second_center':'theta + pi/3',
            'squared_output_difference':s(prior_state_difference),
        },
        'metadata_regeneration_control':{
            'fake_outputs_collide':fake_collision,
            'intrinsic_outputs_differ':intrinsic_separates,
        },
        'recovered_24_state_control':recovered,
        'checks':checks,
        'passed_checks':sum(int(bool(v)) for v in checks.values()),
        'total_checks':len(checks),
        'computed_verdict':'PASS' if all(bool(v) for v in checks.values()) else 'FAIL',
    }


def main():
    result=calculate()
    print(json.dumps(result,indent=2,sort_keys=True))
    if result['computed_verdict']!='PASS':
        raise SystemExit(1)

if __name__=='__main__':
    main()
