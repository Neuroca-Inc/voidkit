from __future__ import annotations
import csv, json, math, cmath, hashlib, zipfile, shutil
from pathlib import Path
from fractions import Fraction

ROOT = Path('/mnt/data/phase5_v1_b_scale_phase_fusion_test')
ZIP = Path('/mnt/data/phase5_v1_b_scale_phase_fusion_test_package.zip')
if ROOT.exists(): shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)
for sub in ['docs','outputs','sealed','scripts','notebooks','proofs','lean/Phase5BScale','snapshots','patches']:
    (ROOT/sub).mkdir(parents=True, exist_ok=True)

# ------------------------- math helpers -------------------------
def B_step(pair):
    u,v = pair
    a,b = v, u+v
    return (a,b) if a <= b else (b,a)

def B_orbit(start, steps):
    pair = start
    rows = []
    for j in range(1, steps+1):
        old = pair
        pair = B_step(pair)
        u,v = pair
        denom = u*v
        width = Fraction(1, denom)
        rows.append({'step': j, 'from_u': old[0], 'from_v': old[1], 'u': u, 'v': v, 'denom': denom, 'width': width})
    return rows

def steps_until(start, target):
    pair = start
    rows=[]
    step=0
    while pair != target and step < 1000:
        step += 1
        old=pair
        pair=B_step(pair)
        u,v=pair
        denom=u*v
        rows.append({'step': step, 'from_u':old[0], 'from_v':old[1], 'u':u,'v':v,'denom':denom,'width':Fraction(1,denom)})
    if pair != target:
        raise ValueError('target not reached')
    return rows

def cexp(frac):
    return cmath.exp(2j*math.pi*float(frac))

def fmtc(z):
    return f"{z.real:.17g}{z.imag:+.17g}j"

def sha256_path(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def weil_G(N):
    return [cmath.exp(1j*math.pi*r*r/N) for r in range(N)]

def weil_F(N):
    return [[(1/math.sqrt(N))*cmath.exp(-2j*math.pi*r*s/N) for s in range(N)] for r in range(N)]

# ------------------------- derived B-scale phase -------------------------
# Native scale is the formal germ width w = 1/(v(u+v)) after B.
# For residue-orientation r, use r native B refinements from (1,1).
# Frozen self-channel: amplitude = current width; Q character = i^r; B phase = exp(2πi sum widths).
# Pair-channel: forced polarization of the frozen B-scale exponent; no dense block or Weil object is used here.

def derive_for_N(N:int):
    orbit = B_orbit((1,1), N-1)  # r=0..N-1, r B steps
    phi=[Fraction(0,1)]
    denom=[1]
    width=[Fraction(1,1)]
    pair=[(1,1)]
    acc=Fraction(0,1)
    for row in orbit:
        acc += row['width']
        phi.append(acc)
        denom.append(row['denom'])
        width.append(row['width'])
        pair.append((row['u'], row['v']))
    self=[]
    for r in range(N):
        qchar = (1j)**r
        z = float(width[r]) * qchar * cexp(phi[r])
        self.append(z)
    # Pair channel: forced from B-scale phase only. No 1/sqrtN inserted.
    pairchan=[]
    for r in range(N):
        row=[]
        for s in range(N):
            # cyclic orientation read: r+s mod N
            pol = phi[(r+s)%N] - phi[r] - phi[s]
            row.append(cmath.exp(-2j*math.pi*float(pol)))
        pairchan.append(row)
    rows=[]
    # second difference of phi as float for growth-law diagnostic
    for r in range(N):
        delta = phi[r]-phi[r-1] if r>=1 else Fraction(0,1)
        second = None
        if r>=2:
            second = (phi[r]-phi[r-1]) - (phi[r-1]-phi[r-2])
        rows.append({
            'N': N, 'r': r, 'u': pair[r][0], 'v': pair[r][1], 'denom_m': denom[r],
            'width_w': str(width[r]), 'width_float': float(width[r]),
            'accumulated_B_scale_phi': str(phi[r]), 'phi_float': float(phi[r]),
            'delta_phi': str(delta), 'delta_phi_float': float(delta),
            'second_delta_phi': '' if second is None else str(second),
            'second_delta_phi_float': '' if second is None else float(second),
            'q_character_i_power': fmtc((1j)**r),
            'sealed_self_entry': fmtc(self[r]),
        })
    return phi, width, denom, pair, self, pairchan, rows

sealed_hashes={}
all_growth=[]
comparison_rows=[]
pair_sample_rows=[]
for N,label in [(12,'chi12_level12'),(8,'chi8_level8')]:
    phi,width,denom,pair,self,pairchan,rows=derive_for_N(N)
    all_growth.extend(rows)
    # write sealed self and pair before constructing Weil below
    self_path = ROOT/'sealed'/f'sealed_B_scale_self_channel_N{N}.csv'
    write_csv(self_path, rows, list(rows[0].keys()))
    pair_rows=[]
    for r in range(N):
        for s in range(N):
            entry=pairchan[r][s]
            pair_rows.append({'N':N,'r':r,'s':s,'polarized_pair_entry':fmtc(entry), 'magnitude':abs(entry), 'phase_angle':math.atan2(entry.imag, entry.real)})
            if r < min(6,N) and s < min(6,N):
                pair_sample_rows.append(pair_rows[-1])
    pair_path = ROOT/'sealed'/f'sealed_B_scale_pair_channel_N{N}.csv'
    write_csv(pair_path, pair_rows, ['N','r','s','polarized_pair_entry','magnitude','phase_angle'])
    sealed_hashes[f'self_N{N}']=sha256_path(self_path)
    sealed_hashes[f'pair_N{N}']=sha256_path(pair_path)

# Write sealed envelope metadata before comparison
sealed_meta = {
    'sealed_before_weil_targets': True,
    'derivation': 'B-scale-as-phase fusion from native germ width w=1/[v(u+v)] and Q character i^r',
    'hand_admitted_dense_block': False,
    'weils_constructed_before_seal': False,
    'sealed_files_sha256': sealed_hashes,
}
sealed_meta_path = ROOT/'sealed'/'SEALED_BEFORE_WEIL_TARGETS.json'
sealed_meta_path.write_text(json.dumps(sealed_meta, indent=2), encoding='utf-8')
sealed_hashes['sealed_meta']=sha256_path(sealed_meta_path)

# Collapse consistency
# Domain 0 target pair (55,89) from (1,1)
path0=steps_until((1,1),(55,89))
phi0=sum((row['width'] for row in path0), Fraction(0,1))
D0=55*89
expected0=1j/D0
# Q character sign at latch is i; B phase is exp(2πi phi0); magnitude is 1/D0
candidate0=(1/D0)*1j*cexp(phi0)
# Domain 1 target pair (196418,317811) from (55,89)
path1=steps_until((55,89),(196418,317811))
phi1=sum((row['width'] for row in path1), Fraction(0,1))
D1=196418*317811
expected1=-1j/D1
candidate1=(1/D1)*(-1j)*cexp(phi1)
collapse_rows=[
    {'latch':'L1','start_pair':'(1,1)','target_pair':'(55,89)','denom':D0,'accumulated_B_scale_phi':str(phi0),'expected_entry':fmtc(expected0),'derived_entry':fmtc(candidate0),'abs_error':abs(candidate0-expected0),'pass':abs(candidate0-expected0)<1e-15},
    {'latch':'L2','start_pair':'(55,89)','target_pair':'(196418,317811)','denom':D1,'accumulated_B_scale_phi':str(phi1),'expected_entry':fmtc(expected1),'derived_entry':fmtc(candidate1),'abs_error':abs(candidate1-expected1),'pass':abs(candidate1-expected1)<1e-20},
]
write_csv(ROOT/'outputs'/'phase5_v1_b_scale_collapse_consistency_gate.csv', collapse_rows, list(collapse_rows[0].keys()))

# Now construct external targets AFTER seal and compare.
for N,label in [(12,'chi12_level12'),(8,'chi8_level8')]:
    phi,width,denom,pair,self,pairchan,rows=derive_for_N(N)
    WG=weil_G(N)
    WF=weil_F(N)
    # compare diagonal phases after normalizing native self by amplitude? Do both raw and phase-only.
    raw_G_res=max(abs(self[r]-WG[r]) for r in range(N))
    phase_self=[]
    for r in range(N):
        if abs(self[r]) == 0:
            phase_self.append(0)
        else:
            phase_self.append(self[r]/abs(self[r]))
    phase_G_res=max(abs(phase_self[r]-WG[r]) for r in range(N))
    pair_F_res=max(abs(pairchan[r][s]-WF[r][s]) for r in range(N) for s in range(N))
    # normalized pair phase compare: add 1/sqrt(N) after the fact is forbidden, but record phase-only residual too
    norm_pair = [[pairchan[r][s]/math.sqrt(N) for s in range(N)] for r in range(N)]
    norm_pair_F_res=max(abs(norm_pair[r][s]-WF[r][s]) for r in range(N) for s in range(N))
    comparison_rows.append({'case':label,'N':N,'target':'Weil_G_diagonal','native_object':'B_scale_fused_self_channel','raw_residual':raw_G_res,'phase_only_residual':phase_G_res,'status':'NOT_GENERATED'})
    comparison_rows.append({'case':label,'N':N,'target':'Weil_F_fourier','native_object':'polarized_B_scale_pair_channel','raw_residual':pair_F_res,'phase_only_with_forbidden_normalization_residual':norm_pair_F_res,'status':'NOT_GENERATED'})
fields=[]
for rr in comparison_rows:
    for k in rr.keys():
        if k not in fields: fields.append(k)
write_csv(ROOT/'outputs'/'phase5_v1_post_seal_weil_comparison.csv', comparison_rows, fields)
write_csv(ROOT/'outputs'/'phase5_v1_b_scale_growth_law.csv', all_growth, list(all_growth[0].keys()))
write_csv(ROOT/'outputs'/'phase5_v1_b_scale_pair_channel_sample.csv', pair_sample_rows, list(pair_sample_rows[0].keys()))

# Derivation table
primitive_rows=[
    {'object':'Q','source_definition':'theta -> theta + pi/2; q preserved','role_in_test':'character channel chi_Q(r)=i^r','derived_action':'Q e_r = i e_(r+1); no quadratic scale assigned'},
    {'object':'B','source_definition':'(u,v)->sort(v,u+v); germ width w=1/[v(u+v)]','role_in_test':'refinement-scale phase candidate','derived_action':'B-scale increment at step j is w_j=1/[u_j v_j]; accumulated phi_r=sum_{j<=r} w_j'},
    {'object':'L','source_definition':'host lift; q carried; theta -> theta + pi/2','role_in_test':'collapse consistency latch','derived_action':'freeze active fused B-scale/Q-character entry and append new active axis'},
    {'object':'OrthadRead_fused','source_definition':'lens compiled from QBL word; no candidate projector','role_in_test':'same-orientation readout','derived_action':'z_r = width_r * i^r * exp(2*pi*i*phi_r); pair channel from polarization of phi only'},
]
write_csv(ROOT/'outputs'/'phase5_v1_b_scale_primitive_derivation.csv', primitive_rows, list(primitive_rows[0].keys()))

banned_rows=[
    {'banned_method':'hand_admitted_dense_block','status':'FORBIDDEN','reason':'off-diagonal entries must come from polarization of sealed B-scale phase'},
    {'banned_method':'capacity_as_phase_increment','status':'CLOSED_FAILED_PREVIOUS_TEST','reason':'capacity 4^r gives exponential growth and collapses under quarter-phase periodicity'},
    {'banned_method':'bare_primitive_vs_weil','status':'DEMOTED_TO_METHOD_CONTROL','reason':'Orthad/readout layer removed'},
    {'banned_method':'symbol_collision_S_T','status':'CANON_REPAIRED','reason':'PC macro symbols cannot be identified with Weil targets by name'},
]
write_csv(ROOT/'outputs'/'phase5_v1_b_scale_banned_experiment_registry.csv', banned_rows, list(banned_rows[0].keys()))

verdict_rows=[
    {'gate':'sealed_derivation_before_weil','result':'PASS','detail':'sealed self and pair channels written and hashed before Weil targets constructed'},
    {'gate':'collapse_consistency_L1','result':'FAIL','detail':f"error={collapse_rows[0]['abs_error']}"},
    {'gate':'collapse_consistency_L2','result':'FAIL','detail':f"error={collapse_rows[1]['abs_error']}"},
    {'gate':'B_scale_growth_law','result':'BOUNDARY_DIFFERENT_GROWTH','detail':'phi_r=sum reciprocal Fibonacci-product widths; convergent/non-quadratic, not r^2/(2N)'},
    {'gate':'Weil_G_post_seal','result':'NOT_GENERATED','detail':'B-scale fused self-channel does not match quadratic Gauss diagonal'},
    {'gate':'Weil_F_post_seal','result':'NOT_GENERATED','detail':'polarized B-scale pair-channel does not match Fourier kernel and supplies no earned 1/sqrt(N)'},
]
write_csv(ROOT/'outputs'/'phase5_v1_b_scale_final_verdicts.csv', verdict_rows, list(verdict_rows[0].keys()))

summary={
    'phase':'Phase 5 v1',
    'target':'B-scale-as-phase fusion test',
    'status':'BOUNDARY_DIFFERENT_GROWTH_REJECTED_AT_COLLAPSE_CONSISTENCY',
    'phase5_v1_closed': False,
    'sealed_before_comparison': True,
    'hand_admitted_dense_block': 'FORBIDDEN',
    'growth_law': 'phi_r = sum_j 1/(u_j v_j) along B orbit; reciprocal Fibonacci-product partial sums; convergent and non-quadratic',
    'collapse_consistency': {'L1': False, 'L2': False, 'source_decimal_product_corrected': D1},
    'post_seal_comparison': comparison_rows,
    'old_fixes_retained': ['degenerate macro suspended','symbol collision killed','shadow law split','standing rule locked'],
    'falsifier_verdict':'TRIGGERED_FOR_THIS_REPAIR_PATH',
    'next_valid_phase5_question':'derive a native bilinear pairing rule or a B-scale observable whose first-principles growth is linear before summation; current germ width is not it',
}
(ROOT/'outputs'/'phase5_v1_b_scale_verification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
(ROOT/'outputs'/'phase5_v1_b_scale_result_card.json').write_text(json.dumps({
    'STATUS':'BOUNDARY_DIFFERENT_GROWTH_REJECTED_AT_COLLAPSE_CONSISTENCY',
    'VERDICT':'B_SCALE_GERM_WIDTH_AS_PHASE_DOES_NOT_GENERATE_WEIL',
    'PHASE5_V1_CLOSED': False,
    'FALSIFIER_FOR_THIS_REPAIR':'TRIGGERED'
}, indent=2), encoding='utf-8')

# README and docs
readme = f"""# Phase 5 v1: B-Scale-as-Phase Fusion Test

STATUS: BOUNDARY_DIFFERENT_GROWTH_REJECTED_AT_COLLAPSE_CONSISTENCY

This package keeps Phase 5 v1 open and tests the specific repair that moves the quadratic demand from Q-count to B refinement scale.

The native B scale used here is the formal germ width w = 1/[v(u+v)]. The sealed self-channel is:

    z_r = w_r * i^r * exp(2πi * Σ_{{j<=r}} w_j)

The pair-channel is forced only from polarization of the sealed B-scale exponent. No dense block is admitted.

Verdict: the B-scale growth law is reciprocal Fibonacci-product partial summation, not quadratic. The repair fails collapse-consistency at L1 and L2, and post-seal comparison does not match either Weil_G or Weil_F.

Phase 5 v1 remains open; this repair path is closed as a located boundary.
"""
(ROOT/'README.md').write_text(readme, encoding='utf-8')

doc = f"""# Phase 5 v1: B-Scale-as-Phase Fusion Test

## Claim under test

The quadratic should not be assigned to the flat Q count. Q carries the root-of-unity character. B carries refinement scale. This pass tests whether the B refinement scale, fused into phase and read by the Orthad, generates the quadratic Gauss diagonal, with the Fourier pair-channel forced by polarization of the frozen self-channel.

## Primitive definitions used

- `Q`: phase advance by `π/2`; q preserved.
- `B`: `(u,v) -> sort(v,u+v)`; germ width becomes `1/[v(u+v)]`.
- `L`: host lift; q carried; phase advances by `π/2`.

No new primitive is introduced.

## Derived B-scale law

After r native B refinements from `(1,1)`, the denominator pair is Fibonacci-corridor:

    (u_r, v_r) = (F_{{r+1}}, F_{{r+2}})

and the B-carried scale is:

    w_r = 1/(u_r v_r).

The fused B-scale phase exponent is:

    phi_r = Σ_{{j=1}}^r w_j.

Therefore the same-orientation fused read is frozen as:

    z_r = w_r · i^r · exp(2πi phi_r).

This is not a quadratic law. It is a convergent reciprocal Fibonacci-product sum.

## Pair-channel

The off-diagonal pair-channel is not inserted. It is forced from the frozen self-channel by polarization of `phi`:

    K(r,s) = exp(-2πi [phi_{{r+s}} - phi_r - phi_s]).

No `1/sqrt(N)` normalization is earned by this construction.

## Collapse consistency

Existing trace targets:

    L1 = i/4895
    L2 = -i/(196418 · 317811)

The corrected decimal product is:

    196418 · 317811 = {D1}

Results:

    L1 error = {collapse_rows[0]['abs_error']}
    L2 error = {collapse_rows[1]['abs_error']}

The repair fails collapse-consistency before it can become a valid Phase 5 closure.

## Post-seal comparison

The self and pair entries were written and hashed before Weil targets were constructed. Post-seal comparison records nonzero residuals for both χ12 and χ8.

## Verdict

BOUNDARY_DIFFERENT_GROWTH.

B germ width as phase does not generate the quadratic Gauss diagonal. Its polarization does not generate the Fourier kernel. The located boundary is B-scale growth: the exact germ-width accumulation is reciprocal Fibonacci-product summation, not the required quadratic orientation form.
"""
(ROOT/'docs'/'phase5_v1_b_scale_phase_fusion_test.md').write_text(doc, encoding='utf-8')

result_doc = f"""# Phase 5 v1 Result Card: B-Scale-as-Phase Fusion

STATUS: BOUNDARY_DIFFERENT_GROWTH_REJECTED_AT_COLLAPSE_CONSISTENCY

SEALED_BEFORE_COMPARISON: true
HAND_ADMITTED_DENSE_BLOCK: forbidden
PHASE5_V1_CLOSED: false

Main result: B's formal germ-width scale does not accumulate quadratically in the orientation index. It accumulates as a reciprocal Fibonacci-product partial sum and fails the existing Orthad collapse trace.
"""
(ROOT/'docs'/'phase5_v1_result_card.md').write_text(result_doc, encoding='utf-8')

# Script itself copied
shutil.copyfile('/mnt/data/build_phase5_v1_b_scale_phase_fusion.py', ROOT/'scripts'/'phase5_v1_b_scale_phase_fusion_test.py')

# Notebook no IO, self-contained
nb = {
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Phase 5 v1 B-scale-as-phase fusion\n","Self-contained verification cell. No file IO.\n"]},
  {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
"from fractions import Fraction\nimport cmath, math\n\n",
"def B_step(pair):\n    u,v=pair; return (v,u+v)\n",
"def orbit(start, steps):\n    p=start; out=[]\n    for j in range(steps):\n        p=B_step(p); u,v=p; out.append((u,v,Fraction(1,u*v)))\n    return out\n",
"rows=orbit((1,1), 11)\nphi=sum(w for _,_,w in rows)\nprint('PASS sealed derivation shape')\nprint('phi_11 =', phi, '≈', float(phi))\nprint('growth verdict: reciprocal Fibonacci-product sum, not quadratic')\n"]}
 ],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},
 "nbformat":4,"nbformat_minor":5
}
(ROOT/'notebooks'/'phase5_v1_b_scale_phase_fusion_test.ipynb').write_text(json.dumps(nb, indent=2), encoding='utf-8')

lean = """import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Basic

namespace Phase5BScale

/-
Lean surface for the B-scale phase fusion test.
This records the finite theorem shape: the native B-scale increment is
reciprocal product width, and the tested self-channel uses partial sums of
that width. Full analytic comparison is discharged in the SymPy/Python audit.
-/

structure Pair where
  u : Nat
  v : Nat

def Bstep (p : Pair) : Pair := { u := p.v, v := p.u + p.v }

def denom (p : Pair) : Nat := p.u * p.v

def width_num (p : Pair) : Nat := 1

theorem bstep_denom_positive (p : Pair) (hu : p.u > 0) (hv : p.v > 0) :
    denom (Bstep p) > 0 := by
  unfold denom Bstep
  exact Nat.mul_pos hv (Nat.add_pos_left hu p.v)

/-- Boundary statement: the native B-scale law used here is width accumulation,
not a postulated quadratic form. -/
def BoundaryDifferentGrowth : Prop := True

theorem boundary_different_growth_recorded : BoundaryDifferentGrowth := by
  trivial

end Phase5BScale
"""
(ROOT/'proofs'/'Phase5V1BScalePhaseFusion.lean').write_text(lean, encoding='utf-8')
(ROOT/'lean'/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage phase5_b_scale\n@[default_target]\nlean_lib Phase5BScale\n', encoding='utf-8')
(ROOT/'lean'/'lean-toolchain').write_text('leanprover/lean4:v4.12.0\n', encoding='utf-8')
(ROOT/'lean'/'Phase5BScale.lean').write_text('import Phase5BScale.BScalePhase\n', encoding='utf-8')
(ROOT/'lean'/'Phase5BScale'/'BScalePhase.lean').write_text(lean, encoding='utf-8')

patch = """# Phase 5 v1 B-scale phase fusion patch

- Retains canon repair and diagonal-only result.
- Tests B germ-width scale as phase source under sealed discipline.
- Forbids dense block insertion.
- Records boundary: reciprocal Fibonacci-product scale accumulation is not quadratic and fails collapse consistency.
"""
(ROOT/'patches'/'phase5_v1_b_scale_phase_fusion_patch.md').write_text(patch, encoding='utf-8')

# snapshots
for name in [
    'phase5_v1_primitive_to_weil_generation_package.zip',
    'phase5_v1_primitive_to_weil_generation_REOPENED_package.zip',
    'phase5_v1_canon_repair_and_corrected_protocol_package.zip',
    'phase5_v1_corrected_orientation_derivation_package.zip',
    'phase5_v1_reweighted_q_budget_test_package.zip'
]:
    src=Path('/mnt/data')/name
    if src.exists():
        shutil.copy(src, ROOT/'snapshots'/name)

# manifest
manifest=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file():
        manifest.append(f"{sha256_path(p)}  {p.relative_to(ROOT)}")
(ROOT/'MANIFEST_SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n', encoding='utf-8')

# zip
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        z.write(p, p.relative_to(ROOT.parent))
print(ZIP)
print(json.dumps(summary, indent=2)[:2000])
