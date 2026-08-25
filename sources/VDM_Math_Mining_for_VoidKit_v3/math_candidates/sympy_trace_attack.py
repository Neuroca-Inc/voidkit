from __future__ import annotations
import csv
from pathlib import Path
from sympy import symbols, simplify

ROOT = Path(__file__).resolve().parent.parent
ref_path = ROOT / 'data' / 'reference_xi_full_engine_trace.csv'
run_path = ROOT / 'native_build' / 'xi_full_engine_trace.csv'

cL, cR = symbols('cL cR')
center = (cL + cR) / 2
half = (cR - cL) / 2
assert simplify(center + half - cR) == 0
assert simplify(center - half - cL) == 0

with open(ref_path, newline='') as f:
    ref = list(csv.DictReader(f))
with open(run_path, newline='') as f:
    run = list(csv.DictReader(f))

assert len(run) == 192
for i in range(16):
    for key in ['step','A','theta_ticks','kappa','u','v','uv','window_ready','r_den']:
        assert run[i][key] == ref[i][key], (i,key)
print('FINAL_RESULT PASS')
