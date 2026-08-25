from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_csv(name):
    with (ROOT/'outputs'/name).open(newline='') as f:
        return list(csv.DictReader(f))

summary = json.loads((ROOT/'outputs'/'phase5_v1_reopened_verification_summary.json').read_text())
gates = read_csv('phase5_v1_corrected_protocol_gate_matrix.csv')

print(json.dumps({
    'status': summary['status'],
    'corrected_claim_verdict': summary['corrected_claim_verdict'],
    'corrected_boundary_triggered': summary['corrected_boundary_triggered'],
    'stop_point': summary['where_corrected_test_stops'],
    'failed_required_gates': [g['gate'] for g in gates if g['status'] == 'FAIL'],
    'not_run_gates': [g['gate'] for g in gates if g['status'] == 'NOT_RUN'],
}, indent=2))
