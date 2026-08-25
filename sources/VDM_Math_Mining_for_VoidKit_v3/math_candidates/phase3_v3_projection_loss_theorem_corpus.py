from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT/'outputs/phase3_v3_verification_summary.json').read_text())
print('GLOBAL_PASS:', summary['global_pass'])
print('WITNESS_COUNT:', summary['witness_count'])
print('STATUS:', summary['status'])
