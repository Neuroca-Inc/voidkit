#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
summary = json.loads((root/'outputs'/'phase5_v7k_verification_summary.json').read_text())
print('STATUS:', summary['status'])
print('GLOBAL_PASS:', summary['global_pass'])
print('MECHANISM_ROWS:', summary['mechanism_rows'])
print('CROSSWALK_ROWS:', summary['crosswalk_rows'])
print('PHASE5_CLOSED:', summary['phase5_closed'])
