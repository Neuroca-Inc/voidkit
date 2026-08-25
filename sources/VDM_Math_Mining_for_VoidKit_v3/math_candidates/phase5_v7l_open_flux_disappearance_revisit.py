#!/usr/bin/env python3
import json, math
from pathlib import Path
root = Path(__file__).resolve().parents[1]
summary = json.loads((root/'outputs'/'phase5_v7l_verification_summary.json').read_text())
checks = summary['hard_numbers']
assert abs(checks['remaining_fraction'] - 0.2) < 1e-12
assert abs(checks['closed_flux_wb'] - 3.2e8) < 1e-6
assert abs(checks['closure_time_hr'] - 2.7777777777777777) < 1e-6
print('PASS phase5_v7l_open_flux_disappearance_revisit')
print(json.dumps(summary, indent=2))
