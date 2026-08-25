#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

PACKAGE_ZIP = Path('/mnt/data/phase5_v8p_canon_first_dual_chart_corrected_rerun_package.zip')
ROOT = Path('/mnt/data/v8p_reaudit/canon_first_dual_chart_corrected_rerun')
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from orthad_canon import run_once
from orthad_canon.application.compiler import apply_b, apply_l, apply_q
from orthad_canon.application.crossing import emit_floor_field, transport_after_l
from orthad_canon.domain.models import AxisState, DualChartLens, ExactEntry, LiftState
from orthad_canon.meta.verify import compare_evidence, evidence_gate

CORRECT_WORD = 'BQQBBBQBQBBQBBL'
EXPECTED_FLOOR_PAIR = (55, 89)
EXPECTED_NEXT_PAIR = (89, 144)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_matrix(matrix):
    return tuple(tuple(ExactEntry(0, 0) for _ in row) for row in matrix)


def replay_word(word: str) -> LiftState:
    s = LiftState(axes=[AxisState(1, 1)])
    for ch in word[:-1]:
        if ch == 'B':
            apply_b(s)
        elif ch == 'Q':
            apply_q(s)
        else:
            raise ValueError(ch)
    emit_floor_field(s)
    if word[-1] != 'L':
        raise ValueError('word must end in L')
    apply_l(s)
    transport_after_l(s)
    return s


def zero_pre_l_omegas_survival() -> bool:
    s = LiftState(axes=[AxisState(34, 55)])
    apply_b(s)
    assert s.lens is not None
    s.lens = replace(
        s.lens,
        omega_plus=zero_matrix(s.lens.omega_plus),
        omega_minus=zero_matrix(s.lens.omega_minus),
    )
    emit_floor_field(s)
    apply_l(s)
    transport_after_l(s)
    return evidence_gate(s)


def zero_post_l_lens_survival() -> bool:
    s = LiftState(axes=[AxisState(34, 55)])
    apply_b(s)
    emit_floor_field(s)
    apply_l(s)
    assert s.lens is not None
    z = zero_matrix(s.lens.pairing)
    s.lens = replace(
        s.lens,
        pairing=z,
        omega_plus=z,
        omega_minus=z,
        transfer_plus_to_minus=z,
        transfer_minus_to_plus=z,
    )
    transport_after_l(s)
    return evidence_gate(s)


def omega_minus_mismatch_count() -> tuple[int, int]:
    s = run_once()
    assert s.lens is not None
    supported = mismatches = 0
    for i, row in enumerate(s.lens.omega_minus):
        for j, e in enumerate(row):
            if e.support:
                supported += 1
                mismatches += int(e != s.lens.pairing[i][j])
    return supported, mismatches


def run_verifier_corruption_test(duplicate_rows: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / 'pkg'
        shutil.copytree(ROOT, dst)
        if duplicate_rows:
            path = dst / 'outputs/channel_readout_after.csv'
            with path.open(newline='') as f:
                rows = list(csv.DictReader(f))
                fields = rows[0].keys()
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows([rows[0]] * 12)
        else:
            (dst / 'outputs/dual_chart_matrices.csv').write_text('')
            (dst / 'outputs/ablation_results.csv').write_text('')
            (dst / 'outputs/ablation_per_channel_evidence.csv').write_text('')
            (dst / 'outputs/provenance_diff.csv').write_text('')
        cp = subprocess.run(
            [sys.executable, 'scripts/verify_evidence.py', '.'],
            cwd=dst,
            capture_output=True,
            text=True,
        )
        try:
            payload = json.loads(cp.stdout)
        except json.JSONDecodeError:
            payload = {'stdout': cp.stdout, 'stderr': cp.stderr}
        return {'exit_code': cp.returncode, 'payload': payload}


def main() -> None:
    baseline = run_once()
    correct = replay_word(CORRECT_WORD)
    correct_rows = compare_evidence(correct)

    frozen_before_next = (correct.axes[0].u, correct.axes[0].v)
    active_after_l = (correct.active.u, correct.active.v)
    apply_b(correct)
    package_next_pair = (correct.active.u, correct.active.v)

    baseline_supported, baseline_mismatches = omega_minus_mismatch_count()

    experiment_source = (ROOT / 'src/orthad_canon/application/experiment.py').read_text()
    compiler_source = (ROOT / 'src/orthad_canon/application/compiler.py').read_text()
    crossing_source = (ROOT / 'src/orthad_canon/application/crossing.py').read_text()
    models_source = (ROOT / 'src/orthad_canon/domain/models.py').read_text()

    results = {
        'subject': str(PACKAGE_ZIP),
        'zip_sha256': sha256(PACKAGE_ZIP),
        'disposition': 'REJECT_CANON_FIRST_CLAIM_RETAIN_NONCANONICAL_TABLE_REPRODUCTION_ONLY',
        'canonical_primitive_law': {
            'expected_first_crossing_word': CORRECT_WORD,
            'expected_floor_pair': list(EXPECTED_FLOOR_PAIR),
            'expected_pair_after_L': list(EXPECTED_FLOOR_PAIR),
            'expected_next_B_pair': list(EXPECTED_NEXT_PAIR),
        },
        'package_baseline': {
            'word': baseline.word,
            'event_sequence': [e['event'] for e in baseline.event_log],
            'frozen_pair': [baseline.axes[0].u, baseline.axes[0].v],
            'frozen_phase_mod4': baseline.axes[0].phase_mod4,
            'reported_axis': baseline.axes[0].lens_axis,
            'active_pair_after_L': [baseline.active.u, baseline.active.v],
            'character_match': f"{sum(r['survival'] for r in compare_evidence(baseline))}/{len(compare_evidence(baseline))}",
            'matrix_shape_before_L': [len(baseline.floor_lens.omega_plus), len(baseline.floor_lens.omega_plus[0])],
            'matrix_shape_after_L': [len(baseline.lens.omega_plus), len(baseline.lens.omega_plus[0])],
        },
        'canonical_word_replayed_through_package': {
            'word': correct.word[:-1],
            'frozen_pair_before_new_domain_step': list(frozen_before_next),
            'frozen_phase_mod4': correct.axes[0].phase_mod4,
            'reported_axis': correct.axes[0].lens_axis,
            'character_match_before_extra_B': f"{sum(r['survival'] for r in correct_rows)}/{len(correct_rows)}",
            'survival_gate': all(r['survival'] for r in correct_rows) and len(correct_rows) == 12,
            'active_pair_immediately_after_L': list(active_after_l),
            'package_next_B_pair': list(package_next_pair),
            'expected_next_B_pair': list(EXPECTED_NEXT_PAIR),
        },
        'new_decisive_findings': {
            'run_once_never_calls_Q': 'apply_q' not in experiment_source,
            'run_once_starts_at_34_55_not_1_1': '(34, 55)' in experiment_source,
            'no_dimension_counter_or_local_Q_capacity_in_state': all(token not in models_source for token in ['dimensional_counter', 'phase_position', 'global_position']),
            'FLOOR_is_synthetic_event': 'event": "FLOOR"' in crossing_source,
            'L_resets_active_pair_to_1_1': 'state.axes.append(AxisState(1, 1))' in compiler_source,
            'axis_label_hardcodes_i_independent_of_phase': 'return f"i/{self.uv}"' in models_source,
            'carrier_12_exists_before_L': [len(baseline.floor_lens.omega_plus), len(baseline.floor_lens.omega_plus[0])] == [12, 12],
            'L_does_not_grow_matrix_rank': len(baseline.floor_lens.omega_plus) == len(baseline.lens.omega_plus),
            'lap_relation_is_logged_string': '"lap_relation": "lap2=-lap1"' in compiler_source,
            'correct_word_breaks_12_of_12_claim': sum(r['survival'] for r in correct_rows) != 12,
        },
        'prior_findings_reproduced': {
            'zero_pre_L_omega_matrices_still_survives': zero_pre_l_omegas_survival(),
            'zero_entire_post_L_lens_still_survives': zero_post_l_lens_survival(),
            'omega_minus_supported_entries': baseline_supported,
            'omega_minus_mismatches_vs_pairing': baseline_mismatches,
            'verifier_accepts_destroyed_matrix_ablation_provenance_evidence': run_verifier_corruption_test(False),
            'verifier_accepts_twelve_duplicate_far_rows': run_verifier_corruption_test(True),
        },
        'retained_facts': {
            'package_is_reproducible_on_its_own_noncanonical_model': True,
            'local_B_arithmetic_34_55_to_55_89': [baseline.axes[0].u, baseline.axes[0].v] == [55, 89],
            'meta_reference_separation_present': True,
            'nonclaims_present': True,
        },
    }
    out = Path('/mnt/data/V8P_REAUDIT_RESULTS_v2.json')
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
