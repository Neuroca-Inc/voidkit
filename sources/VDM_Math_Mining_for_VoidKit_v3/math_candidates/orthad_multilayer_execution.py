from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_V21_SOURCE = Path(__file__).with_name('v21_core.py')


def load_v21(path: Path):
    spec = importlib.util.spec_from_file_location('v21_core', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load v21 source: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class HierarchicalRelationGrid:
    completed: Tuple[object, ...]
    active: object

    def layers(self) -> Tuple[object, ...]:
        return self.completed + (self.active,)

    def validate(self) -> bool:
        layers = self.layers()
        return (
            all(layer.validate() for layer in layers)
            and len({layer.name for layer in layers}) == len(layers)
        )

    def block(self, left: int, right: int):
        layers = self.layers()
        a = layers[left]
        b = layers[right]
        if left == right:
            return a.edges()
        return {(x, y) for x in a.points for y in b.points}

    def blocks(self) -> Dict[str, set]:
        n = len(self.layers())
        return {
            f'{i}_{j}': self.block(i, j)
            for i in range(n)
            for j in range(n)
        }

    def block_counts(self) -> Dict[str, int]:
        n = len(self.layers())
        return {
            f'{i}_{j}': len(self.block(i, j))
            for i in range(n)
            for j in range(n)
        }

    def factor_map(self, index: int, sign: str):
        layer = self.layers()[index]
        if sign == '+':
            return layer.plus_chart()
        if sign == '-':
            return layer.minus_chart()
        raise ValueError(sign)

    def chart_block(self, left: int, right: int, sign: str):
        lm = self.factor_map(left, sign)
        rm = self.factor_map(right, sign)
        return {(lm[x], rm[y]) for x, y in self.block(left, right)}

    def chart_count_residual(self) -> int:
        total = 0
        n = len(self.layers())
        for sign in ('+', '-'):
            for i in range(n):
                for j in range(n):
                    total += abs(len(self.chart_block(i, j, sign)) - len(self.block(i, j)))
        return total

    def transfer_map(self, index: int, direction: str):
        layer = self.layers()[index]
        if direction == '+->-':
            return layer.transfer_plus_to_minus()
        if direction == '-->+':
            return layer.transfer_minus_to_plus()
        raise ValueError(direction)

    def transfer_inverse_residual(self) -> int:
        total = 0
        for i, layer in enumerate(self.layers()):
            forward = self.transfer_map(i, '+->-')
            reverse = self.transfer_map(i, '-->+')
            total += sum(int(reverse[forward[p]] != p) for p in layer.points)
            total += sum(int(forward[reverse[p]] != p) for p in layer.points)
        return total

    def transfer_naturality_residual(self) -> int:
        total = 0
        n = len(self.layers())
        for i in range(n):
            ti = self.transfer_map(i, '+->-')
            for j in range(n):
                tj = self.transfer_map(j, '+->-')
                plus = self.chart_block(i, j, '+')
                transported = {(ti[x], tj[y]) for x, y in plus}
                minus = self.chart_block(i, j, '-')
                total += len(transported.symmetric_difference(minus))
        return total

    def b_update(self, u: int, v: int) -> 'HierarchicalRelationGrid':
        return HierarchicalRelationGrid(self.completed, self.active.b_update(u, v))

    def q_update(self) -> 'HierarchicalRelationGrid':
        return HierarchicalRelationGrid(self.completed, self.active.q_update())

    def l_update(self, new_active_name: str, zero, one) -> 'HierarchicalRelationGrid':
        new_seed = type(self.active)(new_active_name, (zero, one))
        return HierarchicalRelationGrid(self.completed + (self.active,), new_seed)

    def completed_unchanged(self, other: 'HierarchicalRelationGrid') -> bool:
        return self.completed == other.completed

    def old_block_embedding_residual(self, extended: 'HierarchicalRelationGrid') -> int:
        old_n = len(self.layers())
        residual = 0
        for i in range(old_n):
            for j in range(old_n):
                residual += len(self.block(i, j).symmetric_difference(extended.block(i, j)))
        return residual

    def summary(self):
        return {
            'completed_layers': len(self.completed),
            'active_layer': self.active.name,
            'layer_names': [layer.name for layer in self.layers()],
            'point_counts': [len(layer.points) for layer in self.layers()],
            'diagonal_edge_counts': [len(layer.edges()) for layer in self.layers()],
            'total_relation_entries': sum(self.block_counts().values()),
            'chart_count_residual': self.chart_count_residual(),
            'transfer_inverse_residual': self.transfer_inverse_residual(),
            'transfer_naturality_residual': self.transfer_naturality_residual(),
        }


def run(v21_source: Path, l_events: int):
    core = load_v21(v21_source)
    state = core.PrimitiveState(0, 1, 2, 1, 1, 2, 'BQ')
    hierarchy = HierarchicalRelationGrid(
        (),
        core.AxisRelation('domain_0', (core.ZERO, core.Fraction(1, 2) * core.I, core.I)),
    )

    checks = []
    l_records = []
    step_records = []

    def add_check(name: str, passed: bool, detail=None):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    add_check('initial_hierarchy_valid', hierarchy.validate())

    while len(l_records) < l_events:
        primitive = state.selected()
        before_state = state
        before_hierarchy = hierarchy

        if primitive == 'B':
            hierarchy = hierarchy.b_update(state.u, state.v)
            add_check(
                f'step_{len(state.word)+1}_B_completed_unchanged',
                before_hierarchy.completed_unchanged(hierarchy),
            )
        elif primitive == 'Q':
            hierarchy = hierarchy.q_update()
            add_check(
                f'step_{len(state.word)+1}_Q_completed_unchanged',
                before_hierarchy.completed_unchanged(hierarchy),
            )
        elif primitive == 'L':
            hierarchy = hierarchy.l_update(
                f'domain_{state.A + 1}', core.ZERO, core.ONE
            )
            event_number = len(l_records) + 1
            embedding_residual = before_hierarchy.old_block_embedding_residual(hierarchy)
            add_check(f'L{event_number}_old_block_embedding', embedding_residual == 0, embedding_residual)
            add_check(
                f'L{event_number}_completed_stack_exact',
                hierarchy.completed == before_hierarchy.layers(),
            )
            add_check(
                f'L{event_number}_new_seed_exact',
                hierarchy.active.points == (core.ZERO, core.ONE),
            )
            add_check(
                f'L{event_number}_all_mixed_blocks_cartesian',
                all(
                    len(hierarchy.block(i, j))
                    == len(hierarchy.layers()[i].points) * len(hierarchy.layers()[j].points)
                    for i in range(len(hierarchy.layers()))
                    for j in range(len(hierarchy.layers()))
                    if i != j
                ),
            )
            l_records.append({
                'event': event_number,
                'step': len(state.word) + 1,
                'pre_L_A': state.A,
                'completed_axis_name': before_hierarchy.active.name,
                'completed_point_count': len(before_hierarchy.active.points),
                'completed_edge_count': len(before_hierarchy.active.edges()),
                'pre_L_q': [state.u, state.v],
                'pre_L_k': state.k,
                'pre_L_j': state.j,
                'word_after_L': state.word + 'L',
                'embedding_residual': embedding_residual,
                'post_L_summary': hierarchy.summary(),
            })
        else:
            raise RuntimeError(primitive)

        state = state.step()
        add_check(f'step_{len(before_state.word)+1}_hierarchy_valid', hierarchy.validate())
        add_check(
            f'step_{len(before_state.word)+1}_chart_count_residual_zero',
            hierarchy.chart_count_residual() == 0,
            hierarchy.chart_count_residual(),
        )
        add_check(
            f'step_{len(before_state.word)+1}_transfer_inverse_residual_zero',
            hierarchy.transfer_inverse_residual() == 0,
            hierarchy.transfer_inverse_residual(),
        )
        add_check(
            f'step_{len(before_state.word)+1}_transfer_naturality_residual_zero',
            hierarchy.transfer_naturality_residual() == 0,
            hierarchy.transfer_naturality_residual(),
        )

        step_records.append({
            'step': len(before_state.word) + 1,
            'primitive': primitive,
            'A_after': state.A,
            'q_after': [state.u, state.v],
            'k_after': state.k,
            'j_after': state.j,
            'word_length_after': len(state.word),
            'completed_layers_after': len(hierarchy.completed),
            'active_points_after': len(hierarchy.active.points),
        })

    passed = sum(int(c['passed']) for c in checks)
    result = {
        'engine': 'generic finite-depth Orthad hierarchy execution from accepted v21 laws',
        'requested_L_events': l_events,
        'final_state': state.json(),
        'final_hierarchy': hierarchy.summary(),
        'L_events': l_records,
        'steps_executed_from_Xi2': len(step_records),
        'checks_passed': passed,
        'checks_total': len(checks),
        'computed_verdict': 'PASS' if passed == len(checks) else 'FAIL',
        'failed_checks': [c for c in checks if not c['passed']],
        'step_trace': step_records,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v21-source', type=Path, default=DEFAULT_V21_SOURCE)
    parser.add_argument('--l-events', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.l_events < 1:
        raise SystemExit('--l-events must be >= 1')
    result = run(args.v21_source, args.l_events)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    if result['computed_verdict'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
