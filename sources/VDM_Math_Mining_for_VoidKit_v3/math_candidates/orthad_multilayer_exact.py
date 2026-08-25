from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from first_l_hierarchical_relation_grid import AxisRelation, PrimitiveState, ZERO, ONE

@dataclass(frozen=True)
class MultilayerOrthad:
    completed: Tuple[AxisRelation, ...]
    active: AxisRelation

    def b_update(self, u: int, v: int) -> "MultilayerOrthad":
        return MultilayerOrthad(self.completed, self.active.b_update(u, v))

    def q_update(self) -> "MultilayerOrthad":
        return MultilayerOrthad(self.completed, self.active.q_update())

    def l_update(self, next_A: int) -> "MultilayerOrthad":
        return MultilayerOrthad(
            self.completed + (self.active,),
            AxisRelation(f"domain_{next_A}", (ZERO, ONE)),
        )

    def step(self, primitive: str, before: PrimitiveState) -> "MultilayerOrthad":
        if primitive == "B":
            return self.b_update(before.u, before.v)
        if primitive == "Q":
            return self.q_update()
        if primitive == "L":
            return self.l_update(before.A + 1)
        raise ValueError(primitive)

    def layer_sizes(self) -> tuple[int, ...]:
        return tuple(len(layer.points) for layer in self.completed) + (len(self.active.points),)

    def relation_counts(self) -> dict[str, int]:
        sizes = self.layer_sizes()
        total_points = sum(sizes)
        square_sum = sum(n * n for n in sizes)
        within = sum(n * (n - 1) // 2 for n in sizes)
        directed_cross = total_points * total_points - square_sum
        return {
            "total_points": total_points,
            "within_edges": within,
            "directed_cross_placements": directed_cross,
            "total_relation_entries": within + directed_cross,
            "chart_point_entries": total_points,
            "transfer_entries_each_direction": total_points,
        }

    def transfer_square_residual(self) -> int:
        residual = 0
        for layer in self.completed + (self.active,):
            forward = layer.transfer_plus_to_minus()
            reverse = layer.transfer_minus_to_plus()
            residual += sum(int(reverse[forward[p]] != p) for p in layer.points)
        return residual


def initial_state() -> tuple[PrimitiveState, MultilayerOrthad]:
    return (
        PrimitiveState(0, 1, 1, 0, 0, 1, ""),
        MultilayerOrthad((), AxisRelation("domain_0", (ZERO, ONE))),
    )


def execute_ticks(ticks: int) -> tuple[PrimitiveState, MultilayerOrthad, list[dict]]:
    state, orthad = initial_state()
    checkpoints = []
    for tick in range(1, ticks + 1):
        primitive = state.selected()
        before = state
        orthad = orthad.step(primitive, before)
        state = state.step()
        if primitive == "L":
            checkpoints.append({
                "tick": tick,
                "A": state.A,
                "word_length": len(state.word),
                "layer_sizes": list(orthad.layer_sizes()),
                "relation_counts": orthad.relation_counts(),
                "transfer_square_residual": orthad.transfer_square_residual(),
            })
    return state, orthad, checkpoints
