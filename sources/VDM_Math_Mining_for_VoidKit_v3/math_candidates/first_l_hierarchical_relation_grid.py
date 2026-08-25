from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True, order=True)
class ExactComplex:
    re: Fraction
    im: Fraction

    def __add__(self, other: "ExactComplex") -> "ExactComplex":
        return ExactComplex(self.re + other.re, self.im + other.im)

    def __sub__(self, other: "ExactComplex") -> "ExactComplex":
        return ExactComplex(self.re - other.re, self.im - other.im)

    def __mul__(self, other):
        if isinstance(other, int):
            other = Fraction(other, 1)
        if isinstance(other, Fraction):
            return ExactComplex(self.re * other, self.im * other)
        if isinstance(other, ExactComplex):
            return ExactComplex(
                self.re * other.re - self.im * other.im,
                self.re * other.im + self.im * other.re,
            )
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def as_pair(self) -> tuple[str, str]:
        return (fraction_text(self.re), fraction_text(self.im))

    def text(self) -> str:
        if self.im == 0:
            return fraction_text(self.re)
        if self.re == 0:
            if self.im == 1:
                return "i"
            if self.im == -1:
                return "-i"
            return f"{fraction_text(self.im)}*i"
        sign = "+" if self.im > 0 else "-"
        magnitude = self.im if self.im > 0 else -self.im
        imag = "i" if magnitude == 1 else f"{fraction_text(magnitude)}*i"
        return f"{fraction_text(self.re)}{sign}{imag}"


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


ZERO = ExactComplex(Fraction(0), Fraction(0))
ONE = ExactComplex(Fraction(1), Fraction(0))
I = ExactComplex(Fraction(0), Fraction(1))


@dataclass(frozen=True)
class AxisRelation:
    name: str
    points: Tuple[ExactComplex, ...]

    def validate(self) -> bool:
        return (
            len(self.points) >= 2
            and self.points[0] == ZERO
            and len(set(self.points)) == len(self.points)
        )

    def edges(self) -> set[tuple[ExactComplex, ExactComplex]]:
        return {
            (self.points[left], self.points[right])
            for left in range(len(self.points))
            for right in range(left + 1, len(self.points))
        }

    def plus_chart(self) -> dict[ExactComplex, ExactComplex]:
        return {point: point for point in self.points}

    def minus_chart(self) -> dict[ExactComplex, ExactComplex]:
        return {
            point: self.points[len(self.points) - 1 - index]
            for index, point in enumerate(self.points)
        }

    def transfer_plus_to_minus(self) -> dict[ExactComplex, ExactComplex]:
        return compose(self.minus_chart(), invert(self.plus_chart()))

    def transfer_minus_to_plus(self) -> dict[ExactComplex, ExactComplex]:
        return compose(self.plus_chart(), invert(self.minus_chart()))

    def active_determination(self) -> ExactComplex:
        return self.points[1]

    def q_update(self) -> "AxisRelation":
        return AxisRelation(self.name, tuple(I * point for point in self.points))

    def b_update(self, u: int, v: int) -> "AxisRelation":
        if not self.validate():
            raise ValueError("invalid incoming axis relation")
        factor = Fraction(u, u + v)
        if not Fraction(0) < factor < Fraction(1):
            raise ValueError("B factor must lie strictly between zero and one")
        active = self.active_determination()
        new_point = factor * active
        if new_point in self.points:
            raise ValueError("B failed to create a new determination")
        return AxisRelation(
            self.name,
            (self.points[0], new_point) + self.points[1:],
        )

    def json(self) -> dict:
        return {
            "name": self.name,
            "points": [point.text() for point in self.points],
            "edges": sorted([[a.text(), b.text()] for a, b in self.edges()]),
            "plus_chart": map_json(self.plus_chart()),
            "minus_chart": map_json(self.minus_chart()),
            "transfer_plus_to_minus": map_json(self.transfer_plus_to_minus()),
            "transfer_minus_to_plus": map_json(self.transfer_minus_to_plus()),
        }


@dataclass(frozen=True)
class PrimitiveState:
    A: int
    u: int
    v: int
    quarter_turns: int
    k: int
    j: int
    word: str

    def phase_positions(self) -> int:
        return 6 * (2 ** self.A)

    def capacity(self) -> int:
        if self.j == 1:
            return 2
        if self.j == 2:
            return 4
        return 2 ** (2 * self.j)

    def next_pair(self) -> tuple[int, int]:
        return self.v, self.u + self.v

    def can_b(self) -> bool:
        next_u, next_v = self.next_pair()
        if self.k < self.phase_positions() - 1:
            return next_u * next_v <= self.capacity()
        return self.u * self.v < self.capacity()

    def can_q(self) -> bool:
        return self.k < self.phase_positions() - 1

    def selected(self) -> str:
        if self.can_b():
            return "B"
        if self.can_q():
            return "Q"
        return "L"

    def step(self) -> "PrimitiveState":
        primitive = self.selected()
        if primitive == "B":
            next_u, next_v = self.next_pair()
            return PrimitiveState(
                self.A,
                next_u,
                next_v,
                self.quarter_turns,
                self.k,
                self.j,
                self.word + "B",
            )
        if primitive == "Q":
            return PrimitiveState(
                self.A,
                self.u,
                self.v,
                self.quarter_turns + 1,
                self.k + 1,
                self.j + 1,
                self.word + "Q",
            )
        next_A = self.A + 1
        next_j = 1 + 6 * (2 ** next_A - 1)
        return PrimitiveState(
            next_A,
            self.u,
            self.v,
            self.quarter_turns,
            0,
            next_j,
            self.word + "L",
        )

    def json(self) -> dict:
        return {
            "A": self.A,
            "q": [self.u, self.v],
            "theta": f"theta_0+{self.quarter_turns}*pi/2",
            "quarter_turns": self.quarter_turns,
            "k": self.k,
            "j": self.j,
            "word": self.word,
            "phase_positions": self.phase_positions(),
            "capacity": self.capacity(),
            "next_pair": list(self.next_pair()),
            "selected": self.selected(),
        }


@dataclass(frozen=True)
class RelationGrid:
    old: AxisRelation
    active: AxisRelation

    def blocks(self) -> dict[str, set[tuple[ExactComplex, ExactComplex]]]:
        return {
            "old_old": self.old.edges(),
            "old_new": {
                (old_point, new_point)
                for old_point in self.old.points
                for new_point in self.active.points
            },
            "new_old": {
                (new_point, old_point)
                for new_point in self.active.points
                for old_point in self.old.points
            },
            "new_new": self.active.edges(),
        }

    def block_counts(self) -> dict[str, int]:
        return {name: len(values) for name, values in self.blocks().items()}

    def b_update(self, u: int, v: int) -> "RelationGrid":
        return RelationGrid(self.old, self.active.b_update(u, v))

    def q_update(self) -> "RelationGrid":
        return RelationGrid(self.old, self.active.q_update())

    def old_embedding(self) -> dict[ExactComplex, tuple[str, ExactComplex]]:
        return {point: ("old", point) for point in self.old.points}

    def old_recovery(self) -> dict[tuple[str, ExactComplex], ExactComplex]:
        return {("old", point): point for point in self.old.points}

    def old_point_recovery_residual(self) -> int:
        embedding = self.old_embedding()
        recovery = self.old_recovery()
        return sum(int(recovery[embedding[point]] != point) for point in self.old.points)

    def old_edge_recovery_residual(self) -> int:
        recovered = {
            (left, right)
            for left, right in self.blocks()["old_old"]
        }
        return len(recovered.symmetric_difference(self.old.edges()))

    def factor_maps(self, sign: str) -> tuple[dict, dict]:
        if sign == "+":
            return self.old.plus_chart(), self.active.plus_chart()
        if sign == "-":
            return self.old.minus_chart(), self.active.minus_chart()
        raise ValueError(sign)

    def chart_blocks(self, sign: str) -> dict[str, set[tuple[ExactComplex, ExactComplex]]]:
        old_map, active_map = self.factor_maps(sign)
        result: dict[str, set[tuple[ExactComplex, ExactComplex]]] = {}
        for name, block in self.blocks().items():
            if name == "old_old":
                result[name] = {(old_map[a], old_map[b]) for a, b in block}
            elif name == "new_new":
                result[name] = {(active_map[a], active_map[b]) for a, b in block}
            elif name == "old_new":
                result[name] = {(old_map[a], active_map[b]) for a, b in block}
            else:
                result[name] = {(active_map[a], old_map[b]) for a, b in block}
        return result

    def transfer_maps(self) -> dict[str, dict[str, dict[ExactComplex, ExactComplex]]]:
        return {
            "plus_to_minus": {
                "old": self.old.transfer_plus_to_minus(),
                "active": self.active.transfer_plus_to_minus(),
            },
            "minus_to_plus": {
                "old": self.old.transfer_minus_to_plus(),
                "active": self.active.transfer_minus_to_plus(),
            },
        }

    def transfer_square_residual(self) -> int:
        total = 0
        maps = self.transfer_maps()
        for axis_name, points in [("old", self.old.points), ("active", self.active.points)]:
            forward = maps["plus_to_minus"][axis_name]
            reverse = maps["minus_to_plus"][axis_name]
            total += sum(int(reverse[forward[point]] != point) for point in points)
        return total

    def transfer_block_naturality_residual(self) -> int:
        plus = self.chart_blocks("+")
        minus = self.chart_blocks("-")
        old_transfer = self.old.transfer_plus_to_minus()
        active_transfer = self.active.transfer_plus_to_minus()
        transformed: dict[str, set[tuple[ExactComplex, ExactComplex]]] = {}
        for name, block in plus.items():
            if name == "old_old":
                transformed[name] = {(old_transfer[a], old_transfer[b]) for a, b in block}
            elif name == "new_new":
                transformed[name] = {(active_transfer[a], active_transfer[b]) for a, b in block}
            elif name == "old_new":
                transformed[name] = {(old_transfer[a], active_transfer[b]) for a, b in block}
            else:
                transformed[name] = {(active_transfer[a], old_transfer[b]) for a, b in block}
        return sum(len(transformed[name].symmetric_difference(minus[name])) for name in plus)

    def json(self) -> dict:
        return {
            "old": self.old.json(),
            "active": self.active.json(),
            "block_counts": self.block_counts(),
            "blocks": {
                name: sorted([[a.text(), b.text()] for a, b in values])
                for name, values in self.blocks().items()
            },
            "plus_chart_block_counts": {
                name: len(values) for name, values in self.chart_blocks("+").items()
            },
            "minus_chart_block_counts": {
                name: len(values) for name, values in self.chart_blocks("-").items()
            },
            "transfer_maps": {
                direction: {
                    axis: map_json(mapping)
                    for axis, mapping in axis_maps.items()
                }
                for direction, axis_maps in self.transfer_maps().items()
            },
            "old_point_recovery_residual": self.old_point_recovery_residual(),
            "old_edge_recovery_residual": self.old_edge_recovery_residual(),
            "transfer_square_residual": self.transfer_square_residual(),
            "transfer_block_naturality_residual": self.transfer_block_naturality_residual(),
        }


def invert(mapping: dict) -> dict:
    inverse = {value: key for key, value in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("map is not bijective")
    return inverse


def compose(after: dict, before: dict) -> dict:
    return {key: after[before[key]] for key in before}


def map_json(mapping: dict[ExactComplex, ExactComplex]) -> dict[str, str]:
    return {key.text(): value.text() for key, value in mapping.items()}


def run_to_first_l() -> tuple[list[dict], PrimitiveState, AxisRelation]:
    state = PrimitiveState(0, 1, 2, 1, 1, 2, "BQ")
    axis = AxisRelation("domain_0", (ZERO, Fraction(1, 2) * I, I))
    trace: list[dict] = []
    while True:
        primitive = state.selected()
        before = state
        before_axis = axis
        after = state.step()
        if primitive == "B":
            axis = axis.b_update(before.u, before.v)
        elif primitive == "Q":
            axis = axis.q_update()
        trace.append(
            {
                "step": len(before.word) + 1,
                "primitive": primitive,
                "before": before.json(),
                "after": after.json(),
                "before_active": before_axis.active_determination().text(),
                "after_active": axis.active_determination().text(),
                "determination_count": len(axis.points),
                "relation_edge_count": len(axis.edges()),
            }
        )
        state = after
        if primitive == "L":
            return trace, state, axis


def calculate() -> dict:
    trace, post_l_state, completed_old = run_to_first_l()
    expected_word = "BQQBBBQBQBBQBBL"
    expected_old_points = (
        ZERO,
        Fraction(1, 4895) * I,
        Fraction(1, 1870) * I,
        Fraction(1, 714) * I,
        Fraction(1, 273) * I,
        Fraction(1, 104) * I,
        Fraction(1, 40) * I,
        Fraction(1, 15) * I,
        Fraction(1, 6) * I,
        Fraction(1, 2) * I,
        I,
    )

    new_seed = AxisRelation("domain_1_active", (ZERO, ONE))
    first_l_grid = RelationGrid(completed_old, new_seed)

    post_l_b_primitive = post_l_state.selected()
    post_l_b_state = post_l_state.step()
    post_l_b_grid = first_l_grid.b_update(post_l_state.u, post_l_state.v)

    post_l_q_primitive = post_l_b_state.selected()
    post_l_q_state = post_l_b_state.step()
    post_l_q_grid = post_l_b_grid.q_update()

    old_identity_after_b = len(
        first_l_grid.old.edges().symmetric_difference(post_l_b_grid.old.edges())
    )
    old_identity_after_q = len(
        first_l_grid.old.edges().symmetric_difference(post_l_q_grid.old.edges())
    )
    old_chart_identity_after_b = sum(
        int(first_l_grid.old.minus_chart()[point] != post_l_b_grid.old.minus_chart()[point])
        for point in first_l_grid.old.points
    )
    old_chart_identity_after_q = sum(
        int(first_l_grid.old.minus_chart()[point] != post_l_q_grid.old.minus_chart()[point])
        for point in first_l_grid.old.points
    )

    # Negative controls.
    corrupted_points = completed_old.points[:-1]
    corrupted_old = AxisRelation("domain_0", corrupted_points)
    corrupted_grid = RelationGrid(corrupted_old, new_seed)
    corrupted_recovery_difference = len(
        completed_old.edges().symmetric_difference(corrupted_grid.old.edges())
    )

    missing_mixed = set(first_l_grid.blocks()["old_new"])
    missing_mixed.pop()
    missing_mixed_residual = len(
        missing_mixed.symmetric_difference(first_l_grid.blocks()["old_new"])
    )

    wrong_old_q = first_l_grid.old.q_update()
    wrong_q_grid = RelationGrid(wrong_old_q, post_l_b_grid.active.q_update())
    wrong_old_q_residual = len(
        first_l_grid.old.edges().symmetric_difference(wrong_q_grid.old.edges())
    )

    alternative_active = AxisRelation(
        "alternative",
        (ZERO, Fraction(2, 5) * I, I),
    )
    lawful_actual_b = AxisRelation(
        "actual",
        (ZERO, Fraction(1, 2) * I, I),
    ).b_update(2, 3)
    lawful_alternative_b = alternative_active.b_update(2, 3)
    prior_state_dependence = len(
        set(lawful_actual_b.points).symmetric_difference(set(lawful_alternative_b.points))
    )

    # Downstream comparison quantities.
    hierarchy_rows = []
    for A in range(6):
        phase_positions = 6 * (2 ** A)
        layer_count = A + 1
        exact_log_count = 1 + (phase_positions // 6).bit_length() - 1
        hierarchy_rows.append(
            {
                "A": A,
                "phase_positions": phase_positions,
                "layer_count": layer_count,
                "one_plus_log2_ratio": exact_log_count,
                "residual": layer_count - exact_log_count,
            }
        )

    conservation_trace = [
        {
            "state": "after_L",
            "old_points": len(first_l_grid.old.points),
            "old_edges": len(first_l_grid.old.edges()),
            "old_minus_map_entries": len(first_l_grid.old.minus_chart()),
        },
        {
            "state": "after_post_L_B",
            "old_points": len(post_l_b_grid.old.points),
            "old_edges": len(post_l_b_grid.old.edges()),
            "old_minus_map_entries": len(post_l_b_grid.old.minus_chart()),
        },
        {
            "state": "after_post_L_Q",
            "old_points": len(post_l_q_grid.old.points),
            "old_edges": len(post_l_q_grid.old.edges()),
            "old_minus_map_entries": len(post_l_q_grid.old.minus_chart()),
        },
    ]
    conservation_difference = sum(
        abs(row[key] - conservation_trace[0][key])
        for row in conservation_trace[1:]
        for key in ["old_points", "old_edges", "old_minus_map_entries"]
    )

    checks = {
        "path_word_exact": post_l_state.word == expected_word,
        "path_state_exact": (
            post_l_state.A == 1
            and (post_l_state.u, post_l_state.v) == (55, 89)
            and post_l_state.quarter_turns == 5
            and post_l_state.k == 0
            and post_l_state.j == 7
        ),
        "completed_old_points_exact": completed_old.points == expected_old_points,
        "completed_old_edges_exact": len(completed_old.edges()) == 55,
        "completed_old_transfer_order_two": (
            compose(
                completed_old.transfer_minus_to_plus(),
                completed_old.transfer_plus_to_minus(),
            )
            == completed_old.plus_chart()
        ),
        "new_seed_exact": new_seed.points == (ZERO, ONE) and len(new_seed.edges()) == 1,
        "first_l_block_counts_exact": first_l_grid.block_counts()
        == {"old_old": 55, "old_new": 22, "new_old": 22, "new_new": 1},
        "first_l_recovery_exact": (
            first_l_grid.old_point_recovery_residual() == 0
            and first_l_grid.old_edge_recovery_residual() == 0
        ),
        "first_l_chart_counts_exact": (
            first_l_grid.json()["plus_chart_block_counts"] == first_l_grid.block_counts()
            and first_l_grid.json()["minus_chart_block_counts"] == first_l_grid.block_counts()
        ),
        "first_l_transfer_exact": (
            first_l_grid.transfer_square_residual() == 0
            and first_l_grid.transfer_block_naturality_residual() == 0
        ),
        "post_l_B_selected": post_l_b_primitive == "B",
        "post_l_B_state_exact": (
            (post_l_b_state.u, post_l_b_state.v) == (89, 144)
            and post_l_b_state.word == expected_word + "B"
        ),
        "post_l_B_active_exact": post_l_b_grid.active.points
        == (ZERO, Fraction(55, 144) * ONE, ONE),
        "post_l_B_old_preserved": (
            old_identity_after_b == 0 and old_chart_identity_after_b == 0
        ),
        "post_l_Q_selected": post_l_q_primitive == "Q",
        "post_l_Q_state_exact": (
            post_l_q_state.quarter_turns == 6
            and post_l_q_state.k == 1
            and post_l_q_state.j == 8
            and post_l_q_state.word == expected_word + "BQ"
        ),
        "post_l_Q_active_exact": post_l_q_grid.active.points
        == (ZERO, Fraction(55, 144) * I, I),
        "post_l_Q_old_preserved": (
            old_identity_after_q == 0 and old_chart_identity_after_q == 0
        ),
        "post_l_Q_transfer_exact": (
            post_l_q_grid.transfer_square_residual() == 0
            and post_l_q_grid.transfer_block_naturality_residual() == 0
        ),
        "corrupted_old_detected": corrupted_recovery_difference > 0,
        "missing_mixed_detected": missing_mixed_residual == 1,
        "wrong_old_Q_detected": wrong_old_q_residual > 0,
        "actual_prior_dependence": prior_state_dependence > 0,
        "hierarchy_projection_exact_on_tested_domains": all(
            row["residual"] == 0 for row in hierarchy_rows
        ),
        "discrete_old_layer_conservation": conservation_difference == 0,
    }

    result = {
        "research_question": (
            "Does the first internally selected L generate an exact retained old layer, "
            "a new active relation layer, the complete four-block relation grid, and "
            "a bounded post-L B/Q recurrence?"
        ),
        "scope": {
            "interaction": "p5-b3-v21",
            "first_L_complete": True,
            "first_post_L_B_complete": True,
            "first_post_L_Q_complete": True,
            "all_depth_L_recurrence": "OPEN",
            "scalar_mixed_weights": "NOT CLAIMED; mixed blocks are typed placement relations",
        },
        "primitive_trace": trace,
        "post_L_state": post_l_state.json(),
        "completed_old": completed_old.json(),
        "first_L_grid": first_l_grid.json(),
        "post_L_B": {
            "primitive": post_l_b_primitive,
            "state": post_l_b_state.json(),
            "grid": post_l_b_grid.json(),
            "old_edge_residual": old_identity_after_b,
            "old_chart_residual": old_chart_identity_after_b,
        },
        "post_L_Q": {
            "primitive": post_l_q_primitive,
            "state": post_l_q_state.json(),
            "grid": post_l_q_grid.json(),
            "old_edge_residual": old_identity_after_q,
            "old_chart_residual": old_chart_identity_after_q,
        },
        "controls": {
            "corrupted_old_recovery_difference": corrupted_recovery_difference,
            "missing_mixed_residual": missing_mixed_residual,
            "wrong_old_Q_residual": wrong_old_q_residual,
            "lawful_prior_state_output_difference": prior_state_dependence,
        },
        "hierarchy_projection": hierarchy_rows,
        "conservation_trace": conservation_trace,
        "conservation_difference": conservation_difference,
        "checks": checks,
        "passed_checks": sum(int(value) for value in checks.values()),
        "total_checks": len(checks),
        "computed_verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    return result


if __name__ == "__main__":
    result = calculate()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["computed_verdict"] != "PASS":
        raise SystemExit(1)
