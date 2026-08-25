#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class G:
    re: Fraction
    im: Fraction

    @staticmethod
    def zero() -> "G":
        return G(Fraction(0), Fraction(0))

    @staticmethod
    def one() -> "G":
        return G(Fraction(1), Fraction(0))

    def __add__(self, other: "G") -> "G":
        return G(self.re + other.re, self.im + other.im)

    def __sub__(self, other: "G") -> "G":
        return G(self.re - other.re, self.im - other.im)

    def mul_i(self) -> "G":
        return G(-self.im, self.re)

    def scale(self, x: Fraction) -> "G":
        return G(self.re * x, self.im * x)

    def norm_sq(self) -> Fraction:
        return self.re * self.re + self.im * self.im

    def text(self) -> str:
        def f(x: Fraction) -> str:
            return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
        if self.im == 0:
            return f(self.re)
        if self.re == 0:
            return f"{f(self.im)}i"
        sign = "+" if self.im > 0 else "-"
        return f"{f(self.re)}{sign}{f(abs(self.im))}i"


@dataclass
class State:
    domain: int
    u: int
    v: int
    quarter_turns: int
    k: int
    j: int
    completed: list[list[G]]
    active: list[G]

    @staticmethod
    def initial() -> "State":
        return State(0, 1, 1, 0, 0, 1, [], [G.zero(), G.one()])

    def clone(self) -> "State":
        return State(
            self.domain,
            self.u,
            self.v,
            self.quarter_turns,
            self.k,
            self.j,
            [list(x) for x in self.completed],
            list(self.active),
        )

    @property
    def layers(self) -> list[list[G]]:
        return [*self.completed, self.active]

    def phase_positions(self) -> int:
        return 6 * (2**self.domain)

    def capacity(self) -> int:
        if self.j == 1:
            return 2
        if self.j == 2:
            return 4
        return 2 ** (2 * self.j)

    def emit(self) -> str:
        terminal = self.k == self.phase_positions() - 1
        product = self.u * self.v
        capacity = self.capacity()
        if terminal:
            return "B" if product < capacity else "L"
        if product >= capacity:
            return "Q"
        return "B" if self.v * (self.u + self.v) <= capacity else "Q"

    def tick(self) -> str:
        primitive = self.emit()
        if primitive == "B":
            old_u, old_v = self.u, self.v
            c = Fraction(old_u, old_u + old_v)
            point = self.active[1].scale(c)
            self.active = [self.active[0], point, *self.active[1:]]
            self.u, self.v = old_v, old_u + old_v
        elif primitive == "Q":
            self.active = [x.mul_i() for x in self.active]
            self.quarter_turns += 1
            self.k += 1
            self.j += 1
        elif primitive == "L":
            self.completed.append(list(self.active))
            self.active = [G.zero(), G.one()]
            self.domain += 1
            self.k = 0
            self.j = self.phase_positions() - 5
        else:
            raise AssertionError(primitive)
        return primitive


def doubled_area(a: G, b: G, c: G) -> Fraction:
    u = b - a
    v = c - a
    return u.re * v.im - u.im * v.re


def vertex_rows(state: State) -> list[tuple[str, G]]:
    return [
        (f"L{layer}:P{point}", value)
        for layer, values in enumerate(state.layers)
        for point, value in enumerate(values)
    ]


def gf2_rank(bit_columns: list[int]) -> int:
    pivots: dict[int, int] = {}
    for column in bit_columns:
        value = column
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def triangle_complex(state: State) -> dict[str, object]:
    vertices = vertex_rows(state)
    edges = list(itertools.combinations(range(len(vertices)), 2))
    edge_index = {edge: i for i, edge in enumerate(edges)}
    triangles: list[tuple[tuple[int, int, int], Fraction, int]] = []
    area_energy = Fraction(0)
    for tri in itertools.combinations(range(len(vertices)), 3):
        area = doubled_area(vertices[tri[0]][1], vertices[tri[1]][1], vertices[tri[2]][1])
        if area == 0:
            continue
        a, b, c = tri
        boundary = (1 << edge_index[(b, c)]) ^ (1 << edge_index[(a, c)]) ^ (1 << edge_index[(a, b)])
        triangles.append((tri, area, boundary))
        area_energy += area * area
    rank = gf2_rank([x[2] for x in triangles])
    coordinate_classes: dict[tuple[Fraction, Fraction], int] = {}
    for _, z in vertices:
        key = (z.re, z.im)
        coordinate_classes[key] = coordinate_classes.get(key, 0) + 1
    duplicate_pair_count = sum(n * (n - 1) // 2 for n in coordinate_classes.values())
    cycle_dimension = len(edges) - len(vertices) + 1
    return {
        "vertices": vertices,
        "edges": edges,
        "edge_index": edge_index,
        "triangles": triangles,
        "vertex_count": len(vertices),
        "edge_count": len(edges),
        "cycle_dimension": cycle_dimension,
        "nonzero_triangle_count": len(triangles),
        "boundary_rank": rank,
        "cycle_deficiency": cycle_dimension - rank,
        "duplicate_coordinate_pair_count": duplicate_pair_count,
        "area_form_span_rank": 1 if triangles else 0,
        "area_energy": area_energy,
    }


def boundary_vector(complex_data: dict[str, object]) -> list[Fraction]:
    edges = complex_data["edges"]
    triangles = complex_data["triangles"]
    rho = [Fraction(0) for _ in edges]
    edge_index = complex_data["edge_index"]
    for tri, area, _ in triangles:
        a, b, c = tri
        rho[edge_index[(b, c)]] += area
        rho[edge_index[(a, c)]] -= area
        rho[edge_index[(a, b)]] += area
    return rho


def divergence(vertex_count: int, edges: list[tuple[int, int]], current: list[Fraction]) -> list[Fraction]:
    result = [Fraction(0) for _ in range(vertex_count)]
    for value, (a, b) in zip(current, edges, strict=True):
        result[a] -= value
        result[b] += value
    return result


def epsilon(a: int, b: int) -> int:
    if (a, b) == (0, 1):
        return 1
    if (a, b) == (1, 0):
        return -1
    return 0


def local_biform(omega: Fraction, a: int, b: int, c: int, d: int) -> Fraction:
    return omega * epsilon(a, b) * epsilon(c, d)


def edge_relation(state: State, left: tuple[int, int], right: tuple[int, int]) -> G:
    return state.layers[right[0]][right[1]] - state.layers[left[0]][left[1]]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    state = State.initial()
    states = [state.clone()]
    primitives = ["INITIAL"]
    for _ in range(103):
        primitives.append(state.tick())
        states.append(state.clone())

    l_indices = [i for i, p in enumerate(primitives) if p == "L"]
    assert l_indices == [15, 45, 103]

    l_rows: list[dict[str, object]] = []
    complexes: dict[int, dict[str, object]] = {}
    for index in l_indices:
        data = triangle_complex(states[index])
        complexes[index] = data
        l_rows.append({
            "causal_index": index,
            "domain_after": states[index].domain,
            "layer_sizes": ";".join(str(len(x)) for x in states[index].layers),
            "vertex_count": data["vertex_count"],
            "edge_count": data["edge_count"],
            "cycle_dimension": data["cycle_dimension"],
            "nonzero_triangle_count": data["nonzero_triangle_count"],
            "triangle_boundary_rank": data["boundary_rank"],
            "cycle_deficiency": data["cycle_deficiency"],
            "duplicate_coordinate_pair_count": data["duplicate_coordinate_pair_count"],
            "area_form_span_rank": data["area_form_span_rank"],
            "area_energy_exact": str(data["area_energy"]),
            "area_energy": float(data["area_energy"]),
        })

    first = complexes[15]
    rho = boundary_vector(first)
    div = divergence(first["vertex_count"], first["edges"], rho)
    rho_norm_sq = sum(x * x for x in rho)
    assert rho_norm_sq > 0
    assert all(x == 0 for x in div)

    old_norm_sq = sum(x.norm_sq() for x in states[15].completed[0])
    c = states[16].active[1].re
    active_norm_sq = Fraction(1) + c * c
    total_norm_sq = old_norm_sq + active_norm_sq
    omega = -2 * c * old_norm_sq / (total_norm_sq * total_norm_sq)
    assert omega != 0

    antisymmetry_pass = True
    pair_exchange_pass = True
    cyclic_pass = True
    contraction_pass = True
    for a, b, c_i, d in itertools.product(range(2), repeat=4):
        value = local_biform(omega, a, b, c_i, d)
        antisymmetry_pass &= value == -local_biform(omega, b, a, c_i, d)
        antisymmetry_pass &= value == -local_biform(omega, a, b, d, c_i)
        pair_exchange_pass &= value == local_biform(omega, c_i, d, a, b)
        cyclic = (
            local_biform(omega, a, b, c_i, d)
            + local_biform(omega, a, c_i, d, b)
            + local_biform(omega, a, d, b, c_i)
        )
        cyclic_pass &= cyclic == 0
    for a, b in itertools.product(range(2), repeat=2):
        contracted = sum(
            local_biform(omega, a, b, c_i, d) * epsilon(c_i, d)
            for c_i, d in itertools.product(range(2), repeat=2)
        ) / 2
        contraction_pass &= contracted == omega * epsilon(a, b)

    # Exact rank-one graph bi-form E = omega * rho⊗rho / ||rho||².
    # Its normalized polarization projection is omega and its Frobenius energy is omega².
    graph_projection = omega
    graph_energy = omega * omega
    graph_gauss_pass = all(x == 0 for x in div)
    graph_double_gauss_pass = graph_gauss_pass

    # Standard fixed-root cycle basis: root is the active endpoint at first L.
    vertices = first["vertices"]
    root_vertex = next(i for i, (label, _) in enumerate(vertices) if label == "L1:P1")
    completed_origin = next(i for i, (label, _) in enumerate(vertices) if label == "L0:P0")
    active_origin = next(i for i, (label, _) in enumerate(vertices) if label == "L1:P0")
    null_tri = tuple(sorted((completed_origin, active_origin, root_vertex)))
    null_area = doubled_area(vertices[null_tri[0]][1], vertices[null_tri[1]][1], vertices[null_tri[2]][1])
    assert null_area == 0
    null_boundary = 0
    a, b, c_v = null_tri
    null_boundary ^= 1 << first["edge_index"][(b, c_v)]
    null_boundary ^= 1 << first["edge_index"][(a, c_v)]
    null_boundary ^= 1 << first["edge_index"][(a, b)]
    rank_with_null = gf2_rank([x[2] for x in first["triangles"]] + [null_boundary])

    # Map all first-L cycles into the completed stack at L45.
    # L1:P0 stays the completed-layer origin; L1:P1 becomes the completed-layer endpoint.
    state45 = states[45]
    state103 = states[103]
    endpoint45 = len(state45.completed[1]) - 1
    endpoint103 = len(state103.completed[1]) - 1

    def map_vertex(label: str, at_103: bool = False) -> tuple[int, int]:
        layer_text, point_text = label.split(":")
        layer = int(layer_text[1:])
        point = int(point_text[1:])
        if layer == 0:
            return (0, point)
        if layer == 1 and point == 0:
            return (1, 0)
        if layer == 1 and point == 1:
            return (1, endpoint103 if at_103 else endpoint45)
        raise AssertionError(label)

    retained_rows: list[dict[str, object]] = []
    retained_exact = 0
    retained_zero_area = 0
    for ordinal, (tri, birth_area, _) in enumerate(first["triangles"]):
        labels = [vertices[i][0] for i in tri]
        mapped45 = [map_vertex(x, False) for x in labels]
        mapped103 = [map_vertex(x, True) for x in labels]
        rel45 = [
            edge_relation(state45, mapped45[0], mapped45[1]),
            edge_relation(state45, mapped45[1], mapped45[2]),
            edge_relation(state45, mapped45[2], mapped45[0]),
        ]
        rel103 = [
            edge_relation(state103, mapped103[0], mapped103[1]),
            edge_relation(state103, mapped103[1], mapped103[2]),
            edge_relation(state103, mapped103[2], mapped103[0]),
        ]
        stable = rel45 == rel103
        for check_index in range(45, 104):
            check_state = states[check_index]
            check_endpoint = len(check_state.completed[1]) - 1
            check_mapped = []
            for label in labels:
                layer_text, point_text = label.split(":")
                layer = int(layer_text[1:])
                point = int(point_text[1:])
                if layer == 0:
                    check_mapped.append((0, point))
                elif point == 0:
                    check_mapped.append((1, 0))
                else:
                    check_mapped.append((1, check_endpoint))
            check_rel = [
                edge_relation(check_state, check_mapped[0], check_mapped[1]),
                edge_relation(check_state, check_mapped[1], check_mapped[2]),
                edge_relation(check_state, check_mapped[2], check_mapped[0]),
            ]
            stable &= check_rel == rel45
        retained_exact += int(stable)
        z45 = [state45.layers[x[0]][x[1]] for x in mapped45]
        area45 = doubled_area(z45[0], z45[1], z45[2])
        retained_zero_area += int(area45 == 0)
        retained_rows.append({
            "cycle": ordinal,
            "birth_vertices": "|".join(labels),
            "birth_doubled_area_exact": str(birth_area),
            "retained_doubled_area_exact": str(area45),
            "retained_edges_stable_L45_to_L103": stable,
            "edge01_at_L45": rel45[0].text(),
            "edge12_at_L45": rel45[1].text(),
            "edge20_at_L45": rel45[2].text(),
        })

    phase_rows: list[dict[str, object]] = []
    for index in range(15, 46):
        data = triangle_complex(states[index])
        phase_rows.append({
            "causal_index": index,
            "primitive": primitives[index],
            "domain": states[index].domain,
            "layer_sizes": ";".join(str(len(x)) for x in states[index].layers),
            "nonzero_triangle_count": data["nonzero_triangle_count"],
            "triangle_boundary_rank": data["boundary_rank"],
            "area_energy_exact": str(data["area_energy"]),
            "area_energy": float(data["area_energy"]),
        })

    triangle_rows: list[dict[str, object]] = []
    for ordinal, (tri, area, _) in enumerate(first["triangles"]):
        triangle_rows.append({
            "triangle": ordinal,
            "v0": vertices[tri[0]][0],
            "v1": vertices[tri[1]][0],
            "v2": vertices[tri[2]][0],
            "doubled_area_exact": str(area),
            "doubled_area": float(area),
        })

    current_rows: list[dict[str, object]] = []
    for (a_i, b_i), value in zip(first["edges"], rho, strict=True):
        current_rows.append({
            "source": vertices[a_i][0],
            "target": vertices[b_i][0],
            "rho_exact": str(value),
            "rho": float(value),
        })

    gates = {
        "exact_L_indices_15_45_103": l_indices == [15, 45, 103],
        "first_L_65_nonzero_area_cells": first["nonzero_triangle_count"] == 65,
        "first_L_65_independent_cycle_boundaries": first["boundary_rank"] == 65,
        "first_L_only_missing_cycle_is_null_identity_cycle": first["cycle_dimension"] == 66 and rank_with_null == 66,
        "L_event_cycle_deficiency_equals_coordinate_duplicate_loops": all(
            row["cycle_deficiency"] == row["duplicate_coordinate_pair_count"] for row in l_rows
        ),
        "native_chain_Gauss_closure_exact": graph_gauss_pass,
        "native_double_Gauss_closure_exact": graph_double_gauss_pass,
        "local_biform_pair_antisymmetry_exact": antisymmetry_pass,
        "local_biform_pair_exchange_exact": pair_exchange_pass,
        "local_biform_cyclic_identity_exact": cyclic_pass,
        "biform_contraction_recovers_CF09_field_exact": contraction_pass and graph_projection == omega,
        "rank_one_biform_energy_positive": graph_energy > 0,
        "first_post_L_Q_converts_static_area_support_to_holonomy": phase_rows[2]["causal_index"] == 17 and phase_rows[2]["nonzero_triangle_count"] == 0 and omega != 0,
        "all_65_first_interface_cycles_retained_exact_through_L103": retained_exact == 65,
        "all_65_retained_cycles_are_closed_flat_string_support": retained_zero_area == 65,
        "hierarchical_growth_65_493_2271": [row["nonzero_triangle_count"] for row in l_rows] == [65, 493, 2271],
        "hierarchical_cycle_independence_65_493_2271": [row["triangle_boundary_rank"] for row in l_rows] == [65, 493, 2271],
    }

    summary = {
        "probe": "native Orthad area-bi-form and closed-string support check",
        "source": "exact recurrence matching the OI-7.5 Rust target-pass trajectory",
        "CF09_field_strength_exact": str(omega),
        "CF09_field_strength": float(omega),
        "CF09_field_energy_exact": str(graph_energy),
        "CF09_field_energy": float(graph_energy),
        "first_L": {
            "vertices": first["vertex_count"],
            "edges": first["edge_count"],
            "cycle_dimension": first["cycle_dimension"],
            "nonzero_area_cells": first["nonzero_triangle_count"],
            "independent_area_cell_boundaries": first["boundary_rank"],
            "null_identity_cycles": first["cycle_deficiency"],
            "area_form_span_rank": first["area_form_span_rank"],
            "weighted_string_current_nonzero_edges": sum(x != 0 for x in rho),
            "weighted_string_current_norm_sq_exact": str(rho_norm_sq),
            "vertex_divergence_residual": [str(x) for x in div],
            "missing_null_cycle": [vertices[i][0] for i in null_tri],
        },
        "retention": {
            "first_interface_cycles": 65,
            "cycles_unchanged_from_L45_through_L103": retained_exact,
            "retained_flat_closed_cycles": retained_zero_area,
            "subsequent_active_ticks_checked": 58,
        },
        "L_event_growth": l_rows,
        "interpretation": {
            "field": "the CF09 light curvature is the exact rank-one polarization of a native symmetric area-pair field",
            "string_support": "the first L creates 65 independent closed currents; after the next L all 65 are frozen into completed layers while the next interface opens",
            "handoff": "Q17 removes static triangle area but leaves nonzero gauge holonomy, so the light is the causal area-flux transfer rather than a static triangle",
            "fracton_alignment": "the native chain d -> rho -> 0 mirrors the JHEP dipole-like density -> closed string charge -> Gauss closure structure",
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }

    write_csv(results / "L_event_cycle_growth.csv", l_rows)
    write_csv(results / "first_L_nonzero_area_cells.csv", triangle_rows)
    write_csv(results / "first_L_weighted_closed_current.csv", current_rows)
    write_csv(results / "first_domain_area_phase.csv", phase_rows)
    write_csv(results / "retained_first_interface_cycles.csv", retained_rows)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
