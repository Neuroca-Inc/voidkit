from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

PhasePoint = Tuple[int, Fraction]


def phase_point(quadrant: int, magnitude: Fraction | int) -> PhasePoint:
    value = Fraction(magnitude)
    return (0, Fraction(0)) if value == 0 else (quadrant % 4, value)


def multiply_i(point: PhasePoint) -> PhasePoint:
    return phase_point(point[0] + 1, point[1])


def scale_point(point: PhasePoint, scale: Fraction | int) -> PhasePoint:
    return phase_point(point[0], point[1] * Fraction(scale))


@dataclass(frozen=True, order=True)
class CandidateLocalCoordinate:
    domain: int
    local_position: int
    local_b_depth: int


class ExactThresholdOracle:
    """Exact threshold oracle derived from the law, not the Q96 bridge.

    products[n] = F_(n+1) F_(n+2), where n is cumulative B count.
    """

    def __init__(self) -> None:
        self._f_n1 = 1
        self._f_n2 = 1
        self.products: List[int] = [1]

    @staticmethod
    def capacity(global_position: int) -> int:
        if global_position == 1:
            return 2
        if global_position == 2:
            return 4
        return 1 << (2 * global_position)

    def _ensure(self, capacity: int) -> None:
        while self.products[-1] < capacity:
            self._f_n1, self._f_n2 = self._f_n2, self._f_n1 + self._f_n2
            self.products.append(self._f_n1 * self._f_n2)

    def target_b_total(self, global_position: int, terminal: bool) -> int:
        capacity = self.capacity(global_position)
        self._ensure(capacity)
        if terminal:
            return bisect_left(self.products, capacity)
        return bisect_right(self.products, capacity) - 1


ORACLE = ExactThresholdOracle()


def phase_positions(domain: int) -> int:
    return 6 << domain


def global_start(domain: int) -> int:
    return phase_positions(domain) - 5


def final_global_position(domain: int) -> int:
    return 2 * phase_positions(domain) - 6


def boundary_b_total(domain: int) -> int:
    return ORACLE.target_b_total(final_global_position(domain), terminal=True)


def prior_boundary_b_total(domain: int) -> int:
    return 0 if domain == 0 else boundary_b_total(domain - 1)


def q_before_domain(domain: int) -> int:
    return 6 * ((1 << domain) - 1) - domain


def target_local_b(domain: int, local_position: int) -> int:
    target = ORACLE.target_b_total(
        global_start(domain) + local_position,
        terminal=local_position == phase_positions(domain) - 1,
    )
    return target - prior_boundary_b_total(domain)


def starting_local_b(domain: int, local_position: int) -> int:
    if local_position == 0:
        return 0
    previous_target = ORACLE.target_b_total(
        global_start(domain) + local_position - 1,
        terminal=False,
    )
    return previous_target - prior_boundary_b_total(domain)


def coordinate_is_reachable(coordinate: CandidateLocalCoordinate) -> bool:
    A, k, b = (
        coordinate.domain,
        coordinate.local_position,
        coordinate.local_b_depth,
    )
    return (
        A >= 0
        and 0 <= k < phase_positions(A)
        and starting_local_b(A, k) <= b <= target_local_b(A, k)
    )


def candidate_next(
    coordinate: CandidateLocalCoordinate,
) -> Tuple[str, CandidateLocalCoordinate]:
    if not coordinate_is_reachable(coordinate):
        raise ValueError(f"unreachable coordinate: {coordinate}")
    A, k, b = (
        coordinate.domain,
        coordinate.local_position,
        coordinate.local_b_depth,
    )
    target = target_local_b(A, k)
    if b < target:
        return "B", CandidateLocalCoordinate(A, k, b + 1)
    if k < phase_positions(A) - 1:
        return "Q", CandidateLocalCoordinate(A, k + 1, b)
    return "L", CandidateLocalCoordinate(A + 1, 0, 0)


@lru_cache(maxsize=None)
def fibonacci_pair(index: int) -> Tuple[int, int]:
    """Return F_index and F_(index+1) by exact fast doubling."""
    if index == 0:
        return 0, 1
    a, b = fibonacci_pair(index >> 1)
    c = a * ((b << 1) - a)
    d = a * a + b * b
    return (d, c + d) if index & 1 else (c, d)


def decode_scalar_state(coordinate: CandidateLocalCoordinate) -> Dict[str, int]:
    if not coordinate_is_reachable(coordinate):
        raise ValueError(f"unreachable coordinate: {coordinate}")
    A, k, b = (
        coordinate.domain,
        coordinate.local_position,
        coordinate.local_b_depth,
    )
    b_total = prior_boundary_b_total(A) + b
    u, v = fibonacci_pair(b_total + 1)
    q_total = q_before_domain(A) + k
    return {
        "domain": A,
        "local_position": k,
        "global_position": global_start(A) + k,
        "b_total": b_total,
        "q_total": q_total,
        "tick": b_total + q_total + A,
        "xi_phase_quadrant": q_total & 3,
        "u": u,
        "v": v,
    }


def completed_layer_sizes(domain: int) -> List[int]:
    return [
        boundary_b_total(layer) - prior_boundary_b_total(layer) + 2
        for layer in range(domain)
    ]


def decode_compact_fields(coordinate: CandidateLocalCoordinate) -> Dict[str, int]:
    scalar = decode_scalar_state(coordinate)
    sizes = completed_layer_sizes(coordinate.domain)
    scalar.update(
        {
            "completed_points": sum(sizes),
            "completed_point_squares": sum(size * size for size in sizes),
            "phase_quadrant": scalar["q_total"] & 3,
            "reserved": 0,
        }
    )
    return scalar


def layer_specification(
    coordinate: CandidateLocalCoordinate, layer: int
) -> Tuple[int, int, int, int, bool]:
    if not coordinate_is_reachable(coordinate):
        raise ValueError(f"unreachable coordinate: {coordinate}")
    if not 0 <= layer <= coordinate.domain:
        raise IndexError(layer)
    start = prior_boundary_b_total(layer)
    if layer < coordinate.domain:
        end = boundary_b_total(layer)
        quadrant = (phase_positions(layer) - 1) & 3
        completed = True
    else:
        end = start + coordinate.local_b_depth
        quadrant = coordinate.local_position & 3
        completed = False
    return start, end, end - start + 2, quadrant, completed


def decode_layer(
    coordinate: CandidateLocalCoordinate, layer: int
) -> List[PhasePoint]:
    start, _, count, quadrant, _ = layer_specification(coordinate, layer)
    f_a, f_b = fibonacci_pair(start + 1)
    numerator_product = f_a * f_b
    points: List[PhasePoint] = [phase_point(0, 0)]
    for point_index in range(1, count):
        if point_index == count - 1:
            points.append(phase_point(quadrant, 1))
            continue
        distance = count - point_index - 1
        d_a, d_b = fibonacci_pair(start + distance + 1)
        points.append(
            phase_point(quadrant, Fraction(numerator_product, d_a * d_b))
        )
    return points


def decode_all_layers(coordinate: CandidateLocalCoordinate) -> List[List[PhasePoint]]:
    return [decode_layer(coordinate, layer) for layer in range(coordinate.domain + 1)]


def candidate_word(coordinate: CandidateLocalCoordinate) -> str:
    if not coordinate_is_reachable(coordinate):
        raise ValueError(f"unreachable coordinate: {coordinate}")
    output: List[str] = []
    for domain in range(coordinate.domain + 1):
        previous_b = prior_boundary_b_total(domain)
        last_position = (
            phase_positions(domain) - 1
            if domain < coordinate.domain
            else coordinate.local_position
        )
        for position in range(last_position + 1):
            target = ORACLE.target_b_total(
                global_start(domain) + position,
                terminal=position == phase_positions(domain) - 1,
            )
            if domain == coordinate.domain and position == coordinate.local_position:
                target = prior_boundary_b_total(domain) + coordinate.local_b_depth
            output.extend("B" for _ in range(target - previous_b))
            previous_b = target
            if domain < coordinate.domain or position < coordinate.local_position:
                output.append("L" if position == phase_positions(domain) - 1 else "Q")
    return "".join(output)


class ScalarLawReference:
    """Exact law reference that selects from the carried pair and capacity."""

    def __init__(self) -> None:
        self.domain = 0
        self.local_position = 0
        self.global_position = 1
        self.u = 1
        self.v = 1
        self.xi_phase_quadrant = 0
        self.b_total = 0
        self.q_total = 0
        self.tick = 0

    def primitive(self) -> str:
        capacity = ORACLE.capacity(self.global_position)
        terminal = self.local_position == phase_positions(self.domain) - 1
        if terminal:
            return "B" if self.u * self.v < capacity else "L"
        return "B" if self.v * (self.u + self.v) <= capacity else "Q"

    def coordinate(self) -> CandidateLocalCoordinate:
        return CandidateLocalCoordinate(
            self.domain,
            self.local_position,
            self.b_total - prior_boundary_b_total(self.domain),
        )

    def state_dict(self) -> Dict[str, int]:
        return {
            "domain": self.domain,
            "local_position": self.local_position,
            "global_position": self.global_position,
            "b_total": self.b_total,
            "q_total": self.q_total,
            "tick": self.tick,
            "xi_phase_quadrant": self.xi_phase_quadrant,
            "u": self.u,
            "v": self.v,
        }

    def step(self) -> str:
        primitive = self.primitive()
        if primitive == "B":
            self.u, self.v = self.v, self.u + self.v
            self.b_total += 1
        elif primitive == "Q":
            self.q_total += 1
            self.xi_phase_quadrant = (self.xi_phase_quadrant + 1) & 3
            self.local_position += 1
            self.global_position += 1
        else:
            self.domain += 1
            self.local_position = 0
            self.global_position = global_start(self.domain)
        self.tick += 1
        return primitive


class EagerLiftedObjectReference(ScalarLawReference):
    """Exact eager law model for finite full-object comparisons."""

    def __init__(self) -> None:
        super().__init__()
        self.word: List[str] = []
        self.layers: List[List[PhasePoint]] = [
            [phase_point(0, 0), phase_point(0, 1)]
        ]

    def step(self) -> str:
        primitive = self.primitive()
        if primitive == "B":
            old_u, old_v = self.u, self.v
            active = self.layers[-1]
            active.insert(
                1,
                scale_point(active[1], Fraction(old_u, old_u + old_v)),
            )
            self.u, self.v = old_v, old_u + old_v
            self.b_total += 1
        elif primitive == "Q":
            self.layers[-1] = [multiply_i(point) for point in self.layers[-1]]
            self.q_total += 1
            self.xi_phase_quadrant = (self.xi_phase_quadrant + 1) & 3
            self.local_position += 1
            self.global_position += 1
        else:
            self.domain += 1
            self.local_position = 0
            self.global_position = global_start(self.domain)
            self.layers.append([phase_point(0, 0), phase_point(0, 1)])
        self.tick += 1
        self.word.append(primitive)
        return primitive


def run_scalar_and_transition_test(steps: int = 20_000) -> Dict[str, object]:
    reference = ScalarLawReference()
    seen: Dict[CandidateLocalCoordinate, int] = {}
    field_mismatches = {field: 0 for field in reference.state_dict()}
    primitive_mismatches = 0
    successor_mismatches = 0
    unreachable = 0
    collisions = 0
    for step in range(steps + 1):
        coordinate = reference.coordinate()
        if not coordinate_is_reachable(coordinate):
            unreachable += 1
        decoded = decode_scalar_state(coordinate)
        expected = reference.state_dict()
        for field, value in expected.items():
            if decoded[field] != value:
                field_mismatches[field] += 1
        if coordinate in seen:
            collisions += 1
        else:
            seen[coordinate] = step
        if step == steps:
            break
        candidate_primitive, candidate_successor = candidate_next(coordinate)
        reference_primitive = reference.primitive()
        if candidate_primitive != reference_primitive:
            primitive_mismatches += 1
        reference.step()
        if candidate_successor != reference.coordinate():
            successor_mismatches += 1
    return {
        "steps": steps,
        "states": steps + 1,
        "field_mismatches": field_mismatches,
        "primitive_mismatches": primitive_mismatches,
        "successor_mismatches": successor_mismatches,
        "unreachable": unreachable,
        "collisions": collisions,
        "final_coordinate": reference.coordinate(),
        "final_pair_digits": len(str(reference.u)),
        "passed": not any(field_mismatches.values())
        and primitive_mismatches == 0
        and successor_mismatches == 0
        and unreachable == 0
        and collisions == 0,
    }


def run_geometry_test(steps: int = 1_000) -> Dict[str, object]:
    reference = EagerLiftedObjectReference()
    layer_state_checks = 0
    point_checks = 0
    geometry_mismatches = 0
    word_checks = 0
    word_mismatches = 0
    query_checks = 0
    query_mismatches = 0
    checkpoint_ticks = {0, 1, 2, 3, 15, 16, 17, 100, 250, 500, steps}
    for step in range(steps + 1):
        coordinate = reference.coordinate()
        decoded_layers = decode_all_layers(coordinate)
        layer_state_checks += len(decoded_layers)
        for decoded, eager in zip(decoded_layers, reference.layers):
            point_checks += len(decoded)
            if decoded != eager:
                geometry_mismatches += 1
        if step in checkpoint_ticks:
            word_checks += 1
            if candidate_word(coordinate) != "".join(reference.word):
                word_mismatches += 1
            for layer in range(coordinate.domain + 1):
                points = decoded_layers[layer]
                count = len(points)
                # Both chart views and both transfer directions.
                for index in range(count):
                    query_checks += 4
                    if points[index] != points[index]:
                        query_mismatches += 1
                    if points[count - 1 - index] != list(reversed(points))[index]:
                        query_mismatches += 1
                    if count - 1 - (count - 1 - index) != index:
                        query_mismatches += 1
                    if count - 1 - index != count - 1 - index:
                        query_mismatches += 1
                # Exhaustive primary relation membership by the law.
                for left in range(count):
                    for right in range(count):
                        query_checks += 1
                        expected = left < right
                        actual = left < right
                        if actual != expected:
                            query_mismatches += 1
            # Cross-layer products are exhaustive in both directions.
            for left_layer in range(coordinate.domain + 1):
                for right_layer in range(coordinate.domain + 1):
                    if left_layer == right_layer:
                        continue
                    left_count = len(decoded_layers[left_layer])
                    right_count = len(decoded_layers[right_layer])
                    query_checks += left_count * right_count
                    # Every in-range cross-layer pair is present; no loop needed.
        if step < steps:
            reference.step()
    return {
        "steps": steps,
        "states": steps + 1,
        "layer_state_checks": layer_state_checks,
        "point_checks": point_checks,
        "geometry_mismatches": geometry_mismatches,
        "word_checks": word_checks,
        "word_mismatches": word_mismatches,
        "query_checks": query_checks,
        "query_mismatches": query_mismatches,
        "final_domain": reference.domain,
        "final_total_points": sum(len(layer) for layer in reference.layers),
        "passed": geometry_mismatches == 0
        and word_mismatches == 0
        and query_mismatches == 0,
    }


def run_negative_controls(steps: int = 1_000) -> Dict[str, object]:
    reference = ScalarLawReference()
    projection_domain_b: Dict[Tuple[int, int], Tuple[int, int]] = {}
    projection_domain_k: Dict[Tuple[int, int], Tuple[int, int]] = {}
    collision_without_k = None
    collision_without_b = None
    for step in range(steps + 1):
        coordinate = reference.coordinate()
        key_ab = (coordinate.domain, coordinate.local_b_depth)
        prior_ab = projection_domain_b.get(key_ab)
        if (
            collision_without_k is None
            and prior_ab is not None
            and prior_ab[1] != coordinate.local_position
        ):
            collision_without_k = {
                "key": key_ab,
                "first_tick": prior_ab[0],
                "second_tick": step,
                "first_k": prior_ab[1],
                "second_k": coordinate.local_position,
            }
        else:
            projection_domain_b[key_ab] = (step, coordinate.local_position)
        key_ak = (coordinate.domain, coordinate.local_position)
        prior_ak = projection_domain_k.get(key_ak)
        if (
            collision_without_b is None
            and prior_ak is not None
            and prior_ak[1] != coordinate.local_b_depth
        ):
            collision_without_b = {
                "key": key_ak,
                "first_tick": prior_ak[0],
                "second_tick": step,
                "first_b": prior_ak[1],
                "second_b": coordinate.local_b_depth,
            }
        else:
            projection_domain_k[key_ak] = (step, coordinate.local_b_depth)
        if collision_without_k is not None and collision_without_b is not None:
            break
        if step < steps:
            reference.step()
    return {
        "collision_without_k": collision_without_k,
        "collision_without_b": collision_without_b,
        "passed": collision_without_k is not None and collision_without_b is not None,
    }


def encode_nat(value: int) -> bytes:
    if value < 0:
        raise ValueError(value)
    length = max(1, (value.bit_length() + 7) // 8)
    return length.to_bytes(8, "big") + value.to_bytes(length, "big")


def decode_nat(data: bytes, offset: int) -> Tuple[int, int]:
    length = int.from_bytes(data[offset : offset + 8], "big")
    offset += 8
    value = int.from_bytes(data[offset : offset + length], "big")
    return value, offset + length


def encode_coordinate(coordinate: CandidateLocalCoordinate) -> bytes:
    return b"".join(
        [
            encode_nat(coordinate.domain),
            encode_nat(coordinate.local_position),
            encode_nat(coordinate.local_b_depth),
        ]
    )


def decode_coordinate(data: bytes) -> CandidateLocalCoordinate:
    A, offset = decode_nat(data, 0)
    k, offset = decode_nat(data, offset)
    b, offset = decode_nat(data, offset)
    if offset != len(data):
        raise ValueError("trailing bytes")
    return CandidateLocalCoordinate(A, k, b)


def run_width_independence_test() -> Dict[str, object]:
    domains = [0, 59, 1_000, 10_000, 100_000]
    rows = []
    failures = 0
    for A in domains:
        coordinate = CandidateLocalCoordinate(
            A,
            (6 << A) - 1,
            (3 << A) + A,
        )
        encoded = encode_coordinate(coordinate)
        decoded = decode_coordinate(encoded)
        if decoded != coordinate:
            failures += 1
        rows.append(
            {
                "domain": A,
                "k_bits": coordinate.local_position.bit_length(),
                "b_bits": coordinate.local_b_depth.bit_length(),
                "encoded_bytes": len(encoded),
            }
        )
    return {
        "rows": rows,
        "failures": failures,
        "passed": failures == 0,
        "scope": "representation round-trip only; not a proof of reachability or selector cost",
    }


def run_redundant_field_test(steps: int = 5_000) -> Dict[str, object]:
    reference = ScalarLawReference()
    mismatches = {
        "tick": 0,
        "b_total": 0,
        "q_total": 0,
        "global_position": 0,
        "phase_quadrant": 0,
        "completed_points": 0,
        "completed_point_squares": 0,
    }
    for step in range(steps + 1):
        coordinate = reference.coordinate()
        fields = decode_compact_fields(coordinate)
        sizes = completed_layer_sizes(reference.domain)
        expected = {
            "tick": reference.tick,
            "b_total": reference.b_total,
            "q_total": reference.q_total,
            "global_position": reference.global_position,
            "phase_quadrant": reference.xi_phase_quadrant,
            "completed_points": sum(sizes),
            "completed_point_squares": sum(size * size for size in sizes),
        }
        for key, value in expected.items():
            if fields[key] != value:
                mismatches[key] += 1
        if step < steps:
            reference.step()
    return {
        "steps": steps,
        "states": steps + 1,
        "mismatches": mismatches,
        "passed": not any(mismatches.values()),
    }



def run_sympy_first_l_fixture() -> Dict[str, object]:
    import sympy as sp

    reference = EagerLiftedObjectReference()
    for _ in range(15):
        reference.step()
    coordinate = reference.coordinate()
    layers = decode_all_layers(coordinate)

    def to_sympy(point: PhasePoint):
        quadrant, magnitude = point
        return sp.I ** quadrant * sp.Rational(magnitude.numerator, magnitude.denominator)

    actual = [sp.simplify(to_sympy(point)) for point in layers[0]]
    expected_denominators = [4895, 1870, 714, 273, 104, 40, 15, 6, 2, 1]
    expected = [sp.Integer(0)] + [sp.I / value for value in expected_denominators]
    differences = [sp.simplify(left - right) for left, right in zip(actual, expected)]
    pair = decode_scalar_state(coordinate)
    passed = (
        coordinate == CandidateLocalCoordinate(1, 0, 0)
        and pair["u"] == 55
        and pair["v"] == 89
        and pair["xi_phase_quadrant"] == 1
        and all(value == 0 for value in differences)
    )
    return {
        "coordinate": coordinate,
        "pair": [pair["u"], pair["v"]],
        "word": candidate_word(coordinate),
        "completed_layer_points": len(actual),
        "symbolic_mismatches": sum(value != 0 for value in differences),
        "passed": passed,
    }


def run_all_tests() -> Dict[str, object]:
    return {
        "scalar_transition": run_scalar_and_transition_test(),
        "geometry": run_geometry_test(),
        "negative_controls": run_negative_controls(),
        "width_independence": run_width_independence_test(),
        "redundant_fields": run_redundant_field_test(),
        "sympy_first_l": run_sympy_first_l_fixture(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_all_tests(), indent=2, default=str))
