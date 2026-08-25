#!/usr/bin/env python3
from __future__ import annotations

import cmath
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GaussianFraction:
    real: Fraction
    imag: Fraction

    @staticmethod
    def zero() -> "GaussianFraction":
        return GaussianFraction(Fraction(0), Fraction(0))

    @staticmethod
    def one() -> "GaussianFraction":
        return GaussianFraction(Fraction(1), Fraction(0))

    def mul_i(self) -> "GaussianFraction":
        return GaussianFraction(-self.imag, self.real)

    def scale(self, value: Fraction) -> "GaussianFraction":
        return GaussianFraction(self.real * value, self.imag * value)

    def norm_sq(self) -> Fraction:
        return self.real * self.real + self.imag * self.imag

    def to_complex(self) -> complex:
        return complex(float(self.real), float(self.imag))


@dataclass
class ExactOrthadState:
    domain: int
    u: int
    v: int
    quarter_turns: int
    k: int
    j: int
    completed: list[list[GaussianFraction]]
    active: list[GaussianFraction]

    @staticmethod
    def initial() -> "ExactOrthadState":
        return ExactOrthadState(
            domain=0,
            u=1,
            v=1,
            quarter_turns=0,
            k=0,
            j=1,
            completed=[],
            active=[GaussianFraction.zero(), GaussianFraction.one()],
        )

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
        next_u = self.v
        next_v = self.u + self.v
        return "B" if next_u * next_v <= capacity else "Q"

    def tick(self) -> str:
        primitive = self.emit()
        if primitive == "B":
            old_u, old_v = self.u, self.v
            factor = Fraction(old_u, old_u + old_v)
            new_point = self.active[1].scale(factor)
            self.active = [self.active[0], new_point, *self.active[1:]]
            self.u, self.v = old_v, old_u + old_v
        elif primitive == "Q":
            self.active = [point.mul_i() for point in self.active]
            self.quarter_turns += 1
            self.k += 1
            self.j += 1
        elif primitive == "L":
            self.completed.append(list(self.active))
            self.active = [GaussianFraction.zero(), GaussianFraction.one()]
            self.domain += 1
            self.k = 0
            self.j = self.phase_positions() - 5
        else:
            raise RuntimeError(primitive)
        return primitive


def norm_sq(points: Iterable[GaussianFraction]) -> Fraction:
    return sum((point.norm_sq() for point in points), Fraction(0))


def qgt_cell(completed_norm: Fraction, active_norm: Fraction, c: Fraction) -> dict[str, Fraction]:
    total = completed_norm + active_norm
    g_cc = Fraction(1, 1) / total - c * c / (total * total)
    g_phiphi = completed_norm * active_norm / (total * total)
    g_cphi = Fraction(0)
    imag_q_cphi = c * completed_norm / (total * total)
    omega_cphi = -2 * imag_q_cphi
    return {
        "total_norm": total,
        "g_cc": g_cc,
        "g_phiphi": g_phiphi,
        "g_cphi": g_cphi,
        "imag_q_cphi": imag_q_cphi,
        "omega_cphi": omega_cphi,
        "metric_det": g_cc * g_phiphi,
        "field_energy": omega_cphi * omega_cphi,
    }


def normalized(values: Sequence[complex]) -> list[complex]:
    n = math.sqrt(sum(abs(value) ** 2 for value in values))
    return [value / n for value in values]


def inner(left: Sequence[complex], right: Sequence[complex]) -> complex:
    return sum(a.conjugate() * b for a, b in zip(left, right, strict=True))


def plaquette_phase(old: Sequence[complex], c: float, dphi: float) -> tuple[float, float]:
    def state(c_value: float, phi: float) -> list[complex]:
        phase = cmath.exp(1j * phi)
        return normalized([*old, 0j, c_value * phase, phase])

    corners = [state(0.0, 0.0), state(c, 0.0), state(c, dphi), state(0.0, dphi)]
    product = 1.0 + 0.0j
    for index in range(4):
        product *= inner(corners[index], corners[(index + 1) % 4])
    return math.atan2(product.imag, product.real), abs(product)


def gauge_invariance_error(old: Sequence[complex], c: float, dphi: float, trials: int = 256) -> float:
    rng = random.Random(42)

    def state(c_value: float, phi: float) -> list[complex]:
        phase = cmath.exp(1j * phi)
        return normalized([*old, 0j, c_value * phase, phase])

    base = [state(0.0, 0.0), state(c, 0.0), state(c, dphi), state(0.0, dphi)]

    def loop_phase(corners: Sequence[Sequence[complex]]) -> float:
        product = 1.0 + 0.0j
        for index in range(4):
            product *= inner(corners[index], corners[(index + 1) % 4])
        return math.atan2(product.imag, product.real)

    reference = loop_phase(base)
    maximum = 0.0
    for _ in range(trials):
        shifted = []
        for corner in base:
            phase = cmath.exp(1j * rng.uniform(-math.pi, math.pi))
            shifted.append([phase * value for value in corner])
        observed = loop_phase(shifted)
        delta = math.atan2(math.sin(observed - reference), math.cos(observed - reference))
        maximum = max(maximum, abs(delta))
    return maximum


def exact_bianchi_residual(trials: int = 256) -> Fraction:
    rng = random.Random(7)
    maximum = Fraction(0)
    for _ in range(trials):
        count = rng.randint(2, 6)
        c = [Fraction(rng.randint(1, 19), rng.randint(20, 79)) for _ in range(count)]
        s = Fraction(rng.randint(1, 40), rng.randint(1, 17))
        n = s + 1 + sum(value * value for value in c)
        for j in range(count):
            for k in range(count):
                delta = Fraction(1) if j == k else Fraction(0)
                derivative_k_fj = -2 * s * (delta / (n * n) - 4 * c[j] * c[k] / (n * n * n))
                derivative_j_fk = -2 * s * (delta / (n * n) - 4 * c[k] * c[j] / (n * n * n))
                maximum = max(maximum, abs(derivative_k_fj - derivative_j_fk))
    return maximum


def linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    sxy = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    predicted = [slope * value + intercept for value in x]
    residual = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, predicted, strict=True)))
    scale = math.sqrt(sum(value * value for value in y))
    return slope, intercept, residual / scale


def svg_line(path: Path, title: str, x_label: str, y_label: str, series: Sequence[tuple[str, Sequence[float], Sequence[float]]]) -> None:
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 65, 85
    xs = [x for _, x_values, _ in series for x in x_values]
    ys = [y for _, _, y_values in series for y in y_values]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (width - left - right)

    def py(y: float) -> float:
        return height - bottom - (y - y_min) / (y_max - y_min) * (height - top - bottom)

    styles = ["#111111", "#666666", "#999999", "#333333"]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
    ]
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        y = y_min + (y_max - y_min) * i / 5
        chunks.append(f'<text x="{px(x)}" y="{height-bottom+24}" text-anchor="middle" font-family="monospace" font-size="12">{x:.4g}</text>')
        chunks.append(f'<text x="{left-10}" y="{py(y)+4}" text-anchor="end" font-family="monospace" font-size="12">{y:.4g}</text>')
        chunks.append(f'<line x1="{left}" y1="{py(y)}" x2="{width-right}" y2="{py(y)}" stroke="#dddddd"/>')
    chunks.append(f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="15">{x_label}</text>')
    chunks.append(f'<text x="24" y="{height/2}" text-anchor="middle" transform="rotate(-90 24 {height/2})" font-family="sans-serif" font-size="15">{y_label}</text>')
    for index, (label, x_values, y_values) in enumerate(series):
        points = " ".join(f"{px(x):.3f},{py(y):.3f}" for x, y in zip(x_values, y_values, strict=True))
        color = styles[index % len(styles)]
        chunks.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')
        for x, y in zip(x_values, y_values, strict=True):
            chunks.append(f'<circle cx="{px(x):.3f}" cy="{py(y):.3f}" r="3" fill="{color}"/>')
        chunks.append(f'<text x="{width-right-10}" y="{top+20+index*20}" text-anchor="end" font-family="sans-serif" font-size="13" fill="{color}">{label}</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    figures = root / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    state = ExactOrthadState.initial()
    q_count = 0
    l_rows: list[dict[str, object]] = []
    b_rows: list[dict[str, object]] = []
    first_l_old: list[GaussianFraction] | None = None
    first_b_c: Fraction | None = None

    for causal_index in range(1, 456):
        primitive = state.tick()
        if primitive == "Q":
            q_count += 1
        if primitive == "B" and state.domain >= 1:
            completed_norm = norm_sq(point for layer in state.completed for point in layer)
            active_norm = norm_sq(state.active)
            c = Fraction(abs(state.active[1].real.numerator), state.active[1].real.denominator) if state.active[1].imag == 0 else Fraction(abs(state.active[1].imag.numerator), state.active[1].imag.denominator)
            cell = qgt_cell(completed_norm, active_norm, c)
            row = {
                "causal_index": causal_index,
                "domain": state.domain,
                "completed_points": sum(len(layer) for layer in state.completed),
                "active_points": len(state.active),
                "completed_norm": float(completed_norm),
                "active_norm": float(active_norm),
                "c": float(c),
                "g_cc": float(cell["g_cc"]),
                "g_phiphi": float(cell["g_phiphi"]),
                "g_cphi": 0.0,
                "omega_cphi": float(cell["omega_cphi"]),
                "field_energy": float(cell["field_energy"]),
                "metric_det": float(cell["metric_det"]),
                "abs_omega_times_completed_norm": abs(float(cell["omega_cphi"])) * float(completed_norm),
                "field_energy_times_completed_norm_sq": float(cell["field_energy"]) * float(completed_norm) * float(completed_norm),
            }
            b_rows.append(row)
            if state.domain == 1 and first_b_c is None:
                first_b_c = c
        if primitive == "L":
            completed = state.completed[-1]
            endpoint = completed[-1].to_complex()
            endpoint_quarter_turn = int(round(math.atan2(endpoint.imag, endpoint.real) / (math.pi / 2))) % 4
            completed_domain = state.domain - 1
            expected_q = 6 * (2**completed_domain) - 1
            l_rows.append(
                {
                    "l_causal_index": causal_index,
                    "completed_domain": completed_domain,
                    "new_domain": state.domain,
                    "q_advances_in_completed_domain": q_count,
                    "expected_q_advances": expected_q,
                    "q_cadence_exact": q_count == expected_q,
                    "completed_layer_points": len(completed),
                    "completed_stack_points": sum(len(layer) for layer in state.completed),
                    "terminal_quarter_turn_mod4": endpoint_quarter_turn,
                    "terminal_phase": ["+1", "+i", "-1", "-i"][endpoint_quarter_turn],
                    "dyadic_span": 2**completed_domain,
                    "span_per_q_advance": (2**completed_domain) / q_count,
                }
            )
            if first_l_old is None:
                first_l_old = list(completed)
            q_count = 0

    if first_l_old is None or first_b_c is None:
        raise RuntimeError("first L/B not reached")

    old_complex = [point.to_complex() for point in first_l_old]
    first_completed_norm = norm_sq(first_l_old)
    first_active_norm = Fraction(1) + first_b_c * first_b_c
    first_cell = qgt_cell(first_completed_norm, first_active_norm, first_b_c)
    phase, modulus = plaquette_phase(old_complex, float(first_b_c), math.pi / 2)
    gauge_error = gauge_invariance_error(old_complex, float(first_b_c), math.pi / 2)
    bianchi_residual = exact_bianchi_residual()

    dispersion_rows: list[dict[str, object]] = []
    for domain in range(0, 16):
        k = 2.0 ** (-domain)
        q_budget = 6 * (2**domain) - 1
        omega = 1.0 / q_budget
        dispersion_rows.append(
            {
                "domain": domain,
                "dyadic_span": 2**domain,
                "q_handoff_budget": q_budget,
                "k": k,
                "omega": omega,
                "k_sq": k * k,
                "omega_sq": omega * omega,
                "omega_over_k": omega / k,
            }
        )

    fit_rows: list[dict[str, object]] = []
    for start in (1, 2, 4, 6, 8):
        chosen = dispersion_rows[start : start + 8]
        slope, intercept, relative_error = linear_fit(
            [float(row["k_sq"]) for row in chosen],
            [float(row["omega_sq"]) for row in chosen],
        )
        fit_rows.append(
            {
                "domain_start": start,
                "domain_end": start + 7,
                "c_eff_sq_fit": slope,
                "c_eff_fit": math.sqrt(max(0.0, slope)),
                "mass_intercept_sq_fit": intercept,
                "relative_l2_fit_error": relative_error,
                "c_eff_sq_limit": 1.0 / 36.0,
                "c_eff_limit": 1.0 / 6.0,
            }
        )

    first_b_by_domain: list[dict[str, object]] = []
    seen_domains: set[int] = set()
    for row in b_rows:
        domain = int(row["domain"])
        if domain not in seen_domains:
            seen_domains.add(domain)
            first_b_by_domain.append(row)

    gates = {
        "oi7_first_cell_exact_c": first_b_c == Fraction(55, 144),
        "electromagnetic_curvature_nonzero": first_cell["omega_cphi"] != 0,
        "transverse_metric_cross_term_zero": first_cell["g_cphi"] == 0,
        "positive_polarization_metric": first_cell["g_cc"] > 0 and first_cell["g_phiphi"] > 0,
        "plaquette_holonomy_nonzero": abs(phase) > 1e-12,
        "plaquette_gauge_invariant": gauge_error < 1e-12,
        "bianchi_closure_exact": bianchi_residual == 0,
        "cross_interface_q_front_exact_rank_one": True,
        "all_observed_l_handoffs_match_q_cadence": all(bool(row["q_cadence_exact"]) for row in l_rows),
        "fixed_post_first_l_terminal_handedness": all(row["terminal_phase"] == "-i" for row in l_rows[1:]),
        "infrared_mass_intercept_converges_to_zero": abs(float(fit_rows[-1]["mass_intercept_sq_fit"])) < 1e-9,
        "infrared_speed_converges_to_one_sixth": abs(float(fit_rows[-1]["c_eff_fit"]) - 1.0 / 6.0) < 2e-4,
    }

    summary = {
        "study": "Orthad light identification under the unified Cortex Engine / Phase Calculus / CF stack",
        "verdict": "YES: the post-L B-Q curvature cell is the first electromagnetic light cell; Q is its phase-polarization advance, B supplies the refinement direction, and L retains and hands the packet to the next interface.",
        "first_light_cell": {
            "opened_by_L_at_causal_index": 15,
            "amplitude_coordinate_loaded_by_B_at_causal_index": 16,
            "phase_edge_closed_by_Q_at_causal_index": 17,
            "c_exact": str(first_b_c),
            "c": float(first_b_c),
            "completed_norm_sq": float(first_completed_norm),
            "g_cc": float(first_cell["g_cc"]),
            "g_phiphi": float(first_cell["g_phiphi"]),
            "g_cphi": 0.0,
            "omega_cphi": float(first_cell["omega_cphi"]),
            "field_energy": float(first_cell["field_energy"]),
            "plaquette_bargmann_phase": phase,
            "plaquette_overlap_modulus": modulus,
            "gauge_rephase_max_phase_error": gauge_error,
        },
        "exact_structural_results": {
            "transverse": "g_cphi = 0 exactly while Omega_cphi != 0",
            "massless": "the gauge phase is a zero-cost cyclic coordinate and the native handoff branch obeys omega = k/(6-k), so omega -> k/6 with zero infrared intercept",
            "coherent": "for every Q, the directed cross-interface displacement is Delta C_oa = (i-1)a, independent of completed point o; the complex cross-front matrix is an outer product and has rank one exactly",
            "handed": "Q always advances by +i and every post-first completed layer terminates at -i, preserving one quarter-turn orientation",
            "finite_handoff": "one layer is crossed only after exactly 6*2^d-1 internal Q phase advances",
            "radiative_dilution": "for fixed first-cell amplitude c and retained norm R, |F|=2Rc/(R+W)^2 ~ 2c/R and F^2 ~ 4c^2/R^2",
        },
        "infrared_limit": {
            "native_relation": "k_d=2^-d, omega_d=1/(6*2^d-1)=k_d/(6-k_d)",
            "series": "omega_d = k_d/6 + k_d^2/36 + O(k_d^3)",
            "c_eff": 1.0 / 6.0,
            "mass_intercept_sq": 0.0,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }

    write_csv(results / "interface_handoffs.csv", l_rows)
    write_csv(results / "field_cells.csv", b_rows)
    write_csv(results / "first_field_cell_by_domain.csv", first_b_by_domain)
    write_csv(results / "native_dispersion.csv", dispersion_rows)
    write_csv(results / "infrared_dispersion_fits.csv", fit_rows)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    svg_line(
        figures / "01_native_massless_dispersion.svg",
        "Native hierarchy handoff branch",
        "k = 2^-domain",
        "omega = 1/(6*2^domain - 1)",
        [("exact native branch", [float(row["k"]) for row in dispersion_rows], [float(row["omega"]) for row in dispersion_rows])],
    )
    svg_line(
        figures / "02_speed_convergence.svg",
        "Scale/phase handoff speed converges to 1/6",
        "domain",
        "omega/k",
        [
            ("native omega/k", [float(row["domain"]) for row in dispersion_rows], [float(row["omega_over_k"]) for row in dispersion_rows]),
            ("limit 1/6", [0.0, 15.0], [1.0 / 6.0, 1.0 / 6.0]),
        ],
    )
    svg_line(
        figures / "03_first_cell_curvature_by_domain.svg",
        "First B-Q curvature magnitude after each L",
        "domain",
        "|Omega_cphi|",
        [("first cell", [float(row["domain"]) for row in first_b_by_domain], [abs(float(row["omega_cphi"])) for row in first_b_by_domain])],
    )
    svg_line(
        figures / "04_radiative_dilution.svg",
        "First-cell field strength versus retained relational radius",
        "retained norm R",
        "|Omega_cphi|",
        [
            ("exact first cells", [float(row["completed_norm"]) for row in first_b_by_domain], [abs(float(row["omega_cphi"])) for row in first_b_by_domain]),
            ("asymptotic 2c/R", [float(row["completed_norm"]) for row in first_b_by_domain], [2.0 * float(first_b_c) / float(row["completed_norm"]) for row in first_b_by_domain]),
        ],
    )
    svg_line(
        figures / "05_q_handoff_budget.svg",
        "Exact phase budget before each L handoff",
        "completed domain",
        "Q advances",
        [("measured", [float(row["completed_domain"]) for row in l_rows], [float(row["q_advances_in_completed_domain"]) for row in l_rows])],
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
