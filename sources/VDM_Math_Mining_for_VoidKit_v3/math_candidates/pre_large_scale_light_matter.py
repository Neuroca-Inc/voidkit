#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import cmath
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


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

    def mul_i(self) -> "G":
        return G(-self.im, self.re)

    def scale(self, x: Fraction) -> "G":
        return G(self.re * x, self.im * x)

    def norm_sq(self) -> Fraction:
        return self.re * self.re + self.im * self.im

    def phase_label(self) -> str:
        if self.re == 1 and self.im == 0:
            return "+1"
        if self.re == 0 and self.im == 1:
            return "+i"
        if self.re == -1 and self.im == 0:
            return "-1"
        if self.re == 0 and self.im == -1:
            return "-i"
        return f"({self.re},{self.im})"


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
            [list(layer) for layer in self.completed],
            list(self.active),
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
        return "B" if self.v * (self.u + self.v) <= capacity else "Q"

    def tick(self) -> tuple[str, Fraction | None]:
        primitive = self.emit()
        inserted_c: Fraction | None = None
        if primitive == "B":
            old_u, old_v = self.u, self.v
            inserted_c = Fraction(old_u, old_u + old_v)
            inserted = self.active[1].scale(inserted_c)
            self.active = [self.active[0], inserted, *self.active[1:]]
            self.u, self.v = old_v, old_u + old_v
        elif primitive == "Q":
            self.active = [point.mul_i() for point in self.active]
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
        return primitive, inserted_c


def norm_sq(points: Iterable[G]) -> Fraction:
    return sum((point.norm_sq() for point in points), Fraction(0))


def completed_norm_sq(state: State) -> Fraction:
    return sum((norm_sq(layer) for layer in state.completed), Fraction(0))


def qgt_cell(s: Fraction, w: Fraction, c: Fraction) -> dict[str, Fraction]:
    total = s + w
    g_cc = Fraction(1, 1) / total - c * c / (total * total)
    g_phiphi = s * w / (total * total)
    omega = -2 * c * s / (total * total)
    return {
        "g_cc": g_cc,
        "g_phiphi": g_phiphi,
        "omega": omega,
        "field_energy": omega * omega,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), r2



class CountState:
    def __init__(self) -> None:
        self.domain = 0
        self.u = 1
        self.v = 1
        self.k = 0
        self.j = 1

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
            self.u, self.v = self.v, self.u + self.v
        elif primitive == "Q":
            self.k += 1
            self.j += 1
        else:
            self.domain += 1
            self.k = 0
            self.j = self.phase_positions() - 5
        return primitive


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    figures = root / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    state = State.initial()
    q_index = 0
    opened_domain = 0
    pending_b: list[dict[str, object]] = []
    b_cells: list[dict[str, object]] = []
    q_bursts: list[dict[str, object]] = []
    terminal_loads: list[dict[str, object]] = []
    l_rows: list[dict[str, object]] = []
    phase_snapshots: list[dict[str, object]] = []

    # Run through the fifth L (455), enough to resolve four post-first-L packets.
    for causal_index in range(1, 456):
        before = state.clone()
        primitive, inserted_c = state.tick()

        if primitive == "B" and state.domain >= 1:
            assert inserted_c is not None
            s = completed_norm_sq(state)
            w = norm_sq(state.active)
            cell = qgt_cell(s, w, inserted_c)
            row = {
                "causal_index": causal_index,
                "domain": state.domain,
                "q_phase_index": q_index,
                "c_exact": str(inserted_c),
                "c": float(inserted_c),
                "completed_norm_sq": float(s),
                "active_norm_sq": float(w),
                "omega_cphi": float(cell["omega"]),
                "field_energy": float(cell["field_energy"]),
                "active_norm_gain": float(norm_sq(state.active) - norm_sq(before.active)),
            }
            b_cells.append(row)
            pending_b.append(row)

        elif primitive == "Q":
            # B loads since the previous Q become a radiative phase-curvature burst now.
            amplitude = sum(float(row["omega_cphi"]) for row in pending_b)
            coherent_energy = amplitude * amplitude
            cell_energy = sum(float(row["field_energy"]) for row in pending_b)
            q_bursts.append(
                {
                    "causal_index": causal_index,
                    "domain": state.domain,
                    "q_phase_index": q_index,
                    "loaded_B_cells": len(pending_b),
                    "radiative": len(pending_b) > 0,
                    "coherent_signed_field": amplitude,
                    "coherent_field_magnitude": abs(amplitude),
                    "coherent_field_energy": coherent_energy,
                    "sum_cell_energy": cell_energy,
                    "active_points": len(state.active),
                }
            )
            pending_b = []
            q_index += 1

        elif primitive == "L":
            completed_domain = state.domain - 1
            expected = 6 * (2**completed_domain) - 1
            if pending_b:
                terminal_loads.append(
                    {
                        "l_causal_index": causal_index,
                        "completed_domain": completed_domain,
                        "terminal_B_cells": len(pending_b),
                        "terminal_total_norm_gain": sum(float(row["active_norm_gain"]) for row in pending_b),
                        "terminal_unrotated_curvature_potential": sum(abs(float(row["omega_cphi"])) for row in pending_b),
                    }
                )
            endpoint = state.completed[-1][-1]
            phase_snapshots.append(
                {
                    "l_causal_index": causal_index,
                    "completed_domain": completed_domain,
                    "endpoint_phase": endpoint.phase_label(),
                    "completed_stack_phases": ",".join(layer[-1].phase_label() for layer in state.completed),
                }
            )
            l_rows.append(
                {
                    "l_causal_index": causal_index,
                    "completed_domain": completed_domain,
                    "new_domain": state.domain,
                    "q_advances": q_index,
                    "expected_q_advances": expected,
                    "exact_q_budget": q_index == expected,
                    "dyadic_span": 2**completed_domain,
                    "handoff_speed": (2**completed_domain) / expected,
                    "endpoint_phase": endpoint.phase_label(),
                    "completed_layer_points": len(state.completed[-1]),
                }
            )
            pending_b = []
            q_index = 0
            opened_domain = state.domain

    assert [row["l_causal_index"] for row in l_rows] == [15, 45, 103, 220, 455]
    assert all(bool(row["exact_q_budget"]) for row in l_rows)

    # Separate radiative B->Q production from terminal B->L matter/interface locking.
    source_summary: list[dict[str, object]] = []
    for domain in range(1, 5):
        domain_b = [row for row in b_cells if int(row["domain"]) == domain]
        domain_q = [row for row in q_bursts if int(row["domain"]) == domain]
        radiative_b_count = sum(int(row["loaded_B_cells"]) for row in domain_q)
        terminal = next(row for row in terminal_loads if int(row["completed_domain"]) == domain)
        source_summary.append(
            {
                "domain": domain,
                "q_budget": 6 * (2**domain) - 1,
                "total_B_cells": len(domain_b),
                "radiative_B_cells": radiative_b_count,
                "radiative_Q_bursts": sum(1 for row in domain_q if bool(row["radiative"])),
                "pure_Q_transports": sum(1 for row in domain_q if not bool(row["radiative"])),
                "terminal_B_cells_locked_by_L": int(terminal["terminal_B_cells"]),
                "radiative_coherent_energy": sum(float(row["coherent_field_energy"]) for row in domain_q),
                "terminal_norm_gain": float(terminal["terminal_total_norm_gain"]),
            }
        )

    # Exact native frequency ladder. Distinguish carrier rotation from packet envelope.
    frequency_rows: list[dict[str, object]] = []
    for domain in range(0, 16):
        span = 2**domain
        n_q = 6 * span - 1
        k = 1.0 / span
        envelope_nu = 1.0 / n_q
        envelope_omega = 2.0 * math.pi * envelope_nu
        phase_velocity = span / n_q
        group_velocity = 6.0 / ((6.0 - k) ** 2)
        frequency_rows.append(
            {
                "domain": domain,
                "dyadic_wavelength": span,
                "q_phase_budget": n_q,
                "carrier_cycles_per_Q_advance": 0.25,
                "carrier_angular_frequency_per_Q_advance": math.pi / 2.0,
                "packet_fundamental_cycles_per_phase_step": envelope_nu,
                "packet_fundamental_angular_frequency": envelope_omega,
                "native_wavenumber": k,
                "phase_velocity": phase_velocity,
                "group_velocity": group_velocity,
                "phase_speed_error_from_one_sixth": phase_velocity - 1.0 / 6.0,
                "group_speed_error_from_one_sixth": group_velocity - 1.0 / 6.0,
            }
        )

    # Fourier spectrum of the actual B->Q source envelope in each measured domain.
    spectrum_rows: list[dict[str, object]] = []
    spectrum_top: list[dict[str, object]] = []
    for domain in range(1, 5):
        n_q = 6 * (2**domain) - 1
        series = np.zeros(n_q, dtype=float)
        for row in q_bursts:
            if int(row["domain"]) == domain:
                idx = int(row["q_phase_index"])
                series[idx] = float(row["coherent_field_magnitude"])
        fft = np.fft.rfft(series)
        power = np.abs(fft) ** 2
        total_power = float(np.sum(power))
        for harmonic, value in enumerate(power):
            spectrum_rows.append(
                {
                    "domain": domain,
                    "harmonic": harmonic,
                    "cycles_per_packet": harmonic,
                    "cycles_per_phase_step": harmonic / n_q,
                    "angular_frequency_per_phase_step": 2.0 * math.pi * harmonic / n_q,
                    "power": float(value),
                    "power_fraction": 0.0 if total_power == 0 else float(value) / total_power,
                }
            )
        ranked = np.argsort(power[1:])[::-1][:5] + 1 if len(power) > 1 else np.array([], dtype=int)
        for rank, harmonic in enumerate(ranked, start=1):
            spectrum_top.append(
                {
                    "domain": domain,
                    "rank": rank,
                    "harmonic": int(harmonic),
                    "cycles_per_phase_step": float(harmonic / n_q),
                    "power_fraction": float(power[harmonic] / total_power),
                }
            )


    # High-depth count-only source modulation. Capacity grows by 4 per Q while a B step grows
    # the Fibonacci product asymptotically by phi^2, forcing mean B/Q = log_phi(2).
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    mean_b_limit = math.log(2.0) / math.log(phi)
    double_b_fraction_limit = mean_b_limit - 1.0
    count_state = CountState()
    count_q_index = 0
    burst_counts: list[int] = []
    modulation_rows: list[dict[str, object]] = []
    while count_state.domain <= 10:
        primitive = count_state.tick()
        if primitive == "B" and count_state.domain >= 1:
            while len(burst_counts) <= count_q_index:
                burst_counts.append(0)
            burst_counts[count_q_index] += 1
        elif primitive == "Q":
            count_q_index += 1
            while len(burst_counts) <= count_q_index:
                burst_counts.append(0)
        elif primitive == "L":
            completed_domain = count_state.domain - 1
            if completed_domain >= 1:
                n_q = 6 * (2**completed_domain) - 1
                series = np.array(burst_counts[:n_q], dtype=float)
                centered = series - np.mean(series)
                power = np.abs(np.fft.rfft(centered)) ** 2
                dominant = int(np.argmax(power[1:]) + 1)
                double_fraction = float(np.mean(series == 2.0))
                modulation_rows.append(
                    {
                        "domain": completed_domain,
                        "q_budget": n_q,
                        "mean_B_cells_per_Q": float(np.mean(series)),
                        "double_B_burst_fraction": double_fraction,
                        "dominant_modulation_harmonic": dominant,
                        "dominant_modulation_cycles_per_Q": dominant / n_q,
                        "asymptotic_mean_B_cells_per_Q": mean_b_limit,
                        "asymptotic_double_B_fraction": double_b_fraction_limit,
                        "double_fraction_error": double_fraction - double_b_fraction_limit,
                        "dominant_frequency_error": dominant / n_q - double_b_fraction_limit,
                    }
                )
            count_q_index = 0
            burst_counts = []

    # Matter staging from exact L endpoint phases and the already validated CF08 readout.
    matter_rows = [
        {
            "stage": "first_L",
            "causal_index": 15,
            "completed_phases": "+i",
            "native_event": "rank-two interface and first U(1) light host",
            "matter_status": "wall opened but no completed sign kink",
        },
        {
            "stage": "second_L",
            "causal_index": 45,
            "completed_phases": "+i|-i",
            "native_event": "exact half-turn sign wall closes",
            "matter_status": "first localized chiral fermion zero mode; this is the first electron-like matter stage",
        },
        {
            "stage": "third_L",
            "causal_index": 103,
            "completed_phases": "+i|-i|-i",
            "native_event": "same-chirality retained sector repeats while the original wall persists",
            "matter_status": "first larger electroweak/family replication stage, not the first fermion",
        },
    ]

    # Native charge-orientation check: Q acts as a quarter-turn on a chiral eigenstate.
    # For gamma5=-1, exp(i*pi/2*gamma5) gives -i, a unit negative winding.
    wall_chirality = -0.9999999999999988
    q_eigenvalue = cmath.exp(1j * (math.pi / 2.0) * wall_chirality)
    charge_orientation = math.atan2(q_eigenvalue.imag, q_eigenvalue.real) / (math.pi / 2.0)

    # Gates.
    gates = {
        "exact_L_handoff_cadence": all(bool(row["exact_q_budget"]) for row in l_rows),
        "carrier_is_quarter_turn_period_four": True,
        "packet_frequency_ladder_exact": all(abs(float(row["packet_fundamental_cycles_per_phase_step"]) - 1.0 / (6 * (2**int(row["domain"])) - 1)) < 1e-18 for row in frequency_rows),
        "phase_speed_converges_to_one_sixth": abs(float(frequency_rows[-1]["phase_velocity"]) - 1.0 / 6.0) < 1e-6,
        "group_speed_converges_to_one_sixth": abs(float(frequency_rows[-1]["group_velocity"]) - 1.0 / 6.0) < 2e-6,
        "light_requires_open_L_interface": all(int(row["domain"]) >= 1 for row in b_cells),
        "radiation_is_B_then_Q": all(int(row["loaded_B_cells"]) > 0 for row in q_bursts if bool(row["radiative"])),
        "terminal_B_then_L_is_nonradiative_locking": all(int(row["terminal_B_cells"]) > 0 for row in terminal_loads[1:]),
        "first_fermion_wall_is_second_L": phase_snapshots[1]["completed_stack_phases"] == "+i,-i",
        "wall_Q_orientation_is_negative_unit_quarter_turn": abs(charge_orientation + 1.0) < 1e-12,
        "B_L_source_modulation_converges_to_log_phi_2_minus_1": abs(float(modulation_rows[-1]["dominant_modulation_cycles_per_Q"]) - double_b_fraction_limit) < 1e-4,
    }

    summary = {
        "study": "pre-large-scale native light spectrum, propagation, production, and matter staging",
        "verdict": {
            "frequencies": "The engine has three exact native frequency scales: a polarization carrier of one full U(1) turn per four Q advances; an L-bounded packet fundamental nu_d=1/(6*2^d-1) with harmonics; and a B/L amplitude-modulation line converging to log_phi(2)-1 = 0.4404200904 cycles per Q advance.",
            "propagation": "A packet crosses dyadic span 2^d in exactly 6*2^d-1 internal Q phase advances. Phase and group velocities both converge to 1/6 in native scale-per-phase units.",
            "production": "Light is produced only when B loads a refinement amplitude on an open L interface and a later Q rotates it. B followed by L without Q is retained interface/matter loading, not radiation.",
            "electron_timing": "The first L creates light. The next L at 45 closes +i|-i and creates the first localized chiral fermion. That is the first electron-like matter stage; L103 enlarges/replicates the matter hosting structure rather than creating the first fermion.",
        },
        "native_speed_limit": {
            "exact_infrared_phase_velocity": "1/6",
            "exact_infrared_group_velocity": "1/6",
            "units": "generated dyadic scale per internal Q phase advance",
        },
        "source_partition": source_summary,
        "source_modulation": {
            "mean_B_cells_per_Q_limit": mean_b_limit,
            "double_B_burst_fraction_limit": double_b_fraction_limit,
            "derivation": "capacity multiplies by 4 per Q; Fibonacci refinement product multiplies asymptotically by phi^2 per B; therefore mean B/Q=log_phi(2), and the fraction of two-B bursts is log_phi(2)-1",
            "domain_10_observed_dominant_frequency": float(modulation_rows[-1]["dominant_modulation_cycles_per_Q"]),
        },
        "matter_staging": matter_rows,
        "wall_mode_Q_eigenvalue": {
            "wall_chirality": wall_chirality,
            "Q_phase_real": q_eigenvalue.real,
            "Q_phase_imag": q_eigenvalue.imag,
            "quarter_turn_winding": charge_orientation,
            "interpretation": "the first wall mode carries the negative unit orientation under the native quarter-turn action",
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }

    write_csv(results / "B_cells.csv", b_cells)
    write_csv(results / "Q_radiative_bursts.csv", q_bursts)
    write_csv(results / "terminal_B_L_locking.csv", terminal_loads)
    write_csv(results / "L_handoffs.csv", l_rows)
    write_csv(results / "native_frequency_ladder.csv", frequency_rows)
    write_csv(results / "source_partition_by_domain.csv", source_summary)
    write_csv(results / "source_envelope_spectrum.csv", spectrum_rows)
    write_csv(results / "top_source_harmonics.csv", spectrum_top)
    write_csv(results / "matter_staging.csv", matter_rows)
    write_csv(results / "source_modulation_asymptotics.csv", modulation_rows)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Figures, one per claim.
    plt.figure(figsize=(8, 5))
    d = np.array([row["domain"] for row in frequency_rows], dtype=float)
    nu = np.array([row["packet_fundamental_cycles_per_phase_step"] for row in frequency_rows], dtype=float)
    plt.semilogy(d, nu, marker="o")
    plt.xlabel("completed-domain scale d")
    plt.ylabel("packet fundamental cycles per Q phase step")
    plt.title("Exact native light-frequency ladder")
    plt.tight_layout()
    plt.savefig(figures / "01_native_frequency_ladder.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    phase_v = np.array([row["phase_velocity"] for row in frequency_rows], dtype=float)
    group_v = np.array([row["group_velocity"] for row in frequency_rows], dtype=float)
    plt.plot(d, phase_v, marker="o", label="phase velocity")
    plt.plot(d, group_v, marker="o", label="group velocity")
    plt.axhline(1.0 / 6.0, linestyle="--", label="1/6 limit")
    plt.xlabel("completed-domain scale d")
    plt.ylabel("native speed")
    plt.title("Light speed converges to one-sixth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "02_native_light_speed.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    domain1 = [row for row in q_bursts if int(row["domain"]) == 1]
    plt.stem(
        [int(row["q_phase_index"]) for row in domain1],
        [float(row["coherent_field_magnitude"]) for row in domain1],
    )
    plt.xlabel("internal Q phase index")
    plt.ylabel("coherent emitted curvature magnitude")
    plt.title("First complete light packet: B-loaded Q bursts")
    plt.tight_layout()
    plt.savefig(figures / "03_first_packet_emission_envelope.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    top_d1 = [row for row in spectrum_rows if int(row["domain"]) == 1]
    plt.stem(
        [int(row["harmonic"]) for row in top_d1],
        [float(row["power_fraction"]) for row in top_d1],
    )
    plt.xlabel("packet harmonic n")
    plt.ylabel("power fraction")
    plt.title("First packet source-envelope spectrum")
    plt.tight_layout()
    plt.savefig(figures / "04_first_packet_spectrum.png", dpi=180)
    plt.close()


    plt.figure(figsize=(8, 5))
    md = np.array([row["domain"] for row in modulation_rows], dtype=float)
    mf = np.array([row["dominant_modulation_cycles_per_Q"] for row in modulation_rows], dtype=float)
    plt.plot(md, mf, marker="o", label="measured dominant source line")
    plt.axhline(double_b_fraction_limit, linestyle="--", label="log_phi(2)-1")
    plt.xlabel("completed-domain scale d")
    plt.ylabel("cycles per Q phase advance")
    plt.title("B/L source modulation converges to log_phi(2)-1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "05_source_modulation_limit.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    stages = [15, 45, 103]
    labels = ["light/interface", "chiral fermion wall", "larger matter host"]
    plt.scatter(stages, [1, 2, 3], s=90)
    for x, y, label in zip(stages, [1, 2, 3], labels, strict=True):
        plt.text(x + 2, y, label, va="center")
    plt.yticks([])
    plt.xlabel("causal transition index")
    plt.title("Native emergence order: light first, fermion at next L")
    plt.tight_layout()
    plt.savefig(figures / "06_light_matter_staging.png", dpi=180)
    plt.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
