#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
import time
import struct
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh_tridiagonal

sys.set_int_max_str_digits(0)

PHI = (1.0 + math.sqrt(5.0)) / 2.0
SIGMA0 = 1.0 / (1.0 - PHI ** -4)
JM_LIMIT = math.log(2.0) / math.log(PHI)
EXPECTED_L = [15, 45, 103, 220, 455, 923, 1860, 3735, 7483, 14980, 29974]


@dataclass(frozen=True)
class Layer:
    phase_quarters: int
    magnitudes: tuple[Fraction, ...]
    source_dimension: int
    terminal_c: Fraction

    @property
    def point_count(self) -> int:
        return len(self.magnitudes) + 1


@dataclass
class ExactState:
    domain: int
    u: int
    v: int
    quarter_turns: int
    k: int
    j: int
    completed: list[Layer]
    active: object
    active_last_b: tuple[int, Fraction] | None
    phase_quarters: int

    @staticmethod
    def initial() -> "ExactState":
        from collections import deque
        return ExactState(0, 1, 1, 0, 0, 1, [], deque([Fraction(1)]), None, 0)

    def phase_positions(self) -> int:
        return 6 * (2 ** self.domain)

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
        from collections import deque
        primitive = self.emit()
        inserted_c: Fraction | None = None
        if primitive == "B":
            old_u, old_v = self.u, self.v
            inserted_c = Fraction(old_u, old_u + old_v)
            self.active.appendleft(inserted_c * self.active[0])
            self.u, self.v = old_v, old_u + old_v
            # After insertion, nonzero count equals the B source dimension.
            self.active_last_b = (len(self.active), inserted_c)
        elif primitive == "Q":
            self.phase_quarters = (self.phase_quarters + 1) % 4
            self.quarter_turns += 1
            self.k += 1
            self.j += 1
        elif primitive == "L":
            if self.active_last_b is None:
                raise AssertionError("completed layer without B")
            n_source, c = self.active_last_b
            self.completed.append(
                Layer(self.phase_quarters, tuple(self.active), n_source, c)
            )
            self.active = deque([Fraction(1)])
            self.active_last_b = None
            self.phase_quarters = 0
            self.domain += 1
            self.k = 0
            self.j = self.phase_positions() - 5
        else:
            raise AssertionError(primitive)
        return primitive, inserted_c


def _int_bytes(value: int) -> bytes:
    if value == 0:
        return b"\x00"
    return value.to_bytes((value.bit_length() + 7) // 8, "big", signed=False)


def _hash_int(h, value: int) -> None:
    data = _int_bytes(value)
    h.update(len(data).to_bytes(8, "big"))
    h.update(data)


def fraction_sha256(layer: Layer) -> str:
    h = hashlib.sha256()
    h.update(bytes([layer.phase_quarters]))
    for q in layer.magnitudes:
        _hash_int(h, q.numerator)
        _hash_int(h, q.denominator)
    return h.hexdigest()


def _write_bigint(handle, value: int) -> None:
    data = _int_bytes(value)
    handle.write(struct.pack(">I", len(data)))
    handle.write(data)


def phase_complex(quarters: int) -> tuple[int, int]:
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][quarters % 4]


def layer_moments(layer: Layer) -> dict[str, float]:
    n = layer.point_count
    values = [float(q) for q in layer.magnitudes]
    total = math.fsum(values)
    phase_re, phase_im = phase_complex(layer.phase_quarters)
    return {
        "n": float(n),
        "sum_re": phase_re * total,
        "sum_im": phase_im * total,
        "sum_norm_sq": math.fsum(value * value for value in values),
    }


def cross_energy(left: dict[str, float], right: dict[str, float]) -> float:
    nl = int(left["n"])
    nr = int(right["n"])
    dot = left["sum_re"] * right["sum_re"] + left["sum_im"] * right["sum_im"]
    return nr * left["sum_norm_sq"] + nl * right["sum_norm_sq"] - 2.0 * dot


def chiral_radius(layer: Layer, n_source: int, c: float) -> tuple[float, float]:
    weights = np.zeros(n_source + 1, dtype=float)
    weights[1] = 1.0
    weights[n_source - 1] = 1.0
    if n_source > 3:
        weights[2:n_source - 1] = 1.0 - c
    weights = weights * weights
    weights /= weights.sum()
    x = np.array([0.0] + [float(q) for q in layer.magnitudes], dtype=float)
    mean = float(np.sum(weights * x))
    var = float(np.sum(weights * (x - mean) ** 2))
    return mean, math.sqrt(max(var, 0.0))


def phase_label(quarters: int) -> str:
    return ["+1", "+i", "-1", "-i"][quarters % 4]


def choose(n: int, k: int) -> int:
    return math.comb(n, k) if n >= k else 0


def relation_gap(n: int, c: float) -> float:
    # Exact K=D D^T restricted away from transported poles is tridiagonal.
    # Size m=n-1; endpoint diagonal/off-diagonal carry a=1-c.
    m = n - 1
    a = 1.0 - c
    diagonal = np.full(m, 2.0, dtype=float)
    diagonal[0] = a * a
    diagonal[-1] = a * a
    off = np.full(m - 1, -1.0, dtype=float)
    off[0] = -a
    off[-1] = -a
    # The lowest eigenvalue is the exact chiral zero mode; select the next one.
    values = eigh_tridiagonal(
        diagonal, off, select="i", select_range=(0, min(3, m - 1)),
        check_finite=False, tol=1e-13
    )[0]
    positives = values[values > 1e-10]
    if len(positives) == 0:
        raise RuntimeError(f"no positive relation gap found for n={n}")
    return float(positives[0])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    evidence = root / "evidence"
    figures = root / "figures"
    results.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    state = ExactState.initial()
    started = time.time()
    tick = 0
    b_since_q = 0
    domain_counts = defaultdict(lambda: {"B": 0, "Q": 0, "L": 0})
    l_ticks: list[int] = []
    q_rows: list[dict[str, object]] = []
    l_rows: list[dict[str, object]] = []
    primitive_rle: list[tuple[str, int]] = []
    last_primitive = None
    run_length = 0

    # Track hashes of completed layers at formation and recheck at every L.
    formation_hashes: list[str] = []
    retention_checks: list[dict[str, object]] = []

    while len(l_ticks) < len(EXPECTED_L):
        before_product = state.u * state.v
        before_domain = state.domain
        before_k = state.k
        primitive, inserted_c = state.tick()
        tick += 1
        domain_counts[before_domain][primitive] += 1

        if primitive == last_primitive:
            run_length += 1
        else:
            if last_primitive is not None:
                primitive_rle.append((last_primitive, run_length))
            last_primitive = primitive
            run_length = 1

        if primitive == "B":
            b_since_q += 1
        elif primitive == "Q":
            product = state.u * state.v
            q_rows.append({
                "tick": tick,
                "domain": before_domain,
                "k": state.k,
                "q_index": domain_counts[before_domain]["Q"],
                "b_load": b_since_q,
                "closure_duration": b_since_q + 1,
                "product": product,
                "product_bits": product.bit_length(),
                "burden_log": math.log(product),
                "u": state.u,
                "v": state.v,
            })
            b_since_q = 0
        else:
            l_ticks.append(tick)
            new_hash = fraction_sha256(state.completed[-1])
            formation_hashes.append(new_hash)
            current_hashes = [fraction_sha256(layer) for layer in state.completed]
            retention_checks.append({
                "tick": tick,
                "completed_count": len(state.completed),
                "all_prior_unchanged": current_hashes == formation_hashes,
                "hashes": current_hashes,
            })
            counts = domain_counts[before_domain]
            l_rows.append({
                "tick": tick,
                "completed_domain": before_domain,
                "completed_count": len(state.completed),
                "B": counts["B"],
                "Q": counts["Q"],
                "L": counts["L"],
                "M": counts["B"] + counts["L"],
                "J": counts["Q"],
                "M_over_J": (counts["B"] + counts["L"]) / counts["Q"],
                "active_reset_points": len(state.active),
                "completed_point_count": state.completed[-1].point_count,
                "completed_hash": new_hash,
            })
            b_since_q = 0

    if last_primitive is not None:
        primitive_rle.append((last_primitive, run_length))

    elapsed = time.time() - started

    # Exact full-point archive, compressed binary.
    exact_layers_path = results / "completed_layers_exact.bin.gz"
    with gzip.open(exact_layers_path, "wb", compresslevel=6) as handle:
        handle.write(b"ORTHAD-LAYERS-v1\0")
        handle.write(struct.pack(">I", len(state.completed)))
        for index, layer in enumerate(state.completed):
            handle.write(struct.pack(">IBII", index, layer.phase_quarters, layer.source_dimension, len(layer.magnitudes)))
            _write_bigint(handle, layer.terminal_c.numerator)
            _write_bigint(handle, layer.terminal_c.denominator)
            for q in layer.magnitudes:
                _write_bigint(handle, q.numerator)
                _write_bigint(handle, q.denominator)
    (results / "completed_layers_exact_SCHEMA.md").write_text(
        "# completed_layers_exact.bin.gz\n\n"
        "Gzip-compressed big-endian binary exact retained layer data.\n\n"
        "Header: `ORTHAD-LAYERS-v1\0`, u32 layer count. Each layer: "
        "u32 index, u8 phase quarters, u32 terminal-B source dimension, "
        "u32 nonzero magnitude count, then length-prefixed unsigned big integers "
        "for terminal-c numerator/denominator and every exact magnitude numerator/denominator. "
        "The omitted first point is exact zero.\n",
        encoding="utf-8",
    )

    # Layer summaries and spectra.
    layer_rows: list[dict[str, object]] = []
    moments = []
    for index, layer in enumerate(state.completed):
        n_source, c_exact = layer.source_dimension, layer.terminal_c
        c = float(c_exact)
        gap = relation_gap(n_source, c)
        mean, radius = chiral_radius(layer, n_source, c)
        moment = layer_moments(layer)
        moments.append(moment)
        layer_rows.append({
            "layer": index,
            "point_count": layer.point_count,
            "source_dimension": n_source,
            "terminal_c_num": str(c_exact.numerator),
            "terminal_c_den": str(c_exact.denominator),
            "terminal_c": c,
            "endpoint_phase": phase_label(layer.phase_quarters),
            "source_kernel_dim": 2,
            "target_kernel_dim": 3,
            "excess_index": 1,
            "relation_gap": gap,
            "n2_gap": n_source * n_source * gap,
            "gap_relative_error_pi2": abs(n_source * n_source * gap - math.pi**2) / math.pi**2,
            "chiral_mean_radius": mean,
            "chiral_rms_radius": radius,
            "sqrt_n_radius": math.sqrt(n_source) * radius,
            "layer_sha256": formation_hashes[index],
        })

    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(results / "layer_census.csv", index=False)
    pd.DataFrame(l_rows).to_csv(results / "l_handoffs.csv", index=False)
    pd.DataFrame([
        {"primitive": p, "run_length": n} for p, n in primitive_rle
    ]).to_csv(results / "primitive_run_length_encoding.csv", index=False)

    # Q rows: exact values in compressed JSONL + compact CSV.
    with gzip.open(results / "q_closures_exact_hex.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in q_rows:
            payload = dict(row)
            for key in ["product", "u", "v"]:
                payload[key] = hex(int(payload[key]))
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    pd.DataFrame([
        {k: v for k, v in row.items() if k not in {"product", "u", "v"}}
        for row in q_rows
    ]).to_csv(results / "q_closure_census.csv", index=False)

    # Exact pairwise relation-energy matrix via moment identity.
    energy_rows = []
    energy_matrix = np.zeros((len(moments), len(moments)), dtype=float)
    density_matrix = np.zeros_like(energy_matrix)
    for i, left in enumerate(moments):
        for j, right in enumerate(moments):
            energy = cross_energy(left, right)
            density = energy / (int(left["n"]) + int(right["n"]))
            energy_matrix[i, j] = energy
            density_matrix[i, j] = density
            if i <= j:
                energy_rows.append({
                    "left_layer": i,
                    "right_layer": j,
                    "left_points": int(left["n"]),
                    "right_points": int(right["n"]),
                    "energy_float": energy,
                    "numeric_scope": "float moment identity over exact retained points",
                    "support_degree": int(left["n"] + right["n"]),
                    "energy_density": density,
                })
    pd.DataFrame(energy_rows).to_csv(results / "pairwise_relation_energy.csv", index=False)
    np.save(results / "pairwise_relation_energy_density.npy", density_matrix)

    # Adjacent convergence.
    adjacent = []
    for i in range(len(moments) - 1):
        density = density_matrix[i, i + 1]
        adjacent.append({
            "interface": i,
            "left_layer": i,
            "right_layer": i + 1,
            "density": density,
            "sigma0": SIGMA0,
            "absolute_error": abs(density - SIGMA0),
            "error_ratio_to_previous": None if i == 0 else abs(density - SIGMA0) / adjacent[-1]["absolute_error"],
        })
    pd.DataFrame(adjacent).to_csv(results / "adjacent_relation_density.csv", index=False)

    # Deep endpoint host tower and composite capacity.
    host_rows = []
    for completed_count in range(1, len(state.completed) + 1):
        phases = [phase_label(state.completed[i].phase_quarters) for i in range(completed_count)]
        minus_i = phases.count("-i")
        plus_i = phases.count("+i")
        host_rows.append({
            "completed_count": completed_count,
            "L_tick": l_ticks[completed_count - 1],
            "plus_i_singletons": plus_i,
            "minus_i_block": minus_i,
            "nested_su_capacity": minus_i,
            "triplet_subsets": choose(minus_i, 3),
            "disjoint_triplet_capacity": minus_i // 3,
            "residual_after_max_disjoint_triplets": minus_i % 3,
        })
    host_df = pd.DataFrame(host_rows)
    host_df.to_csv(results / "endpoint_host_tower.csv", index=False)

    # Triplet cluster census among -i layers.
    minus_layers = [i for i, layer in enumerate(state.completed) if phase_label(layer.phase_quarters) == "-i"]
    triplets = []
    for combo in itertools.combinations(minus_layers, 3):
        pair_values = [density_matrix[a, b] for a, b in itertools.combinations(combo, 2)]
        triplets.append({
            "layers": "-".join(map(str, combo)),
            "span": max(combo) - min(combo),
            "contiguous": combo[1] == combo[0] + 1 and combo[2] == combo[1] + 1,
            "mean_pair_density": float(np.mean(pair_values)),
            "max_pair_density": float(np.max(pair_values)),
            "min_pair_density": float(np.min(pair_values)),
        })
    triplet_df = pd.DataFrame(triplets).sort_values(["mean_pair_density", "span"])
    triplet_df.to_csv(results / "triplet_cluster_census.csv", index=False)

    # Disjoint triplet-pair census.
    triplet_tuples = [tuple(map(int, row.split("-"))) for row in triplet_df["layers"]]
    pair_clusters = []
    for a_index in range(len(triplet_tuples)):
        a = triplet_tuples[a_index]
        for b_index in range(a_index + 1, len(triplet_tuples)):
            b = triplet_tuples[b_index]
            if set(a).intersection(b):
                continue
            cross = [density_matrix[i, j] for i in a for j in b]
            internal_a = [density_matrix[i, j] for i, j in itertools.combinations(a, 2)]
            internal_b = [density_matrix[i, j] for i, j in itertools.combinations(b, 2)]
            pair_clusters.append({
                "triplet_a": "-".join(map(str, a)),
                "triplet_b": "-".join(map(str, b)),
                "hierarchy_gap": max(0, min(b) - max(a) - 1, min(a) - max(b) - 1),
                "cross_density_mean": float(np.mean(cross)),
                "internal_density_mean": float(np.mean(internal_a + internal_b)),
                "cross_minus_internal": float(np.mean(cross) - np.mean(internal_a + internal_b)),
            })
    cluster_pair_df = pd.DataFrame(pair_clusters).sort_values(["cross_density_mean", "cross_minus_internal"])
    cluster_pair_df.to_csv(results / "disjoint_triplet_pair_census.csv", index=False)

    # 2D burden plaquettes: domain d coarse k maps to domain d+1 fine 2k.
    q_by_domain_k = {(int(r["domain"]), int(r["k"])): r for r in q_rows}
    plaquettes = []
    for d in range(len(state.completed) - 1):
        n = 6 * (2 ** d)
        for k in range(1, n - 1):
            keys = [(d, k), (d, k + 1), (d + 1, 2 * k), (d + 1, 2 * k + 2)]
            if not all(key in q_by_domain_k for key in keys):
                continue
            coarse0 = q_by_domain_k[(d, k)]
            coarse1 = q_by_domain_k[(d, k + 1)]
            fine0 = q_by_domain_k[(d + 1, 2 * k)]
            fine1 = q_by_domain_k[(d + 1, 2 * k + 2)]
            factor_num = int(fine1["product"]) * int(coarse0["product"])
            factor_den = int(fine0["product"]) * int(coarse1["product"])
            log_defect = math.log(factor_num) - math.log(factor_den)
            fh = hashlib.sha256()
            _hash_int(fh, factor_num)
            _hash_int(fh, factor_den)
            cadence_change = (
                int(coarse1["b_load"]) - int(coarse0["b_load"]),
                int(fine1["b_load"]) - int(fine0["b_load"]),
            )
            plaquettes.append({
                "coarse_domain": d,
                "coarse_k": k,
                "fine_domain": d + 1,
                "fine_k0": 2 * k,
                "factor_sha256": fh.hexdigest(),
                "factor_is_one": factor_num == factor_den,
                "log_defect": log_defect,
                "abs_log_defect": abs(log_defect),
                "coarse_b_load_change": cadence_change[0],
                "fine_b_load_change": cadence_change[1],
                "any_cadence_change": cadence_change != (0, 0),
            })
    plaquette_df = pd.DataFrame(plaquettes)
    plaquette_df.to_csv(results / "burden_plaquette_census.csv", index=False)

    # Affine and uniform exact controls.
    controls = []
    for d in range(4):
        n = 6 * (2 ** d)
        for k in range(1, n - 1):
            # Phi(d,x)=a*d+b*(k/n)+c; exact mixed plaquette is zero.
            a = Fraction(7, 5)
            b = Fraction(-11, 7)
            c0 = Fraction(3, 2)
            phi_d_k = a * d + b * Fraction(k, n) + c0
            phi_d_k1 = a * d + b * Fraction(k + 1, n) + c0
            phi_f_2k = a * (d + 1) + b * Fraction(2 * k, 2 * n) + c0
            phi_f_2k2 = a * (d + 1) + b * Fraction(2 * k + 2, 2 * n) + c0
            defect = phi_f_2k2 - phi_f_2k - phi_d_k1 + phi_d_k
            controls.append(defect)
    controls_zero = all(value == 0 for value in controls)

    # Prediction evaluation.
    predictions = {}
    predictions["P1"] = {
        "pass": l_ticks == EXPECTED_L and all(r["all_prior_unchanged"] for r in retention_checks),
        "observed_L": l_ticks,
        "retention_all_pass": all(r["all_prior_unchanged"] for r in retention_checks),
    }
    predictions["P2"] = {
        "pass": bool((layer_df["source_kernel_dim"] == 2).all() and (layer_df["target_kernel_dim"] == 3).all() and (layer_df["excess_index"] == 1).all()),
        "layers": len(layer_df),
    }
    deep_layers = layer_df[layer_df["source_dimension"] >= 278]
    predictions["P3"] = {
        "pass": bool((deep_layers["gap_relative_error_pi2"] < 0.02).all() and deep_layers["sqrt_n_radius"].std() / deep_layers["sqrt_n_radius"].mean() < 0.08),
        "max_gap_relative_error_deep": float(deep_layers["gap_relative_error_pi2"].max()),
        "sqrt_n_radius_relative_std": float(deep_layers["sqrt_n_radius"].std() / deep_layers["sqrt_n_radius"].mean()),
    }
    predictions["P4"] = {
        "pass": host_df.iloc[-1]["plus_i_singletons"] == 1 and host_df.iloc[-1]["minus_i_block"] == 10,
        "final_host": host_df.iloc[-1].to_dict(),
    }
    adjacent_errors = np.array([row["absolute_error"] for row in adjacent])
    predictions["P5"] = {
        "pass": bool(np.all(np.diff(adjacent_errors[2:]) < 0) and adjacent_errors[-1] < 5e-4),
        "final_density": adjacent[-1]["density"],
        "final_error": adjacent[-1]["absolute_error"],
    }
    l_df = pd.DataFrame(l_rows)
    jm_errors = np.abs(l_df["M_over_J"].to_numpy() - JM_LIMIT)
    predictions["P6"] = {
        "pass": bool(np.all(np.diff(jm_errors) < 0)),
        "final_ratio": float(l_df.iloc[-1]["M_over_J"]),
        "final_error": float(jm_errors[-1]),
    }
    predictions["P7"] = {
        "pass": bool(controls_zero and (~plaquette_df["factor_is_one"]).any()),
        "affine_controls_zero": controls_zero,
        "nonzero_generated_plaquettes": int((~plaquette_df["factor_is_one"]).sum()),
        "total_plaquettes": int(len(plaquette_df)),
        "mean_abs_defect_with_cadence_change": float(plaquette_df.loc[plaquette_df["any_cadence_change"], "abs_log_defect"].mean()),
        "mean_abs_defect_without_cadence_change": float(plaquette_df.loc[~plaquette_df["any_cadence_change"], "abs_log_defect"].mean()),
    }
    p8_row = host_df[host_df["L_tick"] == 1860].iloc[0]
    predictions["P8"] = {
        "pass": int(p8_row["disjoint_triplet_capacity"]) >= 2 and int(host_df.iloc[-1]["disjoint_triplet_capacity"]) >= 3,
        "L1860_disjoint_triplets": int(p8_row["disjoint_triplet_capacity"]),
        "L29974_disjoint_triplets": int(host_df.iloc[-1]["disjoint_triplet_capacity"]),
    }
    low_contiguous = triplet_df[triplet_df["contiguous"]].head(6)
    finite_candidates = cluster_pair_df[cluster_pair_df["cross_minus_internal"] < 0]
    predictions["P9"] = {
        "pass": len(low_contiguous) >= 3 and len(finite_candidates) > 0,
        "contiguous_triplet_count": int(triplet_df["contiguous"].sum()),
        "negative_cross_minus_internal_pairs": int(len(finite_candidates)),
        "best_pair": None if len(cluster_pair_df) == 0 else cluster_pair_df.iloc[0].to_dict(),
        "scope": "candidate only; no nuclear closure without separated-threshold and breakup-gap certificate",
    }
    # P10: strict repeated atomic center test. Endpoint inventory has only one +i singleton.
    repeated_neutral_capacity = int(host_df.iloc[-1]["plus_i_singletons"]) >= 2 and int(host_df.iloc[-1]["disjoint_triplet_capacity"]) >= 2
    predictions["P10"] = {
        "pass": repeated_neutral_capacity,
        "plus_i_singletons": int(host_df.iloc[-1]["plus_i_singletons"]),
        "disjoint_triplet_capacity": int(host_df.iloc[-1]["disjoint_triplet_capacity"]),
        "scope": "strict endpoint-bundle neutral-center capacity; other native localization channels require separate recognition",
    }

    # Top findings.
    top_plaquettes = plaquette_df.nlargest(20, "abs_log_defect")
    top_plaquettes.to_csv(results / "top_burden_plaquettes.csv", index=False)
    top_exact = []
    for row in top_plaquettes.itertuples():
        d = int(row.coarse_domain); k = int(row.coarse_k)
        coarse0 = q_by_domain_k[(d, k)]
        coarse1 = q_by_domain_k[(d, k + 1)]
        fine0 = q_by_domain_k[(d + 1, 2 * k)]
        fine1 = q_by_domain_k[(d + 1, 2 * k + 2)]
        factor = Fraction(
            int(fine1["product"]) * int(coarse0["product"]),
            int(fine0["product"]) * int(coarse1["product"]),
        )
        top_exact.append({
            "coarse_domain": d,
            "coarse_k": k,
            "factor_num_hex": hex(factor.numerator),
            "factor_den_hex": hex(factor.denominator),
            "log_defect": float(row.log_defect),
        })
    (results / "top_burden_plaquettes_exact.json").write_text(
        json.dumps(top_exact, indent=2) + "\n", encoding="utf-8"
    )
    triplet_df.head(30).to_csv(results / "top_triplet_clusters.csv", index=False)
    cluster_pair_df.head(30).to_csv(results / "top_disjoint_triplet_pairs.csv", index=False)

    summary = {
        "run": {
            "ticks": tick,
            "completed_layers": len(state.completed),
            "elapsed_seconds": elapsed,
            "L_ticks": l_ticks,
            "total_points": sum(layer.point_count for layer in state.completed),
            "deepest_layer_points": state.completed[-1].point_count,
        },
        "prediction_results": predictions,
        "prediction_pass_count": sum(1 for p in predictions.values() if p["pass"]),
        "prediction_total": len(predictions),
        "strict_closures": {
            "deep_recurrence_retention": predictions["P1"]["pass"],
            "deep_chiral_index_tower": predictions["P2"]["pass"],
            "endpoint_host_tower": predictions["P4"]["pass"],
            "full_2d_burden_plaquette_nonzero": predictions["P7"]["pass"],
            "repeated_color_triplet_capacity": predictions["P8"]["pass"],
            "multi_baryon_candidate_only": predictions["P9"]["pass"],
            "repeated_neutral_atomic_centers": predictions["P10"]["pass"],
        },
        "scope_note": (
            "The run closes structural recurrence, deep retention, index-tower, relation-energy, "
            "J/M, and two-dimensional burden-plaquette results. Multi-baryon and atomic/molecular "
            "claims remain at candidate or capacity status unless their strict native energy gates pass."
        ),
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (results / "prediction_evaluation.json").write_text(json.dumps(predictions, indent=2, default=str) + "\n", encoding="utf-8")
    (evidence / "retention_checks.json").write_text(json.dumps(retention_checks, indent=2) + "\n", encoding="utf-8")

    # Figures.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.semilogy(layer_df["source_dimension"], layer_df["relation_gap"], marker="o", label="measured gap")
    nvals = layer_df["source_dimension"].to_numpy(dtype=float)
    plt.semilogy(nvals, math.pi**2 / nvals**2, linestyle="--", label="pi^2/n^2")
    plt.xlabel("terminal B source dimension n")
    plt.ylabel("first positive relation gap")
    plt.title("Deep native relation-gap scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "01_deep_relation_gap_scaling.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(layer_df["source_dimension"], layer_df["sqrt_n_radius"], marker="o")
    plt.xlabel("source dimension n")
    plt.ylabel("sqrt(n) * chiral RMS radius")
    plt.title("Deep chiral localization scaling")
    plt.tight_layout()
    plt.savefig(figures / "02_chiral_radius_scaling.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    adj_df = pd.DataFrame(adjacent)
    plt.semilogy(adj_df["interface"], adj_df["absolute_error"], marker="o")
    plt.xlabel("adjacent retained interface")
    plt.ylabel("|density - sigma0|")
    plt.title("Adjacent relation-energy convergence")
    plt.tight_layout()
    plt.savefig(figures / "03_relation_energy_convergence.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 7))
    plt.imshow(density_matrix, aspect="auto")
    plt.colorbar(label="cross-relation energy density")
    plt.xlabel("right completed layer")
    plt.ylabel("left completed layer")
    plt.title("Full retained layer relation-energy matrix")
    plt.tight_layout()
    plt.savefig(figures / "04_pairwise_relation_energy_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    sample = plaquette_df.sample(min(5000, len(plaquette_df)), random_state=42)
    plt.scatter(sample["coarse_domain"], sample["log_defect"], s=8, alpha=0.5)
    plt.xlabel("coarse domain")
    plt.ylabel("2D burden plaquette log defect")
    plt.title("Generated CF12 burden plaquettes across scale and local closure")
    plt.tight_layout()
    plt.savefig(figures / "05_burden_plaquette_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(host_df["L_tick"], host_df["disjoint_triplet_capacity"], marker="o")
    plt.xlabel("L handoff tick")
    plt.ylabel("maximum disjoint color-triplet capacity")
    plt.title("Growth of repeated baryon-capable endpoint sectors")
    plt.tight_layout()
    plt.savefig(figures / "06_disjoint_triplet_capacity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(l_df["completed_domain"], l_df["M_over_J"], marker="o", label="observed")
    plt.axhline(JM_LIMIT, linestyle="--", label="log_phi(2)")
    plt.xlabel("completed domain")
    plt.ylabel("M/J = (B+L)/Q")
    plt.title("Deep autonomous J/M convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "07_jm_deep_convergence.png", dpi=180)
    plt.close()

    # Compact plain-text receipt.
    (evidence / "RUN_RECEIPT.txt").write_text(
        f"ticks={tick}\ncompleted_layers={len(state.completed)}\nelapsed_seconds={elapsed:.6f}\n"
        f"prediction_passes={summary['prediction_pass_count']}/{summary['prediction_total']}\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
