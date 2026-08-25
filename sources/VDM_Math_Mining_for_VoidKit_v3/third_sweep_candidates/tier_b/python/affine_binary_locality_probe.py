#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp


@dataclass(frozen=True)
class Witness:
    depth: int
    carry_word_left: str
    carry_word_right: str
    lower_boundary: str
    binary_cut: str
    upper_boundary: str
    cylinder_width: str
    cut_relative_position: str
    left_binary_bit: int
    right_binary_bit: int
    pass_same_carry_prefix: bool
    pass_opposite_binary_bits: bool


def affine_cut(precision: int) -> mp.mpf:
    mp.mp.dps = precision
    phi = (1 + mp.sqrt(5)) / 2
    return -mp.mpf(13) / 2 + mp.log(mp.mpf(4096) / 5) / (2 * mp.log(phi))


def lambda_beta() -> tuple[mp.mpf, mp.mpf]:
    phi = (1 + mp.sqrt(5)) / 2
    lam = mp.log(2) / mp.log(phi)
    beta = mp.log(5) / (2 * mp.log(phi)) - mp.mpf(3) / 2
    return lam, beta


def carry_symbol(y: mp.mpf, p: mp.mpf) -> str:
    y = y % 1
    if p / 2 < y <= p:
        return "9"
    if p < y <= (p + 1) / 2:
        return "7"
    return "8"


def carry_word(y: mp.mpf, p: mp.mpf, depth: int) -> str:
    out: list[str] = []
    for _ in range(depth):
        out.append(carry_symbol(y, p))
        y = (2 * y) % 1
    return "".join(out)


def nearest_refinement_boundaries(p: mp.mpf, depth: int) -> tuple[mp.mpf, mp.mpf]:
    half = mp.mpf("0.5")
    lower_candidates: list[mp.mpf] = []
    upper_candidates: list[mp.mpf] = []
    for k in range(depth + 1):
        denominator = 2**k
        target = denominator * half - p
        m0 = int(mp.floor(target))
        for m in (m0, m0 + 1):
            if 0 <= m < denominator:
                value = (p + m) / denominator
                if value < half:
                    lower_candidates.append(value)
                elif value > half:
                    upper_candidates.append(value)
    if not lower_candidates or not upper_candidates:
        raise RuntimeError(f"failed to bracket binary cut at depth {depth}")
    return max(lower_candidates), min(upper_candidates)


def witness_at_depth(p: mp.mpf, depth: int) -> Witness:
    half = mp.mpf("0.5")
    lower, upper = nearest_refinement_boundaries(p, depth)
    left = (lower + half) / 2
    right = (half + upper) / 2
    left_word = carry_word(left, p, depth)
    right_word = carry_word(right, p, depth)
    width = upper - lower
    relative = (half - lower) / width
    return Witness(
        depth=depth,
        carry_word_left=left_word,
        carry_word_right=right_word,
        lower_boundary=mp.nstr(lower, 60),
        binary_cut=mp.nstr(half, 60),
        upper_boundary=mp.nstr(upper, 60),
        cylinder_width=mp.nstr(width, 60),
        cut_relative_position=mp.nstr(relative, 60),
        left_binary_bit=int(left >= half),
        right_binary_bit=int(right >= half),
        pass_same_carry_prefix=left_word == right_word,
        pass_opposite_binary_bits=int(left >= half) != int(right >= half),
    )


def leavitt_operator_checks(tail_length: int) -> dict[str, object]:
    tails = [format(i, f"0{tail_length}b") for i in range(2**tail_length)]
    relation_failures: list[dict[str, str]] = []
    identity_failures: list[str] = []
    correct_relation_failures = [[0, 0], [0, 0]]
    wrong_relation_failures = [[0, 0], [0, 0]]

    def s(i: str, word: str) -> str:
        return i + word

    def t(i: str, word: str) -> str | None:
        return word[1:] if word.startswith(i) else None

    def wrong_t(_i: str, word: str) -> str | None:
        return word[1:] if word else None

    for tail in tails:
        for i_int, i in enumerate(("0", "1")):
            for j_int, j in enumerate(("0", "1")):
                expected = tail if i == j else None
                observed = t(i, s(j, tail))
                if observed != expected:
                    correct_relation_failures[i_int][j_int] += 1
                    relation_failures.append(
                        {"tail": tail, "i": i, "j": j, "observed": str(observed)}
                    )
                wrong_observed = wrong_t(i, s(j, tail))
                if wrong_observed != expected:
                    wrong_relation_failures[i_int][j_int] += 1

        for first in ("0", "1"):
            word = first + tail
            deleted = t(first, word)
            recovered = s(first, deleted) if deleted is not None else None
            if recovered != word:
                identity_failures.append(word)

    wrong_total = sum(sum(row) for row in wrong_relation_failures)
    return {
        "tail_length": tail_length,
        "basis_tail_count": len(tails),
        "tisj_relation_failures": relation_failures,
        "partition_identity_failures": identity_failures,
        "correct_relation_failure_matrix": correct_relation_failures,
        "negative_control_failure_matrix": wrong_relation_failures,
        "negative_control_failures_detected": wrong_total,
        "pass_tisj": len(relation_failures) == 0,
        "pass_partition_identity": len(identity_failures) == 0,
        "pass_negative_control": wrong_total > 0,
    }


def canonical_orbit_preimage_probe(max_domain: int) -> dict[str, object]:
    lam, beta = lambda_beta()
    p = affine_cut(mp.mp.dps)
    y_values: list[mp.mpf] = []
    for a in range(max_domain + 1):
        j = 6 * (2 ** (a + 1) - 1)
        raw = lam * j + beta
        t = mp.ceil(raw)
        e = raw - t
        y_values.append((e + p) % 1)

    retained_digits = max(40, mp.mp.dps - math.ceil(max_domain * math.log10(2)) - 30)
    tolerance = mp.mpf(10) ** (-retained_digits)
    actual_predecessor_failures: list[int] = []
    alternate_preimage_hits: list[dict[str, int]] = []
    diagnostic_rows: list[dict[str, str | int]] = []
    for a in range(1, max_domain + 1):
        target = y_values[a]
        preimages = [target / 2, (target + 1) / 2]
        actual = y_values[a - 1]
        actual_index = min(range(2), key=lambda i: abs(preimages[i] - actual))
        if abs(preimages[actual_index] - actual) > tolerance:
            actual_predecessor_failures.append(a)
        alternate = preimages[1 - actual_index]
        nearest_distance = min(abs(alternate - yb) for yb in y_values)
        diagnostic_rows.append({
            "target_domain": a,
            "actual_predecessor_error": mp.nstr(abs(preimages[actual_index] - actual), 50),
            "alternate_preimage_nearest_orbit_distance": mp.nstr(nearest_distance, 50),
        })
        for b, yb in enumerate(y_values):
            if abs(alternate - yb) <= tolerance:
                alternate_preimage_hits.append({"target_domain": a, "hit_domain": b})
                break

    repeated_pairs: list[tuple[int, int]] = []
    for i in range(len(y_values)):
        for j in range(i + 1, len(y_values)):
            if abs(y_values[i] - y_values[j]) <= tolerance:
                repeated_pairs.append((i, j))

    return {
        "tested_domain_range": [0, max_domain],
        "actual_predecessor_failures": actual_predecessor_failures,
        "alternate_preimage_hits_in_tested_orbit": alternate_preimage_hits,
        "repeated_orbit_points": repeated_pairs,
        "pass_actual_predecessor": not actual_predecessor_failures,
        "pass_no_alternate_preimage_hit": not alternate_preimage_hits,
        "pass_no_repeated_orbit_point": not repeated_pairs,
        "numeric_tolerance": mp.nstr(tolerance, 30),
        "diagnostic_rows": diagnostic_rows,
        "scope": "finite diagnostic only; the all-depth proof uses irrationality of lambda and orbit injectivity",
    }



def write_orbit_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_orbit_figure(path: Path, orbit_checks: dict[str, object]) -> None:
    rows = orbit_checks["diagnostic_rows"]
    domains = [int(row["target_domain"]) for row in rows]
    actual = [max(float(row["actual_predecessor_error"]), 1e-300) for row in rows]
    alternate = [max(float(row["alternate_preimage_nearest_orbit_distance"]), 1e-300) for row in rows]
    tolerance = float(orbit_checks["numeric_tolerance"])

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.semilogy(domains, actual, label="Actual canonical predecessor error")
    ax.semilogy(domains, alternate, label="Alternate preimage distance to tested orbit")
    ax.axhline(tolerance, linestyle="--", label="Numeric decision tolerance")
    ax.set_xlabel("Target domain A")
    ax.set_ylabel("Circle-coordinate distance")
    ax.set_title("The canonical boundary orbit contains one binary predecessor, not both")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rational_cut_negative_controls(max_depth: int) -> dict[str, object]:
    half = mp.mpf("0.5")
    controls: dict[str, object] = {}
    for label, cut in {"p_equals_zero": mp.mpf("0"), "p_equals_half": half}.items():
        first_hit: dict[str, int] | None = None
        for k in range(max_depth + 1):
            denominator = 2**k
            target = denominator * half - cut
            if target == int(target) and 0 <= int(target) < denominator:
                first_hit = {"depth": k, "m": int(target)}
                break
        controls[label] = {
            "cut": str(cut),
            "first_binary_boundary_hit": first_hit,
            "pass_expected_hit": first_hit is not None,
        }
    return controls


def write_csv(path: Path, witnesses: list[Witness]) -> None:
    rows = [asdict(w) for w in witnesses]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_cut_figure(path: Path, witnesses: list[Witness]) -> None:
    depths = [w.depth for w in witnesses]
    widths = [float(mp.mpf(w.cylinder_width)) for w in witnesses]
    positions = [float(mp.mpf(w.cut_relative_position)) for w in witnesses]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(depths, [-math.log2(width) for width in widths], marker="o")
    axes[0].set_xlabel("Carry refinement depth n")
    axes[0].set_ylabel(r"$-\log_2$ width of the n-cylinder containing $y=1/2$")
    axes[0].set_title("Carry refinement resolves the binary cut only asymptotically")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(depths, positions, marker="o")
    axes[1].axhline(0, linewidth=1)
    axes[1].axhline(1, linewidth=1)
    axes[1].set_xlabel("Carry refinement depth n")
    axes[1].set_ylabel("Relative position of $y=1/2$ inside its carry cylinder")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("The binary cut remains strictly inside one carry cylinder")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_operator_figure(path: Path, checks: dict[str, object]) -> None:
    correct = checks["correct_relation_failure_matrix"]
    wrong = checks["negative_control_failure_matrix"]
    vmax = max(1, max(max(row) for row in wrong))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, matrix, title in (
        (axes[0], correct, r"Correct operators: failures of $t_i s_j=\delta_{ij}I$"),
        (axes[1], wrong, "Negative control: unconditional deletion failures"),
    ):
        image = ax.imshow(matrix, vmin=0, vmax=vmax)
        ax.set_xticks([0, 1], labels=["s₀", "s₁"])
        ax.set_yticks([0, 1], labels=["t₀", "t₁"])
        ax.set_xlabel("Prefix insertion")
        ax.set_ylabel("Prefix deletion")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.colorbar(image, ax=axes, label="Failed binary tails")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-domain", type=int, default=256)
    parser.add_argument("--precision", type=int, default=220)
    parser.add_argument("--tail-length", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cut-figure", type=Path, required=True)
    parser.add_argument("--operator-figure", type=Path, required=True)
    parser.add_argument("--orbit-figure", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cut_figure.parent.mkdir(parents=True, exist_ok=True)
    args.operator_figure.parent.mkdir(parents=True, exist_ok=True)
    args.orbit_figure.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)

    p = affine_cut(args.precision)
    witnesses = [witness_at_depth(p, n) for n in range(1, args.max_depth + 1)]
    operator_checks = leavitt_operator_checks(args.tail_length)
    orbit_checks = canonical_orbit_preimage_probe(args.max_domain)
    negative_controls = rational_cut_negative_controls(args.max_depth)

    count_identity_pass = all(
        sum(2**k for k in range(n + 1)) == 2 ** (n + 1) - 1
        for n in range(1, args.max_depth + 1)
    )
    same_prefix_pass = all(w.pass_same_carry_prefix for w in witnesses)
    opposite_bits_pass = all(w.pass_opposite_binary_bits for w in witnesses)
    pattern_pass = all(
        w.carry_word_left == "7" + "8" * (w.depth - 1)
        for w in witnesses
    )

    theorem_status = {
        "precision_decimal_digits": args.precision,
        "affine_cut_p": mp.nstr(p, 100),
        "p_is_between_zero_and_one": bool(0 < p < 1),
        "tested_depth_range": [1, args.max_depth],
        "refinement_count_identity_pass": count_identity_pass,
        "same_carry_prefix_across_binary_cut_pass": same_prefix_pass,
        "opposite_binary_first_bits_pass": opposite_bits_pass,
        "explicit_7_then_8_pattern_pass": pattern_pass,
        "leavitt_operator_checks": operator_checks,
        "canonical_orbit_preimage_checks": orbit_checks,
        "negative_controls": negative_controls,
        "overall_finite_probe_pass": all(
            [
                count_identity_pass,
                same_prefix_pass,
                opposite_bits_pass,
                pattern_pass,
                operator_checks["pass_tisj"],
                operator_checks["pass_partition_identity"],
                operator_checks["pass_negative_control"],
                orbit_checks["pass_actual_predecessor"],
                orbit_checks["pass_no_alternate_preimage_hit"],
                orbit_checks["pass_no_repeated_orbit_point"],
                all(v["pass_expected_hit"] for v in negative_controls.values()),
            ]
        ),
        "scope": {
            "proved_abstractly": [
                "the binary cut is not a finite carry-refinement boundary",
                "no finite carry prefix determines the first binary bit",
                "the canonical boundary orbit contains only its actual predecessor, not both binary preimages",
            ],
            "certified_finitely": [
                f"carry depths 1..{args.max_depth}",
                f"binary tails of length {args.tail_length}",
                f"canonical boundary domains 0..{args.max_domain}",
            ],
            "open": [
                "an internally generated enlarged lawful retained-state arena supporting both inverse branches",
                "a typed algebraic linearization of those partial maps",
                "a Phase Calculus-native non-sofic group",
            ],
        },
    }

    write_csv(args.output_dir / "binary_cut_witnesses.csv", witnesses)
    (args.output_dir / "theorem_status.json").write_text(
        json.dumps(theorem_status, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "leavitt_operator_checks.json").write_text(
        json.dumps(operator_checks, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "canonical_orbit_preimage_checks.json").write_text(
        json.dumps(orbit_checks, indent=2) + "\n", encoding="utf-8"
    )
    write_orbit_csv(args.output_dir / "canonical_orbit_preimage_distances.csv", orbit_checks["diagnostic_rows"])
    (args.output_dir / "negative_controls.json").write_text(
        json.dumps(negative_controls, indent=2) + "\n", encoding="utf-8"
    )
    write_cut_figure(args.cut_figure, witnesses)
    write_operator_figure(args.operator_figure, operator_checks)
    write_orbit_figure(args.orbit_figure, orbit_checks)

    with args.trace.open("w", encoding="utf-8") as handle:
        for witness in witnesses:
            handle.write(json.dumps({"event": "binary_cut_witness", **asdict(witness)}) + "\n")
        handle.write(json.dumps({"event": "operator_checks", **operator_checks}) + "\n")
        handle.write(json.dumps({"event": "canonical_orbit_preimage_checks", **orbit_checks}) + "\n")
        handle.write(json.dumps({"event": "negative_controls", **negative_controls}) + "\n")

    print(json.dumps(theorem_status, indent=2))
    return 0 if theorem_status["overall_finite_probe_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
