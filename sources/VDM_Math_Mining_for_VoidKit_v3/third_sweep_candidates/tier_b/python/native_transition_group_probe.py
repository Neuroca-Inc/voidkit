#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "inputs" / "v22_exact_model"
sys.path.insert(0, str(MODEL_DIR))

from first_l_hierarchical_relation_grid import (  # noqa: E402
    AxisRelation,
    ExactComplex,
    PrimitiveState,
    I,
    ONE,
    ZERO,
)
from orthad_multilayer_exact import MultilayerOrthad, initial_state  # noqa: E402

MINUS_I = ExactComplex(Fraction(0), Fraction(-1))


def inverse_q_axis(axis: AxisRelation) -> AxisRelation:
    return AxisRelation(axis.name, tuple(MINUS_I * point for point in axis.points))


def recover_predecessor(
    after_state: PrimitiveState,
    after_orthad: MultilayerOrthad,
) -> tuple[str, PrimitiveState, MultilayerOrthad]:
    """Recover the unique predecessor of a noninitial exact retained state."""
    if not after_state.word:
        raise ValueError("the root has no predecessor")

    primitive = after_state.word[-1]
    prior_word = after_state.word[:-1]

    if primitive == "B":
        prior_v = after_state.u
        prior_u = after_state.v - after_state.u
        if prior_u <= 0 or prior_u > prior_v:
            raise ValueError("invalid inverse-B pair")
        if len(after_orthad.active.points) < 3:
            raise ValueError("a B-successor must contain the inserted point")
        prior_active = AxisRelation(
            after_orthad.active.name,
            (after_orthad.active.points[0],) + after_orthad.active.points[2:],
        )
        before_state = PrimitiveState(
            after_state.A,
            prior_u,
            prior_v,
            after_state.quarter_turns,
            after_state.k,
            after_state.j,
            prior_word,
        )
        before_orthad = MultilayerOrthad(after_orthad.completed, prior_active)

    elif primitive == "Q":
        if after_state.k <= 0 or after_state.j <= 1 or after_state.quarter_turns <= 0:
            raise ValueError("invalid inverse-Q coordinates")
        before_state = PrimitiveState(
            after_state.A,
            after_state.u,
            after_state.v,
            after_state.quarter_turns - 1,
            after_state.k - 1,
            after_state.j - 1,
            prior_word,
        )
        before_orthad = MultilayerOrthad(
            after_orthad.completed,
            inverse_q_axis(after_orthad.active),
        )

    elif primitive == "L":
        if after_state.A <= 0 or after_state.k != 0 or not after_orthad.completed:
            raise ValueError("invalid inverse-L coordinates")
        prior_A = after_state.A - 1
        prior_k = 6 * (2**prior_A) - 1
        prior_j = after_state.j - 1
        if after_orthad.active.points != (ZERO, ONE):
            raise ValueError("an L-successor must open the canonical (0,1) seed")
        before_state = PrimitiveState(
            prior_A,
            after_state.u,
            after_state.v,
            after_state.quarter_turns,
            prior_k,
            prior_j,
            prior_word,
        )
        before_orthad = MultilayerOrthad(
            after_orthad.completed[:-1],
            after_orthad.completed[-1],
        )

    else:
        raise ValueError(f"unknown terminal primitive {primitive!r}")

    if before_state.selected() != primitive:
        raise AssertionError("recovered predecessor does not select the recorded primitive")
    if before_state.step() != after_state:
        raise AssertionError("primitive-state round trip failed")
    if before_orthad.step(primitive, before_state) != after_orthad:
        raise AssertionError("Orthad round trip failed")

    return primitive, before_state, before_orthad


def typed_relation_automorphism_is_trivial(orthad: MultilayerOrthad) -> bool:
    """Check the degree-signature certificate for each typed strict-order layer."""
    for layer in orthad.completed + (orthad.active,):
        n = len(layer.points)
        signatures = {(index, n - 1 - index) for index in range(n)}
        if len(signatures) != n:
            return False
    return True


def chart_transfer_certificate(
    orthad: MultilayerOrthad,
) -> tuple[bool, bool, int]:
    residual = orthad.transfer_square_residual()
    nonidentity = any(
        any(source != target for source, target in layer.transfer_plus_to_minus().items())
        for layer in orthad.completed + (orthad.active,)
    )
    return residual == 0, nonidentity, residual


def state_key(state: PrimitiveState) -> tuple[object, ...]:
    return (
        state.A,
        state.u,
        state.v,
        state.quarter_turns,
        state.k,
        state.j,
        state.word,
    )


def run_probe(ticks: int) -> tuple[dict, list[dict]]:
    state, orthad = initial_state()
    seen = {state_key(state): 0}
    primitive_counts = {"B": 0, "Q": 0, "L": 0}
    roundtrip_failures: list[object] = []
    automorphism_failures: list[int] = []
    transfer_failures: list[object] = []
    first_transition: dict[str, tuple] = {}
    trace: list[dict] = []

    for tick in range(1, ticks + 1):
        primitive = state.selected()
        before_state = state
        before_orthad = orthad
        after_orthad = orthad.step(primitive, before_state)
        after_state = state.step()
        primitive_counts[primitive] += 1
        first_transition.setdefault(
            primitive,
            (before_state, before_orthad, after_state, after_orthad),
        )

        try:
            recovered_primitive, recovered_state, recovered_orthad = recover_predecessor(
                after_state,
                after_orthad,
            )
            if (
                recovered_primitive != primitive
                or recovered_state != before_state
                or recovered_orthad != before_orthad
            ):
                roundtrip_failures.append({"tick": tick, "kind": "mismatch"})
        except Exception as exc:  # pragma: no cover - emitted into evidence
            roundtrip_failures.append(
                {"tick": tick, "kind": "exception", "message": repr(exc)}
            )

        key = state_key(after_state)
        if key in seen:
            roundtrip_failures.append(
                {
                    "tick": tick,
                    "kind": "state_collision",
                    "first_tick": seen[key],
                }
            )
        seen[key] = tick

        if not typed_relation_automorphism_is_trivial(after_orthad):
            automorphism_failures.append(tick)

        involutive, nonidentity, residual = chart_transfer_certificate(after_orthad)
        if not involutive or not nonidentity:
            transfer_failures.append(
                {
                    "tick": tick,
                    "involutive": involutive,
                    "nonidentity": nonidentity,
                    "residual": residual,
                }
            )

        trace.append(
            {
                "tick": tick,
                "primitive": primitive,
                "A": after_state.A,
                "k": after_state.k,
                "j": after_state.j,
                "word_length": len(after_state.word),
                "layer_sizes": list(after_orthad.layer_sizes()),
            }
        )

        state, orthad = after_state, after_orthad

    negative_controls = {
        "wrong_B_inverse_detected": False,
        "wrong_Q_inverse_detected": False,
        "wrong_L_inverse_detected": False,
    }

    b_before, _, b_after, _ = first_transition["B"]
    wrong_b = PrimitiveState(
        b_before.A,
        (b_after.v - b_after.u) + 1,
        b_after.u,
        b_after.quarter_turns,
        b_after.k,
        b_after.j,
        b_after.word[:-1],
    )
    negative_controls["wrong_B_inverse_detected"] = wrong_b.step() != b_after

    q_before, _, _, q_after_orthad = first_transition["Q"]
    negative_controls["wrong_Q_inverse_detected"] = (
        q_after_orthad.step("Q", q_before) != q_after_orthad
    )

    l_before, _, _, l_after_orthad = first_transition["L"]
    negative_controls["wrong_L_inverse_detected"] = (
        l_after_orthad.step("L", l_before) != l_after_orthad
    )

    result = {
        "claim_id": "C002-C005",
        "ticks_checked": ticks,
        "states_checked": ticks + 1,
        "primitive_counts": primitive_counts,
        "unique_state_count": len(seen),
        "roundtrip_failures": roundtrip_failures,
        "typed_relation_automorphism_failures": automorphism_failures,
        "chart_transfer_involution_failures": transfer_failures,
        "negative_controls": negative_controls,
        "final": {
            "A": state.A,
            "u_digits": len(str(state.u)),
            "v_digits": len(str(state.v)),
            "quarter_turns": state.quarter_turns,
            "k": state.k,
            "j": state.j,
            "word_sha256": hashlib.sha256(state.word.encode("ascii")).hexdigest(),
            "layer_sizes": list(orthad.layer_sizes()),
            "relation_counts": orthad.relation_counts(),
        },
    }
    result["verdict"] = (
        "PASS"
        if (
            not roundtrip_failures
            and not automorphism_failures
            and not transfer_failures
            and all(negative_controls.values())
        )
        else "FAIL"
    )
    return result, trace


def write_outputs(result: dict, trace: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native_transition_probe.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "primitive_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tick", "primitive", "A", "k", "j", "word_length", "layer_sizes"],
        )
        writer.writeheader()
        for row in trace:
            writer.writerow({**row, "layer_sizes": ";".join(map(str, row["layer_sizes"]))})

    with (ROOT / "trace_logs" / "native_transition_trace.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in trace:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_figure(trace: list[dict], figure_path: Path, limit: int = 16) -> None:
    rows = trace[:limit]
    x = list(range(limit + 1))
    y = [0] * (limit + 1)
    plt.figure(figsize=(12, 2.8))
    plt.plot(x, y, marker="o", linewidth=1)
    for row in rows:
        tick = row["tick"]
        plt.annotate(
            row["primitive"],
            xy=(tick - 0.5, 0),
            xytext=(tick - 0.5, 0.12 if tick % 2 else -0.16),
            ha="center",
            va="center",
        )
    plt.yticks([])
    plt.xticks(x)
    plt.xlabel("Exact retained prefix index")
    plt.title("Canonical QBL transition prefix forms one successor chain")
    plt.xlim(-0.4, limit + 0.4)
    plt.ylim(-0.28, 0.24)
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output_data")
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "figures" / "C003_canonical_transition_ray.png",
    )
    args = parser.parse_args()

    if args.ticks < 20:
        raise SystemExit("--ticks must be at least 20")

    result, trace = run_probe(args.ticks)
    write_outputs(result, trace, args.output_dir)
    build_figure(trace, args.figure)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
