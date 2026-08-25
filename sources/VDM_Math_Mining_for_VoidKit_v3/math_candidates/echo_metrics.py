#!/usr/bin/env python3
"""
CEG Instrumentation — Core Echo Metrics

Model-agnostic CEG computation and echo protocol runner.

Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
Dual-license: academic open / commercial by written permission.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from adapters.base_adapter import EchoAdapter


# ---------------------------------------------------------------------------
# Primitive metrics (no model dependency)
# ---------------------------------------------------------------------------

def ceg(baseline_err: float, assisted_err: float) -> float:
    """Counterfactual Echo Gain.

    CEG = (E_baseline - E_assisted) / E_baseline

    Returns a value in [0, 1].  CEG = 0 means assistance did no better than
    random; CEG = 1 means perfect recovery.

    Parameters
    ----------
    baseline_err : float
        Echo error with random (energy-matched) corrections.
    assisted_err : float
        Echo error with structured assistance corrections.

    Returns
    -------
    float
        Clamped to [0, 1].  Returns 0.0 if baseline_err <= 0.
    """
    if baseline_err <= 0.0:
        return 0.0
    x = (baseline_err - assisted_err) / baseline_err
    return float(max(0.0, min(1.0, x)))


def _lam_key(lam: float) -> str:
    """Canonical string key for lambda values (avoids float-repr issues)."""
    return f"{float(lam):.12g}"


# ---------------------------------------------------------------------------
# Echo protocol runner (model-agnostic via adapter)
# ---------------------------------------------------------------------------

def run_echo_protocol(
    adapter: "EchoAdapter",
    seeds: List[int],
    steps: int,
    dt: float,
    lambdas: List[float],
    budget: float,
    *,
    min_gate_pass_rate: float = 10.0 / 12.0,
    ceg_gate_threshold: float = 0.05,
    enforce_rp1: bool = True,
) -> Dict[str, Any]:
    """Run the full CEG echo protocol.

    Parameters
    ----------
    adapter : EchoAdapter
        Model-specific adapter implementing forward/reverse/norm.
    seeds : list[int]
        Deterministic seeds for reproducibility.
    steps : int
        Number of integration steps in each forward/reverse pass.
    dt : float
        Time step size.
    lambdas : list[float]
        Assistance budget multipliers.  Must include 0.0 as control.
    budget : float
        Base energy budget for per-step corrections (H-norm of correction).
    min_gate_pass_rate : float
        Fraction of seeds that must pass instrument gates (default 10/12).
    ceg_gate_threshold : float
        Minimum median CEG for G5 to pass (default 0.05).
    enforce_rp1 : bool
        If True, skip reverse/assisted runs when G1/G2/G4 fail for a seed.

    Returns
    -------
    dict
        Keys: seeds, lambdas, per_seed, ceg_summary, gate_ledger_per_seed,
        gate_ledger_summary, telemetry_rows, timing.
    """
    t0 = time.time()

    # Ensure 0.0 in lambdas for the control condition
    if 0.0 not in [float(l) for l in lambdas]:
        lambdas = [0.0] + list(lambdas)
    lambdas = sorted(set(float(l) for l in lambdas))

    per_seed: List[Dict[str, Any]] = []
    telemetry_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # --- Initial state ---
        state0 = adapter.initial_state(seed)

        # --- Forward pass ---
        state_fwd = adapter.copy_state(state0)
        fwd_diagnostics: List[Dict[str, Any]] = []
        for j in range(steps):
            state_fwd, diag = adapter.forward_step(state_fwd, dt)
            fwd_diagnostics.append(diag)

        # --- RP-1: Instrument calibration gates ---
        rp1 = adapter.calibration_gates(state0, dt, steps)
        g1_pass = rp1.get("G1_passed", True)
        g2_pass = rp1.get("G2_passed", True)
        g4_pass = rp1.get("G4_passed", True)
        rp1_ok = g1_pass and g2_pass and g4_pass

        if enforce_rp1 and not rp1_ok:
            per_seed.append({
                "seed": seed,
                "baseline_err": {},
                "assisted_err": {},
                "ceg": {},
                "gates_diag": rp1,
                "skipped": True,
            })
            continue

        # --- Reverse pass for each lambda ---
        baseline_errs: Dict[str, float] = {}
        assisted_errs: Dict[str, float] = {}
        work_summaries: Dict[str, Dict[str, float]] = {}
        ceg_map: Dict[str, float] = {}

        for lam in lambdas:
            lam_key = _lam_key(lam)
            actual_budget = budget * lam

            # Baseline reverse: random corrections with same energy budget
            state_bl = adapter.copy_state(state_fwd)
            bl_work = 0.0
            for i in range(steps):
                correction = adapter.random_correction(state_bl, actual_budget, rng)
                state_bl = adapter.apply_correction(state_bl, correction)
                state_bl, _ = adapter.reverse_step(state_bl, dt)
                bl_work += adapter.correction_work(correction)

                telemetry_rows.append({
                    "seed": seed, "lambda": float(lam), "step": i + 1,
                    "mode": "baseline",
                    "err_to_ref": adapter.energy_norm_delta(state_bl, state0),
                    "cum_work": bl_work,
                })

            baseline_err = adapter.energy_norm_delta(state_bl, state0)
            baseline_errs[lam_key] = baseline_err

            # Assisted reverse: structured corrections with same energy budget
            state_as = adapter.copy_state(state_fwd)
            as_work = 0.0
            for i in range(steps):
                correction = adapter.assisted_correction(state_as, state0, actual_budget, rng)
                state_as = adapter.apply_correction(state_as, correction)
                state_as, _ = adapter.reverse_step(state_as, dt)
                as_work += adapter.correction_work(correction)

                telemetry_rows.append({
                    "seed": seed, "lambda": float(lam), "step": i + 1,
                    "mode": "assisted",
                    "err_to_ref": adapter.energy_norm_delta(state_as, state0),
                    "cum_work": as_work,
                })

            assisted_err = adapter.energy_norm_delta(state_as, state0)
            assisted_errs[lam_key] = assisted_err
            work_summaries[lam_key] = {
                "baseline_work": bl_work,
                "assisted_work": as_work,
            }
            ceg_map[lam_key] = ceg(baseline_err, assisted_err)

        # Enforce CEG(0) = 0 by construction
        ceg_map[_lam_key(0.0)] = 0.0

        # --- G3: Energy match (worst-case relative difference across λ > 0) ---
        rels: List[float] = []
        for lam in lambdas:
            if lam <= 0.0:
                continue
            ws = work_summaries.get(_lam_key(lam), {})
            w_b = ws.get("baseline_work", 0.0)
            w_a = ws.get("assisted_work", 0.0)
            denom = max(abs(w_b), 1e-12)
            rels.append(abs((w_a - w_b) / denom))
        rel_diff = max(rels) if rels else 0.0

        per_seed.append({
            "seed": seed,
            "baseline_err": baseline_errs,
            "assisted_err": assisted_errs,
            "work_summaries": work_summaries,
            "gates_diag": {**rp1, "rel_diff": rel_diff},
            "ceg": ceg_map,
            "skipped": False,
        })

    # --- Aggregate CEG summary ---
    ceg_summary: Dict[str, Dict[str, Any]] = {}
    for lam in lambdas:
        k = _lam_key(lam)
        vals = []
        for s in per_seed:
            if s.get("skipped"):
                continue
            v = s.get("ceg", {}).get(k)
            if v is not None:
                vals.append(float(v))
        if vals:
            arr = np.array(vals, dtype=float)
            ceg_summary[k] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "n": int(arr.size),
            }
        else:
            ceg_summary[k] = {"median": 0.0, "mean": 0.0, "n": 0}

    # --- Gate ledger ---
    from gates.echo_gates import (
        gate_noether, gate_h_theorem, gate_energy_match,
        gate_strang_defect, aggregate_gate_ledger,
    )

    gate_ledger_per_seed = []
    for s in per_seed:
        diag = s.get("gates_diag", {})
        gates = [
            gate_noether(diag.get("time_rev_drift", 0.0), tol=diag.get("g1_tol", 1e-12)),
            gate_h_theorem(diag.get("delta_sigma_min", 0.0), tol=diag.get("g2_tol", 1e-12)),
            gate_energy_match(diag.get("rel_diff", 0.0)),
            gate_strang_defect(
                diag.get("slope", 0.0), diag.get("R2", 0.0),
                min_slope=diag.get("min_slope", 2.90),
                min_r2=diag.get("min_r2", 0.999),
            ),
        ]
        failed = [g for g in gates if not g.get("passed", False)]
        gate_ledger_per_seed.append({
            "seed": s["seed"],
            "gates": gates,
            "contradiction": {
                "failed_count": len(failed),
                "failed_gates": [g["gate"] for g in failed],
            } if failed else None,
        })

    agg_ledger = aggregate_gate_ledger(gate_ledger_per_seed, min_gate_pass_rate)

    # G5: CEG positive
    medians = [v["median"] for k, v in ceg_summary.items() if float(k) > 0.0]
    median_max = max(medians) if medians else 0.0
    g5_pass = median_max >= ceg_gate_threshold
    agg_ledger["G5_CEG_Positive"] = {
        "passed": 1 if g5_pass else 0,
        "failed": 0 if g5_pass else 1,
        "n": 1,
        "pass_rate": 1.0 if g5_pass else 0.0,
        "median_max": float(median_max),
        "tol": float(ceg_gate_threshold),
    }

    # Contradiction report
    contradiction = None
    instrument_failures = sum(
        1 for k, v in agg_ledger.items()
        if k != "G5_CEG_Positive"
        and v.get("pass_rate") is not None
        and float(v["pass_rate"]) < min_gate_pass_rate
    )
    if instrument_failures > 0:
        contradiction = {
            "total_failed_gates": instrument_failures,
            "summary": agg_ledger,
        }

    elapsed = time.time() - t0

    return {
        "seeds": seeds,
        "lambdas": lambdas,
        "dt": dt,
        "steps": steps,
        "budget": budget,
        "per_seed": per_seed,
        "ceg_summary": ceg_summary,
        "gate_ledger_per_seed": gate_ledger_per_seed,
        "gate_ledger_summary": agg_ledger,
        "telemetry_rows": telemetry_rows,
        "CONTRADICTION_REPORT": contradiction,
        "timing": {"elapsed_s": elapsed},
    }


def emit_artifacts(
    results: Dict[str, Any],
    output_dir: str = ".",
    tag: str = "ceg_run",
) -> Dict[str, str]:
    """Write JSON log + CSV summary + CSV telemetry to output_dir.

    Returns dict of {artifact_type: filepath}.
    """
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON log
    json_path = out / f"{tag}_results.json"
    # Strip telemetry_rows from JSON for size (they go to CSV)
    json_data = {k: v for k, v in results.items() if k != "telemetry_rows"}
    with json_path.open("w") as f:
        json.dump(json_data, f, indent=2, sort_keys=True, default=str)

    # CSV summary
    csv_summary_path = out / f"{tag}_ceg_summary.csv"
    with csv_summary_path.open("w") as f:
        f.write("lambda,median_ceg,mean_ceg,n\n")
        for k, v in results.get("ceg_summary", {}).items():
            f.write(f"{k},{v['median']},{v['mean']},{v['n']}\n")

    # CSV telemetry
    csv_telem_path = out / f"{tag}_telemetry.csv"
    rows = results.get("telemetry_rows", [])
    if rows:
        cols = list(rows[0].keys())
        with csv_telem_path.open("w") as f:
            f.write(",".join(cols) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

    return {
        "json": str(json_path),
        "csv_summary": str(csv_summary_path),
        "csv_telemetry": str(csv_telem_path),
    }
