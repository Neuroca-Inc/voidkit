#!/usr/bin/env python3
"""
CEG Instrumentation — Echo Gates (G1–G5)

Model-agnostic gate functions for certifying the CEG instrument before
making claims about the observable.

Gate hierarchy:
  G1–G4  =  instrument certification  (must pass before any CEG claim)
  G5     =  outcome metric             (CEG itself)

Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
"""
from __future__ import annotations

from typing import Any, Dict, List


def gate_noether(time_reversal_energy_drift: float, tol: float = 1e-12) -> Dict[str, Any]:
    """G1: J-limb reversibility.

    Checks that the conservative (symplectic) sector preserves energy
    under time-reversal to within numerical precision.

    Parameters
    ----------
    time_reversal_energy_drift : float
        |E(t=T→0) − E(t=0)| after a full forward-then-reverse pass
        through the conservative integrator only.
    tol : float
        Absolute tolerance.  Scale with sqrt(N) * machine_eps * max(H0,1)
        for grid-dependent problems.
    """
    ok = abs(float(time_reversal_energy_drift)) <= float(tol)
    return {
        "gate": "G1_Noether_J",
        "tol": float(tol),
        "drift": float(time_reversal_energy_drift),
        "passed": bool(ok),
    }


def gate_h_theorem(delta_sigma_min: float, tol: float = 1e-12) -> Dict[str, Any]:
    """G2: M-limb entropy monotonicity (discrete H-theorem).

    Checks that the dissipative sector never *increases* the Lyapunov
    functional (entropy non-decrease on metric flow).

    Parameters
    ----------
    delta_sigma_min : float
        Minimum ΔΣ across all M-steps in the forward pass.
        Should be ≥ 0 (entropy never decreases).
    tol : float
        Absolute tolerance for numerical noise.
    """
    ok = float(delta_sigma_min) >= -float(tol)
    return {
        "gate": "G2_H_theorem_M",
        "tol": float(tol),
        "delta_sigma_min": float(delta_sigma_min),
        "passed": bool(ok),
    }


def gate_energy_match(rel_diff: float, tol: float = 1e-4) -> Dict[str, Any]:
    """G3: Energy match between assisted and baseline corrections.

    Ensures the assisted pathway uses the same total energy budget as
    the baseline (random) pathway, so CEG measures structure, not energy.

    Parameters
    ----------
    rel_diff : float
        |W_assisted − W_baseline| / max(|W_baseline|, ε)
    tol : float
        Maximum allowed relative difference.
    """
    ok = abs(float(rel_diff)) <= float(tol)
    return {
        "gate": "G3_EnergyMatch",
        "tol": float(tol),
        "rel_diff": float(rel_diff),
        "passed": bool(ok),
    }


def gate_strang_defect(slope: float, r2: float, min_slope: float = 2.90, min_r2: float = 0.999) -> Dict[str, Any]:
    """G4: Strang splitting convergence order.

    Two-grid refinement test: the splitting error should converge at
    the expected order for your integrator.

    Parameters
    ----------
    slope : float
        Log-log slope of splitting error vs dt from a two-grid study.
    r2 : float
        R² of the log-log fit.
    min_slope : float
        Minimum acceptable slope.  VDM default: 2.90 (near-third-order
        for their specific KG⊕RD split).  Generic Strang: use 1.90.
    min_r2 : float
        Minimum R² (default 0.999).
    """
    ok = (float(slope) >= float(min_slope)) and (float(r2) >= float(min_r2))
    return {
        "gate": "G4_StrangDefect",
        "slope": float(slope),
        "R2": float(r2),
        "min_slope": float(min_slope),
        "min_r2": float(min_r2),
        "passed": bool(ok),
    }


def gate_ceg_positive(
    median_max: float,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """G5: CEG outcome metric.

    Checks that the maximum median CEG (across λ > 0) exceeds the
    pre-registered threshold.

    Parameters
    ----------
    median_max : float
        max over λ>0 of median(CEG(λ)) across seeds.
    threshold : float
        Pre-registered CEG threshold (default 0.05 = 5%).
    """
    ok = float(median_max) >= float(threshold)
    return {
        "gate": "G5_CEG_Positive",
        "median_max": float(median_max),
        "tol": float(threshold),
        "passed": bool(ok),
    }


def aggregate_gate_ledger(
    gate_ledger_per_seed: List[Dict[str, Any]],
    min_pass_rate: float = 10.0 / 12.0,
) -> Dict[str, Any]:
    """Aggregate per-seed gate results into a summary ledger.

    Parameters
    ----------
    gate_ledger_per_seed : list[dict]
        Each entry has 'seed' and 'gates' (list of gate dicts).
    min_pass_rate : float
        Minimum fraction of seeds that must pass each gate.

    Returns
    -------
    dict
        Per-gate aggregates with pass counts, rates, and meets_rate flag.
    """
    tally: Dict[str, Dict[str, int]] = {}
    for entry in gate_ledger_per_seed:
        for g in entry.get("gates", []):
            name = g.get("gate", "unknown")
            if name not in tally:
                tally[name] = {"passed": 0, "failed": 0}
            if g.get("passed", False):
                tally[name]["passed"] += 1
            else:
                tally[name]["failed"] += 1

    agg: Dict[str, Any] = {}
    for name, counts in tally.items():
        total = counts["passed"] + counts["failed"]
        pr = (counts["passed"] / total) if total > 0 else None
        meets = (pr is not None) and (pr >= min_pass_rate)
        agg[name] = {
            "passed": counts["passed"],
            "failed": counts["failed"],
            "n": total,
            "pass_rate": pr,
            "min_pass_rate": min_pass_rate,
            "meets_rate": bool(meets),
        }
    return agg


def check_all_gates(results: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience: check the gate ledger summary and return a report.

    Parameters
    ----------
    results : dict
        Output of run_echo_protocol.

    Returns
    -------
    dict
        all_passed (bool), failed_gates (list), summary (dict).
    """
    ledger = results.get("gate_ledger_summary", {})
    failed = []
    for name, info in ledger.items():
        if not info.get("meets_rate", False) and name != "G5_CEG_Positive":
            failed.append(name)
        if name == "G5_CEG_Positive" and not info.get("passed", info.get("pass_rate", 0) >= 1.0):
            # G5 uses a different check (single-shot, not rate-based)
            pass  # Don't count G5 as instrument failure

    instrument_ok = len(failed) == 0
    g5 = ledger.get("G5_CEG_Positive", {})
    g5_ok = g5.get("passed", g5.get("pass_rate", 0) >= 1.0) if g5 else False

    return {
        "instrument_ok": instrument_ok,
        "g5_passed": bool(g5_ok),
        "all_passed": instrument_ok and g5_ok,
        "failed_gates": failed,
        "summary": ledger,
    }
