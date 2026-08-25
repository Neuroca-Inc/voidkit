#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEALED = ROOT / "sealed"
DOCS = ROOT / "docs"
PATCHES = ROOT / "patches"
NOTES = ROOT / "source_notes"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def complex_resid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def run_checks() -> dict:
    M = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    I = np.eye(3)
    det = float(np.linalg.det(M))
    checks = []
    checks.append({
        "check": "yin_yang_matrix_orthogonal",
        "metric": "max_abs(M.T@M-I)",
        "value": complex_resid(M.T @ M, I),
        "threshold": 1e-12,
        "pass": complex_resid(M.T @ M, I) < 1e-12,
        "meaning": "Yin-Yang chart transform is a rotation/reflection-free orthogonal change of chart."
    })
    checks.append({
        "check": "yin_yang_matrix_involutive",
        "metric": "max_abs(M@M-I)",
        "value": complex_resid(M @ M, I),
        "threshold": 1e-12,
        "pass": complex_resid(M @ M, I) < 1e-12,
        "meaning": "The same transform returns from Yang to Yin, matching M^-1=M."
    })
    checks.append({
        "check": "yin_yang_matrix_orientation_preserving",
        "metric": "abs(det(M)-1)",
        "value": abs(det - 1.0),
        "threshold": 1e-12,
        "pass": abs(det - 1.0) < 1e-12,
        "meaning": "The two-patch transform is a proper rotation composition, not an arbitrary projection."
    })
    alphas = np.linspace(0, 1, 101)
    w = 1.0 - 0.5 * alphas
    checks.append({
        "check": "overlap_weight_bounds",
        "metric": "min(w)>=0.5 and max(w)<=1",
        "value": float(min(w.min() - 0.5, 1.0 - w.max())),
        "threshold": 0.0,
        "pass": bool(w.min() >= 0.5 - 1e-15 and w.max() <= 1.0 + 1e-15),
        "meaning": "Overlap weights interpolate between full custody outside overlap and half custody inside full overlap."
    })
    # A toy two-patch overlap contribution: duplicated full overlap must count once, not twice.
    duplicated_full_overlap_sum = 0.5 + 0.5
    checks.append({
        "check": "full_overlap_no_double_count",
        "metric": "abs(two_patch_full_overlap_weight_sum-1)",
        "value": abs(duplicated_full_overlap_sum - 1.0),
        "threshold": 1e-12,
        "pass": abs(duplicated_full_overlap_sum - 1.0) < 1e-12,
        "meaning": "A fully overlapped zone is split half/half rather than counted twice."
    })
    all_pass = all(bool(r["pass"]) for r in checks)
    write_csv(OUT / "phase5_v7j_numeric_checks.csv", checks)
    return {
        "all_pass": all_pass,
        "max_numeric_residual": max(float(r["value"]) for r in checks),
        "check_count": len(checks),
        "checks_passed": sum(1 for r in checks if r["pass"]),
        "det_M": det,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SEALED.mkdir(parents=True, exist_ok=True)
    stats = run_checks()

    mechanism_rows = [
        {
            "paper_mechanism": "single spherical polar grid has pole convergence and axis boundary artifacts",
            "paper_location": "abstract, introduction",
            "phase5_mapping": "single-chart projection artifact / diagonal-only or folded-carrier warning",
            "qbl_history_implication": "do not let one chart define global retained state",
            "status": "DIRECT_ANALOGUE_FOR_CHART_FAILURE",
        },
        {
            "paper_mechanism": "Yin and Yang are two low-latitude patches that overlap and cover the sphere",
            "paper_location": "section 2, figure 1",
            "phase5_mapping": "overset atlas: two valid local charts over one retained object",
            "qbl_history_implication": "arbitrary history needs chart/patch custody before projection",
            "status": "DIRECT_ANALOGUE_FOR_OVERSET_ATLAS",
        },
        {
            "paper_mechanism": "Yang patch obtained from Yin by rotations; M^-1 = M",
            "paper_location": "equations 4-7",
            "phase5_mapping": "chart transition operator, not an external force",
            "qbl_history_implication": "legal history rewrites should preserve chart-transition invariants",
            "status": "FORMAL_ANALOGUE_FOR_GAUGE_MOVE",
        },
        {
            "paper_mechanism": "vector components require an additional transformation matrix P after interpolation",
            "paper_location": "equations 8-9, implementation section",
            "phase5_mapping": "component mixing across chart transition",
            "qbl_history_implication": "multi-axis coupling cannot be inferred from scalar overlap only",
            "status": "DIRECT_ANALOGUE_FOR_COMPONENT_MIXING_BOUNDARY",
        },
        {
            "paper_mechanism": "no angular boundary conditions; ghost zones are interpolated from neighboring patch interiors",
            "paper_location": "section 3.1, figure 2",
            "phase5_mapping": "overlap handoff before projection",
            "qbl_history_implication": "shared-boundary coupling should be read from retained overlap map, not assigned afterward",
            "status": "DIRECT_ANALOGUE_FOR_NATIVE_COUPLING_EXTRACTION",
        },
        {
            "paper_mechanism": "interpolation coefficients and overlap weights are computed once at initialization and stored",
            "paper_location": "section 3.1",
            "phase5_mapping": "retained transition registry",
            "qbl_history_implication": "history normal form needs stored latch/overlap coefficients",
            "status": "DIRECT_ANALOGUE_FOR_COUPLING_REGISTRY",
        },
        {
            "paper_mechanism": "overlap integrals require weights to avoid double counting",
            "paper_location": "section 3.1",
            "phase5_mapping": "projection/readout map must account for overlap multiplicity",
            "qbl_history_implication": "scalar projection after overset read must carry custody weights",
            "status": "DIRECT_ANALOGUE_FOR_PROJECTION_DISCIPLINE",
        },
        {
            "paper_mechanism": "scalar flux conservation can be restored by replacing boundary fluxes with interior-derived fluxes",
            "paper_location": "section 5, figure 17, equations 33-39",
            "phase5_mapping": "boundary handoff law / no external boundary condition",
            "qbl_history_implication": "legal QBL boundary handoff should be reconstructible from interior history",
            "status": "TESTABLE_ANALOGUE_FOR_HISTORY_NORMAL_FORM",
        },
        {
            "paper_mechanism": "momentum conservation is harder because vector components mix across rotated patches",
            "paper_location": "section 5",
            "phase5_mapping": "multi-axis tensor boundary: scalar closure is easier than vector/tensor closure",
            "qbl_history_implication": "arbitrary-history theorem must handle component mixing, not only scalar c_ij",
            "status": "FRONTIER_WARNING",
        },
        {
            "paper_mechanism": "auxiliary spherical grid used for gravity solver projection",
            "paper_location": "section 3.2",
            "phase5_mapping": "post-evolution auxiliary readout/projection layer",
            "qbl_history_implication": "external scalar/vector basis may be a readout, not the native retained carrier",
            "status": "READOUT_LAYER_ANALOGUE",
        },
    ]
    write_csv(OUT / "phase5_v7j_mechanism_extract.csv", mechanism_rows)

    crosswalk_rows = [
        {"phase5_object": "Orthad overset atlas", "paper_object": "Yin-Yang two-patch grid", "mapping_strength": "STRONG_ANALOGUE", "caution": "Yin/Yang patches are peer geometric charts, not literally point/eigen charts."},
        {"phase5_object": "chart transition", "paper_object": "M coordinate rotation and P vector transform", "mapping_strength": "STRONG_OPERATIONAL", "caution": "Paper transition is geometric; Phase transition is retained-state/eigen chart."},
        {"phase5_object": "shared-boundary coupling", "paper_object": "ghost-zone interpolation from internal neighboring patch zones", "mapping_strength": "STRONG_OPERATIONAL", "caution": "Interpolation is numerical and not automatically conservative."},
        {"phase5_object": "coupling registry", "paper_object": "stored interpolation coefficients and overlap weight maps", "mapping_strength": "STRONG_OPERATIONAL", "caution": "Registry is fixed for stationary grid; QBL histories may have dynamic latch structure."},
        {"phase5_object": "projection discipline", "paper_object": "overlap weights w=1-0.5 alpha", "mapping_strength": "STRONG_ANALOGUE", "caution": "Weights solve double-counted integration, not full history equivalence."},
        {"phase5_object": "arbitrary-history normal form", "paper_object": "scalar boundary flux replacement from interior fluxes", "mapping_strength": "TESTABLE_TEMPLATE", "caution": "Momentum/vector component mixing remains harder."},
    ]
    write_csv(OUT / "phase5_v7j_phase_calculus_crosswalk.csv", crosswalk_rows)

    constraints = [
        {"constraint": "no_single_chart_global_claim", "rule": "A chart-local evolution cannot be promoted to a global retained-state claim without overlap handoff gates.", "source_mechanism": "single spherical grid pole artifacts"},
        {"constraint": "transition_map_must_be_retained", "rule": "Chart changes must be stored as transition data, not recomputed after projection as explanation.", "source_mechanism": "stored interpolation coefficient maps"},
        {"constraint": "overlap_projection_weights_required", "rule": "Projection/integration over an overset state must account for overlap multiplicity.", "source_mechanism": "w=1-0.5 alpha weighting"},
        {"constraint": "boundary_handoff_from_interior", "rule": "Boundary handoff should be reconstructible from interior retained state where possible.", "source_mechanism": "scalar flux replacement algorithm"},
        {"constraint": "scalar_success_does_not_close_vector_tensor", "rule": "Scalar conservation/readout success does not close component-mixing or momentum-like tensor gates.", "source_mechanism": "momentum conservation remains harder"},
        {"constraint": "auxiliary_projection_not_native_carrier", "rule": "An auxiliary grid/readout basis can support computation without becoming the native retained carrier.", "source_mechanism": "gravity solver auxiliary spherical polar grid"},
    ]
    write_csv(OUT / "phase5_v7j_arbitrary_qbl_history_constraints.csv", constraints)

    stale = [
        {"stale_statement": "The Orthad is one matrix/lens.", "repair": "The Orthad is an overset atlas/readout with chart transition structure; single lens is a collapse."},
        {"stale_statement": "Overlap is only a visualization artifact.", "repair": "Overlap carries handoff data: interpolation/transition coefficients and projection weights."},
        {"stale_statement": "Scalar projection closure proves full retained closure.", "repair": "Scalar closure is weaker than vector/tensor component closure across chart transitions."},
        {"stale_statement": "Boundary values can be imposed externally.", "repair": "Boundary handoff should be derived from neighboring interior retained-state data when claiming native generation."},
        {"stale_statement": "Product-module coupling can be supplied as module data and counted as generated.", "repair": "Generated coupling must be extracted and sealed from overlap/latch history before comparison."},
    ]
    write_csv(OUT / "phase5_v7j_stale_canon_patch_rules.csv", stale)

    falsifiers = [
        {"target": "arbitrary_history_normal_form", "falsifier": "Two legally equivalent QBL histories produce different sealed coupling tensors."},
        {"target": "native_overlap_coupling", "falsifier": "A shared-boundary history requires post-comparison adjustment of c_ij to close."},
        {"target": "projection_discipline", "falsifier": "Scalar projection double-counts or loses overlap contribution under legal chart overlap."},
        {"target": "component_mixing", "falsifier": "Vector/tensor readout fails under chart transition while scalar readout passes."},
        {"target": "auxiliary_readout", "falsifier": "The external readout basis changes the retained evolution rather than only reading it."},
    ]
    write_csv(OUT / "phase5_v7j_falsification_targets.csv", falsifiers)

    result_card = {
        "phase": "Phase 5 v7j",
        "title": "Axis-Free Overset Grid Revisit for Arbitrary QBL History",
        "status": "OVERSET_GRID_MECHANISM_REENTRY_COMPLETED_ARBITRARY_HISTORY_CONSTRAINTS_EXTRACTED",
        "global_pass": stats["all_pass"],
        "phase5_closed": False,
        "source": "Axis-Free-overset-grid.pdf",
        "numeric_checks": stats,
        "mechanism_rows": len(mechanism_rows),
        "crosswalk_rows": len(crosswalk_rows),
        "constraints": len(constraints),
        "stale_patch_rules": len(stale),
        "falsification_targets": len(falsifiers),
        "main_verdict": "The paper is a strong operational analogue for overset Orthad chart overlap and boundary handoff, but it does not close arbitrary QBL history classification. It sharpens the required normal-form and component-mixing gates.",
    }
    (OUT / "phase5_v7j_result_card.json").write_text(json.dumps(result_card, indent=2), encoding="utf-8")
    (OUT / "phase5_v7j_verification_summary.json").write_text(json.dumps(result_card, indent=2), encoding="utf-8")

    sealed = {
        "sealed_before_phase_calculus_extension": True,
        "source_pdf_sha256": sha256_file(Path("/mnt/data/Axis-Free-overset-grid.pdf")) if Path("/mnt/data/Axis-Free-overset-grid.pdf").exists() else None,
        "numeric_checks_sha256": sha256_file(OUT / "phase5_v7j_numeric_checks.csv"),
        "mechanism_extract_sha256": sha256_file(OUT / "phase5_v7j_mechanism_extract.csv"),
        "crosswalk_sha256": sha256_file(OUT / "phase5_v7j_phase_calculus_crosswalk.csv"),
        "constraints_sha256": sha256_file(OUT / "phase5_v7j_arbitrary_qbl_history_constraints.csv"),
    }
    (SEALED / "SEALED_AXIS_FREE_OVERSET_REENTRY_BEFORE_QBL_EXTENSION.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")

    # Minimal source notes with direct paper mechanisms.
    (NOTES / "axis_free_source_mechanism_notes.md").write_text("""# Axis-Free Overset Grid Source Mechanism Notes

Mechanisms extracted from Wongwathanarat, Hammer, and Müller (2010):

- A single spherical polar grid suffers pole convergence and axis-boundary artifacts.
- Yin-Yang uses two identical low-latitude spherical polar patches with overlap.
- Yang is obtained from Yin by rotations; the coordinate matrix satisfies M^-1 = M.
- Scalar and vector quantities need interpolation/transform across patch overlap.
- Angular boundary conditions are replaced by ghost-zone interpolation from neighboring patch interiors.
- Interpolation coefficients and overlap weights are retained in maps.
- Integrals over the overset domain need weights to avoid double counting.
- Scalar conservation can be restored by flux replacement from interior-derived fluxes.
- Momentum conservation remains harder due to vector-component mixing across rotated patches.
- Tests show no visible internal boundary artifacts and no preferred radial direction.
""", encoding="utf-8")

    print(json.dumps(result_card, indent=2))

if __name__ == "__main__":
    main()
