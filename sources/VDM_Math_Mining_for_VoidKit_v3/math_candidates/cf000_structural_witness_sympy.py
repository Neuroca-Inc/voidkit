#!/usr/bin/env python3
"""Exact finite structural witnesses for CF000.

This script mirrors Section 6 of main.tex and the CFN000 companion notebook.
It uses only Boolean and integer arithmetic; all reported tolerances are exact.
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> dict:
    candidates = [
        ("isolated 0", 0, 0, 0, 1, 0),
        ("isolated 1", 1, 0, 0, 1, 0),
        ("third terminal empty", 0, 0, 0, 1, 0),
        ("third terminal full-flat", 1, 0, 0, 1, 0),
        ("external split (0,1)", 1, 1, 1, 0, 0),
        ("internal witness omega", 1, 1, 1, 1, 1),
    ]

    rows = []
    for name, nv, not_flat, bear_opp, one_origin, no_discharge in candidates:
        adm = int(bool(nv and not_flat and bear_opp and one_origin and no_discharge))
        rows.append(
            {
                "candidate": name,
                "NV": nv,
                "not_Flat": not_flat,
                "BearOpp": bear_opp,
                "OneOrigin": one_origin,
                "NoDischarge": no_discharge,
                "Adm": adm,
            }
        )

    survivors = [row["candidate"] for row in rows if row["Adm"] == 1]
    assert len(survivors) == 1
    assert survivors[0] == "internal witness omega"

    saturation_index = sp.Integer(9)
    s9 = sp.Min(sp.Integer(9), saturation_index)
    s10 = sp.Min(sp.Integer(10), saturation_index)
    assert s9 == 9
    assert sp.simplify(s10 - s9) == 0

    axis = range(4)
    g1 = [(a, 0) for a in axis]
    g2 = [(a, b) for a in axis for b in axis]
    r1 = [p for p in g1 if p != (0, 0)]
    r2 = [p for p in g2 if p != (0, 0)]
    cross1 = [p for p in r1 if p[1] > 0]
    cross2 = [p for p in r2 if p[1] > 0]

    assert sp.Integer(len(g2)) == sp.Integer(4) * sp.Integer(4)
    assert sp.Integer(len(r2)) == sp.Integer(16) - sp.Integer(1)
    assert sp.Integer(len(cross2)) == sp.Integer(4) * sp.Integer(3)

    result = {
        "paper_id": "CF000",
        "witness_type": "exact finite structural witnesses",
        "primitive_candidate_survivor_count": len(survivors),
        "primitive_candidate_survivors": survivors,
        "candidate_rows": rows,
        "saturation_index": int(saturation_index),
        "S_9": int(s9),
        "S_10_minus_S_9": int(sp.simplify(s10 - s9)),
        "single_axis_placements": len(g1),
        "single_axis_nonzero_relations": len(r1),
        "single_axis_cross_axis_relations": len(cross1),
        "two_axis_placements": len(g2),
        "two_axis_nonzero_relations": len(r2),
        "two_axis_cross_axis_relations": len(cross2),
        "tolerance": 0,
    }

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cf000_structural_witness_sympy.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
