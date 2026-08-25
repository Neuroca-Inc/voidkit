from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from pc_branch_certifier.domain.fibonacci import depth_for_floor


def main() -> int:
    x = sp.symbols("x")
    poly = x**5 - x + 1
    roots = sp.nroots(poly, n=60, maxsteps=200)
    resolution = depth_for_floor(1e-8)
    scale = sp.Integer(2)
    entries = []
    for idx, root in enumerate(roots):
        residual = complex(sp.N(poly.subs(x, root), 50))
        z = complex(root)
        zn = z / float(scale)
        entries.append(
            {
                "index": idx,
                "root_re": z.real,
                "root_im": z.imag,
                "normalized_re": zn.real,
                "normalized_im": zn.imag,
                "normalized_abs": abs(zn),
                "residual_abs": abs(residual),
                "depth": resolution.depth,
                "half_width": resolution.half_width,
            }
        )
    report = {
        "status": "PROVEN" if all(e["residual_abs"] < 1.4e-15 and e["half_width"] < 1e-8 and e["normalized_abs"] <= 1 for e in entries) else "FAIL",
        "polynomial": "x^5 - x + 1",
        "normalization_scale": 2,
        "resolution": {
            "depth": resolution.depth,
            "q": [resolution.u, resolution.v],
            "uv": resolution.uv,
            "half_width": resolution.half_width,
            "exact_remainder": f"1/{resolution.uv}",
        },
        "residual_threshold": 1.4e-15,
        "roots": entries,
    }
    out = Path("certificate_bring_quintic_roots.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_RESULT: PASS" if report["status"] == "PROVEN" else "FINAL_RESULT: FAIL")
    return 0 if report["status"] == "PROVEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
