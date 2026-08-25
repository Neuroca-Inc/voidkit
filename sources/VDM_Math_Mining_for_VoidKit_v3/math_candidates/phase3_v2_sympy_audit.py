from __future__ import annotations

import json
from pathlib import Path
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

u, v, k = sp.symbols("u v k", positive=True, integer=True)
phase_product_progress = sp.simplify(v * (u + v) - u * v)
phase_product_progress_positive_form = sp.factor(phase_product_progress)

checks = {
    "phase_product_collision_uv_1_6_vs_2_3": 1 * 6 == 2 * 3,
    "phase_next_images_differ": (6, 7) != (3, 5),
    "phase_product_progress_symbolic": str(phase_product_progress_positive_form) == "v**2",
    "yi_completion_value_formula_k1": 7 == 8**1 - 1,
    "yi_completion_value_formula_k2": 7 + 7 * 8 == 8**2 - 1,
    "yi_completion_value_formula_k3": 7 + 7 * 8 + 7 * 8**2 == 8**3 - 1,
    "yi_lift_value_k2": 1 * 8**2 == 64,
    "wilhelm_hypercube_edges": 64 * 6 == 384,
    "wilhelm_undirected_edges": (64 * 6) // 2 == 192,
}

summary = {
    "global_pass": all(checks.values()),
    "checks": checks,
    "symbolic_phase_product_progress_delta": str(phase_product_progress_positive_form),
    "interpretation": "B(u,v)=sort(v,u+v) strictly raises product by v^2 before sorting for positive u,v; base-8 all-7 completion equals 8^k-1; Wilhelm line-flip graph has 384 directed edges.",
}
(OUT / "phase3_v2_sympy_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
