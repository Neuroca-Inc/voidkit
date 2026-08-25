# New candidates added from Math.zip

Compared against the previous `VDM_Math_Mining_for_VoidKit` corpus by SHA-256.

## Result

- Input files: 99
- Exact duplicates of prior package: 1 (`asm/qbl_step_wide_v0.S`)
- Added Tier A core math: 10
- Added Tier B algorithmic candidates: 24
- Not added: 64 validation/support/build/persistence/CLI files

## Tier A — direct math / solver mining

- `asm/qbl_b_u64.S` — checked pair refinement `(u,v) -> (v,u+v)`.
- `asm/qbl_inspect_u64.S` — QBL state inspection and derived phase/capacity quantities.
- `asm/qbl_step_u64.S` — exact B/Q/L state-transition selector.
- `clang/orthad_deep_runner.c` — Orthad threshold/domain exploration.
- `clang/qbl_reference.c` — readable C reference for QBL arithmetic, inspection, stepping, recording, and Orthad-local stepping.
- `clang/qbl_step_wide_v0_kernel.c` — fixed-width multiprecision arithmetic and wide QBL stepping.
- `clang/qbl_wide_reference.c` — independent wide-reference helpers.
- `clang/phase_wide_v0.c` — wide phase/world stepping with explicit product arithmetic.
- `rust/lib (11th copy).rs` — Rust wide arithmetic / phase-world implementation.
- `rust/lib (16th copy).rs` — pure Rust QBL reference implementation.

## Tier B — reusable deterministic algorithms

These contain potentially reusable graph, causal-resolution, routing, traversal, and deterministic state-transition algorithms. C and Rust versions coexist where they give independent implementations useful for cross-checking a future Python extraction.

See `MATH_ZIP_AUDIT.csv` for every source file and exclusion reason.
