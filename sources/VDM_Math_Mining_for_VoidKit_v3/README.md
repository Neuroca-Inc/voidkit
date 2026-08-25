# VDM Math Mining for VoidKit — v3

This is the v2 mining corpus plus a third duplicate-aware sweep. `third_sweep.zip` contained 820 files. 183 were exact SHA-256 duplicates of v2; the sweep contained 100 internal duplicate hash groups (189 redundant copies). After semantic review, v3 promotes 11 Tier A candidates, 10 Tier B candidates, and 3 small support dependencies.

The key rule in v3 is **byte-distinct is not automatically new**. Older/reduced QBL kernels, superseded Orthad helper modules, validation artifacts, runtime/persistence code, and repeated Rust snapshots remain excluded even when their hashes differ.

See:
- `THIRD_SWEEP_NEW_CANDIDATES.md` for the promoted material.
- `THIRD_SWEEP_SUPERSEDENCE.md` for important semantic duplicates / older variants.
- `THIRD_SWEEP_AUDIT.csv` for the disposition of all 820 third-sweep files.
- `THIRD_SWEEP_VALIDATION.csv` for static compile/syntax checks.
- `V3_SUMMARY.json` for machine-readable counts.

Validation note: all 19 newly retained Python files syntax-compile; both assembly kernels assemble successfully. The two retained C mining sources could not be syntax-checked standalone because their project headers (`qbl_abi.h`, `xigraph.h`) were not present in the sweep. Rust validation was skipped because `rustc` is unavailable in this environment.

---

## v2 package documentation (preserved)

# VDM Math Mining for VoidKit — v2

This revision merges the original thinned VDM math-mining corpus with a second `Math.zip` scan. The second scan was deduplicated by SHA-256 and filtered by implementation usefulness rather than filename alone.

### v2 delta
- 99 additional input files inspected
- 1 exact duplicate rejected
- 34 new candidates retained (10 Tier A direct math + 24 Tier B deterministic algorithms)
- 64 benchmark/test/CLI/build/persistence/support files rejected

See `NEW_CANDIDATES.md` and `MATH_ZIP_AUDIT.csv`.

---

# VDM Math Mining Corpus for VoidKit

Purpose: aggressively thin the flattened `VDM_Code_Extraction` corpus down to code that is plausibly worth mining for VoidKit mathematical tools, numerical methods, transforms, solvers, exact arithmetic, symbolic derivations, graph/dynamical equations, or reusable scientific algorithms.

## Result

- Original code files inspected: **2051**
- Direct math/solver candidates retained: **264**
- Supporting local dependencies retained: **4**
- Total retained: **268**
- Removed: **1783**
- Retention rate: **13.1%**
- Retained Python syntax check: **260 PASS / 0 FAIL**

## What was deliberately removed

Vendored copies of NumPy/Matplotlib/Pillow/fontTools/pip/etc.; ordinary tests; GUI/plotting code; packaging/build scripts; verification harnesses whose main value is checking a claim rather than implementing the mathematics; logging/control/process infrastructure; and files with too little mathematical content to be useful mining targets.

## Layout

- `math_candidates/` — primary mining targets. Code is copied byte-for-byte from the supplied extraction.
- `support_dependencies/` — small local modules pulled in because retained candidates import them.
- `RETAINED_MANIFEST.csv` — category, reason, LOC, detected definitions/imports, and SHA-256 for every retained file.
- `DROPPED_MANIFEST.csv` — every rejected code file and the filtering reason, so false negatives are recoverable.
- `SYNTAX_CHECK.csv` — parse/compile-only validation of retained Python files. Nothing was imported or executed.

## Important interpretation

This is a **mining corpus**, not a claim that every retained file belongs in VoidKit unchanged. Research-specific Phase Calculus / Orthad / VDM files are intentionally retained when they contain reusable mathematical machinery. The next useful pass is capability extraction: identify individual functions/algorithms, deduplicate equivalent mechanisms, compare them against existing VoidKit modules, and port only the strongest implementations with tests.
