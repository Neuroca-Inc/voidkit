# Repository baseline stabilization pass

**Date:** 2026-08-25  
**Scope:** public repository, packaging, licensing, test-boundary, and stale-tooling cleanup before additional math extraction.

## Authority and invariants

- The uploaded current repository snapshot controlled implementation state for this pass.
- Root `LICENSE` controls software licensing: BSD 3-Clause.
- `MATH_MINING_TODO.md` controls the math-mining roadmap and custody rules.
- Historical `dev-reports/` and mining `sources/` are evidence/history, not current package authority.
- Authored mathematics was not redesigned. SIE, RE-VGSP, VDM, Phase Calculus, and the quarantined legacy equations retain their existing identity and ownership.
- `sources/` remains provenance/ore and must not be executed merely because historical files happen to match pytest naming conventions.

## Changes

1. Restored Python package/build authority with root `pyproject.toml`.
2. Added one Python package-version source, `voidkit/_version.py`, at `0.1.0`.
3. Restored the documented console entry points `voidkit-stats` and `voidkit-diff` through package metadata.
4. Added explicit optional dependency groups and reduced the mandatory core to NumPy + SciPy. The old application-era dependency dump contained multiple packages no longer imported by live Python code.
5. Replaced `requirements.txt` with a convenience full-development install driven by `pyproject.toml`, preventing a second dependency authority.
6. Added pytest collection boundaries so normal `pytest` runs only the maintained `tests/` suite and does not execute research ore under `sources/`.
7. Restored a Rust `Cargo.toml` using dependency versions recorded by the repository's own migration reports plus the crates actually imported by the current `src/` tree.
8. Completed the live-source BSD transition. Current Python/Rust source headers no longer claim a dual proprietary/commercial license. Historical reports and frozen mining sources were deliberately not rewritten.
9. Added `CITATION.cff` with Justin K. Lietz / Neuroca, Inc. attribution and method-level citation guidance.
10. Updated README package version, installation, dependency authority, license, citation guidance, and copyright line.
11. Replaced stale Rust documentation that still presented quarantined legacy void equations as the live VDM API.
12. Updated stale package identity text in the root and advanced-math namespaces without changing mathematical algorithms.
13. Added Python GitHub Actions coverage for Python 3.9, 3.11, and 3.13.
14. Added repository-contract regression tests for licensing, metadata/citation presence, version consistency, and source-ore test isolation.
15. Removed obsolete active-tooling authority from `tools/`: the old Hatch metadata hook, pyproject rewriting script, and FUM migration utilities were moved to `dev-reports/legacy-tools/` and explicitly marked historical. These tools could otherwise overwrite current metadata or act on long-finished namespace migrations.

## Validation

Executed on the final tree:

- Python 3.9 grammar parse: **93 files, 0 errors**.
- `python -m compileall -q voidkit tests`: **PASS**.
- `pytest -q`: **38/38 PASS**.
- Pytest collection: **38 maintained tests, 0 mining-source tests collected**.
- Live license/identity audit: **PASS**; no active dual-license/commercial-permission language, stale personal GitHub URL, or obsolete "Unified" package identity outside intentionally quarantined legacy material.
- Source-tree import sweep: **86/86 discoverable `voidkit` modules imported**.
- Wheel build with local build tooling (`--no-build-isolation` because this environment has no network): **PASS**.
- Wheel: `voidkit-0.1.0-py3-none-any.whl`.
- Wheel SHA-256: `b1a9e64e4d5c77db9431e479ef02d1bdeb02a80e774f545f16a0cfc1210ca870`.
- Fresh wheel install smoke: distribution version **0.1.0**, **86/86 modules imported**.
- Installed `voidkit-stats` CLI: **PASS**.
- Installed `voidkit-diff` CLI: **PASS**.
- Rust static dependency closure: all eight imported external crate families are declared (`pyo3`, `numpy`, `nalgebra`, `ndarray`, `rand`, `rand_distr`, `petgraph`, `approx`).
- Rust module-path audit: **0 missing `pub mod` targets**.
- Source-custody comparison against the uploaded baseline: Python files changed only for license/docs/explicit package-boundary work; license-only Python files preserved executable AST; all 37 changed Rust source files differed only in the leading license header.

## Rust execution boundary

The execution environment does not provide `cargo` or `rustc`. Therefore this pass does **not** claim that the reconstructed Rust manifest compiles. The manifest is grounded in the repository's recorded PyO3 0.22 / NumPy 0.22 / nalgebra 0.33 / ndarray 0.16 dependency set, later recorded petgraph/rand_distr additions, and the current source imports, but the next Rust-capable environment still needs to run:

```bash
cargo test --lib
maturin build --release --features extension-module
```

Until that executes successfully, Rust build status remains unverified rather than assumed.

## Result

The repository now has a coherent v0.1.0 Python packaging/version baseline, BSD licensing and citation custody, deterministic maintained-test collection, a buildable Python wheel, installed CLI validation, current public documentation, and a clear separation between active tooling and historical migration artifacts. The next engineering work can return to restoring and promoting mathematical capabilities without first depending on a drifted repository baseline.
