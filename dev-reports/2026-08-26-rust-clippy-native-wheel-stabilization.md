# Rust Clippy and Native-Wheel Stabilization

Date: 2026-08-26
Scope: Rust quality gate and native extension packaging only

## Controlling evidence

The host run established the following baseline before this pass:

- `RUSTFLAGS="-D warnings" cargo test --lib`: 134 passed, 0 failed, with no rustc warnings.
- `cargo clippy --lib --tests -- -D warnings`: failed on 100 Clippy findings.
- `cargo test --all-targets`: 134 passed, 0 failed.
- `maturin build --release`: compiled the crate but failed manylinux validation because the extension linked `libpython3.13.so.1.0`.

## Invariants protected

- No research equation or authored mechanism is rewritten to satisfy style lints.
- Public Python/Rust call signatures remain stable unless the change is purely representational.
- Floating-point NaN behavior is not silently changed by Clippy rewrites such as `RangeInclusive::contains` or `clamp`.
- The pure-Python `voidkit` distribution remains owned by the root Setuptools configuration.
- The native `voidkit-rust` distribution has an explicit, separate Maturin packaging boundary.
- Compiler warnings and Clippy warnings remain failures in CI.

## Changes

### Clippy cleanup

Semantics-preserving findings were corrected directly:

- removed six unused `PyArrayMethods` imports;
- replaced zero-length comparisons with `is_empty()`;
- replaced fixed test/config `Vec` allocations with arrays where ownership was unnecessary;
- removed PageRank identity mapping and rewrote the dangling-score loop without changing update order;
- replaced `or_insert_with(Vec::new)` with `or_default()` and explicit clone closure with `cloned()`;
- completed the `SpatialHashGrid` `len`/`is_empty` API pair;
- introduced names for the SDE and Gillespie PyO3 tuple return types;
- rewrote test-only index loops where the iterator form is behaviorally identical.

Narrow lint exceptions were retained where the suggested rewrite would be structurally or semantically dishonest:

- `clippy::useless_conversion` is allowed at crate scope because PyO3 0.22 macro expansion emits identity `PyErr` conversions under current Clippy; handwritten warnings remain denied;
- `too_many_arguments` is allowed only on existing public/research-facing APIs and the recursive Simpson state carrier, avoiding API/equation reshaping for lint aesthetics;
- `manual_range_contains` is allowed on the mutation/recombination validation because the suggested range form changes NaN behavior;
- `manual_clamp` is allowed on the adaptive ODE step and quarantined legacy VDM step because `clamp` changes NaN handling;
- `needless_range_loop` is allowed on the Vietoris-Rips symmetric distance-matrix construction to retain the explicit symmetric indexing contract.

### Native wheel boundary

The root `pyproject.toml` remains Setuptools-owned for the pure-Python `voidkit` package.

A dedicated `rust-wheel/pyproject.toml` now owns Maturin packaging for the Cargo crate. It enables the crate's `extension-module` feature explicitly. This is required for the pinned PyO3 0.22 line; that version predates the newer environment-variable extension-module behavior and otherwise links `libpython` on Unix, which fails manylinux validation.

`tools/build_rust_wheel.sh` is the canonical native-wheel build entry point.

### Verification gates

`.github/workflows/rust-tests.yml` now gates:

1. Clippy with all warnings denied on Python 3.9, 3.11, and 3.13;
2. all Rust test targets with compiler warnings denied on the same matrix;
3. native Maturin wheel construction on the same Python matrix.

`tools/validate_rust.sh` mirrors those three host-side gates in one command.

## Validation performed in the artifact environment

- Python test suite: 43 passed.
- Python package compileall: PASS.
- Python package import sweep: 86/86 modules.
- root `pyproject.toml`: TOML parse PASS.
- `Cargo.toml`: TOML parse PASS.
- `rust-wheel/pyproject.toml`: TOML parse PASS.
- shell syntax for `tools/build_rust_wheel.sh`: PASS.
- shell syntax for `tools/validate_rust.sh`: PASS.

Cargo/Clippy/Maturin execution is not available in this artifact environment. The host acceptance command is `./tools/validate_rust.sh`; completion of this pass requires that command to succeed on the Rust-capable host.
