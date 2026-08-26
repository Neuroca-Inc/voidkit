# Rust Offline Compiler Verification

Date: 2026-08-26

## Purpose

Close the previous environment gap by validating VoidKit's Rust crate with an uploaded offline Rust toolchain and vendored dependency closure rather than relying on static inspection alone.

## Verification environment

- rustc 1.96.1 (31fca3adb 2026-06-26)
- cargo 1.96.1 (356927216 2026-06-26)
- clippy 0.1.96 (31fca3adb2 2026-06-26)
- host: x86_64-unknown-linux-gnu
- CPython 3.13.5
- Cargo dependencies resolved exclusively from the uploaded vendored source set

## Executed gates

1. `RUSTFLAGS="-D warnings" cargo clippy --offline --lib --tests -- -D warnings`
   - PASS
   - zero warnings accepted
2. `RUSTFLAGS="-D warnings" cargo test --offline --all-targets`
   - PASS
   - 134 passed, 0 failed
3. `RUSTFLAGS="-D warnings" cargo build --offline --release --features extension-module`
   - PASS
4. Native shared-library dependency inspection
   - PASS
   - `libvoidkit_rust.so` does not link `libpython`
5. CPython extension smoke import
   - PASS
   - `voidkit_rust` imported under CPython 3.13
   - 56 public extension symbols exposed

## Packaging boundary

The environment does not contain the Maturin Python package, so this pass does not claim that an actual `.whl` was produced here. The underlying release extension build, no-libpython condition, and Python import path were executed successfully. The repository's Maturin build remains the authoritative wheel-packaging step on a Maturin-capable environment.

## Reproducibility cleanup

`Cargo.lock` is now retained in the repository baseline, and Rust validation/CI uses `--locked` so dependency drift cannot silently change the validated native surface.

## Mathematical custody

No mathematical or numerical implementation was changed in this verification pass. Changes are limited to dependency locking, validation commands, and this execution record.

## Flaky-test defect discovered by repeated execution

A fresh all-target run exposed `fractal_analysis::fractal_spike_train::tests::test_generate_fractal_spike_train` as probabilistically flaky. Its original parameters produced only about 1.5 expected events over the full test window, so an empty realization was valid and occurred with substantial probability even though the test asserted non-empty output.

The mathematical implementation was not changed. The test parameters were changed so the first 1 ms bin has spike probability exactly 1.0 under the existing implementation (`fractal_dimension = 1`, `k = 1000`, `dt = 1 ms`). This makes the non-empty assertion deterministic while preserving the stochastic implementation path for subsequent bins.

## Final acceptance after stabilization

- Strict Clippy: PASS with `-D warnings`
- Rust all-target suite: PASS, 134/134
- Repetition stress: PASS, 20 consecutive all-target runs (2,680 test executions)
- Release `extension-module` build: PASS with `-D warnings`
- `ldd` check: PASS, no `libpython` dependency
- CPython 3.13 native import: PASS, 56 public extension symbols
- Python repository suite: PASS, 45/45

This closes the compiler-availability gap for the current conversation: subsequent Rust changes can be compiled, linted, tested, and release-built against the uploaded offline toolchain before handoff.
