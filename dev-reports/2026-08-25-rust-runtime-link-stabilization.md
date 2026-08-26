# Rust runtime-link stabilization

**Date:** 2026-08-25
**Scope:** `cargo test --lib` PyO3/libpython runtime loader failure on Linux.

## Observed failure

The Rust crate compiled and linked successfully through creation of the test executable, then the Linux dynamic loader refused to start it because `libpython3.13.so.1.0` was not in the runtime search path.

This is distinct from a Rust compile failure or missing Cargo dependency. PyO3 had already located the Python library well enough to link the executable.

## Root cause

For Rust tests, PyO3 is embedding/linking the selected Python interpreter. Conda/pyenv-style Python installations can report a valid Python library directory to the linker while that directory is absent from the Linux loader's default runtime path. The resulting test executable has a `DT_NEEDED` dependency on libpython but no route to that non-system directory at process startup.

## Repair

Added `build.rs` plus `pyo3-build-config` as a build dependency. On Linux, the build script reads the exact interpreter configuration already selected by PyO3 and emits a test-target-only rpath to its `lib_dir`.

The rpath is intentionally limited to Cargo test executables. Host-specific Python paths are not embedded into distributed extension modules.

The same build script also applies PyO3's extension-module linker arguments for platforms where they are needed.

## Regression protection

Added a Rust GitHub Actions matrix for Python 3.9, 3.11, and 3.13 running `cargo test --lib` with `PYO3_PYTHON=python`.

## Verification boundary

The repair is source-audited against PyO3 0.22's public `InterpreterConfig.lib_dir` API and Cargo's `rustc-link-arg-tests` build-script directive. The packaging environment used to produce this repository does not contain Cargo/Rust, so the repaired native test executable must still be executed on a Rust-capable host before claiming the Rust suite passes.

The pre-repair user run already established that all Rust dependencies compiled and the test binary linked; the remaining failure was process startup due solely to libpython discovery.
