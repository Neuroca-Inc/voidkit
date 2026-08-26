# Rust runtime-link stabilization v2

**Date:** 2026-08-25
**Scope:** follow-up correction for `cargo test --lib` with PyO3 + Conda-style libpython placement.

## Evidence from the target host

The first host run established that the Rust crate and its dependency closure compiled and linked, but the unit-test executable could not start because the Linux dynamic loader could not locate `libpython3.13.so.1.0`.

The first repair attempted to use Cargo's `rustc-link-arg-tests` build-script directive. The next host run falsified that repair before Rust compilation completed:

```text
error: invalid instruction `cargo:rustc-link-arg-tests` ...
The package voidkit-rust ... does not have a test target.
```

## Root cause of the failed repair

`cargo test --lib` builds the package's `[lib]` target in Rust's test compilation mode. Cargo's `rustc-link-arg-tests` build-script instruction is validated against manifest test targets (`[[test]]` / `Target::is_test`) rather than the library target being compiled as a unit-test harness. The package therefore correctly rejected that target-specific directive.

The libpython loader problem itself remains the same: the selected Python library is linkable but its directory is not on the runtime loader's default path.

## Corrected repair

The build script now uses Cargo's general `rustc-link-arg` instruction for ordinary Linux development/test builds. Cargo applies that instruction to test compilations, including the library unit-test harness. The rpath value still comes from the exact `lib_dir` discovered by PyO3.

To avoid contaminating distributed extension modules with a host-specific Conda/pyenv path, the rpath branch is disabled whenever either extension-build signal is present:

- Cargo feature `extension-module`; or
- `PYO3_BUILD_EXTENSION_MODULE`, as used by modern PyO3/Maturin extension builds.

PyO3's `add_extension_module_link_args()` remains in place for extension-module linker requirements on platforms where it is meaningful.

## Regression protection

`tests/test_repository_contracts.py` now asserts that:

- `build.rs` does not reintroduce `rustc-link-arg-tests`;
- the ordinary-build rpath uses `rustc-link-arg`;
- both extension-build guards remain present.

## Verification boundary

The current packaging environment has no Rust toolchain, so this successor repair is source-audited and Python-regression-tested here but still requires one target-host execution of `PYO3_PYTHON="$(command -v python)" cargo test --lib`.

No mathematical Rust implementation was changed by this repair.
