# Rust test-trait scope stabilization

## Trigger

Strict Clippy compilation failed after the prior warning cleanup because six modules use `numpy::PyArrayMethods::readonly()` only inside `#[cfg(test)]` unit tests. Removing the production-scope trait import eliminated non-test warnings but also removed the trait from test compilation.

## Invariant

Production Rust builds must remain warning-clean without removing traits required by unit tests. Test-only extension traits belong in the test module that uses them.

## Change

Added `use numpy::PyArrayMethods;` inside the `mod tests` scope of:

- `src/dynamical_systems/analyze_stability.rs`
- `src/dynamical_systems/find_fixed_points.rs`
- `src/fractal_analysis/fractal_spike_train.rs`
- `src/sde/sde_solver.rs`
- `src/soc_analysis/detect_neuronal_avalanches.rs`
- `src/vdm/diagnostics_formulas.rs`

No executable mathematical logic, public API, numerical constants, equations, or production imports changed.

A repository contract test now guards this failure class.

## Acceptance

Run `./tools/validate_rust.sh`. The expected path is strict Clippy, all Rust targets, then the Maturin extension wheel build.
