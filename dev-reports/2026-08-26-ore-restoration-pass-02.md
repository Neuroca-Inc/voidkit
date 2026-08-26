# VoidKit ore restoration pass 02

Date: 2026-08-26

## Goal

Restore the prior 27-module ore-extraction artifact after the validated v3 extraction batch, while preserving current VoidKit ownership boundaries rather than reproducing historical package names.

## Authority

The source artifact is `voidkit-ore-extraction-v0.1.0.zip`. Its frozen `provenance/SOURCES.json` contains 26 source records mapping to 27 extracted Python modules. All 26 frozen source snapshots were copied under `sources/ore_extraction_v0_1_0/` and rehashed against the recorded SHA-256 values; all 26 matched.

The original ore tests passed 33/33 before reconciliation. Their behavior tests were then adapted only for current namespace ownership and merged into the repository test suite.

## Ownership reconciliation

Historical extraction paths were not treated as automatic architecture authority.

- `voidkit.information.*` -> `voidkit.info_theory.*`
- `voidkit.timeseries.events` -> `voidkit.time_series.events`
- `voidkit.thermo.lit` -> `voidkit.thermodynamics.lit`
- historical `voidkit.wave.kg` was not restored as a duplicate owner; its unique `kg_energy` capability was merged into `voidkit.wave.klein_gordon`
- `voidkit.dynamics.logistic` was already restored from a later validated extraction with the identical authored source hash and was not overwritten

The other ore modules retain their previous capability homes where no conflicting owner existed: `pde`, `stats`, `recurrence`, `signal`, `special`, `structure`, `topology`, `variational`, `graph`, and `wave`.

## Dependency cleanup

Pandas is no longer imported merely to import the statistical modules. DataFrame-returning heavy-tail helpers import pandas locally; the duration-grouping helper also imports it locally. The `statistics` optional dependency group declares pandas explicitly. GWPy remains optional and is imported only by the dedicated GWPy signal loader.

The signal-ridge type annotations were normalized to Python 3.9-compatible syntax without changing runtime behavior.

## Warning cleanup

The source Vuong identity case used `0 * inf`, which produced a NumPy runtime warning while returning `nan`. The reconciled implementation preserves the same `nan` result for an exact zero-variance tie without performing the warning-generating operation; nonzero zero-variance differences retain signed infinity.

## Provenance

- frozen historical source receipt: `sources/ore_extraction_v0_1_0/SOURCES.json`
- current destination reconciliation: `voidkit/provenance/ore_extraction_reconciliation_v0_1_0.json`
- package metadata explicitly includes provenance JSON receipts in built Python wheels

## Scientific custody notes

- `structure.generic.jacobi_residual` remains a source-defined proxy, not a full functional Jacobi proof.
- `topology.vr_graph.beta1_curve` remains graph cycle rank `E - V + C`, not full simplicial persistent homology.
- reaction-diffusion AVF/Picard claims remain bounded by the source's practical convergence scope.
- the cylindrical solver remains scoped to its documented piecewise finite-radius model.
- transfer-entropy unit distinctions in the source family remain explicit.

No VDM or Phase Calculus authored mechanism was renamed or redefined in this pass.

## Final verification

- Python 3.9 grammar parse: 132/132 package files PASS
- source import sweep: 132/132 package modules PASS
- Python tests with warnings promoted to errors: 75/75 PASS
- frozen ore source rehash: 26/26 PASS
- Python wheel build/install smoke: PASS
- wheel provenance receipts included: PASS
- Rust source/Cargo/build metadata compared with the compiler-verified baseline: 0 changes
- strict Rust Clippy (`-D warnings`): PASS
- Rust all-target unit tests: 134/134 PASS

The native Rust/Maturin wheel was already compiler-verified on the byte-identical Rust tree before this Python-only restoration pass. No Rust source or Rust packaging metadata changed here.
