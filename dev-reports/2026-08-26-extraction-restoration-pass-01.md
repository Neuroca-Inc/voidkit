# VoidKit extraction restoration pass 01

Date: 2026-08-26

## Scope

Restore the already-validated v3 extraction batch into the compiler-verified v0.1.0 baseline before mining new mathematics.

## Restored capabilities

- finite permutations: `voidkit/algebra/permutation.py`
- free-group word operations: `voidkit/algebra/free_group.py`
- discrete Heisenberg group arithmetic: `voidkit/algebra/heisenberg.py`
- deterministic interval bisection and real Lambert-W/x+sin brackets: `voidkit/numerical/interval_roots.py`
- periodic Fourier spectral gradient/Laplacian: `voidkit/numerical/spectral.py`
- Klein-Gordon Verlet/stiffness/energy-distance helpers: `voidkit/wave/klein_gordon.py`
- Yin-Yang spherical patch transforms: `voidkit/spatial/yinyang.py`
- exact finite-time logistic flow and logarithmic invariant: `voidkit/dynamics/logistic.py`

## Authority and custody

The restored files come from the previously validated `voidkit-v3-extraction-pass-01` artifact. The provenance receipt at `voidkit/provenance/v3_extraction_pass_01.json` identifies the originating retained source and SHA-256 for each extraction family. Those hashes were rechecked against `VDM_Math_Mining_for_VoidKit_v3(1).zip` and the authored `reaction_exact.py` source before restoration.

The source hashes matched exactly:

- intervals: `c161c02975e9e157cc1b11dd7f4fd3bb57d6c100d490f34e6d6c02809fd43224`
- permutation: `9ff7494c534fdc1f96f3115be86ceb75016774d2f6fb56823c1cd56bca4fa808`
- free-group words: `d59a86979b3515c06e7260b3c86268b14b4368efb635af59030217f7f940609d`
- Heisenberg: `fdc68df28d9348153b107c477440f105fdaa9adc84c17c79535606b50276c645`
- spectral/Klein-Gordon source: `46ef011ba5e0b58e0124a0c8b6978cc652d4added02da2d8798ffe705223d221`
- Yin-Yang: `a7bb89d6710f055dbc8ed01100804e585da71796f4220da2df5c7f5035ddc801`
- exact logistic source: `0c8d93c2ee24d0884b855651b8bceb557c46b8d2d1accaae4187c866e3a4483b`

## Architectural decisions

This pass does not add new mathematical mechanisms. It restores already-extracted capabilities at the destinations previously recorded in the mining ledger. `voidkit.vdm` and `voidkit.phase_calculus` are untouched. The Rust source tree is untouched.

`voidkit.numerical` keeps its existing lazy legacy adapters while adding direct submodules for interval and spectral operations. `voidkit.spatial` keeps `SpatialHashGrid` while adding Yin-Yang transforms.

## Verification burden

The restored regression suite covers analytical identities, invalid brackets, group inverses/commutators, periodic Fourier modes, bounded short-step Klein-Gordon energy error, Yin-Yang Cartesian round-trip, logistic flow composition, invariant preservation, and the `u=0` exponential limit.

The next extraction target is the prior 27-module ore-extraction family. It must be reconciled against existing namespace ownership before merge; duplicate `information`/`info_theory`, `timeseries`/`time_series`, and `thermo`/`thermodynamics` namespaces must not be introduced merely because the earlier extraction artifact used those names.
