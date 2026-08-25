# VoidKit Python Stabilization Pass 01

**Date:** 2026-08-24  
**Status:** CHECKPOINT — not a full-library mathematical certification

## Goal

Stabilize the existing Python mathematical substrate before adding new research methods. Preserve VoidKit as a VDM-branded centralized mathematics library while distinguishing:

- **native/distinctive methods** — uncommon, novel, VDM-derived, or materially improved algorithms;
- **curated adapters** — existing external algorithms retained when centralized access, normalized contracts, dependency isolation, composition, or discoverability add practical research value.

## Verified checkpoint

- Python test suite: **30 passed**.
- Python module import sweep: **85 / 85 modules imported** in the available environment.
- Python 3.9 grammar compatibility: **PASS** via `ast.parse(..., feature_version=(3, 9))`.
- `compileall` over `voidkit/` and `tests/`: **PASS**.
- Optional `ripser` and `scikit-optimize` dependencies no longer break unrelated namespace imports; the requiring functions fail with actionable optional-extra messages when invoked without those packages.
- Wheel build was attempted but could not be completed in the current offline execution environment because the declared Hatchling build backend is not installed locally and cannot be downloaded. This is an environment blocker, not a verified package-build pass.

## Correctness repairs completed

### Fractional calculus

- Replaced the mislabeled Grünwald-Letnikov-style implementation with an L1 Caputo discretization for `0 < alpha < 1`.
- Added invariant coverage that the Caputo derivative of a constant is zero.
- Added a closed-form linear-function reference test.

### Dynamical systems

- Fixed fixed-point solving so unsuccessful nonlinear solves are not returned as equilibria.
- Added residual validation and duplicate-root suppression.
- Reclassified zero-real-part Jacobian cases as nonhyperbolic / linearization-inconclusive rather than automatically calling them centers.

### Stochastic / SDE

- Gillespie reactions sampled beyond `t_max` are no longer executed.
- Added propensity and stoichiometry validation and injectable RNG.
- Euler-Maruyama now keeps returned times consistent with actual step lengths when `dt` does not evenly divide the interval.
- Added scalar, diagonal-vector, and full diffusion-matrix handling.

### Information theory / causal inference

- Corrected the Information Bottleneck objective from `I(X;T) - beta I(X;Y)` to `I(X;T) - beta I(T;Y)` by reconstructing `P(T,Y)` under `T <- X -> Y`.
- Fixed transfer-entropy discretization so maximum-valued observations cannot index outside the final bin.
- Added directed delayed-source validation.
- Added probability/base/domain checks to discrete entropy, mutual information, and KL divergence.

### Fractal / SOC

- Repaired box-counting dimension scale/sign logic and saturation handling.
- Added reference tests recovering approximately dimension 1 for a line and 2 for a planar grid.
- Fixed non-integral spike-train duration/time-grid shape mismatch.
- Replaced the small-rate Bernoulli approximation with the exact one-or-more Poisson event probability per bin.
- Fixed avalanche detection for a spike at time zero.
- Replaced fragile log-histogram power-law fitting with a continuous MLE exponent plus empirical-CCDF R² diagnostic.
- Fixed burst detection so duplicate spike times do not corrupt start-index recovery.

### Numerical adapters

- Quadrature now preserves QUADPACK support for reversed and equal bounds and exposes normalized tolerance controls.
- ODE adapter now supports backward integration, validates initial derivative shape/finiteness, and promotes `success=False` to failure without broad exception laundering.

### Graph / clustering / spatial

- Fixed graph namespace import failure caused by missing `Any`.
- Corrected `detect_communities(method="louvain")` to actually run Louvain rather than silently substituting greedy modularity.
- Reworked temporal spectral clustering to select cluster count from the normalized-Laplacian eigengap rather than raw-affinity eigenvalue gaps.
- Added graph path-score validation.
- Generalized spatial hashing from an effectively 2-D neighboring-cell candidate query to an N-D exact Euclidean radius query.

### Packaging / imports

- Fixed invalid SymPy `subs` import.
- Fixed `void_debt_modulation` package-relative import.
- Added optional dependency groups for symbolic, graphs, causal, TDA, clustering, optimization, plotting, and VDM/Torch surfaces.
- Moved Ripser and scikit-optimize imports behind their actual function boundaries.

### Miscellaneous numerical robustness

- Constant-signal normalized autocorrelation now returns defined zeros instead of NaNs.
- Mutation/recombination accept injectable NumPy generators and validate probabilities/scales.
- Logistic growth trigger uses a numerically stable sigmoid.
- TDA summary metrics separate essential H0/H1 bars from finite H1 persistence totals.

## Deliberately not redefined in this pass

The following surfaces require authoritative mathematical/research context before semantic changes:

- VDM `void_dynamics` equations and domain-modulation claims;
- SIE / RE-VGSP research formulas;
- neuro/STDP/STC formulas;
- structural-plasticity constructs whose comments already identify legacy/non-VDM assumptions;
- simplified-phi / IIT proxy semantics;
- the quadratic `thermodynamics.free_energy` naming and interpretation;
- Rust/PyO3 implementations and Python/Rust semantic parity.

Those areas remain **unverified legacy research code**, not rejected code. They should be checked against the newer source corpus before promotion, correction, removal, or renaming.

## Next stabilization frontier

1. Audit remaining legacy formulas against their actual source authority and newer research artifacts.
2. Build mathematical reference/invariant tests before altering VDM-native equations.
3. Complete optional-extra runtime tests in environments with Ripser and scikit-optimize installed.
4. Reconcile or freeze Rust implementations until Python reference semantics are established for each accelerated function.
5. Revisit namespace structure only after capability correctness and authority are settled.
