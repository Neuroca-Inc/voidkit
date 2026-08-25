# VoidKit Mining Priority

This is a first-pass priority map against the supplied VoidKit package. It is not an instruction to port files wholesale. Mine the mathematical kernels, normalize interfaces, deduplicate against existing VoidKit capabilities, and add focused tests.

## Priority 1 — clear capability expansion

1. `ns3d_pseudospectral_f1a.py`
   - 3D pseudospectral incompressible Navier–Stokes machinery: FFT/IFFT, wave-number construction, dealiasing, incompressibility projection, curl, gradients, RHS, RK4, shell diagnostics.
   - VoidKit currently has generic ODE/integration tools but no PDE or pseudospectral solver family.

2. `spectral_ops.py`
   - Compact periodic spectral gradient/Laplacian and Klein–Gordon Verlet stepping / energy-norm machinery.
   - Good candidate for a new `numerical/spectral.py` or `pde/` surface.

3. `yinyang_transform.py`
   - Yin–Yang spherical-grid generation and exact forward/inverse coordinate transforms.
   - VoidKit has spatial hashing but no spherical overset-grid coordinate tools.

4. `diagnostics.py`
   - Graph shortest-path distance machinery, mass-gap estimation from correlations, and pulse/group-velocity estimation.
   - Distinct from current generic graph metrics and time-series correlations.

5. `phase_calculus_numeric_certificates.py`
   - Standalone numerical certificate routines around Fibonacci corridors and special constants/roots.
   - Mine individual general-purpose numerical kernels rather than Phase-specific orchestration.

6. `lifted_quintic_certifier.py` + `bring_quintic_root_certificate.py` + `unified_quintic_v1_4_validation.py`
   - Exact/numeric quintic-root and algebraic-certification machinery.
   - VoidKit symbolic solving is currently thin; this is a richer algebraic-method source.

## Priority 2 — exact/discrete algebra and transforms

7. `heisenberg.py`
   - Clean discrete Heisenberg group product, inverse, commutator, visibility/order calculations.

8. `fibonacci.py`
   - Balanced-pair refinement and exact resolution certificates.

9. `phase_block_solution.py` / `phase_selector_engine.py` / `phase_native_radix.py`
   - Balanced refinement, endogenous depth/floor selection, block coordinates, and radix/selector logic.
   - Several implementations overlap; extract one canonical primitive set instead of porting all variants.

10. `orthad_birth_transport.py`
    - Exact rational/phase coefficient algebra, Walsh-like phases, transport operators, matrix operations, custody/state transitions.

11. `phase5_v7m_trace_cocycle_normal_form.py`
    - Event independence / Foata normal form / cocycle extraction.

12. `phase5_v7r_fqm_gauge_isometry_classifier.py` and `phase5_v8b_full_fqm_classification_boundary_attack.py`
    - Finite quadratic module / gauge-isometry classification, modular arithmetic, Legendre-symbol and radical logic.

13. `jacobian_sheet_custody.py`
    - Exact witness/equivalence-class machinery around Jacobian sheets.

## Priority 3 — graph-field and dynamical-system mechanisms

14. `void_equations.py`
    - Lattice/stencil construction, weighted graph Laplacian ingredients, node/bond potential derivatives, transport calibration.
    - Much richer than the small legacy `Void_Equations.py` already represented in VoidKit.

15. `vdm_dynamics.py`
    - Telegraph dynamics, ring Laplacian, residual/energy/gradient, debt and thermal state evolution.

16. `connectome.py` + `gauge.py`
    - Coupled node/bond field dynamics and walker/gauge propagation. Mine reusable graph-field kernels rather than the runtime object model.

17. `invariants.py`
    - Logistic-motion invariant and kinetic conversion helpers plus drift checks.

18. `metrics.py`
    - Connectome entropy and streaming ZEMA-style metric tracking.

19. `global_system.py`
    - Small 1-D k-means / cohesion routines are potentially reusable; extract those functions only.

20. `field.py`, `memorymap.py`, `base_decay_map.py`
    - Sparse event-driven decay/spread field mathematics. Relevant if VoidKit expands toward sparse dynamical fields.

## Priority 4 — symbolic identity / physics math mine

The `CF*`, `cf*`, `sympy_*`, `tdahe_*`, `shadow_residual_*`, and much of `phase3_*`–`phase5_*` are retained because they contain exact equations, symbolic identities, modular arithmetic, matrix constructions, transforms, or classification algorithms. They should be mined function-by-function, not treated as coherent VoidKit modules.

Especially promising clusters:

- Contact / Poisson / Jacobi algebra: `CF02_symbolic_checks.py`, `cf05_sympy_verification.py`.
- QGT / metric calculations: `CF01_SSH_Metriplectic_Invariant_sympy.py`, `CF01_TwoBand_QGT_Determinant_Ratio_sympy.py`.
- Covariant multi-filter transforms: `two_filter_covariance.py`, `phase_three_filter_covariance.py`, `phase_four_filter_covariance.py`.
- Shadow/residual coefficient algebra: `shadow_residual_eta_sympy_v59.py`, `shadow_residual_multiplier_v60.py`.
- Exact baryon/operator relation machinery: `native_baryon_hydrogen.py`, `native_baryon_transport_hydrogen.py`, `native_operator_closure.py`.

## Do not port wholesale

The corpus intentionally retains research-specific context where the underlying algorithm may be useful. Before adding anything to VoidKit:

1. Extract the smallest reusable mathematical primitive.
2. Compare it to existing VoidKit functionality and choose one authority.
3. Remove research-specific file IO, plotting, hard-coded experiment paths, and orchestration.
4. Define a small deterministic API.
5. Add positive, edge-case, and negative-control tests.


---

## v3 third-sweep additions

Mine `THIRD_SWEEP_NEW_CANDIDATES.md` in order. The highest-leverage generic additions are `intervals.py`, `permutation.py`, `words.py`, `lib (55th copy).rs`, and the expression-tree search machinery in `run_head_to_head.py`; the strongest exact/research-specific additions are `orthad_closed_form.py`, `candidate_local_coordinate.py`, `jacobian_sheet_custody.py`, `farey_balanced_anchor_kernel.S`, and the Xi/Farey/graph kernels.
