# Third Sweep — New Mining Candidates

This file records only additions promoted from `third_sweep.zip` after exact-hash and semantic duplicate review against v2.

Promoted: **11 Tier A**, **10 Tier B**, plus **3 support dependencies**.

## Tier A — Mine first

1. `intervals.py` — Deterministic real interval root solver and real Lambert-W / x+sin bracketing.
2. `permutation.py` — Generic immutable finite permutation operations: composition, inverse, cycle decomposition.
3. `words.py` — Generic free-group word inversion, reduction, and commutator construction.
4. `run_head_to_head.py` — Expression-tree symbolic-regression / enumerative EML search machinery.
5. `lib (55th copy).rs` — no_std arbitrary-width exact arithmetic: BigNat, BigInt, GaussianRational, gcd/division/encoding.
6. `orthad_closed_form.py` — Execution-free exact Orthad closed form / random access, Fibonacci fast doubling, charts, transfers, cardinalities.
7. `candidate_local_coordinate.py` — Exact local-coordinate encoding/decoding, threshold oracle, capacity/reachability.
8. `jacobian_sheet_custody.py` — Exact symbolic Jacobian/sheet custody, weighted-fiber identity and covariance checks.
9. `qbl_orthad_query.c` — Read-side exact Orthad/QBL query formulas and relation cardinalities without enumeration.
10. `farey_balanced_anchor_kernel.S` — Low-level balanced Farey/Fibonacci trace kernel to an endogenous product floor.
11. `deep_wide_sweep.py` — Exact layered-state math, phase moments, cross energy, chiral radius and tridiagonal relation-gap solver.

## Tier B — Strong specialized mechanisms

1. `certifier.py` — Inverse/root certification service using deterministic interval bisection and high precision.
2. `20260619T190328_orthad_lift_measurement.py` — Partition refinement/inherit-extend measurement, exact coefficients, entropy and residual resolution.
3. `affine_binary_locality_probe.py` — Affine cut/binary-carry symbolic dynamics, refinement boundaries and locality/operator checks.
4. `first_b_germ_sector.py` — Exact SymPy germ intervals, B update, reflection/commutation and recovered matrix controls.
5. `first_b_unimodular_orthad.py` — Concise exact 2x2 unimodular/symplectic shear and alternating-form transport machinery.
6. `native_transition_group_probe.py` — Inverse Q/B/L transition recovery and typed relation-automorphism/chart-transfer checks.
7. `xi_endogenous_floor_burden.py` — Balanced-window depth solver with Fibonacci product floor and endogenous burden recurrence.
8. `xi_step_endogenous_floor.S` — Assembly endogenous-floor update and balanced arithmetic/completion germ.
9. `xigraph_engine.c` — Deterministic graph dynamics: compatibility, debt absorption, edge strengthening, 2D resolution and node birth.
10. `invariants.py` — Trace morphology math: components, polar/radial profiles, harmonic spectra, shell/ring/IoU invariants.

## Support dependencies

- `fibonacci.py` — Support for certifier.py.
- `models.py` — Support data models for certifier.py.
- `readouts.py` — Support radial/packet field construction used by invariants.py.

## Important custody note

- `20260711T162758_v7n_finite_orthad_qgt_jm_split.py` contains top-level filesystem/reporting side effects in its original research form. It is retained **as mining source**, not as an import-safe VoidKit module.
- `deep_wide_sweep.py`, `affine_binary_locality_probe.py`, and several other research scripts also mix reusable math with experiment/report orchestration. Extract the mathematical kernels rather than importing the scripts wholesale.
- Assembly/C/Rust files are retained because they encode useful algorithms; they are not implied to be ready-made VoidKit APIs.
