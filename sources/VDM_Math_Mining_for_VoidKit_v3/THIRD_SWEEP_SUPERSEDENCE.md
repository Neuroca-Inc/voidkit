# Third Sweep — Supersedence / Rejection Notes

These are the most important cases where a byte-distinct file was intentionally **not** treated as a new tool.

## Semantic duplicates / superseded modules

- `carry_oracle.py` — Superseded by orthad_closed_form.py, whose own module contract says it replaces separate carry/interface/read-port modules.
- `interface_oracle.py` — Superseded by orthad_closed_form.py.
- `orthad_read_port.py` — Superseded by orthad_closed_form.py.
- `candidate_global_proof.py` — Substantially overlaps candidate_local_coordinate.py + orthad_closed_form.py; not a distinct mining primitive.
- `orthad_geom.py` — Exact chart/point geometry is already represented by orthad_closed_form.py.
- `operators.py` — Discrete R/S/T/selector transport operations already represented in v2 vdm_dynamics_regressor.py.
- `commutator_service.py` — Application wrapper around algebra already represented more directly; not a new primitive.
- `sheets.py` — One-off finite sheet specialization; generic permutation.py is the reusable primitive.
- `native_color_flux.py` — Duplicate family; accepted_native_color_flux_recurrence.py is byte-identical in this sweep.
- `v21_core.py` — Byte-identical to first_l_hierarchical_relation_grid.py; descriptive canonical file retained instead.
- `xi_frontier_floor_kernel.S` — Semantically equivalent to retained xi_step_endogenous_floor.S; latter is the clearer/canonical variant.

## Older or reduced variants

- `qbl_reference.c` — Older/reduced relative to the richer v2 qbl_reference.c (v2 includes additional CP5 Orthad-local implementation).
- `qbl_step_u64.S` — Older/reduced relative to the v2 kernel, which includes the larger CP5 Orthad-local state/kernel section.
- `qbl_b_u64.S` — Same algorithmic surface as v2 with mainly ABI/version change; no new math capability.
