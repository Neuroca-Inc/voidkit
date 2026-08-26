# Domain Modulation Removal

This change is intentionally narrow.

## Removed

- `voidkit/vdm/void_debt_modulation.py` (formerly `voidkit/void_dynamics/void_debt_modulation.py`)
- the `domain_modulation` argument from the live Python `delta_re_vgsp`, `delta_gdsp`, and `universal_void_dynamics` APIs
- the `domain_modulation` argument and multiplier from the live Rust `delta_re_vgsp` and `delta_gdsp` APIs
- the obsolete domain-modulation regression test

The removed mechanism assigned domain-specific scaling through fixed target sparsity percentages. It is not part of the live VoidKit API after this change.

## Explicitly not changed scientifically

- **SIE remains the Self-Improvement Engine multi-objective reward function/framework.** Its equations and reward behavior were not changed.
- RE-VGSP remains RE-VGSP.
- EHTP/TDA formulas remain in their existing namespace.
- VDM void-equation bodies were not reconciled or redesigned beyond removing the domain multiplier.
- Historical source material under `sources/` remains frozen provenance and may still contain the old `domain_modulation` text.

## Documentation-only SIE naming correction

The Rust module header in `src/neuro/advanced_sie.rs` incorrectly expanded SIE as "Stabilized Information-theoretic Engagement." The implementation directly below that header already describes a stabilized **multi-objective reward function**. The header has been corrected to "Self-Improvement Engine (SIE)" without changing executable code. See `NAMING_CORRECTIONS.md`.
