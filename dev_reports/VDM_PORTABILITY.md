# VDM Portability Rule

This document governs cleanup of `voidkit.vdm` after removal of the arbitrary domain-parameter modulation layer.

## First rule: do not genericize authored mechanisms

The portability audit applies to mathematical subroutines that can stand on their own outside VDM. It does **not** authorize renaming or reclassifying named research mechanisms.

### SIE is not a generic signal utility

SIE is the **Self-Improvement Engine**, an authored **multi-objective reward function/framework**. Its reward combines multiple objectives/signals, including TD error, novelty, habituation, and self-benefit, with downstream reward-dependent plasticity modulation. It must not be renamed to an "adaptive signal" or treated as an anonymous generic helper.

SIE is therefore **out of scope for genericization in this cleanup**. Its mathematical behavior remains unchanged. A documentation-only acronym expansion error in the Rust module was corrected separately and is recorded in `NAMING_CORRECTIONS.md`.

## Current `vdm` scope

| Component | Cleanup status | Custody rule |
|---|---|---|
| SIE | Preserve | Authored multi-objective reward function; do not genericize or reinterpret. |
| RE-VGSP | Preserve | Named learning rule; do not silently recast as generic plasticity. |
| EHTP/TDA formulas | Preserve pending separate review | Do not change semantics merely to make them look general. |
| Diagnostic formulas | Review individually | Extract only calculations whose meaning is independently stateable outside VDM. |
| Void equations | Review individually | Do not call VDM-specific laws generic mathematics. |

## Removed mechanism

The arbitrary domain-parameter modulation layer is removed from the live package. There is no longer a live API that changes RE-VGSP/GDSP coefficients according to a named physics domain or fixed domain sparsity target.

Historical copies under `sources/` remain frozen for provenance and may still contain the removed parameter.

## Rule for future VoidKit mining

For each VDM-originated mathematical component:

1. Determine what the component actually is from its source/specification before classifying it.
2. If it is a named authored mechanism, preserve its identity and semantics.
3. If it contains independently reusable mathematical subroutines, extraction may be additive, with provenance retained.
4. Never replace the named mechanism with the extracted generic subroutine.
5. Packaging/naming corrections must be recorded separately from scientific changes.
