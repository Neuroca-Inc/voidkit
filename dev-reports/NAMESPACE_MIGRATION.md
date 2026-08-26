# Namespace migration

## VDM

The live VDM package namespace has been renamed without changing the contained mathematical mechanisms:

- Python: `voidkit.void_dynamics` → `voidkit.vdm`
- Rust: `src/void_dynamics` → `src/vdm`

CLI entry points and live imports now target `voidkit.vdm`.

This is a namespace change only. SIE remains SIE, the authored multi-objective reward function; RE-VGSP, VDM equations, diagnostics, and other named mechanisms retain their identities.

Historical material under `sources/` and historical migration/report documents may still contain the old path because those files record earlier repository states.

## Phase Calculus

Two explicit namespaces now exist for future Phase Calculus implementation:

- Python: `voidkit.phase_calculus`
- Rust: `src/phase_calculus`

They are intentionally scaffolds. No Phase Calculus equation, kernel, or claim was invented or migrated implicitly during this namespace change.
