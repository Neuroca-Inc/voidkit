# Legacy repository utilities

These scripts are retained as historical migration/packaging evidence. They are **not active repository tooling** and must not be run against the current VoidKit tree without a new audit.

- `audit_fum.ps1` and `vdm_refactor.ps1` belong to the earlier FUM-to-VDM namespace migration.
- `metadata_hooks.py` is an obsolete Hatch metadata hook from the former packaging design.
- `update_pyproject.ps1` can rewrite dependency/author/URL metadata from legacy sources and therefore must not control the current `pyproject.toml`.

Current package and dependency authority is the root `pyproject.toml`; current licensing authority is the root `LICENSE`.
