# VoidKit Rust backend

VoidKit includes a Rust/PyO3 backend for selected mathematical kernels. Python is the primary public research interface; the Rust tree provides native implementations where performance, exactness, or an independent implementation is useful.

The Rust backend is **not** a separate mathematical authority. Authored VDM and Phase Calculus mechanisms retain the same identity and provenance across language implementations.

## Current structure

The `src/` tree follows the same capability-oriented ownership used by the Python package. Current module families include:

```text
src/
├── advanced_math/
├── causal_inference/
├── clustering/
├── dynamical_systems/
├── evolutionary/
├── fractal_analysis/
├── fractional_calculus/
├── graph/
├── iit/
├── info_theory/
├── neuro/
├── numerical/
├── optimization/
├── ot/
├── pathway_analysis/
├── phase_calculus/
├── sde/
├── semantic/
├── soc_analysis/
├── spatial/
├── stochastic/
├── structural_plasticity/
├── tda/
├── thermodynamics/
├── time_series/
├── vdm/
└── lib.rs
```

`src/vdm/easter_eggs/legacy_void_equations.rs` is intentionally quarantined historical material. It is not exported by the current `voidkit_rust` Python module as canonical VDM mathematics.

## Python extension surface

`src/lib.rs` currently registers native functions from numerical methods, SIE, RE-VGSP, VDM diagnostics, descriptive statistics, information theory, thermodynamics, fractional calculus, dynamical systems, SOC analysis, optimal transport, fractal analysis, stochastic simulation, time series, SDEs, evolutionary methods, spatial structures, IIT, semantic analysis, clustering, graph theory, causal inference, pathway analysis, structural plasticity, TDA, and optimization.

Phase Calculus currently has a Rust namespace scaffold; canonical Phase implementations will be promoted deliberately from the research corpus rather than inferred from historical filenames.

## Build

Requirements:

- a current stable Rust toolchain;
- Python 3.9+;
- `maturin` for Python-extension development or wheel builds.

Run native Rust tests without enabling the extension-module link mode. VoidKit treats compiler warnings as stabilization failures, so the canonical validation command denies warnings:

```bash
PYO3_PYTHON="$(command -v python)" RUSTFLAGS="-D warnings" cargo test --lib
```

On Linux, the repository build script gives ordinary Rust development/test targets an rpath to the exact `libpython` directory discovered by PyO3. This matters for Conda, pyenv, and similar Python installations where `libpython` may link successfully but is not on the operating system loader's default runtime search path. Extension-module builds are detected separately and do **not** receive this host-specific rpath, so it is not baked into the distributed Python extension module.

To inspect which interpreter PyO3 selected when diagnosing an environment mismatch:

```bash
PYO3_PRINT_CONFIG=1 cargo build
```

Use `PYO3_PYTHON=/path/to/python` when a specific interpreter must control the build.

Build the separate `voidkit-rust` Python extension wheel:

```bash
python -m pip install maturin
./tools/build_rust_wheel.sh
```

The native wheel has its own packaging boundary at `rust-wheel/pyproject.toml`.
That keeps the root Setuptools configuration authoritative for the pure-Python
`voidkit` distribution while Maturin packages the Cargo crate as `voidkit-rust`.
The native build enables the crate's `extension-module` feature because the
pinned PyO3 0.22 line otherwise links `libpython` on Unix.

For editable native-extension development inside a virtual environment:

```bash
python -m pip install maturin
(cd rust-wheel && maturin develop --release)
```

The Rust dependency manifest is [`Cargo.toml`](Cargo.toml). The package uses PyO3/NumPy bindings together with `nalgebra`, `ndarray`, `petgraph`, and the Rust `rand` ecosystem where those capabilities require them.

## Mathematical custody

A Rust implementation is promoted only when its relationship to the controlling mathematical source is understood. Cross-language implementations are useful as performance implementations and as independent regression/equivalence evidence; they do not authorize silent changes to equations or authored method identity.

In particular:

- SIE remains the Self-Improvement Engine multi-objective reward function;
- RE-VGSP remains an authored VDM mechanism;
- rejected void-debt/domain modulation is not part of the live Rust API;
- legacy hard-coded void equations remain quarantined under `vdm/easter_eggs`;
- Phase Calculus implementations will remain Phase Calculus rather than being flattened into generic utilities merely because portions are reusable.

## License

The Rust backend is licensed under the same **BSD 3-Clause License** as VoidKit. See [`LICENSE`](LICENSE).

## Repository

https://github.com/Neuroca-Inc/voidkit
