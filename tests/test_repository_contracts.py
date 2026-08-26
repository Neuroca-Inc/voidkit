from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_license_and_readme_agree_on_bsd3():
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("BSD 3-Clause License")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "BSD 3-Clause License" in readme
    assert "dual research/commercial license" not in readme


def test_live_sources_do_not_reintroduce_retired_license_terms():
    forbidden = (
        "This research is protected under a dual-license",
        "Commercial use requires written permission",
    )
    live_files = list((ROOT / "voidkit").rglob("*.py")) + list((ROOT / "src").rglob("*.rs"))
    offenders = []
    for path in live_files:
        text = path.read_text(encoding="utf-8")
        if any(term in text for term in forbidden):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_package_metadata_and_citation_files_exist():
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "Cargo.toml").is_file()
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "title: VoidKit" in citation
    assert "license: BSD-3-Clause" in citation


def test_rust_dependency_resolution_is_locked():
    assert (ROOT / "Cargo.lock").is_file()
    workflow = (ROOT / ".github" / "workflows" / "rust-tests.yml").read_text(encoding="utf-8")
    script = (ROOT / "tools" / "validate_rust.sh").read_text(encoding="utf-8")
    wheel_script = (ROOT / "tools" / "build_rust_wheel.sh").read_text(encoding="utf-8")
    assert "cargo clippy --locked" in workflow
    assert "cargo test --locked" in workflow
    assert "cargo clippy --locked" in script
    assert "cargo test --locked" in script
    assert "maturin build --release --locked" in wheel_script


def test_mining_sources_are_not_python_test_roots():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'testpaths = ["tests"]' in pyproject
    assert 'norecursedirs = ["sources"' in pyproject


def test_version_surfaces_are_consistent():
    version_file = (ROOT / "voidkit" / "_version.py").read_text(encoding="utf-8")
    assert '__version__ = "0.1.0"' in version_file
    assert 'version: 0.1.0' in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert '**v0.1.0 (pre-alpha)**' in (ROOT / "README.md").read_text(encoding="utf-8")


def test_rust_runtime_link_contract_uses_lib_unit_test_compatible_directive():
    build_rs = (ROOT / "build.rs").read_text(encoding="utf-8")
    assert "rustc-link-arg-tests" not in build_rs
    assert "cargo:rustc-link-arg=-Wl,-rpath" in build_rs
    assert "CARGO_FEATURE_EXTENSION_MODULE" in build_rs
    assert "PYO3_BUILD_EXTENSION_MODULE" in build_rs

def test_rust_ci_treats_compiler_warnings_as_failures():
    workflow = (ROOT / ".github" / "workflows" / "rust-tests.yml").read_text(encoding="utf-8")
    assert "RUSTFLAGS: -D warnings" in workflow
    assert "cargo test --locked --all-targets" in workflow



def test_rust_clippy_and_all_targets_are_ci_gates():
    workflow = (ROOT / ".github" / "workflows" / "rust-tests.yml").read_text(encoding="utf-8")
    assert "cargo clippy --locked --lib --tests -- -D warnings" in workflow
    assert "cargo test --locked --all-targets" in workflow


def test_native_wheel_has_separate_maturin_packaging_boundary():
    root_pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    rust_pyproject = (ROOT / "rust-wheel" / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert 'build-backend = "setuptools.build_meta"' in root_pyproject
    assert 'build-backend = "maturin"' in rust_pyproject
    assert 'manifest-path = "../Cargo.toml"' in rust_pyproject
    assert 'features = ["extension-module"]' in rust_pyproject
    assert 'name = "voidkit-rust"' in cargo
    assert 'name = "voidkit_rust"' in cargo


def test_rust_validation_script_denies_warnings_and_builds_wheel():
    script = (ROOT / "tools" / "validate_rust.sh").read_text(encoding="utf-8")
    assert 'RUSTFLAGS="${RUSTFLAGS:--D warnings}"' in script
    assert "cargo clippy --locked --lib --tests -- -D warnings" in script
    assert "cargo test --locked --all-targets" in script
    assert "./tools/build_rust_wheel.sh" in script


def test_rust_numpy_test_traits_are_scoped_where_readonly_is_used():
    # These modules intentionally keep PyArrayMethods out of production scope
    # because readonly() is used only by their unit tests. Keep the trait in
    # the cfg(test) module so strict non-test builds do not report it unused.
    test_only_readonly_modules = (
        "src/dynamical_systems/analyze_stability.rs",
        "src/dynamical_systems/find_fixed_points.rs",
        "src/fractal_analysis/fractal_spike_train.rs",
        "src/sde/sde_solver.rs",
        "src/soc_analysis/detect_neuronal_avalanches.rs",
        "src/vdm/diagnostics_formulas.rs",
    )
    for relative in test_only_readonly_modules:
        text = (ROOT / relative).read_text(encoding="utf-8")
        test_body = text.split("mod tests {", 1)[1]
        assert ".readonly()" in test_body
        assert "use numpy::PyArrayMethods;" in test_body
