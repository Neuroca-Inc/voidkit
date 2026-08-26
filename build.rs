fn main() {
    // Keep PyO3's extension-module linker behavior available for manual
    // extension builds (notably macOS). Maturin also configures extension
    // linking when it builds the distributed module.
    pyo3_build_config::add_extension_module_link_args();

    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");

    // Normal Rust binaries and unit-test harnesses embed Python and therefore
    // link to libpython. Conda/pyenv installations often keep libpython outside
    // the system loader's default search path. Add the interpreter's discovered
    // library directory as an rpath for ordinary development/test builds.
    //
    // Do NOT add that host-specific rpath when building a Python extension for
    // distribution. The explicit extension feature and modern Maturin's
    // PYO3_BUILD_EXTENSION_MODULE environment variable both identify that path.
    let extension_build = std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some()
        || std::env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some();

    if !extension_build && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let config = pyo3_build_config::get();
        if let Some(lib_dir) = config.lib_dir.as_deref() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
        }
    }
}
