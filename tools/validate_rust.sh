#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYO3_PYTHON="${PYO3_PYTHON:-$(command -v python)}"
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo is required" >&2
  exit 2
fi
if ! cargo clippy --version >/dev/null 2>&1; then
  echo "ERROR: rustup component add clippy" >&2
  exit 2
fi
if ! command -v maturin >/dev/null 2>&1; then
  echo "ERROR: maturin is required; install with: python -m pip install 'maturin>=1.9,<2.0'" >&2
  exit 2
fi

cargo clippy --locked --lib --tests -- -D warnings
cargo test --locked --all-targets
./tools/build_rust_wheel.sh
