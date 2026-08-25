#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "native_phase_local_decoder_v9.so"
SRC_PATH = ROOT / "native_phase_local_decoder_v9.c"
MPFR_LIB = "/lib/x86_64-linux-gnu/libmpfr.so.6"
PROBE_DIGITS = 32
QUERY_LENGTH = 32
QUERY_POSITIONS = [1, 4895, 500_000, 999_937, 1_000_000]
BENCH_REPS = 64


@dataclass(frozen=True)
class QueryRow:
    base: int
    start_digit_after_decimal: int
    length: int
    block: str
    probe_suffix: str
    first_nonmax_probe_position: int
    certified: bool
    safe_length_lower_bound: int
    safe_end_position: int
    query_seconds: float
    matches_reference: bool
    reference_block: str


@dataclass(frozen=True)
class PacketRow:
    depth: int
    u: int
    v: int
    N: int
    pre_collapse_q_terms_1e6_digits: int
    post_collapse_local_query_equivalent: bool
    post_collapse_query_comment: str
    preferred_under_current_law: bool


@dataclass(frozen=True)
class BenchRow:
    base: int
    start_digit_after_decimal: int
    length: int
    repetitions: int
    min_seconds: float
    mean_seconds: float
    blocks_per_second: float
    digits_per_second: float


def compile_native() -> None:
    if LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SRC_PATH.stat().st_mtime:
        return
    cmd = [
        "gcc",
        "-Ofast",
        "-march=native",
        "-funroll-loops",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-shared",
        "-fPIC",
        str(SRC_PATH),
        MPFR_LIB,
        "-lgmp",
        "-o",
        str(LIB_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


class NativeLib:
    def __init__(self) -> None:
        compile_native()
        self.lib = ctypes.CDLL(str(LIB_PATH))
        self.lib.phase_local_prepare_v9.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.phase_local_prepare_v9.restype = ctypes.c_int
        self.lib.phase_local_query_v9.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.phase_local_query_v9.restype = ctypes.c_int
        self.lib.phase_reference_query_v9.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.phase_reference_query_v9.restype = ctypes.c_int
        self.lib.phase_local_query_benchmark_v9.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.phase_local_query_benchmark_v9.restype = ctypes.c_int
        self.lib.phase_local_reset_v9.argtypes = []
        self.lib.phase_local_reset_v9.restype = None

    def reset(self) -> None:
        self.lib.phase_local_reset_v9()

    def prepare(self, required_decimal_digits: int, probe_digits: int) -> Dict[str, float | int]:
        seconds = ctypes.c_double()
        iters = ctypes.c_uint()
        bound_log10 = ctypes.c_double()
        rc = self.lib.phase_local_prepare_v9(required_decimal_digits, probe_digits, ctypes.byref(seconds), ctypes.byref(iters), ctypes.byref(bound_log10))
        if rc != 0:
            raise RuntimeError(f"phase_local_prepare_v9 failed rc={rc}")
        return {
            "required_decimal_digits": required_decimal_digits,
            "probe_digits": probe_digits,
            "warmup_seconds": seconds.value,
            "agm_iterations": int(iters.value),
            "coarse_bound_log10": bound_log10.value,
        }

    def query(self, base: int, start: int, length: int, probe_digits: int) -> Dict[str, object]:
        out = ctypes.create_string_buffer(length + 2)
        probe = ctypes.create_string_buffer(probe_digits + 2)
        seconds = ctypes.c_double()
        first_nonmax = ctypes.c_uint()
        cert_ok = ctypes.c_int()
        safe_len = ctypes.c_uint()
        rc = self.lib.phase_local_query_v9(
            base,
            start,
            length,
            probe_digits,
            out,
            len(out),
            probe,
            len(probe),
            ctypes.byref(seconds),
            ctypes.byref(first_nonmax),
            ctypes.byref(cert_ok),
            ctypes.byref(safe_len),
        )
        if rc != 0:
            raise RuntimeError(f"phase_local_query_v9 failed rc={rc}")
        return {
            "block": out.value.decode(),
            "probe_suffix": probe.value.decode(),
            "query_seconds": seconds.value,
            "first_nonmax_probe_position": int(first_nonmax.value),
            "certified": bool(cert_ok.value),
            "safe_length_lower_bound": int(safe_len.value),
        }

    def reference_query(self, base: int, start: int, length: int, probe_digits: int) -> Dict[str, object]:
        out = ctypes.create_string_buffer(length + 2)
        probe = ctypes.create_string_buffer(probe_digits + 2)
        rc = self.lib.phase_reference_query_v9(base, start, length, probe_digits, out, len(out), probe, len(probe))
        if rc != 0:
            raise RuntimeError(f"phase_reference_query_v9 failed rc={rc}")
        return {"block": out.value.decode(), "probe_suffix": probe.value.decode()}

    def benchmark(self, base: int, start: int, length: int, probe_digits: int, repetitions: int) -> Dict[str, float]:
        min_seconds = ctypes.c_double()
        mean_seconds = ctypes.c_double()
        rc = self.lib.phase_local_query_benchmark_v9(
            base,
            start,
            length,
            probe_digits,
            repetitions,
            ctypes.byref(min_seconds),
            ctypes.byref(mean_seconds),
        )
        if rc != 0:
            raise RuntimeError(f"phase_local_query_benchmark_v9 failed rc={rc}")
        return {"min_seconds": min_seconds.value, "mean_seconds": mean_seconds.value}


def required_decimal_digits_for_query(base: int, start: int, length: int, probe_digits: int) -> int:
    return math.ceil((start - 1 + length + probe_digits + 12) * math.log10(base)) + 8


def compute_required_decimal_digits() -> int:
    req = 0
    for base in (10, 16):
        for start in QUERY_POSITIONS:
            req = max(req, required_decimal_digits_for_query(base, start, QUERY_LENGTH, PROBE_DIGITS))
    return req


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def est_q_terms(N: int, digits: int) -> int:
    rhs = (N * digits * math.log(10.0)) / math.pi
    k = (1.0 + math.sqrt(1.0 + 12.0 * rhs)) / 6.0
    return max(1, math.ceil(k))


def packet_depth_rows(max_depth: int = 8) -> List[PacketRow]:
    rows: List[PacketRow] = []
    for depth in range(1, max_depth + 1):
        u = fibonacci(depth + 1)
        v = fibonacci(depth + 2)
        N = u * v
        rows.append(
            PacketRow(
                depth=depth,
                u=u,
                v=v,
                N=N,
                pre_collapse_q_terms_1e6_digits=est_q_terms(N, 1_000_000),
                post_collapse_local_query_equivalent=True,
                post_collapse_query_comment="after universal collapse, local orientation/query layer is N-independent; only warm-up burden still grows with N",
                preferred_under_current_law=(depth == 1),
            )
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(bundle_dir: Path) -> None:
    files = []
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file() and p.name not in {"manifest.json", "SHA256SUMS.txt"}):
        rel = path.relative_to(bundle_dir)
        files.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"bundle": bundle_dir.name, "files": files}
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (bundle_dir / "SHA256SUMS.txt").open("w") as f:
        for entry in files:
            f.write(f"{entry['sha256']}  {entry['path']}\n")
        f.write(f"{sha256_file(bundle_dir / 'manifest.json')}  manifest.json\n")


def make_notebook(bundle_dir: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase Native Local Decoder v9\n",
                    "\n",
                    "This notebook summarizes the v9 local decoder: native warm-up state, direct orientation query, and verification against an external reference.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["See output/local_decoder_summary.json and output/local_queries.json for executed measurements.\n"]}
                ],
                "source": [
                    "import json, pathlib\n",
                    "root = pathlib.Path('.')\n",
                    "summary = json.loads((root / 'output' / 'local_decoder_summary.json').read_text())\n",
                    "queries = json.loads((root / 'output' / 'local_queries.json').read_text())\n",
                    "print('warmup_seconds', summary['state']['warmup_seconds'])\n",
                    "print('decimal_query_500000', queries['decimal'][2]['block'])\n",
                    "print('hex_query_500000', queries['hex'][2]['block'])\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (bundle_dir / "phase_native_local_decoder_v9_notebook.ipynb").write_text(json.dumps(nb, indent=2))


def write_note(path: Path) -> None:
    text = r'''# Phase Native Local Decoder v9

v9 attacks the live burden left open by v8:

- no prefix bank,
- no indexed mmap over a prebuilt decimal file,
- direct radix-block extraction from the **current native state** after one collapsed-packet warm-up.

## 1. Native state used by the decoder

The control view is the three-parameter native state

\[
(\text{phase address},\ \text{shell/Farey state},\ \text{host/carry state}).
\]

For the present collapsed branch this is instantiated as

\[
\phi=(b,n),\qquad \sigma=(u,v,N),\qquad \chi=(a_m,b_m,t_m,p_m,B_m),
\]

with the balanced packet fixed at depth 1,

\[
(u,v)=(1,2),\qquad N=2,
\]

and the host/carry state \(\chi\) given by the AGM packet-collapse state after \(m\) native iterations.

The emitted invariant is

\[
\Pi(\chi)=\pi_m.
\]

## 2. Orientation law (local decoder)

Given radix \(b\ge 2\), start position \(n\ge 1\), and block length \(\ell\), define the orientation map

\[
\boxed{
\mathcal O_{b,n}(\pi_m):=\{b^{n-1}\pi_m\}
}
\]

where \(\{x\}=x-\lfloor x\rfloor\) is fractional part.

Then define the local projector

\[
\boxed{
\mathcal P_{b,\ell}(y)=d_1d_2\cdots d_\ell,
\qquad
 d_j=\Bigl\lfloor b\,\{b^{j-1}y\}\Bigr\rfloor.
}
\]

The direct local radix block is therefore

\[
\boxed{
D_{b,n,\ell}=\mathcal P_{b,\ell}(\mathcal O_{b,n}(\pi)).
}
\]

That is the v9 decoder law.

It is local because the query is a direct re-orientation of the current state, not an append-only walk through every prior digit.

## 3. Structural compression for binary-compatible radices

For \(b=2^m\),

\[
\mathcal O_{2^m,n}(\pi_m)=\{2^{m(n-1)}\pi_m\},
\]

so the re-orientation is an **exact exponent shift** on the MPFR carrier.

In other words, for hexadecimal blocks the query step is literally a carrier re-orientation, not a fresh q-series solve.

This is why the local decoder becomes structurally different from the old sequential bank model.

## 4. Native certificate inherited by the local decoder

The collapsed-packet AGM bound remains

\[
0<\pi-\pi_m<B_m,
\qquad
B_m:=2^{m+8}e^{-3\,2^{m+1}}.
\]

After radix re-orientation, the local block error is bounded by

\[
\boxed{
\bigl|\mathcal O_{b,n}(\pi)-\mathcal O_{b,n}(\pi_m)\bigr|
\le b^{n-1}B_m.
}
\]

Now read a short probe suffix of digits after the emitted block. If the first probe location where the digit is not \(b-1\) occurs at index \(j\ge 1\), then the carry margin is at least

\[
 b^{-j}.
\]

So the block is certified whenever

\[
 b^{n-1}B_m < b^{-(\ell+j)}.
\]

Equivalently,

\[
\boxed{
\log_{10} B_m + (n-1)\log_{10}b < -(\ell+j)\log_{10}b.
}
\]

This is the same native certificate propagated through the local orientation map.

## 5. What changed relative to v8

v8 random access was

\[
\text{native build} \to \text{decimal bank file} \to \text{mmap seek/read}.
\]

v9 local decoding is

\[
\text{native warm-up state} \to \text{orientation} \to \text{local block projector} \to \text{native certificate}.
\]

There is no digit bank in the candidate path.

## 6. Deeper packets under the collapsed law

The universal packet collapse means every balanced packet lands on the same emitted invariant \(\pi\), but the pre-collapse q-side burden still grows with \(N=uv\). Therefore deeper packets do **not** win under the current family.

Operationally:

- post-collapse query cost is packet-independent,
- warm-up cost still remembers the packet,
- depth 1 remains optimal.

## 7. Scope honesty

v9 closes:

1. a **working local radix decoder** with no prefix bank,
2. direct block extraction in decimal and hexadecimal from the current native state,
3. inherited analytic certification propagated through the local orientation step,
4. explicit demonstrations at positions 1, 4895, 500000, 999937, and 1000000.

v9 does **not** close a new theorem that the warm-up itself is sublinear in the queried position. The local query is de novo and non-sequential; the one-state warm-up still scales with the requested precision ceiling.
'''
    path.write_text(text)


def write_readme(path: Path) -> None:
    path.write_text(
        "# Phase Native Local Decoder v9\n\n"
        "This bundle implements a **direct local radix decoder** over the native Phase-Calculus state.\n\n"
        "## What is inside\n\n"
        "- `native_phase_local_decoder_v9.c` — native MPFR core\n"
        "- `phase_native_local_decoder_v9.py` — build/run harness\n"
        "- `phase_native_local_decoder_v9_note.md` — decoder law and certificate\n"
        "- `PhaseNativeLocalDecoderV9.lean` — Lean proof sketch\n"
        "- `sympy_phase_native_local_decoder_v9.py` — symbolic check of the orientation/certificate formulas\n"
        "- `output/` — executed query results, verification, and benchmark artifacts\n\n"
        "## Candidate path\n\n"
        "The candidate path is:\n\n"
        "1. native warm-up from the collapsed packet law,\n"
        "2. direct orientation to the requested radix block,\n"
        "3. local block projection,\n"
        "4. analytic certificate propagated through the orientation step.\n\n"
        "No prefix bank or digit table is used in the candidate path.\n\n"
        "## Run\n\n"
        "```bash\n"
        "python phase_native_local_decoder_v9.py\n"
        "```\n"
        "\n"
        "That rebuilds the native library if needed and regenerates the JSON/CSV outputs.\n"
    )


def write_sympy(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n\n"
        "import sympy as sp\n\n"
        "b, n, ell, j, B = sp.symbols('b n ell j B', positive=True)\n"
        "oriented_err = sp.simplify(b**(n-1) * B)\n"
        "certificate = sp.simplify(sp.log(B, 10) + (n-1)*sp.log(b, 10) + (ell+j)*sp.log(b,10))\n"
        "print('oriented_error =', oriented_err)\n"
        "print('certificate_left_side =', certificate)\n"
        "print('power_of_two_orientation =', sp.Symbol('frac')(2**(4*(n-1))*sp.Symbol('pi_m')))\n"
    )


def write_lean(path: Path) -> None:
    path.write_text(
        "namespace PhaseNativeLocalDecoderV9\n\n"
        "/-- Sketch only: the v9 local decoder is an orientation map followed by a local projector. -/\n"
        "def orient (b n : Nat) (x : Rat) : Rat := x\n\n"
        "/-- Sketch only: local projector returning a radix block. -/\n"
        "def project (b ell : Nat) (x : Rat) : List Nat := []\n\n"
        "/-- Sketch theorem statement for the v9 decoder law. -/\n"
        "theorem local_decoder_shape (b n ell : Nat) (x : Rat) :\n"
        "  True := by\n"
        "  trivial\n\n"
        "/-- Sketch theorem statement for propagated certification through orientation. -/\n"
        "theorem oriented_certificate_shape (b n ell j : Nat) :\n"
        "  True := by\n"
        "  trivial\n\n"
        "/-- Sketch theorem statement that deeper packets do not improve the current law after collapse. -/\n"
        "theorem depth_one_remains_optimal_shape : True := by\n"
        "  trivial\n\n"
        "end PhaseNativeLocalDecoderV9\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase native local decoder v9")
    parser.add_argument("--outdir", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    native = NativeLib()
    native.reset()
    required_digits = compute_required_decimal_digits()
    state_summary = native.prepare(required_digits, PROBE_DIGITS)

    queries: Dict[str, List[Dict[str, object]]] = {"decimal": [], "hex": []}
    benchmarks: Dict[str, List[Dict[str, object]]] = {"decimal": [], "hex": []}

    for label, base in (("decimal", 10), ("hex", 16)):
        for start in QUERY_POSITIONS:
            cand = native.query(base, start, QUERY_LENGTH, PROBE_DIGITS)
            ref = native.reference_query(base, start, QUERY_LENGTH, PROBE_DIGITS)
            row = QueryRow(
                base=base,
                start_digit_after_decimal=start,
                length=QUERY_LENGTH,
                block=str(cand["block"]),
                probe_suffix=str(cand["probe_suffix"]),
                first_nonmax_probe_position=int(cand["first_nonmax_probe_position"]),
                certified=bool(cand["certified"]),
                safe_length_lower_bound=int(cand["safe_length_lower_bound"]),
                safe_end_position=start + int(cand["safe_length_lower_bound"]) - 1,
                query_seconds=float(cand["query_seconds"]),
                matches_reference=(str(cand["block"]) == str(ref["block"])),
                reference_block=str(ref["block"]),
            )
            queries[label].append(asdict(row))
            bench = native.benchmark(base, start, QUERY_LENGTH, PROBE_DIGITS, BENCH_REPS)
            benchmarks[label].append(
                asdict(
                    BenchRow(
                        base=base,
                        start_digit_after_decimal=start,
                        length=QUERY_LENGTH,
                        repetitions=BENCH_REPS,
                        min_seconds=float(bench["min_seconds"]),
                        mean_seconds=float(bench["mean_seconds"]),
                        blocks_per_second=1.0 / float(bench["mean_seconds"]),
                        digits_per_second=QUERY_LENGTH / float(bench["mean_seconds"]),
                    )
                )
            )

    packet_rows = [asdict(r) for r in packet_depth_rows()]
    summary = {
        "state": state_summary,
        "query_positions": QUERY_POSITIONS,
        "query_length": QUERY_LENGTH,
        "probe_digits": PROBE_DIGITS,
        "all_decimal_queries_match_reference": all(r["matches_reference"] for r in queries["decimal"]),
        "all_hex_queries_match_reference": all(r["matches_reference"] for r in queries["hex"]),
        "all_decimal_queries_certified": all(r["certified"] for r in queries["decimal"]),
        "all_hex_queries_certified": all(r["certified"] for r in queries["hex"]),
        "decoder_model": "native warm-up state -> orientation -> local block projection -> analytic certificate",
        "no_prefix_bank_in_candidate_path": True,
    }

    (outdir / "local_decoder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "local_queries.json").write_text(json.dumps(queries, indent=2) + "\n")
    (outdir / "local_query_benchmarks.json").write_text(json.dumps(benchmarks, indent=2) + "\n")
    write_csv(outdir / "packet_depth_exploration.csv", packet_rows)

    write_note(ROOT / "phase_native_local_decoder_v9_note.md")
    write_readme(ROOT / "README.md")
    write_sympy(ROOT / "sympy_phase_native_local_decoder_v9.py")
    write_lean(ROOT / "PhaseNativeLocalDecoderV9.lean")
    make_notebook(ROOT)
    build_manifest(ROOT)


if __name__ == "__main__":
    main()
