#!/usr/bin/env python3
"""Cloud hammer for CF10 F1.A production sweeps.

Runs the existing CPU pseudospectral probe script in restartable chunks.
No CUDA path is used. The output gate is the integrated active-front ratio,
the beta tail, the finite boundary/front band, and the xi quotient residual.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "cf10-f1a-hammer"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("cf10-f1a-hammer-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "matplotlib")
    .add_local_dir(".", "/work")
)


def tier_cases(tier: str) -> list[dict]:
    if tier == "n192":
        base = {"N": 192, "dt": 0.00035, "chunk_steps": 48, "chunks": 32, "sample_every": 48}
        return [
            {**base, "nu": 0.0050, "forcing_amp": 0.10, "seed": 1},
            {**base, "nu": 0.0050, "forcing_amp": 0.06, "seed": 1},
            {**base, "nu": 0.0025, "forcing_amp": 0.10, "seed": 1},
            {**base, "nu": 0.0025, "forcing_amp": 0.06, "seed": 1},
            {**base, "nu": 0.0050, "forcing_amp": 0.10, "seed": 2},
            {**base, "nu": 0.0050, "forcing_amp": 0.06, "seed": 2},
            {**base, "nu": 0.0025, "forcing_amp": 0.10, "seed": 2},
            {**base, "nu": 0.0025, "forcing_amp": 0.06, "seed": 2},
        ]
    if tier == "n256":
        base = {"N": 256, "dt": 0.00022, "chunk_steps": 32, "chunks": 48, "sample_every": 32}
        return [
            {**base, "nu": 0.0040, "forcing_amp": 0.08, "seed": 1},
            {**base, "nu": 0.0020, "forcing_amp": 0.08, "seed": 1},
            {**base, "nu": 0.0040, "forcing_amp": 0.05, "seed": 2},
            {**base, "nu": 0.0020, "forcing_amp": 0.05, "seed": 2},
        ]
    raise ValueError(f"unknown tier: {tier}")


@app.function(image=image, volumes={"/results": volume}, timeout=24 * 60 * 60, cpu=8, memory=32768)
def run_case(case: dict) -> dict:
    import subprocess
    import sys
    from pathlib import Path
    import json

    label = f"N{case['N']}_nu{case['nu']}_f{case['forcing_amp']}_seed{case['seed']}".replace(".", "p")
    outdir = Path("/results") / label
    outdir.mkdir(parents=True, exist_ok=True)

    state_in = None
    state_out = outdir / "state_0000.npz"
    summary_out = outdir / "summary_0000.json"

    for chunk in range(case["chunks"]):
        state_out = outdir / f"state_{chunk+1:04d}.npz"
        summary_out = outdir / f"summary_{chunk+1:04d}.json"
        cmd = [
            sys.executable,
            "/work/proof_reproduction/ns3d_fast_f1a.py",
            "chunk",
            "--N", str(case["N"]),
            "--dt", str(case["dt"]),
            "--nu", str(case["nu"]),
            "--seed", str(case["seed"]),
            "--forcing-amp", str(case["forcing_amp"]),
            "--chunk-steps", str(case["chunk_steps"]),
            "--sample-every", str(case["sample_every"]),
            "--state-out", str(state_out),
            "--summary-out", str(summary_out),
            "--sample-final",
        ]
        if state_in is not None:
            cmd.extend(["--state-in", str(state_in)])

        proc = subprocess.run(cmd, cwd="/work", text=True, capture_output=True)
        if proc.returncode != 0:
            (outdir / f"stderr_{chunk+1:04d}.txt").write_text(proc.stderr)
            raise RuntimeError(proc.stderr)

        volume.commit()
        state_in = state_out

        summary = json.loads(summary_out.read_text())
        metrics = summary.get("metrics", {})
        print(json.dumps({
            "label": label,
            "progress": f"{(chunk+1)*case['chunk_steps']}/{case['chunks']*case['chunk_steps']}",
            "time": metrics.get("final_time"),
            "beta": metrics.get("final_beta"),
            "Ppos_over_D": metrics.get("max_positive_tail_pressure_late"),
            "xi_residual": metrics.get("max_quotient_residual_late"),
            "div_l2": metrics.get("max_divergence_l2"),
        }))

    final_summary = json.loads(summary_out.read_text())
    return {"label": label, "case": case, "final_summary": final_summary}


@app.local_entrypoint()
def main(tier: str = "n192"):
    cases = tier_cases(tier)
    print(json.dumps({"event": "stage_start", "tier": tier, "case_count": len(cases)}))
    results = []
    for result in run_case.map(cases):
        print(json.dumps({"event": "case_done", "label": result["label"], "metrics": result["final_summary"].get("metrics", {})}))
        results.append(result)
    Path(f"modal_{tier}_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"event": "stage_done", "tier": tier, "case_count": len(results)}))
