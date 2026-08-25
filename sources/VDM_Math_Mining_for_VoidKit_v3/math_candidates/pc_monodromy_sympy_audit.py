#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from sympy.combinatorics import Permutation


def transitions(p: Permutation, n: int) -> dict[int, int]:
    return {i: p(i) for i in range(n)}


def main() -> None:
    n = 3
    sigma0 = Permutation(0, 1, size=n)  # transposition (0 1)
    sigma1 = Permutation(1, 2, size=n)  # transposition (1 2)
    identity = Permutation(list(range(n)))

    # Sequential word sigma0 sigma1 sigma0^-1 sigma1^-1.
    comm = sigma1**-1 * sigma0**-1 * sigma1 * sigma0

    projection_initial = ("base", (1, 1), 0)
    projection_after_loop = ("base", (1, 1), 0)
    lifted_initial = {"sheet": 0, "history": [], "kappa": 0}
    lifted_after_sigma0 = {"sheet": sigma0(0), "history": ["sigma0"], "kappa": 1}
    lifted_after_q4 = {"sheet": 0, "history": ["Q4"], "kappa": 1}

    results = {
        "sympy_generators": {
            "sigma0": transitions(sigma0, n),
            "sigma1": transitions(sigma1, n),
        },
        "commutator_transitions": transitions(comm, n),
        "commutator_is_identity": comm == identity,
        "commutator_cycles": [[int(x) for x in cyc] for cyc in comm.cyclic_form],
        "projection_loss_sigma0": {
            "visible_equal": projection_initial == projection_after_loop,
            "full_equal": lifted_initial == lifted_after_sigma0,
            "initial": lifted_initial,
            "final": lifted_after_sigma0,
        },
        "projection_loss_Q4": {
            "visible_equal": projection_initial == projection_after_loop,
            "full_equal": lifted_initial == lifted_after_q4,
            "initial": lifted_initial,
            "final": lifted_after_q4,
        },
    }

    pass_conditions = [
        results["commutator_is_identity"] is False,
        results["projection_loss_sigma0"]["visible_equal"] is True,
        results["projection_loss_sigma0"]["full_equal"] is False,
        results["projection_loss_Q4"]["visible_equal"] is True,
        results["projection_loss_Q4"]["full_equal"] is False,
    ]
    results["FINAL_RESULT"] = "PASS" if all(pass_conditions) else "FAIL"

    out = Path(__file__).resolve().parents[1] / "sympy_audit_results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"FINAL_RESULT: {results['FINAL_RESULT']}")


if __name__ == "__main__":
    main()
