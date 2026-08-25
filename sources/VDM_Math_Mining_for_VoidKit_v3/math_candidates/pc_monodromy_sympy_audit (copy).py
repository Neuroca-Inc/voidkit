#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from sympy.combinatorics import Permutation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc_monodromy_certifier.application.certifier import PresentationMonodromyCertifier
from pc_monodromy_certifier.domain.models import BranchGenerator, FinitePresentation
from pc_monodromy_certifier.domain.permutation import Permutation as PCPermutation


def check(name: str, condition: bool) -> None:
    print(f"{name}: {'PASS' if condition else 'FAIL'}")
    if not condition:
        raise AssertionError(name)


def main() -> int:
    gens = {
        "a": BranchGenerator("a", PCPermutation((1, 2, 0, 3, 4)), 1),
        "b": BranchGenerator("b", PCPermutation((0, 1, 3, 4, 2)), 0),
    }
    c = PresentationMonodromyCertifier(5, FinitePresentation(("a", "b")), gens)
    state = c.default_state()
    comm = c.commutator("a", "b")
    witness = c.projection_loss(state, comm.word)
    check("M1 finite word closure", c.transition(("a", "b")).end_sheet in range(5))
    check("M2 sheet commutation", c.transition(("a", "b"), 0).end_sheet == c.evaluate_word(("a", "b"))[0].apply(0))
    check("M3 history retention", c.apply_word(state, ("a", "b")).history == ("a", "b"))
    check("M4 projection loss", witness.same_visible_projection and witness.lifted_state_changed)
    check("M5 commutator nonidentity", not comm.is_identity)
    pa = Permutation([1, 2, 0, 3, 4])
    pb = Permutation([0, 1, 3, 4, 2])
    # independent SymPy commutator nonidentity
    check("SymPy commutator nonidentity", pa * pb * (~pa) * (~pb) != Permutation(4))
    print("FINAL_RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
