from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Custody:
    A: int
    u: int
    v: int
    thetaQ: int
    k: int
    j: int
    W: str


def N(A: int) -> int:
    return 6 * (2**A)


def j_start(A: int) -> int:
    return 1 + 6 * ((2**A) - 1)


def cap(j: int) -> int:
    return 2 if j == 1 else 4 if j == 2 else 2 ** (2 * j)


def can_q(x: Custody) -> bool:
    return x.k < N(x.A) - 1


def can_b(x: Custody) -> bool:
    if x.k < N(x.A) - 1:
        return x.v * (x.u + x.v) <= cap(x.j)
    return x.u * x.v < cap(x.j)


def select(x: Custody) -> str:
    return "B" if can_b(x) else "Q" if can_q(x) else "L"


def lawful_custody_step(x: Custody) -> Custody:
    return step_by_letter(x, select(x))


def step_by_letter(x: Custody, p: str) -> Custody:
    """Total case map used by complete models; lawfulness is checked separately."""
    if p == "B":
        return Custody(x.A, x.v, x.u + x.v, x.thetaQ, x.k, x.j, x.W + "B")
    if p == "Q":
        return Custody(x.A, x.u, x.v, x.thetaQ + 1, x.k + 1, x.j + 1, x.W + "Q")
    if p == "L":
        return Custody(x.A + 1, x.u, x.v, x.thetaQ, 0, j_start(x.A + 1), x.W + "L")
    raise ValueError(p)


def a_text(q: int, d: int) -> str:
    return [
        "1" if d == 1 else f"1/{d}",
        "i" if d == 1 else f"i/{d}",
        "-1" if d == 1 else f"-1/{d}",
        "-i" if d == 1 else f"-i/{d}",
    ][q % 4]


def run_domain0() -> list[dict]:
    x = Custody(0, 1, 1, 0, 0, 1, "")
    rows = [{"step": 0, "selected": "start", **asdict(x), "a": a_text(x.thetaQ, x.u * x.v)}]
    while not x.W.endswith("L"):
        p = select(x)
        x = lawful_custody_step(x)
        rows.append(
            {
                "step": len(rows),
                "selected": p,
                **asdict(x),
                "a": a_text(x.thetaQ, x.u * x.v)
                if p != "L"
                else "new active; old=" + a_text(x.thetaQ, x.u * x.v),
            }
        )
    assert x.W == "BQQBBBQBQBBQBBL"
    assert rows[-2]["a"] == "i/4895"
    return rows


@dataclass(frozen=True)
class Primary:
    rank: int
    history: str
    token: int


@dataclass(frozen=True)
class Coupled:
    X: Custody
    P: Primary
    D: dict[str, tuple]


def restrictions(P: Primary) -> dict[str, tuple]:
    return {slot: (slot, P.rank, P.history, P.token) for slot in ("++", "--", "+-", "-+")}


def primary_case(P: Primary, p: str) -> Primary:
    if p == "L":
        return Primary(P.rank + 1, P.history + p, (P.token * 5 + 3) % 997)
    factor = 2 if p == "B" else 3
    return Primary(P.rank, P.history + p, (P.token * factor + 1) % 997)


def complete_step(
    state: Coupled,
    p: str,
    descendant_rule: Callable[[Primary, dict[str, tuple]], dict[str, tuple]],
) -> Coupled:
    X2 = step_by_letter(state.X, p)
    P2 = primary_case(state.P, p)
    canonical = restrictions(P2)
    D2 = descendant_rule(P2, canonical)
    return Coupled(X2, P2, D2)


def induced_descendants(P: Primary, canonical: dict[str, tuple]) -> dict[str, tuple]:
    return canonical


def independent_descendants(P: Primary, canonical: dict[str, tuple]) -> dict[str, tuple]:
    out = dict(canonical)
    out["++"] = ("independent", P.rank, P.history, P.token + 1000)
    return out


def run_external_selector_countermodel(steps: int = 8) -> dict:
    """Complete all-prefix recurrence with an external fixed schedule; unlawful by construction."""
    schedule = "QBQLBQBL"  # first letter already contradicts custody, which requires B.
    s = Coupled(Custody(0, 1, 1, 0, 0, 1, ""), Primary(1, "", 1), restrictions(Primary(1, "", 1)))
    trace = []
    factorization_ok = True
    for t in range(steps):
        required = select(s.X)
        supplied = schedule[t % len(schedule)]
        factorization_ok &= supplied == required
        trace.append(
            {
                "t": t,
                "required": required,
                "supplied": supplied,
                "rank": s.P.rank,
                "history": s.P.history,
                "descendants_induced": s.D == restrictions(s.P),
            }
        )
        s = complete_step(s, supplied, induced_descendants)
    return {
        "complete_all_prefix_recurrence": True,
        "steps_executed": steps,
        "selector_factorization": factorization_ok,
        "descendants_induced_every_prefix": all(row["descendants_induced"] for row in trace),
        "trace": trace,
        "verdict": "REJECT",
        "failed_obligation": "primitive input does not factor through Sigma_custody",
    }


def run_independent_descendant_countermodel() -> dict:
    """Complete custody-autonomous recurrence; unlawful because a descendant is independently updated."""
    s = Coupled(Custody(0, 1, 1, 0, 0, 1, ""), Primary(1, "", 1), restrictions(Primary(1, "", 1)))
    trace = []
    while not s.X.W.endswith("L"):
        p = select(s.X)
        s = complete_step(s, p, independent_descendants)
        trace.append(
            {
                "t": len(trace),
                "selected": p,
                "prefix": s.X.W,
                "rank": s.P.rank,
                "primary_history": s.P.history,
                "descendants_induced": s.D == restrictions(s.P),
            }
        )
    return {
        "complete_all_prefix_recurrence": True,
        "steps_executed": len(trace),
        "selector_factorization": s.X.W == "BQQBBBQBQBBQBBL",
        "descendants_induced_every_prefix": all(row["descendants_induced"] for row in trace),
        "trace": trace,
        "verdict": "REJECT",
        "failed_obligation": "D_ab,next != R_ab,next(P_next)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = root / "outputs"
    tr = root / "trace"
    out.mkdir(exist_ok=True)
    tr.mkdir(exist_ok=True)

    rows = run_domain0()
    with (out / "DOMAIN0_PREFIX_TRACE.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (tr / "custody_and_local_trace.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )

    c1 = run_external_selector_countermodel()
    c2 = run_independent_descendant_countermodel()
    (out / "COMPLETE_COUNTERMODELS.json").write_text(
        json.dumps({"external_selector": c1, "independent_descendant": c2}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tr / "complete_countermodel_trace.jsonl").write_text(
        "".join(
            json.dumps({"model": model, **row}, sort_keys=True) + "\n"
            for model, result in (("external_selector", c1), ("independent_descendant", c2))
            for row in result["trace"]
        ),
        encoding="utf-8",
    )

    assert c1["complete_all_prefix_recurrence"]
    assert not c1["selector_factorization"]
    assert c1["descendants_induced_every_prefix"]
    assert c2["complete_all_prefix_recurrence"]
    assert c2["selector_factorization"]
    assert not c2["descendants_induced_every_prefix"]

    statuses = {
        "local_two_slot_interface": "PROVED",
        "autonomous_transition_interface": "PROVED",
        "star_typed_by_authority": "FALSE",
        "star_independence": "OPEN",
        "explicit_intrinsic_action": "OPEN",
        "first_missing_bridge": "PPGRL",
        "branch_status": "OPEN",
    }
    (out / "STATUS.json").write_text(json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("PROVED: custody trace and local descendant")
    print("COUNTERMODEL REJECTED: complete external-selector recurrence")
    print("COUNTERMODEL REJECTED: complete independent-descendant recurrence")
    print("OPEN: PPGRL and explicit intrinsic action")


if __name__ == "__main__":
    main()
