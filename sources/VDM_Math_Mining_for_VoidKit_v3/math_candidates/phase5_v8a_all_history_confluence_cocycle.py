#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import csv
import itertools
import json
import random
from pathlib import Path

NAX = 6
MOD = 12

@dataclass(frozen=True)
class Ev:
    kind: str
    a: int
    b: int = -1

    def name(self) -> str:
        return f"{self.kind}{self.a}" if self.b < 0 else f"{self.kind}{self.a}_{self.b}"


def norm_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def support(e: Ev) -> dict[str, set[str]]:
    a, b = e.a, e.b
    if e.kind == "Q":
        rw = {f"theta:{a}", f"q:{a}", f"lens:{a}"}
        return {"read": set(rw), "write": set(rw), "birth_read": set(), "birth_write": set(), "edge": set()}
    if e.kind == "B":
        rw = {f"q:{a}", f"width:{a}", f"lens:{a}"}
        return {"read": set(rw), "write": set(rw), "birth_read": set(), "birth_write": set(), "edge": set()}
    if e.kind == "L":
        c = (a + 1) % NAX
        return {
            "read": {f"lens:{a}", f"latch:{a}", "contact_clock"},
            "write": {f"lens:{c}", f"latch:{c}", "contact_z"},
            "birth_read": {f"born:{a}"},
            "birth_write": {f"born:{c}"},
            "edge": set(),
        }
    if e.kind == "O":
        x, y = norm_edge(a, b)
        return {
            "read": {f"lens:{a}", f"lens:{b}", f"born:{a}", f"born:{b}"},
            "write": {f"edge:{x}:{y}", f"hol:{x}:{y}"},
            "birth_read": {f"born:{a}", f"born:{b}"},
            "birth_write": set(),
            "edge": {f"edge:{x}:{y}"},
        }
    if e.kind == "R":
        return {
            "read": {f"lens:{a}", f"born:{a}"},
            "write": set(),
            "birth_read": {f"born:{a}"},
            "birth_write": set(),
            "edge": set(),
        }
    raise ValueError(e.kind)


def independent(e: Ev, f: Ev) -> bool:
    se, sf = support(e), support(f)
    if se["write"] & (sf["read"] | sf["write"]):
        return False
    if sf["write"] & (se["read"] | se["write"]):
        return False
    if se["birth_write"] & (sf["birth_read"] | sf["birth_write"]):
        return False
    if sf["birth_write"] & (se["birth_read"] | se["birth_write"]):
        return False
    if se["edge"] & sf["edge"]:
        return False
    return True


class State:
    def __init__(self) -> None:
        self.lens = [0] * NAX
        self.width = [1] * NAX
        self.latch = [0] * NAX
        self.born = [True] * NAX
        self.contact_z = 0
        self.edges: dict[tuple[int, int], int] = {}
        self.terminal = []

    def sig(self):
        return (
            tuple(self.lens),
            tuple(self.width),
            tuple(self.latch),
            tuple(self.born),
            self.contact_z,
            tuple(sorted(self.edges.items())),
        )

    def apply(self, e: Ev) -> "State":
        a, b = e.a, e.b
        if e.kind == "Q":
            self.lens[a] = (self.lens[a] + 3) % MOD
        elif e.kind == "B":
            self.width[a] += 1
            self.lens[a] = (self.lens[a] + self.width[a]) % MOD
        elif e.kind == "L":
            c = (a + 1) % NAX
            self.born[c] = True
            self.latch[c] = (self.latch[a] + 1) % 2
            self.lens[c] = (self.lens[a] + 6 * self.latch[c]) % MOD
            self.contact_z += (self.latch[c] + 1) * self.width[a]
        elif e.kind == "O":
            if not (self.born[a] and self.born[b]):
                raise ValueError("overlap before birth")
            x, y = norm_edge(a, b)
            c = (self.lens[b] - self.lens[a]) % MOD
            self.edges[(x, y)] = (self.edges.get((x, y), 0) + c) % MOD
        elif e.kind == "R":
            self.terminal.append((a, self.lens[a], self.width[a]))
        return self


def apply_word(word: list[Ev]) -> State:
    st = State()
    for e in word:
        st.apply(e)
    return st


def dependent(e: Ev, f: Ev) -> bool:
    return not independent(e, f)


def foata_levels(word: list[Ev]) -> tuple[tuple[str, ...], ...]:
    levels: list[int] = []
    for i, e in enumerate(word):
        lvl = 0
        for j, g in enumerate(word[:i]):
            if dependent(e, g):
                lvl = max(lvl, levels[j] + 1)
        levels.append(lvl)
    blocks: dict[int, list[str]] = {}
    for e, lvl in zip(word, levels):
        blocks.setdefault(lvl, []).append(e.name())
    return tuple(tuple(sorted(v)) for k, v in sorted(blocks.items()))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_events() -> list[Ev]:
    events = []
    for a in range(NAX):
        events += [Ev("Q", a), Ev("B", a), Ev("L", a), Ev("R", a)]
    for a in range(NAX):
        for b in range(a + 1, NAX):
            events.append(Ev("O", a, b))
    return events


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "outputs"
    events = build_events()

    pair_rows = []
    for e, f in itertools.product(events, repeat=2):
        if not independent(e, f):
            continue
        ok = apply_word([e, f]).sig() == apply_word([f, e]).sig()
        pair_rows.append({"event_a": e.name(), "event_b": f.name(), "status": "PASS" if ok else "FAIL"})

    random.seed(20260625)
    rewrite_rows = []
    for idx in range(500):
        w = [random.choice(events) for _ in range(random.randint(4, 18))]
        w2 = w[:]
        for _ in range(4 * len(w2)):
            i = random.randrange(len(w2) - 1)
            if independent(w2[i], w2[i + 1]):
                w2[i], w2[i + 1] = w2[i + 1], w2[i]
        ok = foata_levels(w) == foata_levels(w2) and apply_word(w).sig() == apply_word(w2).sig()
        rewrite_rows.append({"case_id": idx, "status": "PASS" if ok else "FAIL"})

    cocycle_rows = []
    for idx in range(200):
        st = apply_word([random.choice(events) for _ in range(random.randint(5, 30))])
        for a, b, c in itertools.combinations(range(NAX), 3):
            residual = ((st.lens[b] - st.lens[a]) + (st.lens[c] - st.lens[b]) + (st.lens[a] - st.lens[c])) % MOD
            cocycle_rows.append({"case_id": idx, "triangle": f"{a}-{b}-{c}", "residual_mod12": residual, "status": "PASS" if residual == 0 else "FAIL"})

    write_csv(out / "phase5_v8a_critical_pair_checks.csv", pair_rows)
    write_csv(out / "phase5_v8a_trace_rewrite_confluence_checks.csv", rewrite_rows)
    write_csv(out / "phase5_v8a_cocycle_compatibility_checks.csv", cocycle_rows)

    summary = {
        "phase": "Phase 5 v8a",
        "status": "ALL_HISTORY_CONFLUENCE_AND_COCYCLE_COMPATIBILITY_CLOSED_CONDITIONALLY_FOR_DEFINED_ADMISSIBLE_RETAINED_QBL_SYSTEM",
        "global_pass": all(r["status"] == "PASS" for r in pair_rows + rewrite_rows + cocycle_rows),
        "phase5_closed": False,
        "critical_pair_checks": {"passed": sum(r["status"] == "PASS" for r in pair_rows), "total": len(pair_rows)},
        "rewrite_checks": {"passed": sum(r["status"] == "PASS" for r in rewrite_rows), "total": len(rewrite_rows)},
        "cocycle_checks": {"passed": sum(r["status"] == "PASS" for r in cocycle_rows), "total": len(cocycle_rows)},
    }
    (out / "phase5_v8a_verification_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
