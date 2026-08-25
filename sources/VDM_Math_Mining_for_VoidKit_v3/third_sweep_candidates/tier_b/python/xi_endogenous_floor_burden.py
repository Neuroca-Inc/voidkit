#!/usr/bin/env python3
"""Endogenous floor_den burden experiments for the xi balanced-window engine.

This script does not install an x/y geometry, prime detector, or graph manager.
It treats floor_den (Delta) as a carried burden and updates it only at host-lift
boundaries from already-carried xi fields such as r_den=uv and A.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
import csv, json, math, hashlib

W = 64
DELTA0 = 4096


def fibs(n: int) -> List[int]:
    F = [0, 1]
    for _ in range(2, n + 2):
        F.append(F[-1] + F[-2])
    return F

F = fibs(128)


def uv_after(B: int, d: int) -> Tuple[int, int, int]:
    """q after d balanced steps from seed (1,B), product uv."""
    if d == 0:
        return 1, B, B
    u = F[d] * B + F[d - 1]
    v = F[d + 1] * B + F[d]
    return u, v, u * v


def depth_for(B: int, Delta: int, W: int = W) -> Tuple[int, int, int, int, bool]:
    """Smallest d in [0,W-1] with uv_d(B)>=Delta, else clamp at W-1."""
    for d in range(W):
        u, v, uv = uv_after(B, d)
        if uv >= Delta:
            return d, u, v, uv, False
    d = W - 1
    u, v, uv = uv_after(B, d)
    return d, u, v, uv, True


def update_delta(rule: str, A: int, Delta: int, d: int, u: int, v: int, uv: int, clamped: bool) -> int:
    """State-only floor update at the host-lift boundary.

    A is the completed block's host class. A+1 is the next host class.
    """
    if rule == "fixed":
        return Delta
    if rule == "frontier_rden":
        return uv
    if rule == "frontier_rden_plus_host":
        return uv + (A + 1)
    if rule == "integrated_rden":
        return Delta + uv
    if rule == "child_frontier":
        return v * (u + v)
    if rule == "germ_cden":
        return 2 * uv
    if rule == "quadratic_host":
        return (A + 1) * (A + 2)
    raise ValueError(rule)


@dataclass
class BlockRow:
    A: int
    B: int
    Delta: int
    depth: int
    u: int
    v: int
    uv: int
    clamped: int
    saturated: int
    next_Delta: int
    hold_slots: int
    word: str


def simulate(rule: str, blocks: int, Delta0: int = DELTA0) -> List[BlockRow]:
    Delta = Delta0
    rows: List[BlockRow] = []
    for A in range(blocks):
        B = max(1, A)
        d, u, v, uv, clamped = depth_for(B, Delta, W)
        saturated = not clamped
        next_Delta = update_delta(rule, A, Delta, d, u, v, uv, clamped)
        hold_slots = (W - 1 - d) if saturated else 0
        word = ("R" * d) + (("S" * hold_slots) if saturated else "") + "T"
        rows.append(BlockRow(A, B, Delta, d, u, v, uv, int(clamped), int(saturated), next_Delta, hold_slots, word))
        Delta = next_Delta
    return rows


def write_csv(path: Path, rows: Iterable[BlockRow]) -> None:
    rows = list(rows)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def summarize(rule: str, rows: List[BlockRow]) -> dict:
    depths = [r.depth for r in rows]
    clamps = sum(r.clamped for r in rows)
    saturated = sum(r.saturated for r in rows)
    collapsed = sum(1 for r in rows if r.depth == 0)
    positive_depth = sum(1 for r in rows if r.depth > 0)
    # Detect eventual constant suffix depth, capped by 2000 rows.
    last_depth = depths[-1]
    suffix = 0
    for d in reversed(depths):
        if d == last_depth:
            suffix += 1
        else:
            break
    # first A where depth becomes permanently equal to last_depth in the observed run
    stable_from = None
    for i in range(len(depths)):
        if all(x == last_depth for x in depths[i:]):
            stable_from = i
            break
    return {
        "rule": rule,
        "blocks": len(rows),
        "initial_Delta": DELTA0,
        "final_Delta": str(rows[-1].next_Delta),
        "first_20_depths": depths[:20],
        "last_20_depths": depths[-20:],
        "depth_counts": dict(sorted(Counter(depths).items())),
        "saturated_blocks": saturated,
        "clamped_blocks": clamps,
        "collapsed_depth0_blocks": collapsed,
        "positive_depth_blocks": positive_depth,
        "last_depth": last_depth,
        "constant_suffix_len": suffix,
        "observed_stable_from_A": stable_from,
        "last_A": rows[-1].A,
        "last_uv": str(rows[-1].uv),
        "last_hold_slots": rows[-1].hold_slots,
    }


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    res.mkdir(exist_ok=True)
    rules = ["fixed", "frontier_rden", "frontier_rden_plus_host", "integrated_rden", "child_frontier", "germ_cden", "quadratic_host"]
    blocks = 100000
    summaries = []
    for rule in rules:
        rows = simulate(rule, blocks)
        # keep first 5000 + last 1000 rows for inspectable CSV; full simulated in summary
        sample = rows[:5000] + rows[-1000:]
        write_csv(res / f"{rule}_blocks_sample.csv", sample)
        summaries.append(summarize(rule, rows))
    with (res / "endogenous_floor_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)
    with (res / "endogenous_floor_summary.txt").open("w") as f:
        for s in summaries:
            f.write(f"RULE {s['rule']}\n")
            for k,v in s.items():
                if k != 'rule':
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

    # Focused exact rows for the conservative frontier law.
    frontier = simulate("frontier_rden", 64)
    write_csv(res / "frontier_rden_first_64_blocks.csv", frontier)

    # SHA manifest after initial outputs; filled later by caller may include figures/notebook.
    paths = sorted([p for p in root.rglob('*') if p.is_file() and 'SHA256SUMS' not in p.name])
    with (res / "SHA256SUMS.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["path","sha256"])
        for p in paths:
            w.writerow([str(p.relative_to(root)), sha256_file(p)])

if __name__ == "__main__":
    main()
