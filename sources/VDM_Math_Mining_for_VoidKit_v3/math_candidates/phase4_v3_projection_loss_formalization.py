#!/usr/bin/env python3
from dataclasses import dataclass
import csv, json
from pathlib import Path

@dataclass(frozen=True)
class Collision:
    domain: str
    projection: str
    x: str
    y: str
    pix: str
    piy: str
    ex: str
    ey: str


def collision_blocks_custody(c: Collision) -> bool:
    return c.pix == c.piy and c.ex != c.ey


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader(open(root/'outputs/phase4_v3_projection_loss_witnesses.csv')))
    failures = []
    for row in rows:
        ok = row['proj_x'] == row['proj_y'] and row['next_x'] != row['next_y'] and row['custody_factorization'] == 'False'
        failures.append(not ok)
    summary = {
        'rows_checked': len(rows),
        'global_pass': not any(failures),
        'failed_rows': sum(failures),
    }
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
