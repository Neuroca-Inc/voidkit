#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mmap
from pathlib import Path


def extract_decimal_block(bank_path: Path, start: int, length: int) -> str:
    if start < 1 or length < 0:
        raise ValueError("start must be >= 1 and length >= 0")
    with bank_path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            offset = 2 + (start - 1)
            return mm[offset: offset + length].decode()
        finally:
            mm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexed random access over a certified native pi bank")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--safe", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    args = parser.parse_args()

    block = extract_decimal_block(args.bank, args.start, args.length)
    payload = {
        "bank": str(args.bank),
        "start_digit_after_decimal": args.start,
        "length": args.length,
        "end_digit_after_decimal": args.start + args.length - 1,
        "certified_under_safe_lower_bound": (args.start + args.length - 1) <= args.safe,
        "block": block,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
