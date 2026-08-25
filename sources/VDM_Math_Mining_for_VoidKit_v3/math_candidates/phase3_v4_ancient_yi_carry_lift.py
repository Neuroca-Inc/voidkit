#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def value_lsd(ds):
    return sum(d*(8**i) for i,d in enumerate(ds))

def digits_lsd(n, k=None):
    ds=[]
    x=n
    if x == 0:
        ds=[0]
    while x:
        ds.append(x % 8)
        x//=8
    if k is not None:
        while len(ds)<k:
            ds.append(0)
    return tuple(ds)

def display(ds):
    return ''.join(str(d) for d in ds)

def succ(ds):
    ds=list(ds)
    i=0
    while True:
        if i == len(ds):
            ds.append(1)
            return tuple(ds), True, i
        ds[i]+=1
        if ds[i] <= 7:
            return tuple(ds), False, i
        ds[i]=0
        i+=1

def run(max_k=5):
    audits=[]
    for k in range(1, max_k+1):
        failures=[]
        for n in range(8**k):
            ds=digits_lsd(n,k)
            nxt,lift,ci=succ(ds)
            if value_lsd(nxt) != n+1:
                failures.append({"n":n,"state":display(ds),"next":display(nxt),"next_value":value_lsd(nxt)})
                break
        all7=tuple([7]*k)
        nxt,lift,ci=succ(all7)
        audits.append({
            "k":k,
            "states":8**k,
            "successor_pass": not failures,
            "completion":display(all7),
            "completion_next":display(nxt),
            "completion_lift_pass": lift and nxt == tuple([0]*k+[1]),
        })
    return {
        "global_pass": all(a["successor_pass"] and a["completion_lift_pass"] for a in audits),
        "audits": audits,
    }

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args=ap.parse_args()
    result=run(args.max_k)
    text=json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
