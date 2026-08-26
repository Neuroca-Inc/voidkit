"""Small reusable statistical primitives recovered from VDM analysis scripts.
"""
from __future__ import annotations
import math
import numpy as np

def powerlaw_alpha_discrete(x: np.ndarray, xmin: int) -> tuple[float, int]:
    x = np.asarray(x)
    x = x[x >= xmin]
    n = len(x)
    if n == 0:
        return (float("nan"), 0)
    denom = np.sum(np.log(x / (xmin - 0.5)))
    if denom <= 0:
        return (float("nan"), n)
    alpha = 1 + n / denom
    return (float(alpha), int(n))


def hellinger(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.linalg.norm(np.sqrt(p + eps) - np.sqrt(q + eps)) / math.sqrt(2))


def detect_avalanches(series: np.ndarray, threshold: float):
    x = np.asarray(series, dtype=np.float64)
    above = x > threshold
    sizes, durs = [], []
    i = 0
    N = len(x)
    while i < N:
        if not above[i]:
            i += 1
            continue
        j = i
        size = 0.0
        while j < N and above[j]:
            size += x[j] - threshold
            j += 1
        sizes.append(size)
        durs.append(j - i)
        i = j
    return np.array(sizes), np.array(durs)


def powerlaw_alpha_continuous(x: np.ndarray, xmin: float):
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    x = x[x >= xmin]
    n = len(x)
    if n == 0:
        return float("nan"), 0
    alpha = 1 + n / np.sum(np.log(x / xmin))
    return float(alpha), int(n)


def size_duration_relation(sizes: np.ndarray, durs: np.ndarray) -> float:
    import pandas as pd

    df = pd.DataFrame({"size": sizes, "dur": durs})
    grp = df.groupby("dur")["size"].mean()
    d = grp.index.values.astype(float)
    s = grp.values.astype(float)
    mask = (d > 0) & (s > 0)
    if mask.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log10(d[mask]), np.log10(s[mask]), 1)
    return float(slope)


def shannon_entropy_from_values(vals):
    vals = np.array(vals, dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def participation_ratio_from_values(vals):
    vals = np.array(vals, dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    p = vals / vals.sum()
    return float(1.0 / np.sum(p*p))


def cosine_similarity_dict(d1, d2, norm1=None, norm2=None):
    if d1 is None or d2 is None or len(d1)==0 or len(d2)==0:
        return np.nan
    if norm1 is None:
        norm1 = math.sqrt(sum(v*v for v in d1.values()))
    if norm2 is None:
        norm2 = math.sqrt(sum(v*v for v in d2.values()))
    if norm1==0 or norm2==0:
        return np.nan
    dot = 0.0
    if len(d1) < len(d2):
        for k,v in d1.items():
            v2 = d2.get(k)
            if v2 is not None:
                dot += v*v2
    else:
        for k,v in d2.items():
            v1 = d1.get(k)
            if v1 is not None:
                dot += v*v1
    return float(dot/(norm1*norm2))

__all__=["hellinger","powerlaw_alpha_discrete","powerlaw_alpha_continuous","detect_avalanches","size_duration_relation","shannon_entropy_from_values","participation_ratio_from_values","cosine_similarity_dict"]
