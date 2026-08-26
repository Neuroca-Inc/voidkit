"""Information, complexity, and simple time-series metrics extracted from the Aura analysis dashboard.

`transfer_entropy` returns natural-log units because that is what the source implementation computes.
Use `transfer_entropy_nats` as the explicit alias.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import numpy as np
from scipy import signal

def gini_coeff(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size == 0: return float("nan")
    mn = x.min()
    if mn < 0: x = x - mn
    s = x.sum()
    if s <= 0: return 0.0
    x = np.sort(x); n = x.size
    i = np.arange(1, n + 1, dtype=float)
    return float(1.0 - (2.0 * np.sum((n + 1 - i) * x)) / (n * s))


def welch_slope(x: np.ndarray, fs: float = 1.0) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size < 64: return float("nan")
    nperseg = int(min(1024, max(64, 2 ** int(np.floor(np.log2(x.size))))))
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    mask = f > 0; f = f[mask]; Pxx = Pxx[mask]
    if f.size < 8: return float("nan")
    b, _ = np.polyfit(np.log10(f), np.log10(Pxx + 1e-24), 1)
    return -float(b)


def perm_entropy(x: np.ndarray, m: int = 4, tau: int = 1) -> float:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size - (m - 1) * tau
    if n <= 1: return float("nan")
    idx = np.arange(m, dtype=np.int64) * tau
    X = x[np.arange(n)[:, None] + idx[None, :]]
    patterns = np.argsort(X, axis=1).astype(np.int64)
    powers = (m ** np.arange(m, dtype=np.int64))
    ids = (patterns * powers[None, :]).sum(axis=1)
    counts = np.bincount(ids, minlength=m ** m).astype(float)
    counts = counts[counts > 0]; p = counts / counts.sum()
    H = float(-(p * np.log(p)).sum())
    Hmax = math.log(math.factorial(m))
    return H / Hmax if Hmax > 0 else float("nan")


def lz_complexity(x: np.ndarray, threshold: Optional[float] = None) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 4: return float("nan")
    thr = threshold if threshold is not None else float(np.median(x))
    s = "".join("1" if v >= thr else "0" for v in x)
    n = len(s); i = 0; c = 1; l = 1; k = 1; kmax = 1
    while True:
        if s[i + k - 1] in s[:i + kmax]:
            k += 1
            if i + k > n:
                c += 1; break
        else:
            kmax = max(k, kmax); i += kmax; k = 1
            if i + 1 > n:
                break
            else:
                kmax = 1
    b = n / math.log2(n + 1) if n > 1 else 1
    return c / b


def transfer_entropy(x: np.ndarray, y: np.ndarray, bins: int = 8, lag: int = 1) -> float:
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    n = min(x.size, y.size)
    if n <= lag + 10: return float("nan")
    x = x[:n]; y = y[:n]
    # quantile bin
    def qbin(v):
        qs = np.nanquantile(v, np.linspace(0, 1, bins + 1))
        qs = np.unique(qs)
        if qs.size <= 2: return np.zeros(v.size, dtype=np.int64)
        return np.digitize(v, qs[1:-1], right=False).astype(np.int64)
    xd = qbin(x); yd = qbin(y)
    K = bins
    y1 = yd[lag:]; y0 = yd[:-lag]; x0 = xd[:-lag]
    idx_y0x0 = y0 * K + x0
    idx_full = y1 * K * K + idx_y0x0
    C_full = np.bincount(idx_full, minlength=K**3).astype(float)
    C_y0x0 = np.bincount(idx_y0x0, minlength=K**2).astype(float)
    C_y1y0 = np.bincount(y1 * K + y0, minlength=K**2).astype(float)
    C_y0   = np.bincount(y0, minlength=K).astype(float)
    N = float(y1.size)
    P = C_full / N; Py0x0 = C_y0x0 / N; Py1y0 = C_y1y0 / N; Py0 = C_y0 / N
    te = 0.0
    nz = np.nonzero(P)[0]
    for idx in nz:
        p = P[idx]
        y1v = idx // (K * K); rem = idx % (K * K); y0v = rem // K; x0v = rem % K
        p_yx = Py0x0[y0v * K + x0v]; p_yy = Py1y0[y1v * K + y0v]; p_y = Py0[y0v]
        if p_yx <= 0 or p_y <= 0: continue
        te += p * np.log((p / p_yx) / max(p_yy / p_y, 1e-12))
    return float(te)


def tc_o_gaussian(X: np.ndarray) -> Tuple[float, float, float]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] < 2:
        return float("nan"), float("nan"), float("nan")
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cov = np.cov(Xs, rowvar=False)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov + 1e-10 * np.eye(d))
    hX = 0.5 * (d * math.log(2 * math.pi * math.e) + logdet) if sign > 0 else float("nan")
    var = np.diag(cov)
    h_marg = 0.5 * (math.log(2 * math.pi * math.e) + np.log(np.clip(var, 1e-12, None)))
    TC = float(np.sum(h_marg) - hX) if math.isfinite(hX) else float("nan")
    h_minus = []
    for i in range(d):
        keep = [j for j in range(d) if j != i]
        cm = cov[np.ix_(keep, keep)]
        s2, ld2 = np.linalg.slogdet(cm + 1e-10 * np.eye(len(keep)))
        hm = 0.5 * (len(keep) * math.log(2 * math.pi * math.e) + ld2) if s2 > 0 else float("nan")
        h_minus.append(hm)
    if any(not math.isfinite(h) for h in h_minus) or not math.isfinite(hX):
        return TC, float("nan"), float("nan")
    DTC = float(sum(h_minus) - (d - 1) * hX)
    O = float(TC - DTC)
    return TC, DTC, O


def avalanches(x: np.ndarray, q: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size < 16: return np.array([]), np.array([])
    thr = float(np.quantile(x, q))
    active = x > thr
    sizes, durs = [], []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]: j += 1
            sizes.append(float(np.sum(x[i:j] - thr)))
            durs.append(j - i)
            i = j
        else:
            i += 1
    return np.array(sizes), np.array(durs)

transfer_entropy_nats = transfer_entropy
__all__=["gini_coeff","welch_slope","perm_entropy","lz_complexity","transfer_entropy","transfer_entropy_nats","tc_o_gaussian","avalanches"]
