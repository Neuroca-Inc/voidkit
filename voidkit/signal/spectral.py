"""Spectral slope helpers.
"""
from __future__ import annotations
import numpy as np
from voidkit.info_theory.complexity import welch_slope

def psd_loglog_slope(signal: np.ndarray, fs: float = 1.0, f_low: float = 1/500, f_high: float = 0.1):
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    N = len(x)
    w = np.hanning(N)
    xw = x * w
    X = np.fft.rfft(xw)
    psd = (np.abs(X) ** 2) / (fs * np.sum(w ** 2))
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    mask = (freqs >= f_low) & (freqs <= f_high) & (psd > 0)
    if mask.sum() < 10:
        return float("nan"), (freqs, psd, mask, float("nan"), float("nan"))
    lf = np.log10(freqs[mask])
    lp = np.log10(psd[mask])
    slope, intercept = np.polyfit(lf, lp, 1)
    return float(slope), (freqs, psd, mask, float(slope), float(intercept))

__all__=["welch_slope","psd_loglog_slope"]
