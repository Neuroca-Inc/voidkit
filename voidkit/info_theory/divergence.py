"""Distribution-divergence utilities.
"""
from __future__ import annotations
import numpy as np
from voidkit.stats.basic import hellinger

def js_div(p, q):
    p = np.asarray(p, dtype=float).flatten(); q = np.asarray(q, dtype=float).flatten()
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

jensen_shannon_divergence = js_div
__all__=["js_div","jensen_shannon_divergence","hellinger"]
