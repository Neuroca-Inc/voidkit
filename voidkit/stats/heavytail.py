"""Discrete heavy-tail diagnostics extracted from the VDM scale-free analysis package.

Includes Clauset-style xmin selection, discrete power-law MLE/KS, bootstrap KS, and Vuong comparisons.
No causal or scale-free claim is made by this utility module.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import optimize, special, stats

def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x)==0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    if np.all(x==0):
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2*np.sum(cumx)/cumx[-1]) / n


def ccdf_int(x):
    import pandas as pd
    x = np.asarray(x, dtype=int)
    x = x[np.isfinite(x)]
    x = x[x>=0]
    if len(x)==0:
        return pd.DataFrame({"k":[], "ccdf":[]})
    maxk = int(x.max())
    counts = np.bincount(x, minlength=maxk+1)
    tail = counts[::-1].cumsum()[::-1] / counts.sum()
    ks = np.arange(len(tail))
    return pd.DataFrame({"k": ks, "ccdf": tail})


def pl_loglik_discrete(alpha, x, xmin):
    x = np.asarray(x, dtype=int)
    x = x[x>=xmin]
    n = len(x)
    if n==0:
        return -np.inf
    z = special.zeta(alpha, xmin)  # Hurwitz zeta
    if not np.isfinite(z) or z<=0:
        return -np.inf
    return -alpha*np.sum(np.log(x)) - n*np.log(z)


def fit_powerlaw_discrete(x, xmin, alpha_bounds=(1.01, 6.0)):
    x = np.asarray(x, dtype=int)
    x = x[x>=xmin]
    if len(x) < 10:
        return None
    def nll(a):
        return -pl_loglik_discrete(a, x, xmin)
    res = optimize.minimize_scalar(nll, bounds=alpha_bounds, method="bounded")
    if not res.success:
        return None
    alpha = float(res.x)
    ll = -float(res.fun)
    return alpha, ll


def ks_distance_powerlaw_discrete(x, xmin, alpha):
    x = np.asarray(x, dtype=int)
    x = x[x>=xmin]
    if len(x)==0:
        return np.nan
    xs = np.sort(x)
    n = len(xs)
    uniq = np.unique(xs)
    emp = np.array([np.searchsorted(xs, k, side="right")/n for k in uniq], dtype=float)
    z_xmin = special.zeta(alpha, xmin)
    theo = 1.0 - (special.zeta(alpha, uniq+1) / z_xmin)
    return float(np.max(np.abs(emp - theo)))


def select_xmin_clauset_discrete(x, xmin_candidates=None, min_tail=50, alpha_bounds=(1.01, 6.0)):
    import pandas as pd
    x = np.asarray(x, dtype=int)
    x = x[np.isfinite(x)]
    x = x[x>0]
    if xmin_candidates is None:
        xmin_candidates = np.unique(x)
    best = None
    rows=[]
    for xmin in xmin_candidates:
        tail = x[x>=xmin]
        n_tail = len(tail)
        if n_tail < min_tail:
            continue
        fit = fit_powerlaw_discrete(x, xmin, alpha_bounds=alpha_bounds)
        if fit is None:
            continue
        alpha, ll = fit
        ks = ks_distance_powerlaw_discrete(x, xmin, alpha)
        rows.append({"xmin": int(xmin), "alpha": alpha, "ks": ks, "n_tail": n_tail, "loglik": ll})
        if best is None or ks < best["ks"]:
            best = {"xmin": int(xmin), "alpha": alpha, "ks": ks, "n_tail": n_tail, "loglik": ll}
    return best, pd.DataFrame(rows).sort_values("ks")


def sample_discrete_powerlaw(alpha, xmin, n, rng):
    z = special.zeta(alpha, xmin)
    xmax = int(min(max(xmin+500, xmin * (n**(1/(alpha-1)))), 20000))
    xs = np.arange(xmin, xmax+1)
    pmf = xs**(-alpha) / z
    cdf = np.cumsum(pmf)
    cdf = cdf / cdf[-1]
    u = rng.random(n)
    idx = np.searchsorted(cdf, u, side="left")
    return xs[idx]


def bootstrap_p_value_ks(x, xmin, alpha, ks_obs, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=int)
    tail = x[x>=xmin]
    n_tail = len(tail)
    if n_tail==0:
        return np.nan
    ks_synth=[]
    for _ in range(n_boot):
        synth = sample_discrete_powerlaw(alpha, xmin, n_tail, rng)
        fit = fit_powerlaw_discrete(synth, xmin)
        if fit is None:
            continue
        a_hat, _ = fit
        ks_b = ks_distance_powerlaw_discrete(synth, xmin, a_hat)
        ks_synth.append(ks_b)
    if len(ks_synth)==0:
        return np.nan
    ks_synth = np.array(ks_synth)
    return float(np.mean(ks_synth >= ks_obs))


def exp_fit_geometric_ll_i(tail, xmin):
    y = tail - xmin
    mean_y = y.mean()
    if mean_y <= 0:
        q = 1e-12
    else:
        q = mean_y/(mean_y+1.0)
    q = min(max(q, 1e-12), 1-1e-12)
    lam = -math.log(q)
    ll_i = math.log(1-q) + y*np.log(q)
    return lam, ll_i


def lognormal_trunc_ll_i(tail, xmin):
    lx = np.log(tail.astype(float))
    mu = float(lx.mean())
    sigma = float(lx.std(ddof=0))
    sigma = max(sigma, 1e-9)
    ll_i = -np.log(tail.astype(float)) - math.log(sigma) - 0.5*math.log(2*math.pi) - ((lx-mu)**2)/(2*sigma**2)
    a = (math.log(xmin) - mu)/sigma
    Z = 1.0 - stats.norm.cdf(a)
    Z = max(Z, 1e-12)
    ll_i = ll_i - math.log(Z)
    return mu, sigma, ll_i


def vuong_test(ll1, ll2):
    d = ll1 - ll2
    n = len(d)
    if n<10:
        return np.nan, np.nan
    sd = d.std(ddof=1)
    if sd==0:
        mean_d = float(d.mean())
        if mean_d == 0.0:
            return float("nan"), 0.0
        return math.copysign(float("inf"), mean_d), 0.0
    z = d.sum() / (sd * math.sqrt(n))
    p = 2*(1-stats.norm.cdf(abs(z)))
    return float(z), float(p)

__all__=["gini","ccdf_int","pl_loglik_discrete","fit_powerlaw_discrete","ks_distance_powerlaw_discrete","select_xmin_clauset_discrete","sample_discrete_powerlaw","bootstrap_p_value_ks","exp_fit_geometric_ll_i","lognormal_trunc_ll_i","vuong_test"]
