"""Weighted graph stationary, spectral-embedding, participation, and projection-grid utilities.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def stationary_distribution(A: sp.csr_matrix, tol=1e-12, maxit=50000, teleport=1e-6) -> np.ndarray:
    """
    Stationary distribution for a weighted random walk on directed A.
    Uses tiny teleportation to ensure ergodicity.
    """
    N = A.shape[0]
    out_w = np.array(A.sum(axis=1)).ravel()
    inv_out = np.zeros(N)
    m = out_w > 0
    inv_out[m] = 1.0 / out_w[m]
    AT = A.transpose().tocsr()

    pi = np.full(N, 1.0/N, dtype=float)
    for _ in range(maxit):
        tmp = pi * inv_out
        nxt = AT.dot(tmp)
        dangling = pi[~m].sum() if np.any(~m) else 0.0
        if dangling:
            nxt += dangling / N
        nxt = (1-teleport)*nxt + teleport*(1.0/N)
        s = nxt.sum()
        if s != 0:
            nxt /= s
        if np.linalg.norm(nxt - pi, 1) < tol:
            pi = nxt
            break
        pi = nxt
    return pi


def spectral_embedding(A_sym: sp.csr_matrix, n_components=3, seed=0):
    """
    Spectral embedding from normalized Laplacian of the symmetrized graph.
    Returns smallest eigenvalues (including the ~0 trivial) and coordinates (skip trivial).
    """
    N = A_sym.shape[0]
    deg = np.array(A_sym.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(deg)
    m = deg > 0
    d_inv_sqrt[m] = 1.0 / np.sqrt(deg[m])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(N, format="csr") - D_inv_sqrt @ A_sym @ D_inv_sqrt

    k = n_components + 1
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=N)
    vals, vecs = spla.eigsh(L, k=k, which="SM", tol=1e-3, maxiter=10000, v0=v0)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    coords = vecs[:, 1:n_components+1]
    return vals, coords


def participation_coefficient(A_sym: sp.csr_matrix, node_to_c: np.ndarray) -> np.ndarray:
    N = A_sym.shape[0]
    deg = np.array(A_sym.sum(axis=1)).ravel()
    nc = node_to_c.max() + 1
    P = np.zeros(N, dtype=float)
    for i in range(N):
        ki = deg[i]
        if ki <= 0:
            continue
        start, end = A_sym.indptr[i], A_sym.indptr[i+1]
        neigh = A_sym.indices[start:end]
        w = A_sym.data[start:end]
        comm = node_to_c[neigh]
        sums = np.bincount(comm, weights=w, minlength=nc)
        P[i] = 1.0 - np.sum((sums/ki)**2)
    return P


def gridify(coords2d: np.ndarray, values: np.ndarray, grid_size=32, padding=0.02):
    x = coords2d[:, 0]
    y = coords2d[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xr = xmax - xmin
    yr = ymax - ymin
    xmin -= padding * xr
    xmax += padding * xr
    ymin -= padding * yr
    ymax += padding * yr
    xn = (x - xmin) / (xmax - xmin + 1e-12)
    yn = (y - ymin) / (ymax - ymin + 1e-12)
    xi = np.clip((xn * grid_size).astype(int), 0, grid_size-1)
    yi = np.clip((yn * grid_size).astype(int), 0, grid_size-1)

    grid = np.zeros((grid_size, grid_size), dtype=float)
    counts = np.zeros((grid_size, grid_size), dtype=int)
    for v, i, j in zip(values, xi, yi):
        grid[j, i] += float(v)
        counts[j, i] += 1
    meta = {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)}
    return grid, counts, meta

__all__=["stationary_distribution","spectral_embedding","participation_coefficient","gridify"]
