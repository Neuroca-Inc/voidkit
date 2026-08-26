"""2D Vietoris-Rips graph-filtration and null/statistical primitives.

The source implementation computes the graph cycle rank beta1 = E - V + C; it is not a full simplicial persistent-homology engine.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

def _rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _ensure_points_array(points_like: np.ndarray) -> np.ndarray:
    A = np.asarray(points_like, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("points must be an array of shape (N,2) with columns [tau, f]")
    return A


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances for Nx2 points. O(N^2)."""
    X = _ensure_points_array(X)
    # (x - y)^2 = x^2 + y^2 - 2 x⋅y
    G = X @ X.T
    sq = np.clip(np.diag(G)[:, None] + np.diag(G)[None, :] - 2.0 * G, a_min=0.0, a_max=None)
    D = np.sqrt(sq, dtype=float)
    return D


def mst_connectivity_radius(D: np.ndarray) -> float:
    """Max edge in a minimum spanning tree (single-linkage connectivity threshold)."""
    n = int(D.shape[0])
    if n <= 1:
        return 0.0
    # Extract upper-triangular edges
    iu, ju = np.triu_indices(n, k=1)
    w = D[iu, ju].astype(float)
    order = np.argsort(w)
    uf = UnionFind(n)
    used = 0
    r_max = 0.0
    for idx in order:
        wij = float(w[idx])
        a = int(iu[idx]); b = int(ju[idx])
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb:
            uf.union(ra, rb)
            used += 1
            if wij > r_max:
                r_max = wij
            if used == n - 1:
                break
    return float(r_max)


def beta1_curve(points: np.ndarray, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute beta1(ε) = E - V + C over a VR-graph filtration on points.
    Returns: beta1 (len eps), E (len eps), C (len eps)
    """
    P = _ensure_points_array(points)
    n = P.shape[0]
    if n < 2:
        return np.zeros_like(eps), np.zeros_like(eps), np.ones_like(eps)
    D = pairwise_distances(P)
    # zero diagonal
    np.fill_diagonal(D, np.inf)
    beta1 = np.zeros_like(eps, dtype=float)
    E_arr = np.zeros_like(eps, dtype=float)
    C_arr = np.zeros_like(eps, dtype=float)
    for k, thr in enumerate(eps):
        # adjacency where dist <= thr
        A = (D <= float(thr))
        # Count edges (undirected)
        E = float(np.count_nonzero(np.triu(A, k=1)))
        # Components via union-find
        uf = UnionFind(n)
        ii, jj = np.where(np.triu(A, k=1))
        for a, b in zip(ii.tolist(), jj.tolist()):
            uf.union(int(a), int(b))
        roots = {uf.find(i) for i in range(n)}
        C = float(len(roots))
        beta1[k] = E - n + C
        E_arr[k] = E
        C_arr[k] = C
    return beta1, E_arr, C_arr


def permute_f(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-shuffle proxy at skeleton level: preserve tau and f marginals, destroy tau–f correlation."""
    P = _ensure_points_array(points)
    tau = P[:, 0].copy()
    f = P[:, 1].copy()
    rng.shuffle(f)
    return np.column_stack([tau, f])


def null_phase_shuffled_curves(points: np.ndarray, eps: np.ndarray, num_sim: int, seed: Optional[int]) -> np.ndarray:
    """
    Generate null beta1(ε) curves by permuting f across points.
    Returns array shape (num_sim, len(eps)).
    """
    rng = _rng(seed)
    curves = np.zeros((int(num_sim), len(eps)), dtype=float)
    for i in range(int(num_sim)):
        Pn = permute_f(points, rng)
        b1, _, _ = beta1_curve(Pn, eps)
        curves[i, :] = b1
    return curves


def bh_fdr(pvals: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    """
    Benjamini–Hochberg FDR control.
    Returns (threshold_p, mask_reject) where mask indicates q ≤ alpha.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, m + 1) / float(m)) * float(alpha)
    below = ranked <= thresh
    if not np.any(below):
        return 0.0, np.zeros_like(pvals, dtype=bool)
    k_max = np.max(np.where(below)[0])
    p_thr = ranked[k_max]
    mask = pvals <= p_thr
    return float(p_thr), mask


def pvals_from_null(obs: np.ndarray, null_curves: np.ndarray) -> np.ndarray:
    """
    Empirical one-sided p-values per ε with mid-p correction for ties:
    Let F_n(x-) = Pr(null < x), T_n(x) = Pr(null = x); p = 1 - (F_n + 0.5 T_n).
    Guarantees p ∈ [1/(2K), 1] when all null_j are identical to obs[j] (prevents p=0 artifacts).
    """
    K = int(null_curves.shape[0])
    p = np.ones_like(obs, dtype=float)
    for j in range(len(obs)):
        null_j = null_curves[:, j]
        less = float(np.mean(null_j < obs[j]))
        eq = float(np.mean(null_j == obs[j]))
        pj = 1.0 - (less + 0.5 * eq)
        if not np.isfinite(pj):
            pj = 1.0
        # clip to [1/(2K), 1] for numerical safety
        lo = 1.0 / (2.0 * max(1, K))
        p[j] = float(np.clip(pj, lo, 1.0))
    return p


def zscore_of_max(obs: np.ndarray, null_curves: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute z-score of the maximum beta1 across epsilons against null maxima distribution.
    Returns (z, mean_null_max, std_null_max).
    """
    max_obs = float(np.max(obs))
    null_max = np.max(null_curves, axis=1)
    mu = float(np.mean(null_max))
    sd = float(np.std(null_max, ddof=1)) if null_max.size > 1 else 0.0
    z = (max_obs - mu) / (sd + 1e-12)
    return float(z), mu, (sd if sd > 0 else 0.0)


def longest_true_run(mask: np.ndarray) -> int:
    m = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    return m

__all__=["pairwise_distances","mst_connectivity_radius","beta1_curve","null_phase_shuffled_curves","bh_fdr","pvals_from_null","zscore_of_max","longest_true_run"]
