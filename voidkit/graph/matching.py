"""Graph spectral-coordinate, feature-matching, and edge-overlap primitives.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment

def compute_spectral_coords(A_und, k=8):
    N=A_und.shape[0]
    deg=np.array(A_und.sum(axis=1)).ravel().astype(float)
    inv_sqrt=np.zeros_like(deg)
    mask=deg>0
    inv_sqrt[mask]=1.0/np.sqrt(deg[mask])
    Dinv=sp.diags(inv_sqrt)
    L = sp.eye(N, format='csr') - Dinv @ A_und @ Dinv

    k_compute=min(N-2, k+6)
    vals, vecs = spla.eigsh(L, k=k_compute, which='SM')
    idx=np.argsort(vals)
    vals=vals[idx]
    vecs=vecs[:,idx]

    nontriv = np.where(vals>1e-6)[0]
    start = int(nontriv[0]) if len(nontriv)>0 else 0
    take = vecs[:, start:start+k]

    for j in range(take.shape[1]):
        if take[:,j].sum() < 0:
            take[:,j] *= -1
    return take


def zscore_pair(X, Y):
    Z=np.vstack([X,Y])
    mu=Z.mean(axis=0)
    sd=Z.std(axis=0)
    sd[sd==0]=1.0
    return (X-mu)/sd, (Y-mu)/sd


def pairwise_cost(X, Y):
    X2=np.sum(X*X, axis=1)[:,None]
    Y2=np.sum(Y*Y, axis=1)[None,:]
    XY=X @ Y.T
    d2=X2 + Y2 - 2*XY
    d2[d2<0]=0
    return np.sqrt(d2)


def match_nodes_features(feats_a, feats_b):
    Xa,Xb=zscore_pair(feats_a,feats_b)
    C=pairwise_cost(Xa,Xb)
    row_ind,col_ind=linear_sum_assignment(C)
    perm=np.empty_like(col_ind)
    perm[row_ind]=col_ind
    return perm, C


def csr_edge_ids(row_ptr, col_idx, N):
    row_idx = np.repeat(np.arange(N, dtype=np.int64), np.diff(row_ptr))
    ids = row_idx * N + col_idx.astype(np.int64)
    return np.unique(ids)


def permute_edge_ids(ids_B, inv_perm, N):
    row = (ids_B // N).astype(np.int64)
    col = (ids_B % N).astype(np.int64)
    row2 = inv_perm[row]
    col2 = inv_perm[col]
    return np.unique(row2 * N + col2)


def edge_jaccard(ids1, ids2):
    inter=np.intersect1d(ids1, ids2, assume_unique=True)
    union=ids1.size + ids2.size - inter.size
    return inter.size/union if union>0 else 1.0, int(inter.size), int(union)


def corr(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if a.std()==0 or b.std()==0:
        return float("nan")
    return float(np.corrcoef(a,b)[0,1])

normalized_laplacian_coords = compute_spectral_coords
__all__=["compute_spectral_coords","normalized_laplacian_coords","zscore_pair","pairwise_cost","match_nodes_features","csr_edge_ids","permute_edge_ids","edge_jaccard","corr"]
