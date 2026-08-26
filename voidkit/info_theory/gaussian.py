"""Gaussian covariance-based information measures extracted from the Aura analysis suite.
`gaussian_mi_matrix` is retained under its source name; it returns (TC, DTC, O).
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import det

def gaussian_mi_matrix(X):
    X=np.asarray(X)
    cov=np.cov(X, rowvar=False)
    if cov.ndim==0:
        return 0.0
    cov=np.atleast_2d(cov)+np.eye(cov.shape[0])*1e-9
    n=cov.shape[0]
    tc = 0.5*np.log(np.prod(np.diag(cov))/det(cov))
    dtc = 0.0
    for i in range(n):
        idx=[j for j in range(n) if j!=i]
        cov_wo=np.atleast_2d(np.cov(X[:,idx], rowvar=False))+np.eye(n-1)*1e-9
        dtc += 0.5*np.log(det(cov_wo)/det(cov))
    return tc, dtc, tc-dtc


def mi_gaussian_xy(x,y):
    xy=np.c_[x,y]
    cov=np.cov(xy, rowvar=False)+np.eye(xy.shape[1])*1e-9
    k=x.shape[1]
    covx=cov[:k,:k]; covy=cov[k:,k:]
    return 0.5*np.log(det(covx)*det(covy)/det(cov))

gaussian_tc_dtc_o = gaussian_mi_matrix
__all__=["gaussian_mi_matrix","gaussian_tc_dtc_o","mi_gaussian_xy"]
