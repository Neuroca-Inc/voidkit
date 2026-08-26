"""Recurrence-quantification metrics extracted from a VDM analysis harness.
Input is an already-constructed recurrence matrix.
"""
from __future__ import annotations
import numpy as np

def rqa_metrics(R: np.ndarray, lmin: int = 2, vmin: int = 2) -> dict:
    N = R.shape[0]
    RR = (R.sum() - N) / (N * (N - 1))

    diag_lengths = []
    for k in range(-N + 1, N):
        if k == 0:
            continue
        diag = np.diagonal(R, offset=k)
        run = 0
        for val in diag:
            if val:
                run += 1
            else:
                if run > 0:
                    diag_lengths.append(run)
                    run = 0
        if run > 0:
            diag_lengths.append(run)
    diag_lengths = np.array(diag_lengths, dtype=int)
    if len(diag_lengths) == 0:
        DET = float("nan")
        Lmean = float("nan")
        Lent = float("nan")
    else:
        det_points = diag_lengths[diag_lengths >= lmin].sum()
        all_points = diag_lengths.sum()
        DET = det_points / all_points if all_points > 0 else float("nan")
        Lmean = diag_lengths[diag_lengths >= lmin].mean() if np.any(diag_lengths >= lmin) else float("nan")
        if np.any(diag_lengths >= lmin):
            lengths = diag_lengths[diag_lengths >= lmin]
            vals, counts = np.unique(lengths, return_counts=True)
            p = counts / counts.sum()
            Lent = float(-(p * np.log(p)).sum())
        else:
            Lent = float("nan")

    vert_lengths = []
    for j in range(N):
        col = R[:, j]
        run = 0
        for i, val in enumerate(col):
            if i == j:
                if run > 0:
                    vert_lengths.append(run)
                    run = 0
                continue
            if val:
                run += 1
            else:
                if run > 0:
                    vert_lengths.append(run)
                    run = 0
        if run > 0:
            vert_lengths.append(run)
    vert_lengths = np.array(vert_lengths, dtype=int)
    if len(vert_lengths) == 0:
        LAM = float("nan")
        TT = float("nan")
    else:
        lam_points = vert_lengths[vert_lengths >= vmin].sum()
        all_points = vert_lengths.sum()
        LAM = lam_points / all_points if all_points > 0 else float("nan")
        TT = vert_lengths[vert_lengths >= vmin].mean() if np.any(vert_lengths >= vmin) else float("nan")

    return {
        "RR": float(RR),
        "DET": float(DET),
        "Lmean": float(Lmean),
        "Lent": float(Lent),
        "LAM": float(LAM),
        "TT": float(TT),
        "n_diag_lines": int(len(diag_lengths)),
        "n_vert_lines": int(len(vert_lengths)),
    }

__all__=["rqa_metrics"]
