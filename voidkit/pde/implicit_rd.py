"""Structure-aware 1D reaction-diffusion solvers extracted from the VDM conservation harness.

Includes Crank-Nicolson diffusion, exact-logistic Strang splitting, and the source dense Newton/backtracking discrete-gradient step.
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from voidkit.dynamics.logistic import reaction_exact_step

def laplacian_periodic_1d(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)


def energy_potential_Vhat(W: np.ndarray, r: float, ucoef: float) -> np.ndarray:
    # Vhat'(W) = -f(W) = -(r W - u W^2) => Vhat(W) = -(r/2) W^2 + (u/3) W^3 + C
    return -(r / 2.0) * (W ** 2) + (ucoef / 3.0) * (W ** 3)


def discrete_lyapunov_Lh(W: np.ndarray, dx: float, D: float, r: float, ucoef: float) -> float:
    # Edge-based gradient consistent with 3-point Laplacian
    diff = (np.roll(W, -1) - W) / dx
    grad_sq = diff * diff
    return float(np.sum(0.5 * D * grad_sq + energy_potential_Vhat(W, r, ucoef)) * dx)


def diffusion_CN_step_periodic(W: np.ndarray, dt: float, dx: float, D: float) -> np.ndarray:
    """Crank-Nicolson diffusion half/whole step via spectral diagonalization (periodic)."""
    if D == 0.0 or dt == 0.0:
        return W.copy()
    N = W.size
    k = np.fft.fftfreq(N, d=dx)  # cycles per unit length
    # Symbol of discrete Laplacian: lambda = -4 sin^2(pi k dx)/(dx^2)
    theta = 2.0 * np.pi * k * dx
    lam = -4.0 * (np.sin(0.5 * theta) ** 2) / (dx * dx)
    alpha = 0.5 * dt * D
    G = (1.0 + alpha * lam) / (1.0 - alpha * lam)
    W_hat = np.fft.fft(W)
    Wn1_hat = G * W_hat
    Wn1 = np.fft.ifft(Wn1_hat).real
    return Wn1


def strang_step(W: np.ndarray, dt: float, dx: float, D: float, r: float, ucoef: float) -> np.ndarray:
    """Strang split: 1/2 diffusion (CN), exact reaction, 1/2 diffusion (CN)."""
    W_half = diffusion_CN_step_periodic(W, 0.5 * dt, dx, D)
    W_react = reaction_exact_step(W_half, r, ucoef, dt)
    W_out = diffusion_CN_step_periodic(W_react, 0.5 * dt, dx, D)
    return W_out


def dg_rd_step(Wn: np.ndarray, dt: float, dx: float, D: float, r: float, u: float, tol: float = 1e-12, max_iter: int = 20) -> np.ndarray:
    """Discrete-gradient RD implicit step (AVF for reaction, midpoint Laplacian), Newton solve (dense)."""
    N = Wn.size
    W1 = Wn.copy()
    def lap(x):
        return laplacian_periodic_1d(x, dx)
    for it in range(max_iter):
        mid = 0.5 * (W1 + Wn)
        # overline f (AVF) for logistic: r*(Wn + 0.5*(W1-Wn)) - u*((Wn^2 + Wn W1 + W1^2)/3)
        dW = (W1 - Wn)
        over_f = r * (Wn + 0.5 * dW) - u * ((Wn * Wn + Wn * W1 + W1 * W1) / 3.0)
        F = W1 - Wn - dt * (D * lap(mid) + over_f)
        res = np.linalg.norm(F, ord=np.inf)
        if res <= tol:
            break
        # Build dense Jacobian: I - dt*(0.5 D L + 0.5 r I - u*(Wn/3 + 2/3 W1) I)
        # Laplacian linear operator with periodic stencil
        J = np.eye(N)
        # Add - dt * 0.5 D L
        coeff = - dt * 0.5 * D / (dx * dx)
        for i in range(N):
            J[i, i] += - coeff * (-2.0)
            J[i, (i - 1) % N] += - coeff * (1.0)
            J[i, (i + 1) % N] += - coeff * (1.0)
        # Add - dt * (0.5 r I - u*(Wn/3 + 2/3 W1) I)
        diag_add = - dt * (0.5 * r - u * (Wn / 3.0 + (2.0 / 3.0) * W1))
        J[np.arange(N), np.arange(N)] += diag_add
        d = np.linalg.solve(J, -F)
        W1 = W1 + d
        if np.linalg.norm(d, ord=np.inf) <= tol * 0.1:
            break
    return W1


def dg_rd_step_with_stats(Wn: np.ndarray, dt: float, dx: float, D: float, r: float, u: float,
                          tol: float = 1e-12, max_iter: int = 20, max_backtracks: int = 10,
                          lap_operator: str = "stencil") -> tuple[np.ndarray, Dict[str, Any]]:
    """DG RD step with Newton iteration stats and simple backtracking line search.

    lap_operator: 'stencil' (3-pt periodic) or 'spectral' (FFT-based circulant). Default 'stencil'.
    """
    N = Wn.size
    W1 = Wn.copy()
    stats = {"iters": 0, "final_residual_inf": None, "backtracks": 0, "converged": False}
    # Prepare Laplacian operator according to mode
    lap_mode = str(lap_operator or "stencil").lower()
    if lap_mode == "spectral":
        N = Wn.size
        k_cyc = np.fft.fftfreq(N, d=dx)
        omega_sq = (2.0 * np.pi) ** 2 * (k_cyc ** 2)
        lam_spec = - omega_sq  # symbol for ∂xx
        # Dense circulant matrix of spectral Laplacian (real)
        kernel = np.fft.ifft(lam_spec).real
        C_spec = np.empty((N, N), dtype=float)
        for i in range(N):
            C_spec[i, :] = np.roll(kernel, i)
        def lap(x):
            # Use dense circulant for consistency with Jacobian
            return C_spec @ x
    else:
        C_spec = None
        def lap(x):
            return laplacian_periodic_1d(x, dx)
    prev_res = None
    for it in range(1, max_iter + 1):
        mid = 0.5 * (W1 + Wn)
        dW = (W1 - Wn)
        over_f = r * (Wn + 0.5 * dW) - u * ((Wn * Wn + Wn * W1 + W1 * W1) / 3.0)
        F = W1 - Wn - dt * (D * lap(mid) + over_f)
        res = float(np.linalg.norm(F, ord=np.inf))
        if res <= tol:
            stats.update({"iters": it, "final_residual_inf": res, "converged": True})
            break
        # Build dense Jacobian
        J = np.eye(N)
        if lap_mode == "spectral":
            # Add - dt * 0.5 * D * L_spec
            J += (- dt * 0.5 * D) * C_spec
        else:
            coeff = - dt * 0.5 * D / (dx * dx)
            for i in range(N):
                J[i, i] += - coeff * (-2.0)
                J[i, (i - 1) % N] += - coeff * (1.0)
                J[i, (i + 1) % N] += - coeff * (1.0)
        diag_add = - dt * (0.5 * r - u * (Wn / 3.0 + (2.0 / 3.0) * W1))
        J[np.arange(N), np.arange(N)] += diag_add
        d = np.linalg.solve(J, -F)
        # Backtracking line search to ensure residual decrease
        step = 1.0
        W_trial = W1 + step * d
        # Evaluate residual at trial
        mid_t = 0.5 * (W_trial + Wn)
        over_f_t = r * (Wn + 0.5 * (W_trial - Wn)) - u * ((Wn * Wn + Wn * W_trial + W_trial * W_trial) / 3.0)
        F_t = W_trial - Wn - dt * (D * lap(mid_t) + over_f_t)
        res_t = float(np.linalg.norm(F_t, ord=np.inf))
        bt = 0
        while res_t > res and bt < max_backtracks:
            step *= 0.5
            W_trial = W1 + step * d
            mid_t = 0.5 * (W_trial + Wn)
            over_f_t = r * (Wn + 0.5 * (W_trial - Wn)) - u * ((Wn * Wn + Wn * W_trial + W_trial * W_trial) / 3.0)
            F_t = W_trial - Wn - dt * (D * lap(mid_t) + over_f_t)
            res_t = float(np.linalg.norm(F_t, ord=np.inf))
            bt += 1
        if bt > 0:
            stats["backtracks"] = stats.get("backtracks", 0) + bt
        W1 = W_trial
        prev_res = res
        stats.update({"iters": it, "final_residual_inf": res_t})
        if np.linalg.norm(step * d, ord=np.inf) <= tol * 0.1:
            # Step small enough
            stats["converged"] = True
            break
    return W1, stats

__all__=["laplacian_periodic_1d","energy_potential_Vhat","discrete_lyapunov_Lh","dg_rd_step","dg_rd_step_with_stats","diffusion_CN_step_periodic","strang_step"]
