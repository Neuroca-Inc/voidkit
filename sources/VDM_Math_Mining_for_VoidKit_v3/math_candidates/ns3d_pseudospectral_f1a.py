#!/usr/bin/env python3
"""Direct F1.A pilot for 3D incompressible Navier--Stokes.

This is a compact periodic pseudo-spectral solver designed for the CF10
companion package. It is intentionally a pilot, not a production turbulence
study. It measures the objects that F1.A actually needs:

  * dyadic-shell enstrophy tails,
  * the vorticity nonlinear transfer term,
  * positive active-tail production versus viscous tail dissipation,
  * divergence residuals, energy, and enstrophy.

Domain: [0, 2*pi)^3. Fourier derivatives use integer wavenumbers.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass(frozen=True)
class SolverConfig:
    N: int = 32
    steps: int = 64
    dt: float = 0.006
    nu: float = 0.02
    seed: int = 7
    sample_every: int = 4
    init_amp: float = 0.30
    tail_start_shell: int = 2


def fftn(a: np.ndarray) -> np.ndarray:
    return np.fft.fftn(a, axes=(0, 1, 2), norm="ortho")


def ifftn(a: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(a, axes=(0, 1, 2), norm="ortho")


def wave_numbers(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1 = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return kx, ky, kz, k2


def dealias_mask(N: int, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    cutoff = N / 3.0
    return (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff) & (np.abs(kz) <= cutoff)


def project_incompressible(uhat: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, k2: np.ndarray) -> np.ndarray:
    kdotu = kx * uhat[..., 0] + ky * uhat[..., 1] + kz * uhat[..., 2]
    out = uhat.copy()
    nz = k2 > 0
    out[..., 0][nz] -= kx[nz] * kdotu[nz] / k2[nz]
    out[..., 1][nz] -= ky[nz] * kdotu[nz] / k2[nz]
    out[..., 2][nz] -= kz[nz] * kdotu[nz] / k2[nz]
    out[~nz, :] = 0.0
    return out


def curl_hat(vhat: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    out = np.empty_like(vhat)
    out[..., 0] = 1j * (ky * vhat[..., 2] - kz * vhat[..., 1])
    out[..., 1] = 1j * (kz * vhat[..., 0] - kx * vhat[..., 2])
    out[..., 2] = 1j * (kx * vhat[..., 1] - ky * vhat[..., 0])
    return out


def gradient_phys(vhat: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    """Return grad_j v_i in physical space with shape (..., component, derivative)."""
    ks = [kx, ky, kz]
    grad = np.empty(vhat.shape + (3,), dtype=float)
    for comp in range(3):
        for der, kval in enumerate(ks):
            grad[..., comp, der] = ifftn(1j * kval * vhat[..., comp]).real
    return grad


def rhs(uhat: np.ndarray, cfg: SolverConfig, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, k2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    u = ifftn(uhat).real
    grad_u = gradient_phys(uhat, kx, ky, kz)
    adv = np.empty_like(u)
    # adv_i = u_j d_j u_i
    for i in range(3):
        adv[..., i] = u[..., 0] * grad_u[..., i, 0] + u[..., 1] * grad_u[..., i, 1] + u[..., 2] * grad_u[..., i, 2]
    nhat = -fftn(adv)
    nhat *= mask[..., None]
    nhat = project_incompressible(nhat, kx, ky, kz, k2)
    return nhat - cfg.nu * k2[..., None] * uhat


def rk4_step(uhat: np.ndarray, cfg: SolverConfig, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, k2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dt = cfg.dt
    k1 = rhs(uhat, cfg, kx, ky, kz, k2, mask)
    k2r = rhs(uhat + 0.5 * dt * k1, cfg, kx, ky, kz, k2, mask)
    k3 = rhs(uhat + 0.5 * dt * k2r, cfg, kx, ky, kz, k2, mask)
    k4 = rhs(uhat + dt * k3, cfg, kx, ky, kz, k2, mask)
    out = uhat + (dt / 6.0) * (k1 + 2.0 * k2r + 2.0 * k3 + k4)
    out *= mask[..., None]
    return project_incompressible(out, kx, ky, kz, k2)


def init_field(cfg: SolverConfig, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, k2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    u = rng.normal(size=(cfg.N, cfg.N, cfg.N, 3))
    uhat = fftn(u)
    envelope = np.exp(-0.5 * (np.sqrt(k2) / 4.0) ** 2)
    low = (np.sqrt(k2) <= 8.0)
    uhat *= (envelope * low * mask)[..., None]
    uhat = project_incompressible(uhat, kx, ky, kz, k2)
    energy = 0.5 * float(np.sum(np.abs(uhat) ** 2))
    if energy > 0:
        uhat *= cfg.init_amp / np.sqrt(2.0 * energy)
    return uhat


def shell_indices(kmag: np.ndarray) -> list[np.ndarray]:
    maxk = int(np.floor(float(np.max(kmag))))
    max_shell = int(np.floor(np.log2(max(maxk, 1)))) + 1
    shells: list[np.ndarray] = []
    for j in range(max_shell + 1):
        lo = 2 ** j
        hi = 2 ** (j + 1)
        shells.append((kmag >= lo) & (kmag < hi))
    return shells


def fit_beta(shell_enstrophy: np.ndarray, start: int) -> dict[str, float | int | None]:
    js = np.arange(len(shell_enstrophy), dtype=float)
    valid = (np.arange(len(shell_enstrophy)) >= start) & (shell_enstrophy > 1e-28)
    if int(np.sum(valid)) < 3:
        valid = shell_enstrophy > 1e-28
    if int(np.sum(valid)) < 3:
        return {"beta": None, "r2": None, "n_shells": int(np.sum(valid))}
    x = js[valid]
    y = np.log2(shell_enstrophy[valid])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"beta": float(-slope), "r2": float(r2), "n_shells": int(np.sum(valid))}


def shell_diagnostics(uhat: np.ndarray, cfg: SolverConfig, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, k2: np.ndarray, shells: list[np.ndarray]) -> dict[str, Any]:
    omega_hat = curl_hat(uhat, kx, ky, kz)
    omega = ifftn(omega_hat).real
    grad_u = gradient_phys(uhat, kx, ky, kz)
    grad_w = gradient_phys(omega_hat, kx, ky, kz)

    stretch_minus_adv = np.empty_like(omega)
    # Nomega_i = omega_j d_j u_i - u_j d_j omega_i
    u = ifftn(uhat).real
    for i in range(3):
        stretch = omega[..., 0] * grad_u[..., i, 0] + omega[..., 1] * grad_u[..., i, 1] + omega[..., 2] * grad_u[..., i, 2]
        adv = u[..., 0] * grad_w[..., i, 0] + u[..., 1] * grad_w[..., i, 1] + u[..., 2] * grad_w[..., i, 2]
        stretch_minus_adv[..., i] = stretch - adv
    n_omega_hat = fftn(stretch_minus_adv)

    enst = []
    transfer = []
    diss = []
    for mask_shell in shells:
        om_shell = omega_hat[mask_shell, :]
        n_shell = n_omega_hat[mask_shell, :]
        k2_shell = k2[mask_shell]
        shell_enst = float(np.sum(np.abs(om_shell) ** 2))
        shell_transfer = float(2.0 * np.real(np.sum(np.conj(om_shell) * n_shell)))
        shell_diss = float(2.0 * cfg.nu * np.sum(k2_shell[:, None] * np.abs(om_shell) ** 2))
        enst.append(shell_enst)
        transfer.append(shell_transfer)
        diss.append(shell_diss)
    enst_arr = np.array(enst, dtype=float)
    transfer_arr = np.array(transfer, dtype=float)
    diss_arr = np.array(diss, dtype=float)
    beta_info = fit_beta(enst_arr, cfg.tail_start_shell)
    tail = np.arange(len(enst_arr)) >= cfg.tail_start_shell
    pos_tail = float(np.sum(np.maximum(transfer_arr[tail], 0.0)))
    diss_tail = float(np.sum(diss_arr[tail]))
    integrated_pressure = pos_tail / max(diss_tail, 1e-30)
    pointwise_pressure = float(np.max(np.maximum(transfer_arr[tail], 0.0) / np.maximum(diss_arr[tail], 1e-30))) if np.any(tail) else 0.0
    div_hat = 1j * (kx * uhat[..., 0] + ky * uhat[..., 1] + kz * uhat[..., 2])
    return {
        "dyadic_shells": list(range(len(enst_arr))),
        "shell_enstrophy": enst_arr.tolist(),
        "shell_transfer": transfer_arr.tolist(),
        "shell_dissipation": diss_arr.tolist(),
        "dyadic_beta": beta_info["beta"],
        "dyadic_beta_r2": beta_info["r2"],
        "dyadic_beta_shell_count": beta_info["n_shells"],
        "positive_tail_pressure": integrated_pressure,
        "max_pointwise_tail_pressure": pointwise_pressure,
        "energy": float(0.5 * np.sum(np.abs(uhat) ** 2)),
        "enstrophy": float(np.sum(np.abs(omega_hat) ** 2)),
        "divergence_l2": float(np.sqrt(np.sum(np.abs(div_hat) ** 2))),
    }


def run(cfg: SolverConfig) -> dict[str, Any]:
    kx, ky, kz, k2 = wave_numbers(cfg.N)
    mask = dealias_mask(cfg.N, kx, ky, kz)
    kmag = np.sqrt(k2)
    shells = shell_indices(kmag)
    uhat = init_field(cfg, kx, ky, kz, k2, mask)
    samples: list[dict[str, Any]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.sample_every == 0 or step == cfg.steps:
            d = shell_diagnostics(uhat, cfg, kx, ky, kz, k2, shells)
            d["step"] = step
            d["time"] = step * cfg.dt
            samples.append(d)
        if step < cfg.steps:
            uhat = rk4_step(uhat, cfg, kx, ky, kz, k2, mask)
    betas = np.array([np.nan if s["dyadic_beta"] is None else s["dyadic_beta"] for s in samples], dtype=float)
    pressures = np.array([s["positive_tail_pressure"] for s in samples], dtype=float)
    pointwise = np.array([s["max_pointwise_tail_pressure"] for s in samples], dtype=float)
    divs = np.array([s["divergence_l2"] for s in samples], dtype=float)
    energies = np.array([s["energy"] for s in samples], dtype=float)
    enst = np.array([s["enstrophy"] for s in samples], dtype=float)
    finite_betas = betas[np.isfinite(betas)]
    late_slice = slice(max(0, len(samples) // 2), None)
    late_betas = betas[late_slice]
    late_betas = late_betas[np.isfinite(late_betas)]
    late_pressures = pressures[late_slice]
    late_pointwise = pointwise[late_slice]
    return {
        "status": "pilot_attack_not_validation",
        "method": "periodic incompressible dealiased 3D pseudospectral Navier-Stokes, RK4",
        "config": asdict(cfg),
        "samples": samples,
        "metrics": {
            "min_beta_all": float(np.min(finite_betas)) if finite_betas.size else None,
            "median_beta_all": float(np.median(finite_betas)) if finite_betas.size else None,
            "min_beta_late": float(np.min(late_betas)) if late_betas.size else None,
            "median_beta_late": float(np.median(late_betas)) if late_betas.size else None,
            "final_beta": float(finite_betas[-1]) if finite_betas.size else None,
            "max_positive_tail_pressure_all": float(np.max(pressures)),
            "max_positive_tail_pressure_late": float(np.max(late_pressures)),
            "max_pointwise_tail_pressure_all": float(np.max(pointwise)),
            "max_pointwise_tail_pressure_late": float(np.max(late_pointwise)),
            "max_divergence_l2": float(np.max(divs)),
            "initial_energy": float(energies[0]),
            "final_energy": float(energies[-1]),
            "initial_enstrophy": float(enst[0]),
            "final_enstrophy": float(enst[-1]),
            "energy_decay_fraction": float((energies[0] - energies[-1]) / max(energies[0], 1e-30)),
        },
    }


def make_figures(results_by_label: dict[str, dict[str, Any]], figdir: Path) -> None:
    if plt is None:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    markers = ["o", "s", "^"]
    grays = ["0.15", "0.45", "0.65"]
    for idx, (label, res) in enumerate(results_by_label.items()):
        xs = [s["time"] for s in res["samples"]]
        ys = [np.nan if s["dyadic_beta"] is None else s["dyadic_beta"] for s in res["samples"]]
        ax.plot(xs, ys, marker=markers[idx % len(markers)], color=grays[idx % len(grays)], linewidth=1.2, markersize=3.5, label=label)
    ax.axhline(3.0, color="0.75", linestyle="--", linewidth=1.0, label=r"$\beta=3$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"dyadic tail exponent $\beta(t)$")
    ax.set_title("Direct 3D pilot: measured shell-tail exponent")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "f1a_pilot_tail_exponent.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for idx, (label, res) in enumerate(results_by_label.items()):
        xs = [s["time"] for s in res["samples"]]
        ys = [s["positive_tail_pressure"] for s in res["samples"]]
        ax.plot(xs, ys, marker=markers[idx % len(markers)], color=grays[idx % len(grays)], linewidth=1.2, markersize=3.5, label=label)
    ax.axhline(1.0, color="0.75", linestyle="--", linewidth=1.0, label="unit pressure")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\mathcal{P}_{\geq K}^{+}/\mathcal{D}_{\geq K}$")
    ax.set_title("Direct 3D pilot: active-tail transfer pressure")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "f1a_pilot_transfer_pressure.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for idx, (label, res) in enumerate(results_by_label.items()):
        final = res["samples"][-1]
        shells = np.array(final["dyadic_shells"], dtype=float)
        vals = np.array(final["shell_enstrophy"], dtype=float)
        ok = vals > 1e-28
        ax.semilogy(shells[ok], vals[ok], marker=markers[idx % len(markers)], color=grays[idx % len(grays)], linewidth=1.2, markersize=3.5, label=label)
    ax.set_xlabel("dyadic shell index")
    ax.set_ylabel(r"final shell enstrophy $\Omega_k$")
    ax.set_title("Direct 3D pilot: final resolved tail")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "f1a_pilot_final_shell_tail.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.006)
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-every", type=int, default=4)
    ap.add_argument("--init-amp", type=float, default=0.30)
    ap.add_argument("--tail-start-shell", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--figdir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    cfg = SolverConfig(N=args.N, steps=args.steps, dt=args.dt, nu=args.nu, seed=args.seed, sample_every=args.sample_every, init_amp=args.init_amp, tail_start_shell=args.tail_start_shell)
    res = run(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    if args.figures:
        make_figures({f"N={cfg.N}": res}, args.figdir)
    print(json.dumps({"status": res["status"], "config": res["config"], "metrics": res["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
