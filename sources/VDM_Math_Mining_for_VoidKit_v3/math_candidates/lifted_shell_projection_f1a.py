#!/usr/bin/env python3
"""Retained lifted-shell F1.A mechanism scanner.

This script is deliberately not a dense 3D Navier--Stokes validation. It is the
second CF10 attack path: evolve a retained shell state cheaply, keep branch
history/monodromy until final readout, and project to F1.A observables only at
sample boundaries. The reduced state is admissible only after a future quotient
residual is measured against full 3D evolution.
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
class ShellConfig:
    K: int = 18
    steps: int = 5000
    dt: float = 3.0e-4
    nu: float = 1.0e-5
    forcing: float = 1.3e-2
    cascade_rate: float = 8.0
    hierarchy_strength: float = 3.0
    front_shell: int = 6
    sample_every: int = 60
    floor: float = 1.0e-18


def fit_beta(omega: np.ndarray, start: int) -> dict[str, Any]:
    idx = np.arange(len(omega), dtype=float)
    valid = (idx >= start) & (omega > 1e-18)
    if int(np.sum(valid)) < 3:
        return {"beta": None, "r2": None, "n_shells": int(np.sum(valid))}
    x = idx[valid]
    y = np.log2(omega[valid])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"beta": float(-slope), "r2": 1.0 if ss_tot == 0 else float(1 - ss_res / ss_tot), "n_shells": int(np.sum(valid))}


def fibonacci_corridor(max_shell: int) -> list[int]:
    """Return the retained Fibonacci refinement corridor inside the shell range."""
    out: list[int] = []
    a, b = 1, 2
    while a < max_shell:
        out.append(a)
        a, b = b, a + b
    return out


def shadow_residual(Omega: np.ndarray, mu: np.ndarray, gamma: np.ndarray, cfg: ShellConfig, hierarchy: bool) -> float:
    """Projection honesty metric: full lifted step minus memoryless shadow step.

    This is the reduced-shell analogue of R_{Pi,G}=Pi(E X)-G(Pi X).
    A large value means amplitude-only projection lost active lifted state.
    """
    full_next, _, _, _ = step_state(Omega.copy(), mu.copy(), gamma.copy(), cfg, hierarchy)
    shadow_next, _, _, _ = step_state(Omega.copy(), np.zeros_like(mu), np.zeros_like(gamma), cfg, hierarchy)
    return float(np.linalg.norm(full_next - shadow_next) / max(np.linalg.norm(full_next), cfg.floor))


def step_state(Omega: np.ndarray, mu: np.ndarray, gamma: np.ndarray, cfg: ShellConfig, hierarchy: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    K = cfg.K
    idx = np.arange(K, dtype=float)
    # Low-shell forcing with mild saturation.
    force = np.zeros(K)
    force[:3] = cfg.forcing * np.array([1.0, 0.55, 0.20]) / (1.0 + 12.0 * Omega[:3])

    # Adjacent-shell nonlinear cascade pressure.
    shoulder = 1.0 + 0.12 * np.sin(0.4 * idx + 0.2 * float(np.sum(mu[:3])))
    raw_flux = cfg.cascade_rate * shoulder * np.maximum(Omega, 0.0) ** 1.45
    raw_flux *= 1.0 / (1.0 + 0.25 * idx)
    if hierarchy:
        # Boundary hosting / interface throttle. This is the retained hierarchy channel.
        throttle = np.ones(K)
        far = np.maximum(idx - cfg.front_shell, 0.0)
        throttle *= np.exp(-cfg.hierarchy_strength * far)
        throttle *= 1.0 / (1.0 + 0.10 * np.maximum(gamma, 0.0))
    else:
        throttle = np.ones(K)
    flux = raw_flux * throttle
    flux[-1] = 0.0

    incoming = np.zeros(K)
    outgoing = np.zeros(K)
    incoming[1:] = flux[:-1]
    outgoing[:-1] = flux[:-1]
    diss = 2.0 * cfg.nu * (2.0 ** (2.0 * idx)) * Omega
    dOmega = force + incoming - outgoing - diss
    nextOmega = np.maximum(Omega + cfg.dt * dOmega, cfg.floor)

    # Retained monodromy / branch-history coordinate: signed cascade imbalance normalized
    # against a global scale, not against tiny shell amplitudes.
    imbalance = incoming - outgoing
    scale = max(float(np.max(Omega)), cfg.floor)
    mu_rhs = imbalance / max(scale, cfg.floor)
    nextMu = mu + cfg.dt * mu_rhs

    # Interface-hosting memory grows when far-tail flux asks to cross the front.
    gamma_rhs = np.zeros(K)
    gamma_rhs += np.maximum(idx - cfg.front_shell, 0.0) * np.maximum(incoming, 0.0)
    gamma_rhs -= 0.02 * gamma
    nextGamma = np.maximum(gamma + cfg.dt * gamma_rhs, 0.0)

    diag = {
        "positive_tail_production": float(np.sum(np.maximum(incoming[cfg.front_shell:] - outgoing[cfg.front_shell:], 0.0))),
        "tail_dissipation": float(np.sum(diss[cfg.front_shell:])),
        "tail_flux": float(np.sum(flux[cfg.front_shell:])),
    }
    return nextOmega, nextMu, nextGamma, diag


def run_case(cfg: ShellConfig, hierarchy: bool) -> dict[str, Any]:
    idx = np.arange(cfg.K, dtype=float)
    # Smooth low-shell seed with a small high-shell shoulder.
    Omega = 0.07 * np.exp(-0.70 * idx) + 8e-5 * np.exp(-0.12 * (idx - 5.0) ** 2)
    mu = np.zeros(cfg.K)
    gamma = np.zeros(cfg.K)
    samples = []
    for n in range(cfg.steps + 1):
        if n % cfg.sample_every == 0 or n == cfg.steps:
            beta = fit_beta(Omega, cfg.front_shell)
            tail = Omega[cfg.front_shell:]
            ultra = Omega[min(cfg.front_shell + 3, cfg.K):]
            far_mu = mu[min(cfg.front_shell + 3, cfg.K):]
            monodromy_total_var = float(np.sum(np.abs(np.diff(mu))))
            monodromy_tail_var = float(np.sum(np.abs(np.diff(mu[cfg.front_shell:]))))
            monodromy_far_tail_var = float(np.sum(np.abs(np.diff(far_mu)))) if far_mu.size >= 2 else 0.0
            # Estimate production/dissipation and quotient-shadow residual at the sampled state.
            _, _, _, diag = step_state(Omega.copy(), mu.copy(), gamma.copy(), cfg, hierarchy)
            pressure = diag["positive_tail_production"] / max(diag["tail_dissipation"], cfg.floor)
            shadow = shadow_residual(Omega, mu, gamma, cfg, hierarchy)
            samples.append({
                "step": n,
                "time": n * cfg.dt,
                "dyadic_beta": beta["beta"],
                "dyadic_beta_r2": beta["r2"],
                "tail_mass": float(np.sum(tail)),
                "ultra_tail_mass": float(np.sum(ultra)),
                "total_enstrophy_proxy": float(np.sum(Omega)),
                "tail_pressure_ratio": float(pressure),
                "monodromy_total_variation": monodromy_total_var,
                "monodromy_tail_variation": monodromy_tail_var,
                "monodromy_far_tail_variation": monodromy_far_tail_var,
                "monodromy_max_abs": float(np.max(np.abs(mu))),
                "shadow_residual_l2": shadow,
                "omega": Omega.tolist(),
                "mu": mu.tolist(),
                "gamma": gamma.tolist(),
            })
        if n < cfg.steps:
            Omega, mu, gamma, _ = step_state(Omega, mu, gamma, cfg, hierarchy)
    late = samples[len(samples) // 2:]
    betas = np.array([np.nan if s["dyadic_beta"] is None else s["dyadic_beta"] for s in late], dtype=float)
    betas = betas[np.isfinite(betas)]
    pressures = np.array([s["tail_pressure_ratio"] for s in late], dtype=float)
    shadows = np.array([s["shadow_residual_l2"] for s in late], dtype=float)
    final = samples[-1]
    return {
        "samples": samples,
        "metrics": {
            "median_beta_late": float(np.median(betas)) if betas.size else None,
            "min_beta_late": float(np.min(betas)) if betas.size else None,
            "max_tail_production_ratio_late": float(np.max(pressures)) if pressures.size else None,
            "final_tail_mass": final["tail_mass"],
            "final_ultra_tail_mass": final["ultra_tail_mass"],
            "final_total_enstrophy_proxy": final["total_enstrophy_proxy"],
            "final_monodromy_total_variation": final["monodromy_total_variation"],
            "final_monodromy_tail_variation": final["monodromy_tail_variation"],
            "final_monodromy_far_tail_variation": final["monodromy_far_tail_variation"],
            "final_monodromy_max_abs": final["monodromy_max_abs"],
            "max_shadow_residual_late": float(np.max(shadows)) if shadows.size else None,
            "final_shadow_residual_l2": final["shadow_residual_l2"],
        },
    }


def run(cfg: ShellConfig) -> dict[str, Any]:
    hierarchy = run_case(cfg, True)
    open_flux = run_case(cfg, False)
    return {
        "status": "lifted_shell_projection_surrogate_not_full_ns_validation",
        "retained_state_channels": ["Omega_k", "Phi_k", "D_k", "Gamma_k", "mu_k"],
        "projection_rule": "Evolve lifted shell state first; project to beta, tail pressure, and shell masses at sampling boundaries only.",
        "fibonacci_corridor_shells": fibonacci_corridor(cfg.K),
        "orthogonal_rearticulation_rule": "When same-shell cascade hosting saturates, flux is re-hosted into the hierarchy channel Gamma_k rather than discharged.",
        "shadow_residual_definition": "||Pi_shell(E_lifted X) - G_shadow(Pi_shell X)||_2 / ||Pi_shell(E_lifted X)||_2, with G_shadow dropping Gamma_k and mu_k.",
        "quotient_condition": "Use only as a mechanism scanner until Pi_shell(E_NS(u)) - G_shell(Pi_shell(u)) is measured against full 3D trajectories.",
        "config": asdict(cfg),
        "hierarchical_tail": hierarchy,
        "open_flux_tail": open_flux,
    }


def make_figures(result: dict[str, Any], figdir: Path) -> None:
    if plt is None:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    h = result["hierarchical_tail"]["samples"]
    o = result["open_flux_tail"]["samples"]
    t = [s["time"] for s in h]
    beta_h = [np.nan if s["dyadic_beta"] is None else s["dyadic_beta"] for s in h]
    beta_o = [np.nan if s["dyadic_beta"] is None else s["dyadic_beta"] for s in o]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(t, beta_h, marker="o", color="0.15", markersize=3.5, linewidth=1.2, label="hierarchy-throttled lift")
    ax.plot(t, beta_o, marker="s", color="0.55", markersize=3.5, linewidth=1.2, label="open shell flux")
    ax.axhline(3.0, color="0.75", linestyle="--", linewidth=1.0, label=r"$\beta=3$")
    ax.set_xlabel("lifted-shell time")
    ax.set_ylabel(r"projected tail exponent $\beta(t)$")
    ax.set_title("Lifted-shell projection: tail exponent")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "lifted_shell_tail_exponent.png", dpi=220)
    plt.close(fig)

    final_h = np.array(h[-1]["omega"], dtype=float)
    final_o = np.array(o[-1]["omega"], dtype=float)
    idx = np.arange(len(final_h))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.semilogy(idx, final_h, marker="o", color="0.15", markersize=3.5, linewidth=1.2, label="hierarchy-throttled lift")
    ax.semilogy(idx, final_o, marker="s", color="0.55", markersize=3.5, linewidth=1.2, label="open shell flux")
    ax.set_xlabel("shell index")
    ax.set_ylabel(r"projected shell enstrophy $\Omega_k$")
    ax.set_title("Lifted-shell projection: final shell state")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "lifted_shell_final_tail.png", dpi=220)
    plt.close(fig)

    mu_h = [s["monodromy_far_tail_variation"] for s in h]
    mu_o = [s["monodromy_far_tail_variation"] for s in o]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.semilogy(t, mu_h, marker="o", color="0.15", markersize=3.5, linewidth=1.2, label="hierarchy-throttled lift")
    ax.semilogy(t, mu_o, marker="s", color="0.55", markersize=3.5, linewidth=1.2, label="open shell flux")
    ax.set_xlabel("lifted-shell time")
    ax.set_ylabel("far-tail monodromy variation")
    ax.set_title("Lifted-shell projection: retained branch history")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "lifted_shell_monodromy_variation.png", dpi=220)
    plt.close(fig)

    sr_h = [s["shadow_residual_l2"] for s in h]
    sr_o = [s["shadow_residual_l2"] for s in o]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.semilogy(t, sr_h, marker="o", color="0.15", markersize=3.5, linewidth=1.2, label="hierarchy-throttled lift")
    ax.semilogy(t, sr_o, marker="s", color="0.55", markersize=3.5, linewidth=1.2, label="open shell flux")
    ax.set_xlabel("lifted-shell time")
    ax.set_ylabel("shadow residual")
    ax.set_title("Lifted-shell projection: quotient-shadow residual")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "lifted_shell_shadow_residual.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--figdir", type=Path, default=Path("figures"))
    ap.add_argument("--K", type=int, default=18)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--dt", type=float, default=3e-4)
    ap.add_argument("--nu", type=float, default=1e-5)
    ap.add_argument("--forcing", type=float, default=1.3e-2)
    ap.add_argument("--cascade-rate", type=float, default=8.0)
    ap.add_argument("--hierarchy-strength", type=float, default=3.0)
    ap.add_argument("--front-shell", type=int, default=6)
    args = ap.parse_args()
    cfg = ShellConfig(K=args.K, steps=args.steps, dt=args.dt, nu=args.nu, forcing=args.forcing, cascade_rate=args.cascade_rate, hierarchy_strength=args.hierarchy_strength, front_shell=args.front_shell)
    result = run(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    if args.figures:
        make_figures(result, args.figdir)
    print(json.dumps({
        "status": result["status"],
        "hierarchical_tail": result["hierarchical_tail"]["metrics"],
        "open_flux_tail": result["open_flux_tail"]["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
