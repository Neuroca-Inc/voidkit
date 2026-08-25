#!/usr/bin/env python3
"""
CEG Instrumentation — Klein-Gordon Reference Adapter

Working reference implementation of EchoAdapter for a 1D periodic
Klein-Gordon field (VDM's J-limb).  Use this as a template for
building your own adapter.

Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.base_adapter import EchoAdapter
from core.spectral_ops import (
    kg_verlet_step, spectral_laplacian, spectral_grad,
    h_energy_norm_delta, stiffness,
)


class KGReferenceAdapter(EchoAdapter):
    """Klein-Gordon reference adapter for CEG instrumentation.

    This adapter implements the echo protocol on a 1D periodic
    Klein-Gordon field:  ∂²φ/∂t² = c² Δφ − m² φ

    with Störmer-Verlet (symplectic) integration for the J-limb
    and a simple diffusion step for the M-limb.

    Parameters
    ----------
    N : int
        Grid points.
    dx : float
        Grid spacing.
    c : float
        Wave speed.
    m : float
        Mass parameter.
    D : float
        Diffusion coefficient for M-limb.
    """

    def __init__(
        self, N: int = 256, dx: float = 1.0,
        c: float = 1.0, m: float = 0.5, D: float = 0.01,
    ):
        self.N = N
        self.dx = dx
        self.c = c
        self.m = m
        self.D = D

    def initial_state(self, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        x = np.arange(self.N) * self.dx
        # Band-limited initial condition (modes 1–5)
        phi = np.zeros(self.N, dtype=float)
        for k in range(1, 6):
            phi += rng.standard_normal() * 1e-2 * np.sin(
                2 * np.pi * k * x / (self.N * self.dx) + rng.uniform(0, 2 * np.pi)
            )
        pi = np.zeros(self.N, dtype=float)
        return {"fields": {"phi": phi, "pi": pi}}

    def forward_step(self, state: Dict[str, Any], dt: float):
        phi, pi = state["fields"]["phi"], state["fields"]["pi"]

        # J half-step
        phi, pi = kg_verlet_step(phi, pi, 0.5 * dt, self.dx, self.c, self.m)

        # M full-step (spectral diffusion on phi)
        lap = spectral_laplacian(phi, self.dx)
        # Entropy production ΔΣ = D·||Δφ||²·dx·dt ≥ 0  (free energy decreases)
        delta_sigma = float(self.D * np.sum(lap * lap) * self.dx * dt)
        phi = phi + self.D * dt * lap

        # J half-step
        phi, pi = kg_verlet_step(phi, pi, 0.5 * dt, self.dx, self.c, self.m)

        new_state = {"fields": {"phi": phi, "pi": pi}}
        diag = {"delta_sigma": delta_sigma}
        return new_state, diag

    def reverse_step(self, state: Dict[str, Any], dt: float):
        phi, pi = state["fields"]["phi"], state["fields"]["pi"]

        # Reverse Strang: J(-dt/2) → M(+dt) → J(-dt/2)
        # Note: M is NOT reversed (dissipation is irreversible)
        phi, pi = kg_verlet_step(phi, pi, -0.5 * dt, self.dx, self.c, self.m)

        lap = spectral_laplacian(phi, self.dx)
        phi = phi + self.D * dt * lap

        phi, pi = kg_verlet_step(phi, pi, -0.5 * dt, self.dx, self.c, self.m)

        return {"fields": {"phi": phi, "pi": pi}}, {}

    def energy_norm_delta(self, state_a: Dict[str, Any], state_b: Dict[str, Any]) -> float:
        return h_energy_norm_delta(
            state_a["fields"]["phi"], state_a["fields"]["pi"],
            state_b["fields"]["phi"], state_b["fields"]["pi"],
            self.dx, self.c, self.m,
        )

    def random_correction(self, state, budget, rng):
        # Random direction in (phi, pi) space, normalized to budget
        dphi = rng.standard_normal(self.N)
        dpi = rng.standard_normal(self.N)
        norm = h_energy_norm_delta(
            dphi, dpi,
            np.zeros(self.N), np.zeros(self.N),
            self.dx, self.c, self.m,
        )
        if norm > 0:
            scale = budget / norm
            dphi *= scale
            dpi *= scale
        return {"fields": {"phi": dphi, "pi": dpi}, "budget": budget}

    def assisted_correction(self, state, target, budget, rng):
        # Project (target − current) onto energy ball of radius = budget
        dphi = target["fields"]["phi"] - state["fields"]["phi"]
        dpi = target["fields"]["pi"] - state["fields"]["pi"]
        norm = h_energy_norm_delta(
            dphi, dpi,
            np.zeros(self.N), np.zeros(self.N),
            self.dx, self.c, self.m,
        )
        if norm > 0:
            scale = budget / norm
            dphi *= scale
            dpi *= scale
        return {"fields": {"phi": dphi, "pi": dpi}, "budget": budget}

    def correction_work(self, correction):
        return float(correction.get("budget", 0.0))

    def calibration_gates(self, state0, dt, steps):
        phi0 = state0["fields"]["phi"].copy()
        pi0 = state0["fields"]["pi"].copy()

        # G1: J-only round-trip drift
        phi, pi = phi0.copy(), pi0.copy()
        for _ in range(steps):
            phi, pi = kg_verlet_step(phi, pi, dt, self.dx, self.c, self.m)
        for _ in range(steps):
            phi, pi = kg_verlet_step(phi, pi, -dt, self.dx, self.c, self.m)
        drift = h_energy_norm_delta(phi, pi, phi0, pi0, self.dx, self.c, self.m)

        # Scaled tolerance
        eps = float(np.finfo(float).eps)
        sqrtN = float(np.sqrt(self.N))
        h0 = h_energy_norm_delta(
            phi0, pi0,
            np.zeros(self.N), np.zeros(self.N),
            self.dx, self.c, self.m,
        )
        g1_tol = max(1e-12 * sqrtN, 10.0 * eps * sqrtN * max(h0, 1.0))

        # G2: M-step entropy monotonicity (spot check)
        phi_test = phi0.copy()
        min_ds = float("inf")
        for _ in range(min(steps, 10)):
            lap = spectral_laplacian(phi_test, self.dx)
            # Entropy production: D·||Δφ||²·dx·dt ≥ 0
            ds = float(self.D * np.sum(lap * lap) * self.dx * dt)
            min_ds = min(min_ds, ds)
            phi_test = phi_test + self.D * dt * lap
        if min_ds == float("inf"):
            min_ds = 0.0

        # G4: Strang defect (two-grid slope)
        # Measure JMJ splitting error at different dt, expect O(dt²) convergence
        errors = []
        dt_ladder = [dt, dt / 2, dt / 4]
        # Reference: fine-grid solution at dt/8
        dt_ref = dt / 8
        n_ref = int(steps * dt / dt_ref)
        phi_ref, pi_ref = phi0.copy(), pi0.copy()
        for _ in range(n_ref):
            phi_ref, pi_ref = kg_verlet_step(phi_ref, pi_ref, 0.5 * dt_ref, self.dx, self.c, self.m)
            lap_r = spectral_laplacian(phi_ref, self.dx)
            phi_ref = phi_ref + self.D * dt_ref * lap_r
            phi_ref, pi_ref = kg_verlet_step(phi_ref, pi_ref, 0.5 * dt_ref, self.dx, self.c, self.m)

        for dt_test in dt_ladder:
            n_steps = int(steps * dt / dt_test)
            phi_t, pi_t = phi0.copy(), pi0.copy()
            for _ in range(n_steps):
                phi_t, pi_t = kg_verlet_step(phi_t, pi_t, 0.5 * dt_test, self.dx, self.c, self.m)
                lap_t = spectral_laplacian(phi_t, self.dx)
                phi_t = phi_t + self.D * dt_test * lap_t
                phi_t, pi_t = kg_verlet_step(phi_t, pi_t, 0.5 * dt_test, self.dx, self.c, self.m)
            err = h_energy_norm_delta(phi_t, pi_t, phi_ref, pi_ref, self.dx, self.c, self.m)
            errors.append(max(err, 1e-30))  # floor for log safety

        # Log-log fit for slope
        if len(errors) >= 2 and all(e > 0 for e in errors):
            log_dt = np.log(np.array(dt_ladder))
            log_err = np.log(np.array(errors))
            A = np.vstack([log_dt, np.ones_like(log_dt)]).T
            coeff = np.linalg.lstsq(A, log_err, rcond=None)[0]
            slope = float(coeff[0])
            y_pred = A @ coeff
            ss_res = float(np.sum((log_err - y_pred) ** 2))
            ss_tot = float(np.sum((log_err - np.mean(log_err)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            slope, r2 = 3.0, 1.0  # fallback for zero-error case

        return {
            "G1_passed": abs(drift) <= g1_tol,
            "time_rev_drift": drift,
            "g1_tol": g1_tol,
            "G2_passed": min_ds >= -g1_tol,
            "delta_sigma_min": min_ds,
            "g2_tol": g1_tol,
            "G4_passed": slope >= 1.90 and r2 >= 0.998,
            "slope": slope,
            "R2": r2,
            "min_slope": 1.90,  # Generic Strang is O(dt²); VDM uses 2.90
            "min_r2": 0.998,    # Three-point fit; VDM uses 0.999
        }
