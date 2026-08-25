#!/usr/bin/env python3
"""
CEG Instrumentation — Base Adapter (Abstract)

Implement this interface to plug your model into the CEG measurement instrument.

The adapter decouples the CEG protocol from any specific dynamical system.
You provide: how your model evolves forward, reverses, generates corrections,
and measures distances.  The instrument handles the rest.

Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class EchoAdapter(ABC):
    """Abstract base class for CEG echo adapters.

    A "state" is a dict whose internal structure is entirely up to you.
    The only requirement is that the methods below are consistent with
    each other: the norm measures the same thing the integrators evolve,
    and corrections live in the same space.

    Minimal state contract:
        state = {"fields": <your_data>, ...}

    You can add any auxiliary keys you need (e.g., hidden states, time
    indices, metadata).
    """

    @abstractmethod
    def initial_state(self, seed: int) -> Dict[str, Any]:
        """Generate a deterministic initial state from the given seed.

        This should produce a reproducible state.  Use the seed to
        initialize any random components of the initial condition.

        Returns
        -------
        dict
            State dictionary.
        """
        ...

    @abstractmethod
    def forward_step(self, state: Dict[str, Any], dt: float) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Advance the state by one forward time step.

        Parameters
        ----------
        state : dict
            Current state.
        dt : float
            Time step size.

        Returns
        -------
        tuple[dict, dict]
            (new_state, diagnostics).
            diagnostics should include at minimum:
              - "delta_sigma": change in entropy/Lyapunov (for G2)
                Set to 0.0 if your model has no dissipative sector.
        """
        ...

    @abstractmethod
    def reverse_step(self, state: Dict[str, Any], dt: float) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Advance the state by one *reverse* time step.

        This is the time-reversed dynamics.  For a Hamiltonian system,
        this is just forward_step with dt → −dt (and possibly flipping
        momenta).  For a dissipative system, this reverses the
        conservative sector but keeps dissipation in the forward direction.

        Returns
        -------
        tuple[dict, dict]
            (new_state, diagnostics).
        """
        ...

    @abstractmethod
    def energy_norm_delta(self, state_a: Dict[str, Any], state_b: Dict[str, Any]) -> float:
        """Compute ||state_a − state_b|| in your model's energy norm.

        This defines how echo error is measured.  It should be:
        - Non-negative
        - Zero iff state_a == state_b
        - Consistent with your model's conserved quantities

        For a field theory this is typically:
            sqrt( ∫ [δπ² + c²(∇δφ)² + m²δφ²] dx )

        For a neural ODE / Liquid model this might be:
            sqrt( Σ_i (h_a[i] − h_b[i])² )
        weighted by your model's metric tensor.

        Returns
        -------
        float
            Non-negative scalar.
        """
        ...

    @abstractmethod
    def random_correction(
        self, state: Dict[str, Any], budget: float, rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Generate a random correction with the given energy budget.

        The correction should have ||correction||_H = budget (in the
        same norm as energy_norm_delta).  This is the null hypothesis:
        if you inject random noise with the same energy as your
        structured assistance, does the echo improve?

        Parameters
        ----------
        state : dict
            Current state (for computing the correction direction).
        budget : float
            Target energy norm of the correction.
        rng : np.random.Generator
            Seeded RNG for reproducibility.

        Returns
        -------
        dict
            Correction state dict (same structure as state).
        """
        ...

    @abstractmethod
    def assisted_correction(
        self, state: Dict[str, Any], target: Dict[str, Any],
        budget: float, rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Generate a *structured* correction toward the target state.

        This is your model's "assistance mechanism."  It knows the
        target state and can steer the correction intelligently, but
        must respect the same energy budget as the random correction.

        For a field theory: project (target − current) onto the energy
        ball of radius = budget.

        For a Liquid AI model: use the model's internal estimate of
        the gradient toward target, clamped to the budget.

        Parameters
        ----------
        state : dict
            Current state during reverse pass.
        target : dict
            The initial state we're trying to recover (the echo target).
        budget : float
            Energy budget (same as random_correction).
        rng : np.random.Generator
            Seeded RNG.

        Returns
        -------
        dict
            Correction state dict.
        """
        ...

    def apply_correction(self, state: Dict[str, Any], correction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a correction to the state.

        Default: element-wise addition of 'fields'.  Override if your
        model needs something else (e.g., constrained manifold projection).
        """
        new_state = dict(state)
        if "fields" in state and "fields" in correction:
            new_state["fields"] = {
                k: state["fields"][k] + correction["fields"].get(k, 0.0)
                for k in state["fields"]
            }
        return new_state

    def correction_work(self, correction: Dict[str, Any]) -> float:
        """Compute the energy cost of a correction.

        Default: returns the 'budget' key if present, else 0.0.
        Override for precise energy accounting.
        """
        return float(correction.get("budget", 0.0))

    def copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a state dict.

        Default: copies 'fields' with np.copy, shallow-copies the rest.
        Override if your state has non-trivial structure.
        """
        new = dict(state)
        if "fields" in state:
            new["fields"] = {
                k: np.copy(v) if isinstance(v, np.ndarray) else v
                for k, v in state["fields"].items()
            }
        return new

    def calibration_gates(
        self, state0: Dict[str, Any], dt: float, steps: int,
    ) -> Dict[str, Any]:
        """Run RP-1 calibration: instrument gate diagnostics.

        Default implementation returns all-pass with zeros (override
        to provide real diagnostics from your model).

        Should return a dict with at least:
          - G1_passed, time_rev_drift, g1_tol
          - G2_passed, delta_sigma_min, g2_tol
          - G4_passed, slope, R2

        Returns
        -------
        dict
            Gate diagnostics.
        """
        return {
            "G1_passed": True,
            "time_rev_drift": 0.0,
            "g1_tol": 1e-12,
            "G2_passed": True,
            "delta_sigma_min": 0.0,
            "g2_tol": 1e-12,
            "G4_passed": True,
            "slope": 3.0,
            "R2": 1.0,
        }
