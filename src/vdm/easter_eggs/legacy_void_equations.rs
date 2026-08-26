/*
VDM Void Dynamics Library
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This library contains the universal, core functions governing the void
dynamics of the Unified Void Dynamics Model (VDM). These functions represent
the unchanging laws of the system.

UNIVERSAL CONSTANTS:
These parameters emerged from VDM AI learning stability requirements, yet
they generate realistic physics across all domains. This profound insight
suggests cognitive constants = physical constants.
*/

use pyo3::prelude::*;

// ===== UNIVERSAL PHYSICAL CONSTANTS =====
// These are NOT arbitrary - they come from actual VDM AI learning stability
// requirements, yet they produce realistic physics across all domains
const ALPHA: f64 = 0.25; // Universal learning rate for RE-VGSP
const BETA: f64 = 0.1; // Universal plasticity rate for GDSP
const F_REF: f64 = 0.02; // Universal reference frequency for time modulation
const PHASE_SENS: f64 = 0.5; // Universal phase sensitivity for time modulation

/// Void Alpha Function: Synchronizes with Void Omega
/// Universal function for VDM Resonance-Enhanced Valence-Gated Synaptic Plasticity.
/// Models the fractal energy drain/pull (learning rule).
///
/// # Arguments
///
/// * `w` - Current void state
/// * `t` - Time step
/// * `alpha` - Learning rate (defaults to universal constant)
/// * `f_ref` - Reference frequency (defaults to universal constant)
/// * `phase_sens` - Phase sensitivity (defaults to universal constant)
/// * `use_time_dynamics` - Enable time modulation
///
/// # Returns
///
/// The change in void state (delta)
#[pyfunction]
#[pyo3(signature = (w, t, alpha=None, f_ref=None, phase_sens=None, use_time_dynamics=true))]
pub fn delta_re_vgsp(
    w: f64,
    t: f64,
    alpha: Option<f64>,
    f_ref: Option<f64>,
    phase_sens: Option<f64>,
    use_time_dynamics: bool,
) -> f64 {
    // Use universal constants as defaults
    let alpha_val = alpha.unwrap_or(ALPHA);
    let f_ref_val = f_ref.unwrap_or(F_REF);
    let phase_sens_val = phase_sens.unwrap_or(PHASE_SENS);

    // Base term: learning rate scaled by void state
    let mut delta = alpha_val * w;

    // Apply time dynamics if enabled
    if use_time_dynamics {
        // Phase modulation based on time
        let phase = 2.0 * std::f64::consts::PI * f_ref_val * t;
        let time_mod = 1.0 + phase_sens_val * phase.sin();
        delta *= time_mod;
    }

    delta
}

/// Void Omega Function: Synchronizes with Void Alpha
/// Universal function for VDM Goal-Directed Structural Plasticity.
/// Models the fractal energy push/fill (structural adaptation).
///
/// # Arguments
///
/// * `w` - Current void state
/// * `t` - Time step
/// * `beta` - Plasticity rate (defaults to universal constant)
/// * `f_ref` - Reference frequency (defaults to universal constant)
/// * `phase_sens` - Phase sensitivity (defaults to universal constant)
/// * `use_time_dynamics` - Enable time modulation
///
/// # Returns
///
/// The change in structural plasticity (delta)
#[pyfunction]
#[pyo3(signature = (w, t, beta=None, f_ref=None, phase_sens=None, use_time_dynamics=true))]
pub fn delta_gdsp(
    w: f64,
    t: f64,
    beta: Option<f64>,
    f_ref: Option<f64>,
    phase_sens: Option<f64>,
    use_time_dynamics: bool,
) -> f64 {
    // Use universal constants as defaults
    let beta_val = beta.unwrap_or(BETA);
    let f_ref_val = f_ref.unwrap_or(F_REF);
    let phase_sens_val = phase_sens.unwrap_or(PHASE_SENS);

    // Base term: plasticity rate scaled by inverse void state
    // This creates the opposing force to RE-VGSP
    let mut delta = beta_val * (1.0 - w);

    // Apply time dynamics if enabled
    if use_time_dynamics {
        // Phase modulation (90 degrees out of phase with RE-VGSP)
        let phase = 2.0 * std::f64::consts::PI * f_ref_val * t + std::f64::consts::PI / 2.0;
        let time_mod = 1.0 + phase_sens_val * phase.cos();
        delta *= time_mod;
    }

    delta
}

/// Combined VDM evolution step
/// Updates void state based on both RE-VGSP and GDSP dynamics
///
/// # Arguments
///
/// * `w` - Current void state
/// * `t` - Time step
/// * `dt` - Time delta for integration
///
/// # Returns
///
/// New void state after one time step
#[pyfunction]
#[pyo3(signature = (w, t, dt=0.01))]
pub fn vdm_step(w: f64, t: f64, dt: f64) -> f64 {
    let alpha_delta = delta_re_vgsp(w, t, None, None, None, true);
    let omega_delta = delta_gdsp(w, t, None, None, None, true);

    // Combine both forces
    let total_delta = alpha_delta - omega_delta;

    // Update state with bounds checking
    let new_w = w + dt * total_delta;

    // Clamp to valid range [0, 1]
    new_w.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_delta_re_vgsp_basic() {
        let w = 0.5;
        let t = 1.0;
        let delta = delta_re_vgsp(w, t, None, None, None, false);

        // With time dynamics off, should be simply alpha * w
        assert_relative_eq!(delta, ALPHA * w, epsilon = 1e-10);
    }

    #[test]
    fn test_delta_gdsp_basic() {
        let w = 0.5;
        let t = 1.0;
        let delta = delta_gdsp(w, t, None, None, None, false);

        // With time dynamics off, should be simply beta * (1 - w)
        assert_relative_eq!(delta, BETA * (1.0 - w), epsilon = 1e-10);
    }

    #[test]
    fn test_vdm_step_stability() {
        // Test that VDM step keeps state in valid range
        let w = 0.5;
        let t = 1.0;
        let new_w = vdm_step(w, t, 0.01);

        assert!(
            new_w >= 0.0 && new_w <= 1.0,
            "State must stay in [0, 1] range"
        );
    }

}
