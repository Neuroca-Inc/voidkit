/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! RE-VGSP Learning Rule Formula Implementations
//! 
//! Provides pure, canonical implementations of the mathematical formulas
//! for the RE-VGSP (Resonance-Enhanced Void-Guided Structural Plasticity) learning rule.

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

/// Calculates the effective learning rate, modulated by the global reward signal.
/// 
/// A positive reward enables learning; a negative reward could enable anti-learning.
/// 
/// Ref: Blueprint Rule 2, `eta_effective(total_reward)`
/// Time Complexity: O(1)
/// 
/// # Arguments
/// * `base_eta` - Base learning rate
/// * `total_reward` - Global reward signal from SIE
/// 
/// # Returns
/// Effective learning rate: base_eta * total_reward
#[pyfunction]
#[pyo3(signature = (base_eta, total_reward))]
pub fn calculate_modulated_learning_rate(
    base_eta: f64,
    total_reward: f64,
) -> PyResult<f64> {
    Ok(base_eta * total_reward)
}

/// Calculates the effective eligibility trace decay factor, modulated by the
/// local network resonance (Phase-Locking Value).
/// 
/// High resonance (high PLV) leads to more stable traces (higher gamma, closer to 1.0).
/// 
/// Ref: Blueprint Rule 2, `gamma(PLV)`
/// Time Complexity: O(1)
/// 
/// # Arguments
/// * `base_gamma` - Base trace decay factor
/// * `plv` - Phase-Locking Value (0.0 to 1.0)
/// 
/// # Returns
/// Effective gamma: base_gamma * (0.5 + 0.5 * plv)
#[pyfunction]
#[pyo3(signature = (base_gamma, plv))]
pub fn calculate_modulated_trace_decay(
    base_gamma: f64,
    plv: f64,
) -> PyResult<f64> {
    // PLV of 1.0 (perfect sync) -> gamma is base_gamma
    // PLV of 0.0 (no sync) -> gamma is base_gamma * 0.5 (faster decay)
    Ok(base_gamma * (0.5 + 0.5 * plv))
}

/// Calculates the phase-sensitive Plasticity Impulse (PI) for a batch of
/// pre-post spike pairs.
/// 
/// Ref: Blueprint Rule 8.1
/// Time Complexity: O(k) where k is the number of spike pairs
/// 
/// # Arguments
/// * `delta_t` - Time differences between pre and post spikes (ms)
/// * `phase_pre` - Phase values of pre-synaptic spikes (radians)
/// * `phase_post` - Phase values of post-synaptic spikes (radians)
/// 
/// # Returns
/// Array of plasticity impulse values
#[pyfunction]
#[pyo3(signature = (delta_t, phase_pre, phase_post))]
pub fn calculate_plasticity_impulse<'py>(
    py: Python<'py>,
    delta_t: PyReadonlyArray1<f64>,
    phase_pre: PyReadonlyArray1<f64>,
    phase_post: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let delta_t = delta_t.as_array();
    let phase_pre = phase_pre.as_array();
    let phase_post = phase_post.as_array();
    
    if delta_t.len() != phase_pre.len() || delta_t.len() != phase_post.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input arrays must have the same length"
        ));
    }
    
    let mut result = Vec::with_capacity(delta_t.len());
    
    for i in 0..delta_t.len() {
        let base_pi = (-delta_t[i].abs() / 10.0).exp();
        let phase_diff_cos = (phase_pre[i] - phase_post[i]).cos();
        let phase_modulation = (1.0 + phase_diff_cos) / 2.0;
        result.push(base_pi * phase_modulation);
    }
    
    Ok(PyArray1::from_vec_bound(py, result))
}

/// Updates the eligibility traces for a batch of synapses.
/// 
/// Ref: Blueprint Rule 2 & 2.1
/// Time Complexity: O(N) where N is number of synapses
/// 
/// # Arguments
/// * `e_ij_prev` - Previous eligibility trace values
/// * `pi` - Plasticity impulse values
/// * `gamma_eff` - Effective trace decay factor
/// 
/// # Returns
/// Updated eligibility traces: gamma_eff * e_ij_prev + pi
#[pyfunction]
#[pyo3(signature = (e_ij_prev, pi, gamma_eff))]
pub fn update_eligibility_trace<'py>(
    py: Python<'py>,
    e_ij_prev: PyReadonlyArray1<f64>,
    pi: PyReadonlyArray1<f64>,
    gamma_eff: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let e_prev = e_ij_prev.as_array();
    let pi_arr = pi.as_array();
    
    if e_prev.len() != pi_arr.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "e_ij_prev and pi must have the same length"
        ));
    }
    
    let result: Vec<f64> = e_prev.iter()
        .zip(pi_arr.iter())
        .map(|(e, p)| gamma_eff * e + p)
        .collect();
    
    Ok(PyArray1::from_vec_bound(py, result))
}

/// Calculates the final weight change for a batch of synapses using the
/// effective learning rate and eligibility traces.
/// 
/// Ref: Blueprint Rule 2
/// Time Complexity: O(N) where N is number of synapses
/// 
/// # Arguments
/// * `e_ij` - Eligibility trace values
/// * `w_ij` - Current weight values
/// * `eta_eff` - Effective learning rate
/// * `lambda_decay` - Weight decay parameter
/// 
/// # Returns
/// Weight changes: eta_eff * e_ij - lambda_decay * w_ij
#[pyfunction]
#[pyo3(signature = (e_ij, w_ij, eta_eff, lambda_decay))]
pub fn calculate_weight_change<'py>(
    py: Python<'py>,
    e_ij: PyReadonlyArray1<f64>,
    w_ij: PyReadonlyArray1<f64>,
    eta_eff: f64,
    lambda_decay: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let e = e_ij.as_array();
    let w = w_ij.as_array();
    
    if e.len() != w.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "e_ij and w_ij must have the same length"
        ));
    }
    
    let result: Vec<f64> = e.iter()
        .zip(w.iter())
        .map(|(e_val, w_val)| {
            let reinforcement = eta_eff * e_val;
            let decay = lambda_decay * w_val;
            reinforcement - decay
        })
        .collect();
    
    Ok(PyArray1::from_vec_bound(py, result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_calculate_modulated_learning_rate() {
        let eta = calculate_modulated_learning_rate(0.01, 2.0).unwrap();
        assert_relative_eq!(eta, 0.02, epsilon = 1e-10);
        
        let eta_neg = calculate_modulated_learning_rate(0.01, -1.0).unwrap();
        assert_relative_eq!(eta_neg, -0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_modulated_trace_decay() {
        // High PLV (perfect sync) should give full base_gamma
        let gamma_high = calculate_modulated_trace_decay(0.9, 1.0).unwrap();
        assert_relative_eq!(gamma_high, 0.9, epsilon = 1e-10);
        
        // Low PLV (no sync) should give reduced gamma
        let gamma_low = calculate_modulated_trace_decay(0.9, 0.0).unwrap();
        assert_relative_eq!(gamma_low, 0.45, epsilon = 1e-10);
    }
}
