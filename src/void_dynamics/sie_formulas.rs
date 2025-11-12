/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Self-Improvement Engine (SIE) Formula Implementations
//! 
//! Provides pure, canonical implementations of the mathematical formulas
//! for the Self-Improvement Engine (SIE) as specified in the VDM Blueprint.

use pyo3::prelude::*;
use ndarray::Array1;

/// Calculates the Temporal Difference (TD) error for a state transition.
/// 
/// Ref: Blueprint Rule 3 (Component of the SIE)
/// Time Complexity: O(1)
/// 
/// # Arguments
/// * `v_current` - Current state value
/// * `r_external` - External reward
/// * `v_next` - Next state value
/// * `gamma` - Discount factor
/// 
/// # Returns
/// TD error: R_external + (gamma * V_next) - V_current
#[pyfunction]
#[pyo3(signature = (v_current, r_external, v_next, gamma))]
pub fn calculate_td_error(
    v_current: f64,
    r_external: f64,
    v_next: f64,
    gamma: f64,
) -> PyResult<f64> {
    Ok(r_external + (gamma * v_next) - v_current)
}

/// Calculates the novelty score for a state based on its visitation count.
/// 
/// Ref: Blueprint Rule 3 (Component of the SIE)
/// Time Complexity: O(1)
/// 
/// # Arguments
/// * `n_s` - Number of times the state has been visited
/// 
/// # Returns
/// Novelty score: 1.0 / (N_s + epsilon)
#[pyfunction]
#[pyo3(signature = (n_s))]
pub fn calculate_novelty_score(n_s: i32) -> PyResult<f64> {
    let epsilon = 1e-6;
    Ok(1.0 / (n_s as f64 + epsilon))
}

/// Calculates a habituation score based on the frequency of the current
/// input in recent history.
/// 
/// Ref: Blueprint Rule 3 (Component of the SIE)
/// Time Complexity: O(1)
/// 
/// # Arguments
/// * `recent_count` - Count of occurrences in recent history
/// * `history_length` - Length of the history window
/// 
/// # Returns
/// Habituation score: min(recent_count / history_length, 1.0)
#[pyfunction]
#[pyo3(signature = (recent_count, history_length))]
pub fn calculate_habituation_score(
    recent_count: i32,
    history_length: i32,
) -> PyResult<f64> {
    if history_length == 0 {
        return Ok(0.0);
    }
    
    let score = (recent_count as f64) / (history_length as f64);
    Ok(score.min(1.0))
}

/// Calculates the Homeostatic Stability Index (HSI).
/// 
/// Ref: Blueprint Rule 3.1 & VDM Nomenclature
/// Time Complexity: O(N) where N is number of neurons
/// 
/// # Arguments
/// * `firing_rates` - Array of firing rates for each neuron
/// * `target_var` - Target variance for homeostatic regulation
/// 
/// # Returns
/// HSI: 1.0 - (|current_var - target_var| / target_var)
#[pyfunction]
#[pyo3(signature = (firing_rates, target_var))]
pub fn calculate_hsi(
    firing_rates: Vec<f64>,
    target_var: f64,
) -> PyResult<f64> {
    if firing_rates.is_empty() {
        return Ok(0.0);
    }
    
    let arr = Array1::from_vec(firing_rates);
    let mean = arr.mean().unwrap();
    let variance = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap();
    
    let hsi = 1.0 - ((variance - target_var).abs() / target_var);
    Ok(hsi)
}

/// Calculates the composite total_reward signal from its four weighted,
/// normalized components.
/// 
/// Ref: Blueprint Rule 3
/// Time Complexity: O(1)
/// 
/// # Arguments
/// * `w_td` - Weight for TD error component
/// * `td_error_norm` - Normalized TD error
/// * `w_nov` - Weight for novelty component
/// * `novelty_norm` - Normalized novelty score
/// * `w_hab` - Weight for habituation component
/// * `habituation_norm` - Normalized habituation score
/// * `w_hsi` - Weight for HSI component
/// * `hsi_norm` - Normalized HSI
/// 
/// # Returns
/// Total reward signal
#[pyfunction]
#[pyo3(signature = (w_td, td_error_norm, w_nov, novelty_norm, w_hab, habituation_norm, w_hsi, hsi_norm))]
pub fn calculate_total_reward(
    w_td: f64,
    td_error_norm: f64,
    w_nov: f64,
    novelty_norm: f64,
    w_hab: f64,
    habituation_norm: f64,
    w_hsi: f64,
    hsi_norm: f64,
) -> PyResult<f64> {
    let reward = w_td * td_error_norm 
        + w_nov * novelty_norm 
        - w_hab * habituation_norm 
        + w_hsi * hsi_norm;
    
    Ok(reward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_calculate_td_error() {
        let td_error = calculate_td_error(1.0, 0.5, 2.0, 0.9).unwrap();
        assert_relative_eq!(td_error, 0.5 + 0.9 * 2.0 - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_novelty_score() {
        let novelty = calculate_novelty_score(0).unwrap();
        assert!(novelty > 0.0);
        
        let novelty2 = calculate_novelty_score(10).unwrap();
        assert!(novelty2 < novelty);
    }

    #[test]
    fn test_calculate_habituation_score() {
        let hab = calculate_habituation_score(5, 10).unwrap();
        assert_relative_eq!(hab, 0.5, epsilon = 1e-10);
        
        let hab_zero = calculate_habituation_score(0, 0).unwrap();
        assert_eq!(hab_zero, 0.0);
    }

    #[test]
    fn test_calculate_hsi() {
        let rates = vec![1.0, 2.0, 3.0, 4.0];
        let target_var = 1.25; // Actual variance of the data
        let hsi = calculate_hsi(rates, target_var).unwrap();
        assert_relative_eq!(hsi, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_calculate_total_reward() {
        let reward = calculate_total_reward(
            1.0, 0.5,  // TD component
            1.0, 0.3,  // Novelty component
            1.0, 0.2,  // Habituation component (subtracted)
            1.0, 0.1   // HSI component
        ).unwrap();
        
        assert_relative_eq!(reward, 0.5 + 0.3 - 0.2 + 0.1, epsilon = 1e-10);
    }
}
