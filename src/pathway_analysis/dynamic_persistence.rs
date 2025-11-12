/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArrayMethods};

/// Calculates a dynamic persistence threshold based on input diversity.
///
/// thresh_p(t) = base_threshold - alpha * input_diversity
///
/// # Arguments
/// * `base_threshold` - The base persistence threshold (default: 0.9)
/// * `input_diversity` - A measure of the diversity of recent inputs (default: 0.0)
/// * `alpha` - The scaling factor for input diversity (default: 0.05)
///
/// # Returns
/// The calculated dynamic persistence threshold
///
/// # Example
/// ```
/// let threshold = calculate_dynamic_persistence_threshold(0.9, 0.2, 0.05);
/// assert!((threshold - 0.89).abs() < 1e-6);
/// ```
#[pyfunction]
#[pyo3(signature = (base_threshold=0.9, input_diversity=0.0, alpha=0.05))]
pub fn calculate_dynamic_persistence_threshold(
    base_threshold: f64,
    input_diversity: f64,
    alpha: f64,
) -> f64 {
    base_threshold - alpha * input_diversity
}

/// Calculates a score to predict potential interference with persistent pathways.
///
/// I_score = mean(spike_rates[persistent_paths] * (1 - output_diversity))
///
/// # Arguments
/// * `spike_rates_persistent` - The spike rates of the neurons in the persistent pathways
/// * `output_diversity` - A measure of the diversity of the network's output
///
/// # Returns
/// The calculated interference score
///
/// # Example
/// ```python
/// import numpy as np
/// spike_rates = np.array([0.5, 0.6, 0.7])
/// score = calculate_interference_score(spike_rates, 0.3)
/// ```
#[pyfunction]
pub fn calculate_interference_score(
    spike_rates_persistent: PyReadonlyArray1<f64>,
    output_diversity: f64,
) -> PyResult<f64> {
    let spike_rates = spike_rates_persistent.as_slice()?;
    
    if spike_rates.is_empty() {
        return Ok(0.0);
    }

    let sum: f64 = spike_rates
        .iter()
        .map(|&rate| rate * (1.0 - output_diversity))
        .sum();

    Ok(sum / spike_rates.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_calculate_dynamic_persistence_threshold_default() {
        let result = calculate_dynamic_persistence_threshold(0.9, 0.0, 0.05);
        assert_relative_eq!(result, 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_dynamic_persistence_threshold_with_diversity() {
        let result = calculate_dynamic_persistence_threshold(0.9, 0.2, 0.05);
        assert_relative_eq!(result, 0.89, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_dynamic_persistence_threshold_high_diversity() {
        let result = calculate_dynamic_persistence_threshold(0.9, 1.0, 0.1);
        assert_relative_eq!(result, 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_dynamic_persistence_threshold_negative_result() {
        // Test case where result can go negative
        let result = calculate_dynamic_persistence_threshold(0.5, 10.0, 0.1);
        assert_relative_eq!(result, -0.5, epsilon = 1e-10);
    }
}
