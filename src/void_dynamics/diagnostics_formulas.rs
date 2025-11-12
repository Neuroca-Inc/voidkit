/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Diagnostic Formula Implementations
//! 
//! Provides pure, canonical implementations of the mathematical formulas
//! for the Introspection Probe's pathology detection and the ADC's adaptive scheduling.

use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

/// Calculates the pathology score for a locus (subgraph) to identify
/// inefficient, high-activity, low-output regions.
/// 
/// Ref: Blueprint Rule 4.1
/// Time Complexity: O(k) where k is the number of nodes in the locus
/// 
/// # Arguments
/// * `spike_rates` - Array of firing rates for neurons in the locus
/// * `output_diversity` - Array of output diversity values (0.0 to 1.0)
/// 
/// # Returns
/// Mean pathology score: mean(spike_rates * (1 - output_diversity))
#[pyfunction]
#[pyo3(signature = (spike_rates, output_diversity))]
pub fn calculate_pathology_score(
    spike_rates: PyReadonlyArray1<f64>,
    output_diversity: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let rates = spike_rates.as_array();
    let diversity = output_diversity.as_array();
    
    if rates.len() != diversity.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "spike_rates and output_diversity must have the same length"
        ));
    }
    
    if rates.is_empty() {
        return Ok(0.0);
    }
    
    let sum: f64 = rates.iter()
        .zip(diversity.iter())
        .map(|(r, d)| r * (1.0 - d))
        .sum();
    
    Ok(sum / rates.len() as f64)
}

/// Calculates the entropy of the graph based on its degree distribution.
/// 
/// Used by Introspection Probe for global health monitoring and ADC for scheduling.
/// 
/// Ref: Blueprint Rule 4.1
/// Time Complexity: O(N) where N is the number of nodes in the graph
/// 
/// # Arguments
/// * `degree_distribution` - Array of node degrees (must be non-negative)
/// 
/// # Returns
/// Shannon entropy: -sum(p * log(p)) where p is normalized degree distribution
#[pyfunction]
#[pyo3(signature = (degree_distribution))]
pub fn calculate_graph_entropy(
    degree_distribution: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let degrees = degree_distribution.as_array();
    
    if degrees.is_empty() {
        return Ok(0.0);
    }
    
    // Normalize to get probabilities
    let total: f64 = degrees.iter().sum();
    
    if total <= 0.0 {
        return Ok(0.0);
    }
    
    // Calculate entropy, filtering out zero probabilities
    let entropy: f64 = degrees.iter()
        .filter(|&&d| d > 0.0)
        .map(|&d| {
            let p = d / total;
            -p * p.ln()
        })
        .sum();
    
    Ok(entropy)
}

/// Calculates the timestep for the next scheduled ADC cartography event,
/// based on the current graph entropy.
/// 
/// Ref: Blueprint Rule 7
/// Time Complexity: O(1)
/// 
/// # Arguments
/// * `graph_entropy` - Current graph entropy value
/// * `alpha` - Scaling parameter (higher alpha = more sensitive to entropy)
/// * `base_interval` - Base interval in timesteps (default: 100000)
/// 
/// # Returns
/// Time interval until next cartography event (in timesteps)
#[pyfunction]
#[pyo3(signature = (graph_entropy, alpha, base_interval = 100000))]
pub fn calculate_cartography_time(
    graph_entropy: f64,
    alpha: f64,
    base_interval: i32,
) -> PyResult<i32> {
    let t_territory = (base_interval as f64) * (-alpha * graph_entropy).exp();
    Ok(t_territory as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_calculate_pathology_score() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let rates = vec![1.0, 2.0, 3.0];
            let diversity = vec![0.5, 0.5, 0.5];
            
            let rates_py = numpy::PyArray1::from_vec_bound(py, rates);
            let diversity_py = numpy::PyArray1::from_vec_bound(py, diversity);
            
            let score = calculate_pathology_score(
                rates_py.readonly(),
                diversity_py.readonly()
            ).unwrap();
            
            // Expected: mean((1.0 * 0.5) + (2.0 * 0.5) + (3.0 * 0.5)) = mean(0.5 + 1.0 + 1.5) = 1.0
            assert_relative_eq!(score, 1.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_calculate_graph_entropy() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Uniform distribution should have maximum entropy
            let uniform = vec![1.0, 1.0, 1.0, 1.0];
            let uniform_py = numpy::PyArray1::from_vec_bound(py, uniform);
            
            let entropy = calculate_graph_entropy(uniform_py.readonly()).unwrap();
            
            // For uniform distribution: -4 * (0.25 * ln(0.25)) = ln(4) ≈ 1.386
            assert_relative_eq!(entropy, 4.0_f64.ln(), epsilon = 1e-6);
        });
    }

    #[test]
    fn test_calculate_cartography_time() {
        // High entropy should give longer intervals
        let t_high = calculate_cartography_time(2.0, 0.5, 100000).unwrap();
        
        // Low entropy should give shorter intervals
        let t_low = calculate_cartography_time(0.5, 0.5, 100000).unwrap();
        
        assert!(t_high < t_low);
        assert!(t_high > 0);
    }
}
