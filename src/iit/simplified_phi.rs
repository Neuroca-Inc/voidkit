// Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
//
// This research is protected under a dual-license to foster open academic
// research while ensuring commercial applications are aligned with the project's ethical principles.
// Commercial use requires written permission from Justin K. Lietz.
// See LICENSE file for full terms.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Calculates a simplified version of the integrated information (Φ) metric.
///
/// The formula is: Φ = Σ_i,j log₂(1 + |I(i,j)| / H(j))
/// where:
/// - I(i,j) = w_ij × spike_rate_j (information flow from i to j)
/// - H(j) = -p_j × log₂(p_j) - (1 - p_j) × log₂(1 - p_j) (binary entropy)
///
/// # Arguments
///
/// * `weights` - 2D array of synaptic weights (w_ij), shape (n_neurons, n_neurons)
/// * `spike_rates` - 1D array of spike rates for each neuron, shape (n_neurons,)
/// * `spike_probabilities` - 1D array of spike probabilities for each neuron, shape (n_neurons,)
///
/// # Returns
///
/// The calculated simplified Φ value as a float
///
/// # Example
///
/// ```
/// use ndarray::{array, Array1, Array2};
/// use voidkit_rust::iit::calculate_simplified_phi;
///
/// let weights = array![[0.5, 0.3], [0.2, 0.4]];
/// let spike_rates = array![0.8, 0.6];
/// let spike_probabilities = array![0.5, 0.5];
///
/// let phi = calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
/// assert!(phi > 0.0);
/// ```
pub fn calculate_simplified_phi(
    weights: &Array2<f64>,
    spike_rates: &Array1<f64>,
    spike_probabilities: &Array1<f64>,
) -> f64 {
    let n_neurons = weights.nrows();
    
    // Validate input dimensions
    assert_eq!(weights.ncols(), n_neurons, "Weight matrix must be square");
    assert_eq!(spike_rates.len(), n_neurons, "Spike rates length must match neuron count");
    assert_eq!(spike_probabilities.len(), n_neurons, "Spike probabilities length must match neuron count");
    
    let mut phi = 0.0;
    
    for j in 0..n_neurons {
        // Calculate binary entropy H(j) for neuron j
        let p_j = spike_probabilities[j];
        let h_j = if p_j > 0.0 && p_j < 1.0 {
            -(p_j * p_j.log2() + (1.0 - p_j) * (1.0 - p_j).log2())
        } else {
            0.0
        };
        
        // Only process if entropy is positive
        if h_j > 0.0 {
            for i in 0..n_neurons {
                // Calculate information flow I(i,j)
                let i_ij = weights[[i, j]] * spike_rates[j];
                
                // Add contribution to total Phi
                phi += (1.0 + i_ij.abs() / h_j).log2();
            }
        }
    }
    
    phi
}

/// Python wrapper for calculate_simplified_phi
#[pyfunction]
#[pyo3(name = "calculate_simplified_phi")]
pub fn calculate_simplified_phi_py<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyArray2<f64>>,
    spike_rates: &Bound<'py, PyArray1<f64>>,
    spike_probabilities: &Bound<'py, PyArray1<f64>>,
) -> PyResult<f64> {
    // Convert numpy arrays to ndarray
    let weights_array = weights.readonly().as_array().to_owned();
    let spike_rates_array = spike_rates.readonly().as_array().to_owned();
    let spike_probabilities_array = spike_probabilities.readonly().as_array().to_owned();
    
    // Call the Rust function
    Ok(calculate_simplified_phi(&weights_array, &spike_rates_array, &spike_probabilities_array))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_calculate_simplified_phi_basic() {
        // Simple 2-neuron system with uniform probabilities
        let weights = array![[0.5, 0.3], [0.2, 0.4]];
        let spike_rates = array![0.8, 0.6];
        let spike_probabilities = array![0.5, 0.5];
        
        let phi = calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
        
        // Phi should be positive for connected system
        assert!(phi > 0.0);
        
        // Manual calculation for verification:
        // H(0) = H(1) = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        // I(0,0) = 0.5 * 0.8 = 0.4, I(1,0) = 0.2 * 0.8 = 0.16
        // I(0,1) = 0.3 * 0.6 = 0.18, I(1,1) = 0.4 * 0.6 = 0.24
        // phi = log2(1 + 0.4/1) + log2(1 + 0.16/1) + log2(1 + 0.18/1) + log2(1 + 0.24/1)
        let expected = (1.4_f64).log2() + (1.16_f64).log2() + (1.18_f64).log2() + (1.24_f64).log2();
        assert_relative_eq!(phi, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_simplified_phi_zero_probabilities() {
        // Test with neurons that have zero or one probabilities (no entropy)
        let weights = array![[0.5, 0.3], [0.2, 0.4]];
        let spike_rates = array![0.8, 0.6];
        let spike_probabilities = array![0.0, 1.0];
        
        let phi = calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
        
        // Should be zero since all neurons have zero entropy
        assert_relative_eq!(phi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_simplified_phi_isolated_system() {
        // Test with no connections (zero weights)
        let weights = array![[0.0, 0.0], [0.0, 0.0]];
        let spike_rates = array![0.8, 0.6];
        let spike_probabilities = array![0.5, 0.5];
        
        let phi = calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
        
        // With zero weights, all I(i,j) = 0, so phi = n*n*log2(1) = 0
        assert_relative_eq!(phi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_simplified_phi_asymmetric() {
        // Test with asymmetric connections and different probabilities
        let weights = array![[0.8, 0.1], [0.0, 0.6]];
        let spike_rates = array![1.0, 0.5];
        let spike_probabilities = array![0.3, 0.7];
        
        let phi = calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
        
        // Should be positive and reflect asymmetric structure
        assert!(phi > 0.0);
        
        // H(0) = -0.3*log2(0.3) - 0.7*log2(0.7) ≈ 0.881
        // H(1) = -0.7*log2(0.7) - 0.3*log2(0.3) ≈ 0.881
        let h_0 = -(0.3 * 0.3_f64.log2() + 0.7 * 0.7_f64.log2());
        let h_1 = -(0.7 * 0.7_f64.log2() + 0.3 * 0.3_f64.log2());
        
        // I(0,0) = 0.8 * 1.0 = 0.8, I(1,0) = 0.0 * 1.0 = 0.0
        // I(0,1) = 0.1 * 0.5 = 0.05, I(1,1) = 0.6 * 0.5 = 0.3
        let expected = (1.0 + 0.8 / h_0).log2() + (1.0 + 0.0 / h_0).log2()
                     + (1.0 + 0.05 / h_1).log2() + (1.0 + 0.3 / h_1).log2();
        
        assert_relative_eq!(phi, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_simplified_phi_large_system() {
        // Test with larger system (5 neurons)
        let weights = array![
            [0.1, 0.2, 0.3, 0.1, 0.0],
            [0.2, 0.1, 0.1, 0.2, 0.1],
            [0.0, 0.3, 0.2, 0.1, 0.2],
            [0.1, 0.0, 0.2, 0.1, 0.3],
            [0.2, 0.1, 0.0, 0.2, 0.1]
        ];
        let spike_rates = array![0.7, 0.8, 0.6, 0.9, 0.5];
        let spike_probabilities = array![0.4, 0.5, 0.6, 0.3, 0.7];
        
        let phi = calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
        
        // Should be positive for this connected system
        assert!(phi > 0.0);
        // Should scale with system size
        assert!(phi > 2.0);  // Rough sanity check
    }

    #[test]
    #[should_panic(expected = "Weight matrix must be square")]
    fn test_invalid_weight_dimensions() {
        let weights = array![[0.5, 0.3, 0.2], [0.2, 0.4, 0.1]];
        let spike_rates = array![0.8, 0.6];
        let spike_probabilities = array![0.5, 0.5];
        
        calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
    }

    #[test]
    #[should_panic(expected = "Spike rates length must match neuron count")]
    fn test_invalid_spike_rates_length() {
        let weights = array![[0.5, 0.3], [0.2, 0.4]];
        let spike_rates = array![0.8, 0.6, 0.7];
        let spike_probabilities = array![0.5, 0.5];
        
        calculate_simplified_phi(&weights, &spike_rates, &spike_probabilities);
    }
}
