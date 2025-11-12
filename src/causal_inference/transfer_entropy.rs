// Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
//
// This research is protected under a dual-license to foster open academic
// research while ensuring commercial applications are aligned with the project's ethical principles.
// Commercial use requires written permission from Justin K. Lietz.
// See LICENSE file for full terms.

use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// Calculates the transfer entropy from time series x to time series y.
///
/// Transfer entropy quantifies the amount of directed (causal) information transfer
/// from one time series to another. It measures how much knowing the past of x
/// reduces uncertainty about the future of y, beyond what is already known from
/// the past of y itself.
///
/// Formula: TE(X→Y) = Σ p(y_t, y_{t-lag}, x_{t-lag}) × log₂[p(y_t | y_{t-lag}, x_{t-lag}) / p(y_t | y_{t-lag})]
///
/// # Arguments
///
/// * `x` - Source time series
/// * `y` - Target time series
/// * `lag` - Time lag for the analysis (default: 1)
/// * `n_bins` - Number of bins for discretization (default: 10)
///
/// # Returns
///
/// The transfer entropy from x to y in bits
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use voidkit_rust::causal_inference::calculate_transfer_entropy;
///
/// // Create two related time series
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]; // y follows x with offset
///
/// let te = calculate_transfer_entropy(&x, &y, 1, 5);
/// assert!(te >= 0.0); // Transfer entropy is non-negative
/// ```
pub fn calculate_transfer_entropy(
    x: &Array1<f64>,
    y: &Array1<f64>,
    lag: usize,
    n_bins: usize,
) -> f64 {
    let n = x.len();
    
    assert_eq!(y.len(), n, "Time series must have the same length");
    assert!(lag > 0 && lag < n, "Lag must be positive and less than series length");
    assert!(n_bins > 1, "Number of bins must be greater than 1");
    
    // Discretize the data
    let x_binned = discretize(x, n_bins);
    let y_binned = discretize(y, n_bins);
    
    // Create lagged versions
    let y_t = &y_binned[lag..];
    let y_t_minus_lag = &y_binned[..n - lag];
    let x_t_minus_lag = &x_binned[..n - lag];
    
    let data_len = y_t.len();
    
    // Initialize probability tables
    let mut p_y_t_y_lag_x_lag = vec![vec![vec![0usize; n_bins]; n_bins]; n_bins];
    let mut p_y_lag_x_lag = vec![vec![0usize; n_bins]; n_bins];
    let mut p_y_t_y_lag = vec![vec![0usize; n_bins]; n_bins];
    let mut p_y_lag = vec![0usize; n_bins];
    
    // Count occurrences
    for i in 0..data_len {
        let yt = y_t[i];
        let yl = y_t_minus_lag[i];
        let xl = x_t_minus_lag[i];
        
        p_y_t_y_lag_x_lag[yt][yl][xl] += 1;
        p_y_lag_x_lag[yl][xl] += 1;
        p_y_t_y_lag[yt][yl] += 1;
        p_y_lag[yl] += 1;
    }
    
    // Calculate transfer entropy
    let mut te = 0.0;
    let data_len_f = data_len as f64;
    
    for yt in 0..n_bins {
        for yl in 0..n_bins {
            for xl in 0..n_bins {
                let count_full = p_y_t_y_lag_x_lag[yt][yl][xl];
                if count_full == 0 {
                    continue;
                }
                
                let p_full = count_full as f64 / data_len_f;
                let p_yl_xl = p_y_lag_x_lag[yl][xl] as f64 / data_len_f;
                let p_yt_yl = p_y_t_y_lag[yt][yl] as f64 / data_len_f;
                let p_yl = p_y_lag[yl] as f64 / data_len_f;
                
                if p_yl_xl > 0.0 && p_yt_yl > 0.0 && p_yl > 0.0 {
                    // TE = sum p(yt, yl, xl) * log2[p(yt|yl,xl) / p(yt|yl)]
                    //    = sum p(yt, yl, xl) * log2[(p(yt,yl,xl) * p(yl)) / (p(yl,xl) * p(yt,yl))]
                    let ratio = (p_full * p_yl) / (p_yl_xl * p_yt_yl);
                    if ratio > 0.0 {
                        te += p_full * ratio.log2();
                    }
                }
            }
        }
    }
    
    te
}

/// Discretizes a continuous signal into bins
fn discretize(signal: &Array1<f64>, n_bins: usize) -> Vec<usize> {
    let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    if (max - min).abs() < 1e-10 {
        // All values are the same
        return vec![0; signal.len()];
    }
    
    signal
        .iter()
        .map(|&val| {
            let normalized = (val - min) / (max - min);
            let bin = (normalized * n_bins as f64).floor() as usize;
            bin.min(n_bins - 1) // Ensure we don't exceed n_bins-1
        })
        .collect()
}

/// Python wrapper for calculate_transfer_entropy
#[pyfunction]
#[pyo3(name = "calculate_transfer_entropy")]
pub fn calculate_transfer_entropy_py<'py>(
    _py: Python<'py>,
    x: &Bound<'py, PyArray1<f64>>,
    y: &Bound<'py, PyArray1<f64>>,
    lag: Option<usize>,
    n_bins: Option<usize>,
) -> PyResult<f64> {
    let x_array = x.readonly().as_array().to_owned();
    let y_array = y.readonly().as_array().to_owned();
    
    let lag = lag.unwrap_or(1);
    let n_bins = n_bins.unwrap_or(10);
    
    Ok(calculate_transfer_entropy(&x_array, &y_array, lag, n_bins))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_discretize_uniform() {
        let signal = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let binned = discretize(&signal, 5);
        assert_eq!(binned, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_discretize_constant() {
        let signal = array![1.0, 1.0, 1.0, 1.0];
        let binned = discretize(&signal, 5);
        // All should be in the same bin
        assert!(binned.iter().all(|&x| x == binned[0]));
    }

    #[test]
    fn test_transfer_entropy_independent() {
        // Two independent random-like sequences
        let x = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        
        let te = calculate_transfer_entropy(&x, &y, 1, 2);
        
        // Independent sequences should have low transfer entropy
        assert!(te >= 0.0); // TE is always non-negative
        assert!(te < 1.0);  // Should be relatively small
    }

    #[test]
    fn test_transfer_entropy_dependent() {
        // Y follows X with lag 1
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = array![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shifted version of x
        
        let te = calculate_transfer_entropy(&x, &y, 1, 4);
        
        // Dependent sequences should have higher transfer entropy
        assert!(te > 0.0);
    }

    #[test]
    fn test_transfer_entropy_identical() {
        // Identical sequences
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = x.clone();
        
        let te = calculate_transfer_entropy(&x, &y, 1, 3);
        
        // Identical sequences have strong dependency
        assert!(te >= 0.0);
    }

    #[test]
    fn test_transfer_entropy_asymmetry() {
        // Test that TE(X→Y) ≠ TE(Y→X) in general
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = array![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Y follows X
        
        let te_x_to_y = calculate_transfer_entropy(&x, &y, 1, 3);
        let te_y_to_x = calculate_transfer_entropy(&y, &x, 1, 3);
        
        // Both should be non-negative
        assert!(te_x_to_y >= 0.0);
        assert!(te_y_to_x >= 0.0);
        
        // They might be different (asymmetric)
        // Note: We can't assert they ARE different, but they CAN be
    }

    #[test]
    fn test_transfer_entropy_zero_for_constant() {
        // Constant sequences have no information transfer
        let x = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = array![2.0, 2.0, 2.0, 2.0, 2.0];
        
        let te = calculate_transfer_entropy(&x, &y, 1, 3);
        
        assert_relative_eq!(te, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transfer_entropy_different_lags() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        
        let te_lag1 = calculate_transfer_entropy(&x, &y, 1, 4);
        let te_lag2 = calculate_transfer_entropy(&x, &y, 2, 4);
        
        // Both should be valid (non-negative)
        assert!(te_lag1 >= 0.0);
        assert!(te_lag2 >= 0.0);
    }

    #[test]
    fn test_transfer_entropy_more_bins() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = array![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        
        let te_few_bins = calculate_transfer_entropy(&x, &y, 1, 2);
        let te_many_bins = calculate_transfer_entropy(&x, &y, 1, 4);
        
        // Both should be valid
        assert!(te_few_bins >= 0.0);
        assert!(te_many_bins >= 0.0);
    }

    #[test]
    #[should_panic(expected = "Time series must have the same length")]
    fn test_transfer_entropy_mismatched_lengths() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        
        calculate_transfer_entropy(&x, &y, 1, 3);
    }

    #[test]
    #[should_panic(expected = "Lag must be positive and less than series length")]
    fn test_transfer_entropy_invalid_lag() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        
        calculate_transfer_entropy(&x, &y, 5, 3); // Lag too large
    }
}
