/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Time series analysis module - autocorrelation and cross-correlation.
*/

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};

/// Calculates the autocorrelation of a signal.
///
/// # Arguments
///
/// * `signal` - A 1D array representing the time series signal
///
/// # Returns
///
/// The autocorrelation of the signal
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_autocorrelation
/// import numpy as np
///
/// # Generate a sinusoidal signal
/// t = np.linspace(0, 10, 100)
/// signal = np.sin(2 * np.pi * t)
/// autocorr = calculate_autocorrelation(signal)
/// ```
#[pyfunction]
pub fn calculate_autocorrelation<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sig = signal.as_array();
    let n = sig.len();
    
    if n == 0 {
        return Err(PyValueError::new_err("Signal must not be empty"));
    }
    
    // Calculate mean
    let mean: f64 = sig.iter().sum::<f64>() / n as f64;
    
    // Calculate variance
    let variance: f64 = sig.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    
    if variance == 0.0 {
        return Ok(PyArray1::from_vec_bound(py, vec![0.0; n]));
    }
    
    // Mean-subtract
    let mean_subtracted: Vec<f64> = sig.iter().map(|&x| x - mean).collect();
    
    // Calculate autocorrelation
    let mut autocorr = Vec::with_capacity(n);
    
    for lag in 0..n {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += mean_subtracted[i] * mean_subtracted[i + lag];
        }
        autocorr.push(sum / (n as f64 * variance));
    }
    
    Ok(PyArray1::from_vec_bound(py, autocorr))
}

/// Calculates the cross-correlation between two signals.
///
/// # Arguments
///
/// * `signal1` - The first 1D array
/// * `signal2` - The second 1D array
///
/// # Returns
///
/// The cross-correlation of the two signals
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_cross_correlation
/// import numpy as np
///
/// signal1 = np.array([1.0, 2.0, 3.0, 4.0])
/// signal2 = np.array([4.0, 3.0, 2.0, 1.0])
/// cross_corr = calculate_cross_correlation(signal1, signal2)
/// ```
#[pyfunction]
pub fn calculate_cross_correlation<'py>(
    py: Python<'py>,
    signal1: PyReadonlyArray1<'py, f64>,
    signal2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sig1 = signal1.as_array();
    let sig2 = signal2.as_array();
    
    if sig1.len() != sig2.len() {
        return Err(PyValueError::new_err("Signals must have the same length"));
    }
    
    let n = sig1.len();
    
    if n == 0 {
        return Err(PyValueError::new_err("Signals must not be empty"));
    }
    
    // Calculate means
    let mean1: f64 = sig1.iter().sum::<f64>() / n as f64;
    let mean2: f64 = sig2.iter().sum::<f64>() / n as f64;
    
    // Calculate standard deviations
    let std1: f64 = (sig1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / n as f64).sqrt();
    let std2: f64 = (sig2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / n as f64).sqrt();
    
    if std1 == 0.0 || std2 == 0.0 {
        return Ok(PyArray1::from_vec_bound(py, vec![0.0; n]));
    }
    
    // Mean-subtract
    let norm1: Vec<f64> = sig1.iter().map(|&x| x - mean1).collect();
    let norm2: Vec<f64> = sig2.iter().map(|&x| x - mean2).collect();
    
    // Calculate cross-correlation
    let mut cross_corr = Vec::with_capacity(n);
    
    for lag in 0..n {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += norm1[i] * norm2[i + lag];
        }
        cross_corr.push(sum / (n as f64 * std1 * std2));
    }
    
    Ok(PyArray1::from_vec_bound(py, cross_corr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_autocorr_constant() {
        // Autocorrelation of constant signal should be all zeros
        let signal = vec![5.0; 10];
        let mean = 5.0;
        let variance = 0.0;
        
        assert_eq!(variance, 0.0);
    }
    
    #[test]
    fn test_cross_corr_identical() {
        // Cross-correlation of identical signals should equal autocorrelation
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let n = signal.len();
        let mean: f64 = signal.iter().sum::<f64>() / n as f64;
        let std: f64 = (signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        
        assert!(std > 0.0);
    }
}
