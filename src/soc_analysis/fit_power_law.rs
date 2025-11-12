/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

SOC (Self-Organized Criticality) analysis module - power law fitting.
*/

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;

/// Fits a power-law distribution to data using linear regression on log-log plot.
///
/// # Arguments
///
/// * `data` - A 1D array of the data to be fitted (e.g., avalanche sizes)
///
/// # Returns
///
/// A tuple containing:
/// * The exponent of the power law
/// * The R-squared value of the fit
///
/// # Examples
///
/// ```python
/// from voidkit_rust import fit_power_law
/// import numpy as np
///
/// # Generate power-law distributed data
/// data = np.random.pareto(2.0, 1000) + 1
/// exponent, r_squared = fit_power_law(data)
/// print(f"Power law exponent: {exponent}, R²: {r_squared}")
/// ```
#[pyfunction]
pub fn fit_power_law(
    data: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    let data_array = data.as_array();
    
    if data_array.len() < 10 {
        return Err(PyValueError::new_err("Data must have at least 10 points"));
    }
    
    // Find min and max for binning
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    
    for &val in data_array.iter() {
        if val > 0.0 {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }
    
    if min_val == f64::INFINITY || max_val == f64::NEG_INFINITY {
        return Err(PyValueError::new_err("Data must contain positive values"));
    }
    
    // Create log-spaced bins
    let n_bins = (data_array.len() / 10).max(10);
    let log_min = min_val.ln();
    let log_max = max_val.ln();
    let log_step = (log_max - log_min) / n_bins as f64;
    
    // Create histogram
    let mut counts = vec![0; n_bins];
    let mut bin_centers = Vec::new();
    
    for i in 0..n_bins {
        let log_edge = log_min + i as f64 * log_step;
        let log_edge_next = log_min + (i + 1) as f64 * log_step;
        let center = ((log_edge + log_edge_next) / 2.0).exp();
        bin_centers.push(center);
    }
    
    // Fill histogram
    for &val in data_array.iter() {
        if val > 0.0 {
            let log_val = val.ln();
            let bin_idx = ((log_val - log_min) / log_step) as usize;
            if bin_idx < n_bins {
                counts[bin_idx] += 1;
            }
        }
    }
    
    // Filter non-zero bins for log-log fit
    let mut log_x = Vec::new();
    let mut log_y = Vec::new();
    
    for i in 0..n_bins {
        if counts[i] > 0 {
            log_x.push(bin_centers[i].log10());
            log_y.push((counts[i] as f64).log10());
        }
    }
    
    if log_x.len() < 3 {
        return Err(PyValueError::new_err("Not enough non-zero bins for fitting"));
    }
    
    // Linear regression on log-log data
    let n = log_x.len() as f64;
    let sum_x: f64 = log_x.iter().sum();
    let sum_y: f64 = log_y.iter().sum();
    let sum_xy: f64 = log_x.iter().zip(&log_y).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_x.iter().map(|x| x * x).sum();
    
    // Calculate slope (exponent)
    let exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - exponent * sum_x) / n;
    
    // Calculate R-squared
    let y_mean = sum_y / n;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    
    for i in 0..log_x.len() {
        let y_pred = exponent * log_x[i] + intercept;
        ss_tot += (log_y[i] - y_mean).powi(2);
        ss_res += (log_y[i] - y_pred).powi(2);
    }
    
    let r_squared = 1.0 - (ss_res / ss_tot);
    
    Ok((exponent, r_squared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_linear_regression() {
        // Test linear regression calculation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(&y).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        assert_relative_eq!(slope, 2.0, epsilon = 1e-10);
    }
}
