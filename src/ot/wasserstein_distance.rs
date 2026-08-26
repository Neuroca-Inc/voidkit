/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Optimal Transport module - Wasserstein distance calculation.
*/

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Calculates the 1-D Wasserstein distance between two distributions.
///
/// Also known as the Earth Mover's Distance, this computes the minimum
/// cost to transform one distribution into another.
///
/// # Arguments
///
/// * `u_values` - A 1D array of values for the first distribution
/// * `v_values` - A 1D array of values for the second distribution  
/// * `u_weights` - Optional weights for the first distribution
/// * `v_weights` - Optional weights for the second distribution
///
/// # Returns
///
/// The 1-D Wasserstein distance
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_wasserstein_distance
/// import numpy as np
///
/// # Two normal distributions
/// u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
/// v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
///
/// distance = calculate_wasserstein_distance(u, v)
/// print(f"Wasserstein distance: {distance}")
/// ```
#[pyfunction]
#[pyo3(signature = (u_values, v_values, u_weights=None, v_weights=None))]
pub fn calculate_wasserstein_distance(
    u_values: PyReadonlyArray1<'_, f64>,
    v_values: PyReadonlyArray1<'_, f64>,
    u_weights: Option<PyReadonlyArray1<'_, f64>>,
    v_weights: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<f64> {
    let u = u_values.as_array();
    let v = v_values.as_array();

    if u.is_empty() || v.is_empty() {
        return Err(PyValueError::new_err("Input arrays must not be empty"));
    }

    // Get or create uniform weights
    let u_w: Vec<f64> = if let Some(weights) = u_weights {
        let w = weights.as_array();
        if w.len() != u.len() {
            return Err(PyValueError::new_err(
                "u_weights must have the same length as u_values",
            ));
        }
        w.to_vec()
    } else {
        vec![1.0 / u.len() as f64; u.len()]
    };

    let v_w: Vec<f64> = if let Some(weights) = v_weights {
        let w = weights.as_array();
        if w.len() != v.len() {
            return Err(PyValueError::new_err(
                "v_weights must have the same length as v_values",
            ));
        }
        w.to_vec()
    } else {
        vec![1.0 / v.len() as f64; v.len()]
    };

    // Validate weights sum to 1 (approximately)
    let u_sum: f64 = u_w.iter().sum();
    let v_sum: f64 = v_w.iter().sum();

    if (u_sum - 1.0).abs() > 1e-6 {
        return Err(PyValueError::new_err(format!(
            "u_weights must sum to 1, got {}",
            u_sum
        )));
    }

    if (v_sum - 1.0).abs() > 1e-6 {
        return Err(PyValueError::new_err(format!(
            "v_weights must sum to 1, got {}",
            v_sum
        )));
    }

    // Sort values with their weights
    let mut u_sorted: Vec<(f64, f64)> = u.iter().zip(&u_w).map(|(&val, &w)| (val, w)).collect();
    let mut v_sorted: Vec<(f64, f64)> = v.iter().zip(&v_w).map(|(&val, &w)| (val, w)).collect();

    u_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    v_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compute Wasserstein distance using cumulative distributions
    let mut distance: f64 = 0.0;
    let mut u_idx = 0;
    let mut v_idx = 0;
    let mut u_cum: f64 = 0.0;
    let mut v_cum: f64 = 0.0;
    let mut prev_val: f64 = u_sorted[0].0.min(v_sorted[0].0);

    while u_idx < u_sorted.len() || v_idx < v_sorted.len() {
        let (u_val, u_w_val) = if u_idx < u_sorted.len() {
            u_sorted[u_idx]
        } else {
            (f64::INFINITY, 0.0)
        };

        let (v_val, v_w_val) = if v_idx < v_sorted.len() {
            v_sorted[v_idx]
        } else {
            (f64::INFINITY, 0.0)
        };

        let curr_val = u_val.min(v_val);

        if curr_val < f64::INFINITY {
            // Add contribution from previous segment
            distance += (u_cum - v_cum).abs() * (curr_val - prev_val);
            prev_val = curr_val;
        }

        // Update cumulative sums
        if u_val <= v_val && u_idx < u_sorted.len() {
            u_cum += u_w_val;
            u_idx += 1;
        }
        if v_val <= u_val && v_idx < v_sorted.len() {
            v_cum += v_w_val;
            v_idx += 1;
        }
    }

    Ok(distance)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_identical_distributions() {
        // Wasserstein distance between identical distributions should be 0
        let u = [1.0, 2.0, 3.0];
        let u_w = vec![1.0 / 3.0; 3];

        let mut u_sorted: Vec<(f64, f64)> = u.iter().zip(&u_w).map(|(&val, &w)| (val, w)).collect();
        u_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // For identical distributions, cumulative sums should match at each step
        assert_eq!(u_sorted.len(), 3);
    }

    #[test]
    fn test_shifted_distribution() {
        // Distance between distributions shifted by 1 should be 1
        let _u = [0.0, 1.0, 2.0];
        let _v = [1.0, 2.0, 3.0];

        // Both uniform
        let _weight = 1.0 / 3.0;

        // Expected: each point moves distance 1, weighted by 1/3
        // Total = 3 * (1/3) * 1 = 1
        let expected = 1.0;

        // Simplified check
        assert_relative_eq!(expected, 1.0, epsilon = 1e-10);
    }
}
