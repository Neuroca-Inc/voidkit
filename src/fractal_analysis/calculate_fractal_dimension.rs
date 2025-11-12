/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Fractal analysis module - fractal dimension calculation using box-counting.
*/

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray2;
use std::collections::HashMap;

/// Calculates the fractal dimension of a point set using the box-counting algorithm.
///
/// # Arguments
///
/// * `points` - A 2D array of shape (n_points, n_features) representing the data
/// * `threshold` - The threshold for the number of points in a box (unused in current impl)
///
/// # Returns
///
/// The estimated fractal dimension
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_fractal_dimension
/// import numpy as np
///
/// # Generate a 2D point cloud
/// points = np.random.rand(1000, 2)
/// dimension = calculate_fractal_dimension(points)
/// print(f"Fractal dimension: {dimension}")
/// ```
#[pyfunction]
#[pyo3(signature = (points, _threshold=0.9))]
pub fn calculate_fractal_dimension(
    points: PyReadonlyArray2<'_, f64>,
    _threshold: f64,
) -> PyResult<f64> {
    let points_array = points.as_array();
    let shape = points_array.shape();
    let (n_points, n_dims) = (shape[0], shape[1]);
    
    if n_points == 0 {
        return Err(PyValueError::new_err("Points array must not be empty"));
    }
    
    // Find bounding box
    let mut min_coords = vec![f64::INFINITY; n_dims];
    let mut max_coords = vec![f64::NEG_INFINITY; n_dims];
    
    for i in 0..n_points {
        for j in 0..n_dims {
            let val = points_array[[i, j]];
            min_coords[j] = min_coords[j].min(val);
            max_coords[j] = max_coords[j].max(val);
        }
    }
    
    // Generate scales using logspace
    let num_scales = 10;
    let mut scales = Vec::new();
    for i in 0..num_scales {
        let exponent = 0.01 + (1.0 - 0.01) * (i as f64) / (num_scales as f64);
        scales.push(2.0_f64.powf(exponent));
    }
    
    let mut counts = Vec::new();
    
    for &scale in &scales {
        let mut grid: HashMap<Vec<i64>, usize> = HashMap::new();
        
        for i in 0..n_points {
            let mut box_index = Vec::new();
            
            for j in 0..n_dims {
                let val = points_array[[i, j]];
                let box_size = (max_coords[j] - min_coords[j]) / scale;
                let idx = ((val - min_coords[j]) / box_size).floor() as i64;
                box_index.push(idx);
            }
            
            *grid.entry(box_index).or_insert(0) += 1;
        }
        
        counts.push(grid.len() as f64);
    }
    
    // Fit a line to log-log plot: log(counts) = slope * log(scales) + intercept
    let n = scales.len() as f64;
    let log_scales: Vec<f64> = scales.iter().map(|s| s.ln()).collect();
    let log_counts: Vec<f64> = counts.iter().map(|c| c.ln()).collect();
    
    let sum_x: f64 = log_scales.iter().sum();
    let sum_y: f64 = log_counts.iter().sum();
    let sum_xy: f64 = log_scales.iter().zip(&log_counts).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_scales.iter().map(|x| x * x).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    
    // Fractal dimension is the negative of the slope
    Ok(-slope)
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
        
        assert!((slope - 2.0).abs() < 1e-10);
    }
}
