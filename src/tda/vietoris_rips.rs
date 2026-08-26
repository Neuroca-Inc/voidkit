/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Bound;
use numpy::PyReadonlyArray2;
use std::collections::HashSet;

/// Constructs a Vietoris-Rips complex from a set of points up to a given dimension.
///
/// # Arguments
/// * `points` - A 2D array of shape (n_points, n_features) representing the data
/// * `max_edge_length` - The maximum edge length to consider for the complex
/// * `max_dim` - The maximum dimension of the simplices to include (default: 2)
///
/// # Returns
/// A list of simplices, where each simplex is represented as a tuple
/// containing the vertices (as a list) and the filtration value at which it appears
///
/// # Example
/// ```python
/// import numpy as np
/// points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
/// complex = construct_vietoris_rips(points, 1.5, max_dim=2)
/// ```
#[allow(clippy::needless_range_loop)]
#[pyfunction]
#[pyo3(signature = (points, max_edge_length, max_dim=2))]
pub fn construct_vietoris_rips<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    max_edge_length: f64,
    max_dim: usize,
) -> PyResult<Bound<'py, PyList>> {
    let points_array = points.as_array();
    let n_points = points_array.shape()[0];
    let n_features = points_array.shape()[1];

    if n_points == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "points array cannot be empty",
        ));
    }

    if max_edge_length < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_edge_length must be non-negative",
        ));
    }

    // Compute pairwise distance matrix
    let mut dist_matrix = vec![vec![0.0; n_points]; n_points];
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            let mut sum_sq = 0.0;
            for k in 0..n_features {
                let diff = points_array[[i, k]] - points_array[[j, k]];
                sum_sq += diff * diff;
            }
            let dist = sum_sq.sqrt();
            dist_matrix[i][j] = dist;
            dist_matrix[j][i] = dist;
        }
    }

    let rips_complex = PyList::new_bound(py, &[] as &[i32]);

    // 0-simplices (vertices)
    for i in 0..n_points {
        let vertex_list = PyList::new_bound(py, vec![i]);
        rips_complex.append((vertex_list, 0.0))?;
    }

    // 1-simplices (edges)
    let mut edges = HashSet::new();
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            if dist_matrix[i][j] <= max_edge_length {
                let edge_list = PyList::new_bound(py, vec![i, j]);
                rips_complex.append((edge_list, dist_matrix[i][j]))?;
                edges.insert((i.min(j), i.max(j)));
            }
        }
    }

    // Higher-dimensional simplices
    if max_dim > 1 {
        for dim in 2..=max_dim {
            // Generate all possible (dim+1)-simplices
            let mut simplices = Vec::new();
            generate_combinations(n_points, dim + 1, &mut vec![], 0, &mut simplices);

            for simplex in simplices {
                // Check if all edges of this simplex exist
                let mut is_valid = true;
                let mut max_dist = 0.0;

                for i in 0..simplex.len() {
                    for j in (i + 1)..simplex.len() {
                        let vi = simplex[i];
                        let vj = simplex[j];
                        let dist = dist_matrix[vi][vj];

                        if dist > max_edge_length {
                            is_valid = false;
                            break;
                        }
                        max_dist = f64::max(max_dist, dist);
                    }
                    if !is_valid {
                        break;
                    }
                }

                if is_valid {
                    let simplex_list = PyList::new_bound(py, simplex);
                    rips_complex.append((simplex_list, max_dist))?;
                }
            }
        }
    }

    Ok(rips_complex)
}

/// Helper function to generate all k-combinations of n elements
fn generate_combinations(n: usize, k: usize, current: &mut Vec<usize>, start: usize, result: &mut Vec<Vec<usize>>) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }

    for i in start..n {
        current.push(i);
        generate_combinations(n, k, current, i + 1, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {

    // Note: Tests requiring Python/numpy integration are disabled pending
    // test infrastructure updates for PyO3 0.22 API
    
    /*
    #[test]
    fn test_construct_vietoris_rips_basic() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Three points forming an equilateral triangle
            let points = vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.5, 0.866],
            ];
            let points_array = numpy::PyArray2::from_vec2(py, &points).unwrap();
            
            let result = construct_vietoris_rips(py, points_array.readonly(), 1.5, 2).unwrap();
            
            // Should have vertices (3), edges (3), and triangles (1)
            // Total: 3 + 3 + 1 = 7 simplices
            assert!(result.len() >= 6); // At least vertices + edges
        });
    }

    #[test]
    fn test_construct_vietoris_rips_no_edges() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Points far apart, no edges should form
            let points = vec![
                vec![0.0, 0.0],
                vec![10.0, 0.0],
                vec![0.0, 10.0],
            ];
            let points_array = numpy::PyArray2::from_vec2(py, &points).unwrap();
            
            let result = construct_vietoris_rips(py, points_array.readonly(), 1.0, 2).unwrap();
            
            // Should only have 3 vertices (0-simplices)
            assert_eq!(result.len(), 3);
        });
    }

    #[test]
    fn test_construct_vietoris_rips_2d_max_dim_0() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let points = vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
            ];
            let points_array = numpy::PyArray2::from_vec2(py, &points).unwrap();
            
            let result = construct_vietoris_rips(py, points_array.readonly(), 2.0, 0).unwrap();
            
            // With max_dim=0, should only have vertices
            assert_eq!(result.len(), 2);
        });
    }
    */
}
