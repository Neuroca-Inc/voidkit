/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use numpy::PyReadonlyArray2;

/// Calculates TDA metrics from persistence diagrams.
///
/// # Arguments
/// * `h0_diagram` - Persistence diagram for 0-dimensional features (connected components)
/// * `h1_diagram` - Persistence diagram for 1-dimensional features (loops/cycles), optional
///
/// # Returns
/// A dictionary containing the calculated TDA metrics:
/// - 'total_b1_persistence': The sum of persistence of 1-dimensional features
/// - 'component_count': The number of connected components (0-dimensional features)
///
/// # Note
/// In persistence diagrams, infinite persistence is represented by f64::INFINITY.
/// The input diagrams should be Nx2 arrays where each row is [birth, death].
///
/// # Example
/// ```python
/// import numpy as np
/// h0 = np.array([[0.0, np.inf], [0.5, 1.0]])
/// h1 = np.array([[0.3, 0.8], [0.4, 0.9]])
/// metrics = calculate_tda_metrics(h0, h1)
/// assert metrics['component_count'] == 1  # One infinite component
/// ```
#[pyfunction]
#[pyo3(signature = (h0_diagram, h1_diagram=None))]
pub fn calculate_tda_metrics<'py>(
    py: Python<'py>,
    h0_diagram: PyReadonlyArray2<f64>,
    h1_diagram: Option<PyReadonlyArray2<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let h0 = h0_diagram.as_array();
    
    // Validate H0 diagram
    if h0.shape()[1] != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Persistence diagrams must have shape (n, 2)",
        ));
    }

    let metrics = PyDict::new_bound(py);

    // H0: Connected components
    // Count points with infinite persistence (death = inf)
    let mut component_count = 0;
    for i in 0..h0.shape()[0] {
        let death = h0[[i, 1]];
        if death.is_infinite() && death.is_sign_positive() {
            component_count += 1;
        }
    }
    metrics.set_item("component_count", component_count)?;

    // H1: Loops/cycles
    let total_b1_persistence = if let Some(h1_array) = h1_diagram {
        let h1 = h1_array.as_array();
        
        if h1.shape()[1] != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Persistence diagrams must have shape (n, 2)",
            ));
        }

        let mut persistence_sum = 0.0;
        for i in 0..h1.shape()[0] {
            let birth = h1[[i, 0]];
            let death = h1[[i, 1]];
            
            // Only count finite persistence values
            if !death.is_infinite() {
                persistence_sum += death - birth;
            }
        }
        persistence_sum
    } else {
        0.0
    };
    
    metrics.set_item("total_b1_persistence", total_b1_persistence)?;

    Ok(metrics)
}

#[cfg(test)]
mod tests {

    // Note: Tests requiring Python/numpy integration are disabled pending
    // test infrastructure updates for PyO3 0.22 API
    
    /*
    #[test]
    fn test_calculate_tda_metrics_basic() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // H0: 1 infinite component, 2 finite components
            let h0 = vec![
                vec![0.0, f64::INFINITY],
                vec![0.5, 1.0],
                vec![0.3, 0.8],
            ];
            
            // H1: 2 loops with persistence 0.5 and 0.6
            let h1 = vec![
                vec![0.3, 0.8],  // persistence = 0.5
                vec![0.4, 1.0],  // persistence = 0.6
            ];
            
            let h0_array = numpy::PyArray2::from_vec2(py, &h0).unwrap();
            let h1_array = numpy::PyArray2::from_vec2(py, &h1).unwrap();
            
            let metrics = calculate_tda_metrics(
                py,
                h0_array.readonly(),
                Some(h1_array.readonly()),
            ).unwrap();
            
            let component_count: i32 = metrics.get_item("component_count").unwrap().unwrap().extract().unwrap();
            let total_b1: f64 = metrics.get_item("total_b1_persistence").unwrap().unwrap().extract().unwrap();
            
            assert_eq!(component_count, 1);  // Only 1 infinite component
            assert_relative_eq!(total_b1, 1.1, epsilon = 1e-10);  // 0.5 + 0.6
        });
    }

    #[test]
    fn test_calculate_tda_metrics_no_h1() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let h0 = vec![
                vec![0.0, f64::INFINITY],
                vec![0.5, f64::INFINITY],
            ];
            
            let h0_array = numpy::PyArray2::from_vec2(py, &h0).unwrap();
            
            let metrics = calculate_tda_metrics(
                py,
                h0_array.readonly(),
                None,
            ).unwrap();
            
            let component_count: i32 = metrics.get_item("component_count").unwrap().unwrap().extract().unwrap();
            let total_b1: f64 = metrics.get_item("total_b1_persistence").unwrap().unwrap().extract().unwrap();
            
            assert_eq!(component_count, 2);
            assert_relative_eq!(total_b1, 0.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_calculate_tda_metrics_no_infinite_components() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // All finite components
            let h0 = vec![
                vec![0.0, 0.5],
                vec![0.3, 0.8],
            ];
            
            let h0_array = numpy::PyArray2::from_vec2(py, &h0).unwrap();
            
            let metrics = calculate_tda_metrics(
                py,
                h0_array.readonly(),
                None,
            ).unwrap();
            
            let component_count: i32 = metrics.get_item("component_count").unwrap().unwrap().extract().unwrap();
            
            assert_eq!(component_count, 0);
        });
    }

    #[test]
    fn test_calculate_tda_metrics_empty_diagrams() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let h0 = vec![vec![0.0, f64::INFINITY]];
            let h0_array = numpy::PyArray2::from_vec2(py, &h0).unwrap();
            
            let metrics = calculate_tda_metrics(
                py,
                h0_array.readonly(),
                None,
            ).unwrap();
            
            assert!(metrics.contains("component_count").unwrap());
            assert!(metrics.contains("total_b1_persistence").unwrap());
        });
    }
    */
}
