/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

//! Dynamical Systems: Stability Analysis
//!
//! Analyzes the stability of fixed points by examining eigenvalues of the Jacobian.

use nalgebra::DMatrix;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Analyzes the stability of a fixed point by examining the eigenvalues of the Jacobian.
///
/// # Arguments
/// * `jacobian` - The Jacobian matrix at the fixed point (NxN array)
///
/// # Returns
/// Dictionary containing:
/// - 'eigenvalues': Complex eigenvalues as (real, imag) tuples
/// - 'stability_type': String describing stability type
#[pyfunction]
#[pyo3(signature = (jacobian))]
pub fn analyze_stability<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray2<f64>,
) -> PyResult<HashMap<String, PyObject>> {
    let jac_arr = jacobian.as_array();
    let (nrows, ncols) = (jac_arr.nrows(), jac_arr.ncols());

    if nrows != ncols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Jacobian matrix must be square",
        ));
    }

    // Convert to nalgebra matrix
    let mut jac_matrix = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            jac_matrix[(i, j)] = jac_arr[[i, j]];
        }
    }

    // Compute eigenvalues
    let eigenvalues = jac_matrix.complex_eigenvalues();

    // Extract real and imaginary parts
    let mut real_parts = Vec::new();
    let mut imag_parts = Vec::new();
    let mut eig_tuples = Vec::new();

    for eig in eigenvalues.iter() {
        real_parts.push(eig.re);
        imag_parts.push(eig.im);
        eig_tuples.push((eig.re, eig.im).to_object(py));
    }

    // Determine stability type
    let all_negative_real = real_parts.iter().all(|&r| r < 0.0);
    let all_positive_real = real_parts.iter().all(|&r| r > 0.0);
    let all_zero_imag = imag_parts.iter().all(|&i| i.abs() < 1e-10);
    let mixed_signs = real_parts.iter().any(|&r| r > 0.0) && real_parts.iter().any(|&r| r < 0.0);

    let stability_type = if all_negative_real {
        if all_zero_imag {
            "Stable Node"
        } else {
            "Stable Spiral"
        }
    } else if all_positive_real {
        if all_zero_imag {
            "Unstable Node"
        } else {
            "Unstable Spiral"
        }
    } else if mixed_signs {
        "Saddle Point"
    } else {
        "Center (Marginally Stable)"
    };

    // Build result dictionary
    let mut result = HashMap::new();
    result.insert("eigenvalues".to_string(), eig_tuples.to_object(py));
    result.insert("stability_type".to_string(), stability_type.to_object(py));

    Ok(result)
}

#[cfg(test)]
mod tests {
    use numpy::PyArrayMethods;
    use super::*;

    #[test]
    fn test_analyze_stability_stable_node() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Stable node: both eigenvalues negative
            let jac = vec![vec![-1.0, 0.0], vec![0.0, -2.0]];

            let jac_py = numpy::PyArray2::from_vec2_bound(py, &jac).unwrap();

            let result = analyze_stability(py, jac_py.readonly()).unwrap();

            let stability_type: String = result.get("stability_type").unwrap().extract(py).unwrap();

            assert_eq!(stability_type, "Stable Node");
        });
    }

    #[test]
    fn test_analyze_stability_saddle() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Saddle point: eigenvalues have opposite signs
            let jac = vec![vec![1.0, 0.0], vec![0.0, -1.0]];

            let jac_py = numpy::PyArray2::from_vec2_bound(py, &jac).unwrap();

            let result = analyze_stability(py, jac_py.readonly()).unwrap();

            let stability_type: String = result.get("stability_type").unwrap().extract(py).unwrap();

            assert_eq!(stability_type, "Saddle Point");
        });
    }
}
