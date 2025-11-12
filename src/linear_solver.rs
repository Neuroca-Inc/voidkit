/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Solves a system of linear equations in the form Ax = b using high-performance
linear algebra routines.
*/

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use nalgebra::{DMatrix, DVector};

/// Solves a system of linear equations in the form Ax = b.
///
/// This function uses LU decomposition to solve the linear system efficiently.
/// The implementation leverages nalgebra for high-performance linear algebra.
///
/// # Arguments
///
/// * `matrix_a` - Coefficient matrix of the linear system (must be square and non-singular)
/// * `vector_b` - Right-hand side vector of the linear system
///
/// # Returns
///
/// Solution vector x that satisfies Ax = b
///
/// # Errors
///
/// * `TypeError` - If inputs are not valid arrays
/// * `ValueError` - If matrix is not square, shapes are incompatible, or matrix is singular
///
/// # Examples
///
/// ```python
/// import numpy as np
/// from voidkit_rust import linear_system_solver
///
/// # Solve 3x + 2y = 7, x + y = 3
/// A = np.array([[3.0, 2.0], [1.0, 1.0]])
/// b = np.array([7.0, 3.0])
/// x = linear_system_solver(A, b)
/// print(f"Solution: x = {x[0]}, y = {x[1]}")
/// ```
#[pyfunction]
pub fn linear_system_solver<'py>(
    py: Python<'py>,
    matrix_a: PyReadonlyArray2<'py, f64>,
    vector_b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Get array views
    let a = matrix_a.as_array();
    let b = vector_b.as_array();
    
    // Validate dimensions
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    
    if rows != cols {
        return Err(PyValueError::new_err(format!(
            "matrix_A must be square, got shape ({}, {})", rows, cols
        )));
    }
    
    if rows != b.len() {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: matrix_A has {} rows, but vector_b has {} elements",
            rows, b.len()
        )));
    }
    
    if rows == 0 || b.len() == 0 {
        return Err(PyValueError::new_err("Empty arrays are not valid inputs"));
    }
    
    // Check for NaN or infinity values
    for &val in a.iter() {
        if !val.is_finite() {
            return Err(PyValueError::new_err("matrix_A contains NaN or infinity values"));
        }
    }
    
    for &val in b.iter() {
        if !val.is_finite() {
            return Err(PyValueError::new_err("vector_b contains NaN or infinity values"));
        }
    }
    
    // Convert to nalgebra types
    let matrix = DMatrix::from_row_slice(rows, cols, a.as_slice().unwrap());
    let vec = DVector::from_row_slice(b.as_slice().unwrap());
    
    // Solve using LU decomposition
    match matrix.lu().solve(&vec) {
        Some(solution) => {
            // Convert back to numpy array
            let solution_slice = solution.as_slice();
            Ok(PyArray1::from_slice_bound(py, solution_slice))
        }
        None => {
            Err(PyValueError::new_err(
                "The coefficient matrix is singular. The system has no unique solution."
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simple_2x2_system() {
        // Test: 3x + 2y = 7, x + y = 3
        // Expected solution: x = 1, y = 2
        use nalgebra::{DMatrix, DVector};
        
        let a = DMatrix::from_row_slice(2, 2, &[3.0, 2.0, 1.0, 1.0]);
        let b = DVector::from_row_slice(&[7.0, 3.0]);
        
        let solution = a.lu().solve(&b).unwrap();
        
        assert_relative_eq!(solution[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_identity_matrix() {
        use nalgebra::{DMatrix, DVector};
        
        let a = DMatrix::identity(3, 3);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        
        let solution = a.lu().solve(&b).unwrap();
        
        assert_relative_eq!(solution[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(solution[2], 3.0, epsilon = 1e-10);
    }
}
