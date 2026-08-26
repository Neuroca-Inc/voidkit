/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Dynamical systems module - Jacobian calculation using finite differences.
*/

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Calculates the Jacobian matrix of a dynamical system at a given point.
///
/// Uses finite difference approximation to compute partial derivatives.
///
/// # Arguments
///
/// * `func` - A Python callable representing the dynamical system
/// * `point` - The point at which to calculate the Jacobian
/// * `epsilon` - The step size for the finite difference approximation
///
/// # Returns
///
/// The Jacobian matrix at the given point
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_jacobian
/// import numpy as np
///
/// # Define a 2D dynamical system
/// def system(x):
///     return np.array([x[0]**2 + x[1], x[0] - x[1]**2])
///
/// point = np.array([1.0, 1.0])
/// J = calculate_jacobian(system, point, epsilon=1e-6)
/// print(J)
/// ```
#[pyfunction]
#[pyo3(signature = (func, point, epsilon=1e-6))]
pub fn calculate_jacobian<'py>(
    py: Python<'py>,
    func: PyObject,
    point: PyReadonlyArray1<'py, f64>,
    epsilon: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let point_array = point.as_array();
    let n = point_array.len();

    if !func.bind(py).is_callable() {
        return Err(PyValueError::new_err("func must be callable"));
    }

    if epsilon <= 0.0 {
        return Err(PyValueError::new_err("epsilon must be positive"));
    }

    // Evaluate function at the base point
    let point_vec: Vec<f64> = point_array.to_vec();
    let point_py = PyArray1::from_vec_bound(py, point_vec.clone());
    let f0_obj = func.call1(py, (point_py,))?;
    let f0: Vec<f64> = f0_obj.extract(py)?;
    let m = f0.len();

    // Compute Jacobian using finite differences
    let mut jacobian = vec![0.0; m * n];

    for i in 0..n {
        // Create perturbed point
        let mut p_plus = point_vec.clone();
        p_plus[i] += epsilon;

        let p_plus_py = PyArray1::from_vec_bound(py, p_plus);
        let f_plus_obj = func.call1(py, (p_plus_py,))?;
        let f_plus: Vec<f64> = f_plus_obj.extract(py)?;

        if f_plus.len() != m {
            return Err(PyValueError::new_err("Function output dimension changed"));
        }

        // Compute partial derivative: ∂f/∂x_i
        for j in 0..m {
            jacobian[j * n + i] = (f_plus[j] - f0[j]) / epsilon;
        }
    }

    // Convert to 2D array
    use ndarray::Array2;
    let jac_ndarray = Array2::from_shape_vec((m, n), jacobian)
        .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;

    Ok(PyArray2::from_owned_array_bound(py, jac_ndarray))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_jacobian_linear_system() {
        // For a linear system f(x) = Ax, the Jacobian is simply A
        // f([x, y]) = [2x + 3y, x - y]
        // J = [[2, 3], [1, -1]]

        let x = vec![1.0, 1.0];
        let f0 = [2.0 * x[0] + 3.0 * x[1], x[0] - x[1]];

        let epsilon = 1e-6;
        let mut x_plus = x.clone();
        x_plus[0] += epsilon;
        let f_plus = [2.0 * x_plus[0] + 3.0 * x_plus[1], x_plus[0] - x_plus[1]];

        let df_dx0 = (f_plus[0] - f0[0]) / epsilon;
        let df_dx1 = (f_plus[1] - f0[1]) / epsilon;

        assert_relative_eq!(df_dx0, 2.0, epsilon = 1e-5);
        assert_relative_eq!(df_dx1, 1.0, epsilon = 1e-5);
    }
}
