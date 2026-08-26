/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

//! Dynamical Systems: Fixed Point Finding
//!
//! Finds fixed points (equilibria) of dynamical systems using Newton's method.

use nalgebra::DVector;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Finds fixed points (equilibria) of a dynamical system using Newton's method.
///
/// For a dynamical system dx/dt = f(x), fixed points satisfy f(x) = 0.
///
/// # Arguments
/// * `func` - Python callable representing the system: f(x) -> dx/dt
/// * `initial_guesses` - List of initial guess vectors for fixed points
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `max_iter` - Maximum number of iterations (default: 100)
///
/// # Returns
/// Array of found fixed points (one per initial guess)
#[pyfunction]
#[pyo3(signature = (func, initial_guesses, tol = 1e-6, max_iter = 100))]
pub fn find_fixed_points<'py>(
    py: Python<'py>,
    func: PyObject,
    initial_guesses: Vec<PyReadonlyArray1<f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if initial_guesses.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "initial_guesses cannot be empty",
        ));
    }

    let dim = initial_guesses[0].as_array().len();
    let mut fixed_points = Vec::new();

    for guess in &initial_guesses {
        let guess_arr = guess.as_array();
        let dim = guess_arr.len();

        // Start from initial guess
        let mut x = DVector::from_vec(guess_arr.to_vec());
        let mut converged = false;

        for _ in 0..max_iter {
            // Evaluate f(x)
            let x_py = PyArray1::from_vec_bound(py, x.as_slice().to_vec());
            let f_x: Vec<f64> = func.call1(py, (x_py,))?.extract(py)?;
            let f_vec = DVector::from_vec(f_x);

            // Check convergence: |f(x)| < tol
            let norm = f_vec.norm();
            if norm < tol {
                converged = true;
                break;
            }

            // Compute Jacobian numerically using finite differences
            let epsilon = 1e-8;
            let mut jacobian = nalgebra::DMatrix::zeros(dim, dim);

            for j in 0..dim {
                // Perturb x[j]
                let mut x_plus = x.clone();
                x_plus[j] += epsilon;

                // Evaluate f(x + epsilon * e_j)
                let x_plus_py = PyArray1::from_vec_bound(py, x_plus.as_slice().to_vec());
                let f_plus: Vec<f64> = func.call1(py, (x_plus_py,))?.extract(py)?;

                // Finite difference approximation
                for i in 0..dim {
                    jacobian[(i, j)] = (f_plus[i] - f_vec[i]) / epsilon;
                }
            }

            // Newton step: x_new = x - J^(-1) * f(x)
            match jacobian.lu().solve(&f_vec) {
                Some(delta) => {
                    x -= delta;
                }
                None => {
                    // Singular Jacobian, use gradient descent step instead
                    x -= f_vec * 0.01;
                }
            }
        }

        if !converged {
            eprintln!("Warning: Fixed point search did not converge for one initial guess");
        }

        fixed_points.extend(x.as_slice());
    }

    // Reshape to (n_guesses, dim)
    let result = PyArray2::from_vec2_bound(
        py,
        &fixed_points
            .chunks(dim)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use numpy::PyArrayMethods;
    use super::*;

    #[test]
    fn test_find_fixed_points_simple() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Simple 1D system: dx/dt = -x (fixed point at x=0)
            let code = r#"
def func(x):
    return [-x[0]]
"#;
            let locals = pyo3::types::PyDict::new_bound(py);
            py.run_bound(code, None, Some(&locals)).unwrap();
            let func: PyObject = locals.get_item("func").unwrap().unwrap().into();

            // Initial guess near the fixed point
            let guess: Vec<f64> = vec![1.0];
            let guess_py = numpy::PyArray1::from_vec_bound(py, guess);
            let guesses = vec![guess_py.readonly()];

            let fixed_pts = find_fixed_points(py, func, guesses, 1e-6, 100).unwrap();

            let pts = fixed_pts.readonly();
            let pts_arr = pts.as_array();

            // Should converge to x=0
            assert!((pts_arr[[0, 0]]).abs() < 1e-5);
        });
    }
}
