/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Numerically calculate the definite integral of a given function over a specified interval.
*/

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Numerically calculate the definite integral of a given function over a specified interval.
///
/// This function uses adaptive quadrature methods to compute the definite integral
/// with error control. The implementation is designed for high performance and accuracy.
///
/// # Arguments
///
/// * `func` - A Python callable that accepts a float and returns a float
/// * `a` - The lower limit of integration
/// * `b` - The upper limit of integration
/// * `args` - Additional arguments to pass to the function (optional)
///
/// # Returns
///
/// A tuple containing:
/// * The computed value of the definite integral
/// * The estimated absolute error in the result
///
/// # Errors
///
/// * `TypeError` - If func is not callable, or if a or b are not numbers
/// * `ValueError` - If a >= b
///
/// # Examples
///
/// ```python
/// from voidkit_rust import numerical_integrate
///
/// # Simple integral of x^2 from 0 to 1 (equals 1/3)
/// result, error = numerical_integrate(lambda x: x**2, 0, 1)
/// print(f"Result: {result:.6f}, Error: {error:.6e}")
/// ```
#[pyfunction]
#[pyo3(signature = (func, a, b, args=None))]
pub fn numerical_integrate(
    py: Python<'_>,
    func: PyObject,
    a: f64,
    b: f64,
    args: Option<&Bound<'_, PyTuple>>,
) -> PyResult<(f64, f64)> {
    // Input validation
    if !func.bind(py).is_callable() {
        return Err(PyTypeError::new_err("func must be a callable"));
    }

    if !a.is_finite() || !b.is_finite() {
        return Err(PyTypeError::new_err(
            "Lower limit 'a' and upper limit 'b' must be finite numbers",
        ));
    }

    if a >= b {
        return Err(PyValueError::new_err(
            "Lower limit 'a' must be less than upper limit 'b'",
        ));
    }

    // Create a wrapper function for the integrand
    let integrand = |x: f64| -> PyResult<f64> {
        Python::with_gil(|py| {
            let x_obj = x.into_py(py);
            let result = if let Some(extra_args) = args {
                let mut call_args = vec![x_obj];
                for arg in extra_args.iter() {
                    call_args.push(arg.clone().unbind());
                }
                let args_tuple = PyTuple::new_bound(py, &call_args);
                func.call1(py, args_tuple)
            } else {
                func.call1(py, (x_obj,))
            }?;

            result.extract::<f64>(py)
        })
    };

    // Perform adaptive quadrature integration
    match adaptive_quadrature(&integrand, a, b, 1e-8, 1e-10) {
        Ok((result, error)) => Ok((result, error)),
        Err(e) => Err(e),
    }
}

/// Adaptive quadrature integration using Simpson's rule
fn adaptive_quadrature<F>(f: &F, a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> PyResult<(f64, f64)>
where
    F: Fn(f64) -> PyResult<f64>,
{
    const MAX_DEPTH: usize = 50;

    let fa = f(a)?;
    let fb = f(b)?;
    let fc = f((a + b) / 2.0)?;

    // Initial Simpson's rule estimate
    let h = b - a;
    let s = h / 6.0 * (fa + 4.0 * fc + fb);

    adaptive_simpson_recursive(f, a, b, fa, fb, fc, s, abs_tol, rel_tol, 0, MAX_DEPTH)
}

/// Recursive adaptive Simpson's rule
fn adaptive_simpson_recursive<F>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fc: f64,
    s: f64,
    abs_tol: f64,
    rel_tol: f64,
    depth: usize,
    max_depth: usize,
) -> PyResult<(f64, f64)>
where
    F: Fn(f64) -> PyResult<f64>,
{
    if depth >= max_depth {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Integration failed to converge: maximum recursion depth reached",
        ));
    }

    let c = (a + b) / 2.0;
    let h = b - a;
    let d = (a + c) / 2.0;
    let e = (c + b) / 2.0;

    let fd = f(d)?;
    let fe = f(e)?;

    // Simpson's rule on left and right halves
    let s_left = h / 12.0 * (fa + 4.0 * fd + fc);
    let s_right = h / 12.0 * (fc + 4.0 * fe + fb);
    let s2 = s_left + s_right;

    // Error estimate
    let error = (s2 - s).abs() / 15.0;
    let tol = abs_tol + rel_tol * s2.abs();

    if error < tol {
        // Accept the result
        Ok((s2, error))
    } else {
        // Subdivide further
        let (left_result, left_error) = adaptive_simpson_recursive(
            f,
            a,
            c,
            fa,
            fc,
            fd,
            s_left,
            abs_tol / 2.0,
            rel_tol,
            depth + 1,
            max_depth,
        )?;
        let (right_result, right_error) = adaptive_simpson_recursive(
            f,
            c,
            b,
            fc,
            fb,
            fe,
            s_right,
            abs_tol / 2.0,
            rel_tol,
            depth + 1,
            max_depth,
        )?;

        Ok((left_result + right_result, left_error + right_error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_x_squared_integral() {
        // Integral of x^2 from 0 to 1 should be 1/3
        let f = |x: f64| -> PyResult<f64> { Ok(x * x) };
        let (result, error) = adaptive_quadrature(&f, 0.0, 1.0, 1e-8, 1e-10).unwrap();

        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-8);
        assert!(error < 1e-8);
    }

    #[test]
    fn test_sine_integral() {
        // Integral of sin(x) from 0 to π should be 2
        let f = |x: f64| -> PyResult<f64> { Ok(x.sin()) };
        let (result, error) =
            adaptive_quadrature(&f, 0.0, std::f64::consts::PI, 1e-8, 1e-10).unwrap();

        assert_relative_eq!(result, 2.0, epsilon = 1e-8);
        assert!(error < 1e-7);
    }

    #[test]
    fn test_exponential_integral() {
        // Integral of e^(-x) from 0 to 1 should be 1 - 1/e ≈ 0.632120559
        let f = |x: f64| -> PyResult<f64> { Ok((-x).exp()) };
        let (result, error) = adaptive_quadrature(&f, 0.0, 1.0, 1e-8, 1e-10).unwrap();

        let expected = 1.0 - (1.0_f64).exp().recip();
        assert_relative_eq!(result, expected, epsilon = 1e-8);
        assert!(error < 1e-8);
    }
}
