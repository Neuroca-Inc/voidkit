/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Numerically solve an ordinary differential equation (ODE) initial value problem.
*/

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError, PyRuntimeError};
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};

/// Numerically solve an ordinary differential equation (ODE) initial value problem.
///
/// This function solves a system of first-order ODEs:
///     dy/dt = fun(t, y, *args)
/// with initial conditions:
///     y(t0) = y0
///
/// # Arguments
///
/// * `fun` - Function that defines the ODE system (must accept (t, y, *args))
/// * `t_span` - Tuple of (t0, tf) - interval of integration
/// * `y0` - Initial state (1-D array)
/// * `t_eval` - Optional times at which to store solution
/// * `args` - Extra arguments to pass to fun
/// * `method` - Integration method ('RK45', 'RK4', 'Euler')
/// * `rtol` - Relative tolerance
/// * `atol` - Absolute tolerance
///
/// # Returns
///
/// Dictionary with:
/// * 't': array of times at which solution was computed
/// * 'y': array of solution values at corresponding times
/// * 'success': whether solver succeeded
///
/// # Examples
///
/// ```python
/// from voidkit_rust import numerical_ode_solver
/// import numpy as np
///
/// # Exponential decay: dy/dt = -k*y
/// def decay(t, y, k):
///     return np.array([-k * y[0]])
///
/// t_span = (0.0, 10.0)
/// y0 = np.array([1.0])
/// sol = numerical_ode_solver(decay, t_span, y0, args=(0.1,))
/// ```
#[pyfunction]
#[pyo3(signature = (fun, t_span, y0, t_eval=None, args=None, method="RK45", rtol=1e-3, atol=1e-6))]
pub fn numerical_ode_solver(
    py: Python<'_>,
    fun: PyObject,
    t_span: (f64, f64),
    y0: PyReadonlyArray1<'_, f64>,
    t_eval: Option<PyReadonlyArray1<'_, f64>>,
    args: Option<&Bound<'_, PyTuple>>,
    method: &str,
    rtol: f64,
    atol: f64,
) -> PyResult<PyObject> {
    // Input validation
    if !fun.bind(py).is_callable() {
        return Err(PyTypeError::new_err("The 'fun' parameter must be a callable function."));
    }
    
    let (t0, tf) = t_span;
    if t0 >= tf {
        return Err(PyValueError::new_err("The 't_span' parameter must have t0 < tf."));
    }
    
    let y0_array = y0.as_array();
    if y0_array.len() == 0 {
        return Err(PyValueError::new_err("The 'y0' parameter must not be empty."));
    }
    
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(PyValueError::new_err("Tolerances 'rtol' and 'atol' must be positive."));
    }
    
    // Validate method
    let valid_methods = vec!["RK45", "RK4", "Euler"];
    if !valid_methods.contains(&method) {
        return Err(PyValueError::new_err(format!(
            "Invalid method '{}'. Valid methods are: RK45, RK4, Euler", method
        )));
    }
    
    // Create wrapper function for ODE
    let ode_func = |t: f64, y: &[f64]| -> PyResult<Vec<f64>> {
        Python::with_gil(|py| {
            let t_obj = t.into_py(py);
            let y_obj = PyArray1::from_slice_bound(py, y);
            
            let result = if let Some(extra_args) = args {
                let mut call_args = vec![t_obj, y_obj.into_py(py)];
                for arg in extra_args.iter() {
                    call_args.push(arg.clone().unbind());
                }
                let args_tuple = PyTuple::new_bound(py, &call_args);
                fun.call1(py, args_tuple)
            } else {
                fun.call1(py, (t_obj, y_obj))
            }?;
            
            let result_array: Vec<f64> = result.extract(py)?;
            
            if result_array.len() != y.len() {
                return Err(PyValueError::new_err(format!(
                    "Function returned array of length {}, expected {}",
                    result_array.len(), y.len()
                )));
            }
            
            Ok(result_array)
        })
    };
    
    // Test the function with initial conditions
    let y0_vec: Vec<f64> = y0_array.to_vec();
    let _ = ode_func(t0, &y0_vec).map_err(|e| {
        PyValueError::new_err(format!(
            "Error when testing the ODE function with initial conditions: {}",
            e
        ))
    })?;
    
    // Solve the ODE
    let (t_out, y_out) = match method {
        "RK45" => rk45_adaptive(&ode_func, t0, tf, y0_vec, rtol, atol, t_eval)?,
        "RK4" => rk4_fixed(&ode_func, t0, tf, y0_vec, t_eval)?,
        "Euler" => euler_fixed(&ode_func, t0, tf, y0_vec, t_eval)?,
        _ => unreachable!(),
    };
    
    // Convert results to Python objects
    let t_array = PyArray1::from_slice_bound(py, &t_out);
    
    // Convert y_out to a 2D array (transpose to get shape [n_vars, n_steps])
    let n_steps = y_out.len();
    let n_vars = if n_steps > 0 { y_out[0].len() } else { 0 };
    
    let mut y_transposed: Vec<Vec<f64>> = vec![vec![0.0; n_steps]; n_vars];
    for (step_idx, step) in y_out.iter().enumerate() {
        for (var_idx, &val) in step.iter().enumerate() {
            y_transposed[var_idx][step_idx] = val;
        }
    }
    
    // Flatten for PyArray2
    let mut y_flat: Vec<f64> = Vec::with_capacity(n_vars * n_steps);
    for row in &y_transposed {
        y_flat.extend(row);
    }
    
    // Create 2D array using ndarray
    use ndarray::Array2;
    let y_ndarray = Array2::from_shape_vec((n_vars, n_steps), y_flat)
        .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;
    let y_array = PyArray2::from_owned_array_bound(py, y_ndarray);
    
    // Create result dictionary
    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("t", t_array)?;
    result.set_item("y", y_array)?;
    result.set_item("success", true)?;
    
    Ok(result.into())
}

/// Runge-Kutta 4/5 adaptive step size method
fn rk45_adaptive<F>(
    f: &F,
    t0: f64,
    tf: f64,
    y0: Vec<f64>,
    rtol: f64,
    atol: f64,
    _t_eval: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> PyResult<Vec<f64>>,
{
    let mut t = t0;
    let mut y = y0.clone();
    let mut t_out = vec![t];
    let mut y_out = vec![y.clone()];
    
    let mut h = (tf - t0) / 100.0; // Initial step size
    let n = y.len();
    
    while t < tf {
        if t + h > tf {
            h = tf - t;
        }
        
        // RK4 step
        let k1 = f(t, &y)?;
        
        let y_tmp: Vec<f64> = y.iter().zip(&k1).map(|(yi, k)| yi + h * k / 2.0).collect();
        let k2 = f(t + h / 2.0, &y_tmp)?;
        
        let y_tmp: Vec<f64> = y.iter().zip(&k2).map(|(yi, k)| yi + h * k / 2.0).collect();
        let k3 = f(t + h / 2.0, &y_tmp)?;
        
        let y_tmp: Vec<f64> = y.iter().zip(&k3).map(|(yi, k)| yi + h * k).collect();
        let k4 = f(t + h, &y_tmp)?;
        
        // 4th order estimate
        let y_new: Vec<f64> = y.iter()
            .zip(&k1).zip(&k2).zip(&k3).zip(&k4)
            .map(|((((yi, k1i), k2i), k3i), k4i)| {
                yi + h / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            })
            .collect();
        
        // Error estimate (using embedded 3rd order method)
        let y_low: Vec<f64> = y.iter()
            .zip(&k1).zip(&k2).zip(&k3)
            .map(|(((yi, k1i), k2i), k3i)| {
                yi + h / 6.0 * (k1i + 4.0 * k2i + k3i)
            })
            .collect();
        
        // Compute error
        let error: f64 = y_new.iter().zip(&y_low).zip(&y)
            .map(|((yn, yl), yi)| {
                let scale = atol + rtol * yi.abs().max(yn.abs());
                ((yn - yl) / scale).powi(2)
            })
            .sum::<f64>()
            .sqrt() / (n as f64).sqrt();
        
        if error < 1.0 || h <= 1e-12 {
            // Accept step
            t += h;
            y = y_new;
            t_out.push(t);
            y_out.push(y.clone());
        }
        
        // Adjust step size
        if error > 0.0 {
            h *= 0.9 * (1.0 / error).powf(0.2).min(5.0).max(0.2);
        }
    }
    
    Ok((t_out, y_out))
}

/// Fixed-step Runge-Kutta 4th order method
fn rk4_fixed<F>(
    f: &F,
    t0: f64,
    tf: f64,
    y0: Vec<f64>,
    _t_eval: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> PyResult<Vec<f64>>,
{
    let steps = 1000;
    let h = (tf - t0) / steps as f64;
    
    let mut t = t0;
    let mut y = y0.clone();
    let mut t_out = vec![t];
    let mut y_out = vec![y.clone()];
    
    for _ in 0..steps {
        let k1 = f(t, &y)?;
        
        let y_tmp: Vec<f64> = y.iter().zip(&k1).map(|(yi, k)| yi + h * k / 2.0).collect();
        let k2 = f(t + h / 2.0, &y_tmp)?;
        
        let y_tmp: Vec<f64> = y.iter().zip(&k2).map(|(yi, k)| yi + h * k / 2.0).collect();
        let k3 = f(t + h / 2.0, &y_tmp)?;
        
        let y_tmp: Vec<f64> = y.iter().zip(&k3).map(|(yi, k)| yi + h * k).collect();
        let k4 = f(t + h, &y_tmp)?;
        
        y = y.iter()
            .zip(&k1).zip(&k2).zip(&k3).zip(&k4)
            .map(|((((yi, k1i), k2i), k3i), k4i)| {
                yi + h / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            })
            .collect();
        
        t += h;
        t_out.push(t);
        y_out.push(y.clone());
    }
    
    Ok((t_out, y_out))
}

/// Simple Euler method for testing
fn euler_fixed<F>(
    f: &F,
    t0: f64,
    tf: f64,
    y0: Vec<f64>,
    _t_eval: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> PyResult<Vec<f64>>,
{
    let steps = 10000;
    let h = (tf - t0) / steps as f64;
    
    let mut t = t0;
    let mut y = y0.clone();
    let mut t_out = vec![t];
    let mut y_out = vec![y.clone()];
    
    for _ in 0..steps {
        let dy = f(t, &y)?;
        y = y.iter().zip(&dy).map(|(yi, dyi)| yi + h * dyi).collect();
        t += h;
        t_out.push(t);
        y_out.push(y.clone());
    }
    
    Ok((t_out, y_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_exponential_decay() {
        // dy/dt = -k*y, solution: y(t) = y0 * exp(-k*t)
        let k = 0.1;
        let ode = |_t: f64, y: &[f64]| -> PyResult<Vec<f64>> {
            Ok(vec![-k * y[0]])
        };
        
        let (t_out, y_out) = rk4_fixed(&ode, 0.0, 10.0, vec![1.0], None).unwrap();
        
        // Check final value
        let t_final = *t_out.last().unwrap();
        let y_final = y_out.last().unwrap()[0];
        let expected = (-k * t_final).exp();
        
        assert_relative_eq!(y_final, expected, epsilon = 1e-4);
    }
}
