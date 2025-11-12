/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Stochastic Differential Equation (SDE) Solver
//! 
//! Solves systems of SDEs using the Euler-Maruyama method.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Solves a system of stochastic differential equations (SDEs) using the
/// Euler-Maruyama method.
/// 
/// For SDEs of the form: dx/dt = drift_func(x) + diffusion_func(x) * dW_t
/// where dW_t is Wiener process noise.
/// 
/// # Arguments
/// * `drift_func` - Python callable for drift: f(x) -> dx/dt (deterministic part)
/// * `diffusion_func` - Python callable for diffusion: g(x) -> noise coefficient
/// * `initial_state` - Initial state vector
/// * `t_span` - Time span tuple (t_start, t_end)
/// * `dt` - Time step size
/// 
/// # Returns
/// Tuple of (times, states) where:
/// - times: 1D array of time points
/// - states: 2D array where each row is the state at that time
#[pyfunction]
#[pyo3(signature = (drift_func, diffusion_func, initial_state, t_span, dt))]
pub fn sde_solver<'py>(
    py: Python<'py>,
    drift_func: PyObject,
    diffusion_func: PyObject,
    initial_state: PyReadonlyArray1<f64>,
    t_span: (f64, f64),
    dt: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (t_start, t_end) = t_span;
    
    if t_end <= t_start {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "t_end must be greater than t_start"
        ));
    }
    
    if dt <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "dt must be positive"
        ));
    }
    
    let initial = initial_state.as_array();
    let n_dim = initial.len();
    let n_steps = ((t_end - t_start) / dt) as usize;
    
    // Initialize time array
    let mut times = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        times.push(t_start + (i as f64) * dt);
    }
    
    // Initialize state array
    let mut states = Vec::with_capacity((n_steps + 1) * n_dim);
    states.extend(initial.to_vec());
    
    // Random number generator for Wiener process
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, dt.sqrt()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create normal distribution: {}", e))
    })?;
    
    // Current state vector
    let mut current_state = initial.to_vec();
    
    // Euler-Maruyama iteration
    for _step in 0..n_steps {
        // Convert current state to Python array
        let state_py = PyArray1::from_vec_bound(py, current_state.clone());
        
        // Evaluate drift function
        let drift: Vec<f64> = drift_func.call1(py, (state_py.clone(),))?.extract(py)?;
        
        // Evaluate diffusion function
        let diffusion: Vec<f64> = diffusion_func.call1(py, (state_py,))?.extract(py)?;
        
        if drift.len() != n_dim || diffusion.len() != n_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "drift_func and diffusion_func must return arrays of the same length as initial_state"
            ));
        }
        
        // Generate Wiener increments
        let dw: Vec<f64> = (0..n_dim)
            .map(|_| normal.sample(&mut rng))
            .collect();
        
        // Euler-Maruyama step: x[i+1] = x[i] + drift * dt + diffusion * dW
        for i in 0..n_dim {
            current_state[i] += drift[i] * dt + diffusion[i] * dw[i];
        }
        
        states.extend(&current_state);
    }
    
    // Convert to numpy arrays
    let times_py = PyArray1::from_vec_bound(py, times);
    
    // Reshape states to (n_steps + 1, n_dim)
    let states_2d: Vec<Vec<f64>> = states
        .chunks(n_dim)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    let states_py = PyArray2::from_vec2_bound(py, &states_2d)?;
    
    Ok((times_py, states_py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sde_solver_deterministic() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Test with zero diffusion (should behave like ODE)
            // dx/dt = -x (exponential decay)
            let code = r#"
def drift(x):
    return [-x[0]]

def diffusion(x):
    return [0.0]
"#;
            let locals = pyo3::types::PyDict::new_bound(py);
            py.run_bound(code, None, Some(&locals)).unwrap();
            
            let drift: PyObject = locals.get_item("drift").unwrap().unwrap().into();
            let diffusion: PyObject = locals.get_item("diffusion").unwrap().unwrap().into();
            
            let initial = numpy::PyArray1::from_vec_bound(py, vec![1.0]);
            let t_span = (0.0, 1.0);
            let dt = 0.1;
            
            let (times, states) = sde_solver(
                py,
                drift,
                diffusion,
                initial.readonly(),
                t_span,
                dt
            ).unwrap();
            
            let times_arr = times.readonly();
            let states_arr = states.readonly();
            
            // Check dimensions
            assert_eq!(times_arr.len(), 11); // 0.0, 0.1, ..., 1.0
            assert_eq!(states_arr.shape(), [11, 1]);
            
            // Check initial state
            assert!((states_arr[[0, 0]] - 1.0).abs() < 1e-10);
            
            // Final state should be approximately exp(-1) ≈ 0.368
            // (with some numerical error from Euler method)
            let final_val = states_arr[[10, 0]];
            assert!(final_val < 0.5 && final_val > 0.2);
        });
    }
}
