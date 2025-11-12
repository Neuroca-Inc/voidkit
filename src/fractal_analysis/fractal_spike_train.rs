/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Fractal Spike Train Generation
//! 
//! Generates spike trains based on fractal dynamics.

use pyo3::prelude::*;
use numpy::{PyArray1};
use rand::thread_rng;
use rand::Rng;

/// Generates a spike train based on a fractal dynamics rule.
/// 
/// spike_rate_i = k * D_f * e^(-t/τ_f)
/// 
/// Spikes are generated using a Poisson process with the time-varying rate.
/// 
/// # Arguments
/// * `fractal_dimension` - The fractal dimension D_f
/// * `k` - Scaling factor
/// * `tau_f` - Time constant for exponential decay
/// * `duration` - Duration of spike train to generate
/// * `dt` - Time step (default: 1.0 ms)
/// 
/// # Returns
/// Array of spike times
#[pyfunction]
#[pyo3(signature = (fractal_dimension, k, tau_f, duration, dt = 1.0))]
pub fn generate_fractal_spike_train<'py>(
    py: Python<'py>,
    fractal_dimension: f64,
    k: f64,
    tau_f: f64,
    duration: f64,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if duration <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "duration must be positive"
        ));
    }
    
    if dt <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "dt must be positive"
        ));
    }
    
    let n_steps = (duration / dt) as usize;
    let mut spike_times = Vec::new();
    let mut rng = thread_rng();
    
    for i in 0..n_steps {
        let t = i as f64 * dt;
        
        // Calculate spike rate at this time
        let spike_rate = k * fractal_dimension * (-t / tau_f).exp();
        
        // Generate spike with Poisson probability
        // Assuming dt is in ms, convert rate to probability
        let spike_probability = spike_rate * dt / 1000.0;
        
        if rng.gen::<f64>() < spike_probability {
            spike_times.push(t);
        }
    }
    
    Ok(PyArray1::from_vec_bound(py, spike_times))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_fractal_spike_train() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let fractal_dim = 1.5;
            let k = 10.0;
            let tau_f = 100.0;
            let duration = 1000.0;
            let dt = 1.0;
            
            let spike_train = generate_fractal_spike_train(
                py,
                fractal_dim,
                k,
                tau_f,
                duration,
                dt
            ).unwrap();
            
            let times = spike_train.readonly();
            let times_arr = times.as_array();
            
            // Should generate some spikes
            assert!(times_arr.len() > 0);
            
            // All spike times should be within duration
            for &spike_time in times_arr.iter() {
                assert!(spike_time >= 0.0 && spike_time < duration);
            }
            
            // Spikes should be monotonically increasing
            for i in 1..times_arr.len() {
                assert!(times_arr[i] > times_arr[i-1]);
            }
        });
    }

    #[test]
    fn test_zero_duration() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let result = generate_fractal_spike_train(
                py, 1.5, 10.0, 100.0, 0.0, 1.0
            );
            
            assert!(result.is_err());
        });
    }
}
