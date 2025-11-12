/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Mutation Operator for Evolutionary Algorithms
//! 
//! Applies random mutations to weights/parameters.

use pyo3::prelude::*;
use numpy::{PyArray, PyReadonlyArray, IntoPyArray};
use ndarray::ArrayD;
use rand::thread_rng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Applies random mutations to a set of weights.
/// 
/// # Arguments
/// * `weights` - The weights to mutate (any dimensional array)
/// * `mutation_rate` - Probability of each weight being mutated (default: 0.01)
/// * `mutation_scale` - Standard deviation of Gaussian noise (default: 0.1)
/// 
/// # Returns
/// Mutated weights array
#[pyfunction]
#[pyo3(signature = (weights, mutation_rate = 0.01, mutation_scale = 0.1))]
pub fn apply_mutation<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray<'py, f64, ndarray::IxDyn>,
    mutation_rate: f64,
    mutation_scale: f64,
) -> PyResult<Bound<'py, PyArray<f64, ndarray::IxDyn>>> {
    if mutation_rate < 0.0 || mutation_rate > 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mutation_rate must be between 0 and 1"
        ));
    }
    
    if mutation_scale < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mutation_scale must be non-negative"
        ));
    }
    
    let weights_arr = weights.as_array();
    let mut mutated = weights_arr.to_owned();
    
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, mutation_scale).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid mutation_scale: {}", e))
    })?;
    
    // Apply mutations
    for elem in mutated.iter_mut() {
        if rng.gen::<f64>() < mutation_rate {
            let mutation = normal.sample(&mut rng);
            *elem += mutation;
        }
    }
    
    Ok(mutated.into_pyarray_bound(py))
}
