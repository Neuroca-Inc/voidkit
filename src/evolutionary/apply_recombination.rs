/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

//! Recombination/Crossover Operator for Evolutionary Algorithms
//!
//! Performs crossover between two sets of weights.

use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use pyo3::prelude::*;
use rand::thread_rng;
use rand::Rng;

/// Performs crossover/recombination between two sets of weights.
///
/// # Arguments
/// * `weights1` - First set of weights
/// * `weights2` - Second set of weights
/// * `recombination_prob` - Probability of choosing from first set (default: 0.5)
///
/// # Returns
/// New weights after recombination
#[allow(clippy::manual_range_contains)]
#[pyfunction]
#[pyo3(signature = (weights1, weights2, recombination_prob = 0.5))]
pub fn apply_recombination<'py>(
    py: Python<'py>,
    weights1: PyReadonlyArray<'py, f64, ndarray::IxDyn>,
    weights2: PyReadonlyArray<'py, f64, ndarray::IxDyn>,
    recombination_prob: f64,
) -> PyResult<Bound<'py, PyArray<f64, ndarray::IxDyn>>> {
    if recombination_prob < 0.0 || recombination_prob > 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "recombination_prob must be between 0 and 1",
        ));
    }

    let w1 = weights1.as_array();
    let w2 = weights2.as_array();

    if w1.shape() != w2.shape() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Weight arrays must have the same shape",
        ));
    }

    // Start with copy of weights1
    let mut result = w1.to_owned();
    let mut rng = thread_rng();

    // Perform recombination
    for (r_elem, &w2_elem) in result.iter_mut().zip(w2.iter()) {
        if rng.gen::<f64>() >= recombination_prob {
            *r_elem = w2_elem;
        }
    }

    Ok(result.into_pyarray_bound(py))
}
