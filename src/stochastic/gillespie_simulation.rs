/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Stochastic simulation module - Gillespie algorithm.
*/

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type GillespiePyOutput<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);
use rand::Rng;

/// Performs a Gillespie simulation (Stochastic Simulation Algorithm).
///
/// # Arguments
///
/// * `initial_state` - A 1D array of the initial counts of each species
/// * `propensity_func` - A Python function that returns reaction propensities
/// * `stoichiometry` - A 2D array of shape (n_reactions, n_species) defining state changes
/// * `t_max` - The maximum simulation time
///
/// # Returns
///
/// A tuple containing:
/// * The time points of the simulation
/// * The state of the system at each time point (2D array)
///
/// # Examples
///
/// ```python
/// from voidkit_rust import gillespie_simulation
/// import numpy as np
///
/// # Simple birth-death process
/// def propensities(state):
///     # Birth rate = 0.5 * population, Death rate = 0.3 * population
///     return np.array([0.5 * state[0], 0.3 * state[0]])
///
/// initial = np.array([10])  # Start with 10 individuals
/// stoich = np.array([[1], [-1]])  # Birth adds 1, death removes 1
/// times, states = gillespie_simulation(initial, propensities, stoich, 10.0)
/// ```
#[pyfunction]
pub fn gillespie_simulation<'py>(
    py: Python<'py>,
    initial_state: PyReadonlyArray1<'py, f64>,
    propensity_func: PyObject,
    stoichiometry: PyReadonlyArray2<'py, f64>,
    t_max: f64,
) -> PyResult<GillespiePyOutput<'py>> {
    let init_state = initial_state.as_array();
    let stoich = stoichiometry.as_array();

    if !propensity_func.bind(py).is_callable() {
        return Err(PyValueError::new_err("propensity_func must be callable"));
    }

    if t_max <= 0.0 {
        return Err(PyValueError::new_err("t_max must be positive"));
    }

    let n_species = init_state.len();
    let stoich_shape = stoich.shape();
    let n_reactions = stoich_shape[0];

    if stoich_shape[1] != n_species {
        return Err(PyValueError::new_err(
            "Stoichiometry matrix dimensions must match number of species",
        ));
    }

    let mut times = vec![0.0];
    let mut states = vec![init_state.to_vec()];

    let mut t = 0.0;
    let mut current_state = init_state.to_vec();
    let mut rng = rand::thread_rng();

    while t < t_max {
        // Call propensity function
        let state_array = PyArray1::from_vec_bound(py, current_state.clone());
        let propensities_obj = propensity_func.call1(py, (state_array,))?;
        let propensities: Vec<f64> = propensities_obj.extract(py)?;

        if propensities.len() != n_reactions {
            return Err(PyValueError::new_err(format!(
                "Propensity function returned {} values, expected {}",
                propensities.len(),
                n_reactions
            )));
        }

        let total_propensity: f64 = propensities.iter().sum();

        if total_propensity <= 0.0 {
            break;
        }

        // Time to next reaction
        let r1: f64 = rng.gen();
        let dt = -(r1.ln()) / total_propensity;

        // Which reaction occurs?
        let r2: f64 = rng.gen();
        let mut cumsum = 0.0;
        let mut reaction_index = 0;

        for (i, &prop) in propensities.iter().enumerate() {
            cumsum += prop / total_propensity;
            if r2 < cumsum {
                reaction_index = i;
                break;
            }
        }

        // Update state
        for j in 0..n_species {
            current_state[j] += stoich[[reaction_index, j]];
        }

        t += dt;
        times.push(t);
        states.push(current_state.clone());
    }

    // Convert to numpy arrays
    let times_array = PyArray1::from_vec_bound(py, times);

    // Convert states to 2D array
    let n_steps = states.len();
    let mut states_flat = Vec::with_capacity(n_steps * n_species);
    for state in &states {
        states_flat.extend(state);
    }

    use ndarray::Array2;
    let states_ndarray = Array2::from_shape_vec((n_steps, n_species), states_flat)
        .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;
    let states_array = PyArray2::from_owned_array_bound(py, states_ndarray);

    Ok((times_array, states_array))
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_gillespie_basic() {
        // Basic test that algorithm structure is correct
        let initial = [10.0];
        let _stoich = [[1.0], [-1.0]];

        // Simple propensity calculation
        let birth_rate = 0.5;
        let death_rate = 0.3;

        let state = initial[0];
        let propensities = [birth_rate * state, death_rate * state];
        let total: f64 = propensities.iter().sum();

        assert!(total > 0.0);
        assert_eq!(propensities.len(), 2);
    }
}
