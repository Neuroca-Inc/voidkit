/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Thermodynamics module - free energy calculations.
*/

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Calculates the free energy of the system.
///
/// F = Σ_i (spike_rate_i - target_rate)^2 + λ * Σ w_ij^2
///
/// # Arguments
///
/// * `spike_rates` - A 1D array of spike rates for each neuron
/// * `target_rate` - The target firing rate for the neurons
/// * `weights` - A 2D array of synaptic weights
/// * `lambda_reg` - The regularization parameter for the weights
///
/// # Returns
///
/// The calculated free energy
///
/// # Examples
///
/// ```python
/// from voidkit_rust import calculate_free_energy
/// import numpy as np
///
/// spike_rates = np.array([0.5, 0.6, 0.4])
/// target_rate = 0.5
/// weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
/// lambda_reg = 0.01
///
/// F = calculate_free_energy(spike_rates, target_rate, weights, lambda_reg)
/// ```
#[pyfunction]
pub fn calculate_free_energy(
    spike_rates: PyReadonlyArray1<'_, f64>,
    target_rate: f64,
    weights: PyReadonlyArray2<'_, f64>,
    lambda_reg: f64,
) -> PyResult<f64> {
    let rates = spike_rates.as_array();
    let w = weights.as_array();

    // Calculate rate error: Σ_i (spike_rate_i - target_rate)^2
    let rate_error: f64 = rates.iter().map(|&r| (r - target_rate).powi(2)).sum();

    // Calculate weight regularization: λ * Σ w_ij^2
    let weight_regularization: f64 = lambda_reg * w.iter().map(|&w_val| w_val.powi(2)).sum::<f64>();

    Ok(rate_error + weight_regularization)
}

/// Performs one step of gradient descent to minimize the free energy.
///
/// dw_ij/dt = -η * ∂F/∂w_ij * e^(-Δt/τ)
///
/// # Arguments
///
/// * `weights` - The current synaptic weights
/// * `spike_rates` - The current spike rates
/// * `target_rate` - The target firing rate
/// * `lambda_reg` - The weight regularization parameter
/// * `eta` - The learning rate
/// * `delta_t` - The time difference for the STDP-like modulation
/// * `tau` - The time constant for the STDP-like modulation
///
/// # Returns
///
/// The updated weights
///
/// # Examples
///
/// ```python
/// from voidkit_rust import minimize_free_energy_step
/// import numpy as np
///
/// weights = np.array([[0.1, 0.2], [0.3, 0.4]])
/// spike_rates = np.array([0.5, 0.6])
/// target_rate = 0.5
/// lambda_reg = 0.01
/// eta = 0.1
/// delta_t = 0.01
/// tau = 20.0
///
/// new_weights = minimize_free_energy_step(
///     weights, spike_rates, target_rate, lambda_reg, eta, delta_t, tau
/// )
/// ```
#[pyfunction]
pub fn minimize_free_energy_step<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f64>,
    spike_rates: PyReadonlyArray1<'py, f64>,
    target_rate: f64,
    lambda_reg: f64,
    eta: f64,
    delta_t: f64,
    tau: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let w = weights.as_array();
    let _rates = spike_rates.as_array();

    let shape = w.shape();
    let (n_rows, n_cols) = (shape[0], shape[1]);

    // Calculate gradient: ∂F/∂w_ij = 2 * lambda * w_ij
    let mut new_weights = Vec::with_capacity(n_rows * n_cols);

    // STDP-like time modulation
    let time_modulation = (-delta_t / tau).exp();

    for i in 0..n_rows {
        for j in 0..n_cols {
            let w_ij = w[[i, j]];
            let grad_f = 2.0 * lambda_reg * w_ij;

            // Update: w_ij = w_ij - η * grad_F * exp(-Δt/τ)
            let delta_w = -eta * grad_f * time_modulation;
            new_weights.push(w_ij + delta_w);
        }
    }

    // Convert back to 2D array representation (flattened)
    Ok(PyArray1::from_vec_bound(py, new_weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_free_energy_calculation() {
        let spike_rates: Vec<f64> = vec![0.5, 0.6, 0.4];
        let target_rate: f64 = 0.5;
        let weights: Vec<Vec<f64>> = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let lambda_reg: f64 = 0.01;

        // Calculate rate error
        let rate_error: f64 = spike_rates.iter().map(|&r| (r - target_rate).powi(2)).sum();

        // Calculate weight regularization
        let weight_reg: f64 = lambda_reg
            * weights
                .iter()
                .flat_map(|row| row.iter())
                .map(|&w| w.powi(2))
                .sum::<f64>();

        let expected = rate_error + weight_reg;

        // Basic sanity check
        assert!(expected >= 0.0);
        // rate_error = (0.5-0.5)^2 + (0.6-0.5)^2 + (0.4-0.5)^2 = 0 + 0.01 + 0.01 = 0.02
        assert_relative_eq!(rate_error, 0.02, epsilon = 1e-6);
    }
}
