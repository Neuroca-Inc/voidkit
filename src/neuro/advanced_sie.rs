/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles.
Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Self-Improvement Engine (SIE) multi-objective reward functions.
//!
//! This module provides the stabilized SIE reward calculation and its
//! reward-dependent STDP modulation support.

use pyo3::prelude::*;

/// Calculates a stabilized multi-objective reward function.
///
/// The reward function combines temporal difference error, novelty, habituation,
/// and self-benefit components:
///
/// R_tot(t) = w_r * TD_error(t) + w_n * novelty(t) * (1 - tanh(habituation(t))) + w_s * self_benefit(t)
///
/// where w_r = w_r_base * e^(-λ * |external_reward|)
///
/// # Arguments
///
/// * `td_error` - Temporal difference error signal
/// * `novelty` - Novelty score for the current state
/// * `habituation` - Habituation level (higher = more habituated)
/// * `self_benefit` - Self-benefit component
/// * `external_reward` - External reward signal
/// * `w_r_base` - Base weight for TD error component (default: 0.6)
/// * `w_n` - Weight for novelty component (default: 0.3)
/// * `w_s` - Weight for self-benefit component (default: 0.1)
/// * `lambda_reg` - Regularization parameter for external reward (default: 0.05)
///
/// # Returns
///
/// The calculated stabilized reward value
///
/// # Example
///
/// ```
/// use voidkit_rust::neuro::advanced_sie::calculate_stabilized_reward;
///
/// let reward = calculate_stabilized_reward(
///     0.5,    // td_error
///     0.8,    // novelty
///     0.2,    // habituation
///     0.3,    // self_benefit
///     1.0,    // external_reward
///     0.6,    // w_r_base
///     0.3,    // w_n
///     0.1,    // w_s
///     0.05    // lambda_reg
/// );
/// ```
#[pyfunction]
#[pyo3(signature = (
    td_error,
    novelty,
    habituation,
    self_benefit,
    external_reward,
    w_r_base = 0.6,
    w_n = 0.3,
    w_s = 0.1,
    lambda_reg = 0.05
))]
pub fn calculate_stabilized_reward(
    td_error: f64,
    novelty: f64,
    habituation: f64,
    self_benefit: f64,
    external_reward: f64,
    w_r_base: f64,
    w_n: f64,
    w_s: f64,
    lambda_reg: f64,
) -> PyResult<f64> {
    // Calculate modulated weight for TD error based on external reward
    let w_r = w_r_base * (-lambda_reg * external_reward.abs()).exp();

    // Calculate total reward combining all components
    let reward = w_r * td_error + w_n * novelty * (1.0 - habituation.tanh()) + w_s * self_benefit;

    Ok(reward)
}

/// Calculates the STDP weight change with quadratic reward modulation.
///
/// The weight change is computed as:
///
/// Δw_ij = η * (1 + β * R_tot²) * e^(-Δt/τ)
///
/// This combines spike-timing-dependent plasticity with reward-dependent modulation,
/// where the modulation scales quadratically with the total reward signal.
///
/// # Arguments
///
/// * `eta_base` - Base learning rate (default: 0.12)
/// * `beta` - Quadratic reward modulation factor (default: 0.15)
/// * `tau` - Time constant for STDP decay (default: 15.0)
/// * `delta_t` - Time difference between pre- and post-synaptic spikes
/// * `total_reward` - Total reward signal
///
/// # Returns
///
/// The calculated STDP weight change
///
/// # Example
///
/// ```
/// use voidkit_rust::neuro::advanced_sie::apply_quadratic_stdp_modulation;
///
/// let delta_w = apply_quadratic_stdp_modulation(
///     0.12,   // eta_base
///     0.15,   // beta
///     15.0,   // tau
///     5.0,    // delta_t
///     0.5     // total_reward
/// );
/// ```
#[pyfunction]
#[pyo3(signature = (
    eta_base = 0.12,
    beta = 0.15,
    tau = 15.0,
    delta_t = 0.0,
    total_reward = 0.0
))]
pub fn apply_quadratic_stdp_modulation(
    eta_base: f64,
    beta: f64,
    tau: f64,
    delta_t: f64,
    total_reward: f64,
) -> PyResult<f64> {
    // Calculate reward-modulated learning rate
    let eta = eta_base * (1.0 + beta * total_reward.powi(2));

    // Apply temporal decay based on spike timing
    let delta_w = eta * (-delta_t / tau).exp();

    Ok(delta_w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_calculate_stabilized_reward_basic() {
        // Test with simple values
        let reward = calculate_stabilized_reward(
            1.0,  // td_error
            0.5,  // novelty
            0.0,  // habituation (no habituation)
            0.2,  // self_benefit
            0.0,  // external_reward (no external reward)
            0.6,  // w_r_base
            0.3,  // w_n
            0.1,  // w_s
            0.05, // lambda_reg
        )
        .unwrap();

        // Expected: 0.6 * 1.0 + 0.3 * 0.5 * (1 - tanh(0)) + 0.1 * 0.2
        // = 0.6 + 0.15 + 0.02 = 0.77
        assert_relative_eq!(reward, 0.77, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_stabilized_reward_with_habituation() {
        // Test with habituation effect
        let reward = calculate_stabilized_reward(
            0.0,  // td_error
            1.0,  // novelty
            2.0,  // habituation (high)
            0.0,  // self_benefit
            0.0,  // external_reward
            0.6,  // w_r_base
            0.3,  // w_n
            0.1,  // w_s
            0.05, // lambda_reg
        )
        .unwrap();

        // Expected: 0.3 * 1.0 * (1 - tanh(2.0))
        let expected = 0.3 * (1.0 - 2.0_f64.tanh());
        assert_relative_eq!(reward, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_stabilized_reward_external_modulation() {
        // Test external reward modulation
        let reward1 =
            calculate_stabilized_reward(1.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.3, 0.1, 0.05).unwrap();

        let reward2 =
            calculate_stabilized_reward(1.0, 0.0, 0.0, 0.0, 5.0, 0.6, 0.3, 0.1, 0.05).unwrap();

        // With external reward, w_r should be reduced
        // reward2 should be less than reward1
        assert!(reward2 < reward1);
    }

    #[test]
    fn test_apply_quadratic_stdp_modulation_zero_reward() {
        // Test with no reward modulation
        let delta_w = apply_quadratic_stdp_modulation(
            0.1,  // eta_base
            0.15, // beta
            10.0, // tau
            0.0,  // delta_t (simultaneous spikes)
            0.0,  // total_reward
        )
        .unwrap();

        // Expected: 0.1 * (1 + 0.15 * 0) * exp(0) = 0.1
        assert_relative_eq!(delta_w, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_quadratic_stdp_modulation_with_reward() {
        // Test with reward modulation
        let delta_w = apply_quadratic_stdp_modulation(
            0.1,  // eta_base
            0.2,  // beta
            10.0, // tau
            0.0,  // delta_t
            1.0,  // total_reward
        )
        .unwrap();

        // Expected: 0.1 * (1 + 0.2 * 1.0) * exp(0) = 0.12
        assert_relative_eq!(delta_w, 0.12, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_quadratic_stdp_modulation_temporal_decay() {
        // Test temporal decay
        let delta_w1 = apply_quadratic_stdp_modulation(0.1, 0.0, 10.0, 0.0, 0.0).unwrap();

        let delta_w2 = apply_quadratic_stdp_modulation(0.1, 0.0, 10.0, 10.0, 0.0).unwrap();

        let delta_w3 = apply_quadratic_stdp_modulation(0.1, 0.0, 10.0, 20.0, 0.0).unwrap();

        // Should decay exponentially with increasing delta_t
        assert!(delta_w1 > delta_w2);
        assert!(delta_w2 > delta_w3);

        // Check specific decay value for delta_t = 10
        let expected = 0.1 * (-1.0_f64).exp(); // exp(-10/10) = exp(-1)
        assert_relative_eq!(delta_w2, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_quadratic_stdp_modulation_quadratic_scaling() {
        // Test quadratic scaling of reward
        let delta_w1 = apply_quadratic_stdp_modulation(0.1, 0.2, 10.0, 0.0, 1.0).unwrap();

        let delta_w2 = apply_quadratic_stdp_modulation(0.1, 0.2, 10.0, 0.0, 2.0).unwrap();

        // Reward 2 should give 4x modulation compared to reward 1
        // eta1 = 0.1 * (1 + 0.2 * 1) = 0.12
        // eta2 = 0.1 * (1 + 0.2 * 4) = 0.18
        assert_relative_eq!(delta_w1, 0.12, epsilon = 1e-10);
        assert_relative_eq!(delta_w2, 0.18, epsilon = 1e-10);
    }
}
