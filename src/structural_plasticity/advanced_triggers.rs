/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

use pyo3::prelude::*;

/// Calculates an advanced, biologically-inspired growth trigger.
///
/// G(c,t) = σ(κ * (avg_reward[c] - 0.5) + ν * burst_score[c] + ρ * bdnf_proxy[c])
///
/// where σ is the sigmoid function.
///
/// # Arguments
/// * `avg_reward` - The average reward of the cluster
/// * `burst_score` - The burst score of the cluster
/// * `bdnf_proxy` - The BDNF proxy level of the cluster
/// * `kappa` - Weight for reward component (default: 2.0)
/// * `nu` - Weight for burst score component (default: 0.8)
/// * `rho` - Weight for BDNF proxy component (default: 0.5)
///
/// # Returns
/// The calculated growth trigger value (0.0 to 1.0)
#[pyfunction]
#[pyo3(signature = (avg_reward, burst_score, bdnf_proxy, kappa=2.0, nu=0.8, rho=0.5))]
pub fn calculate_advanced_growth_trigger(
    avg_reward: f64,
    burst_score: f64,
    bdnf_proxy: f64,
    kappa: f64,
    nu: f64,
    rho: f64,
) -> f64 {
    let arg = kappa * (avg_reward - 0.5) + nu * burst_score + rho * bdnf_proxy;
    sigmoid(arg)
}

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_zero() {
        let result = sigmoid(0.0);
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_positive() {
        let result = sigmoid(2.0);
        assert_relative_eq!(result, 0.8807970779778823, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_negative() {
        let result = sigmoid(-2.0);
        assert_relative_eq!(result, 0.11920292202211755, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_advanced_growth_trigger_default() {
        let result = calculate_advanced_growth_trigger(0.5, 0.0, 0.0, 2.0, 0.8, 0.5);
        // arg = 2.0 * (0.5 - 0.5) + 0.8 * 0.0 + 0.5 * 0.0 = 0.0
        // sigmoid(0.0) = 0.5
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_advanced_growth_trigger_high_reward() {
        let result = calculate_advanced_growth_trigger(1.0, 0.5, 0.3, 2.0, 0.8, 0.5);
        // arg = 2.0 * (1.0 - 0.5) + 0.8 * 0.5 + 0.5 * 0.3 = 1.0 + 0.4 + 0.15 = 1.55
        let expected = sigmoid(1.55);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_advanced_growth_trigger_low_reward() {
        let result = calculate_advanced_growth_trigger(0.0, 0.0, 0.0, 2.0, 0.8, 0.5);
        // arg = 2.0 * (0.0 - 0.5) + 0.8 * 0.0 + 0.5 * 0.0 = -1.0
        let expected = sigmoid(-1.0);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_advanced_growth_trigger_all_positive() {
        let result = calculate_advanced_growth_trigger(0.8, 0.6, 0.4, 2.0, 0.8, 0.5);
        // arg = 2.0 * (0.8 - 0.5) + 0.8 * 0.6 + 0.5 * 0.4 = 0.6 + 0.48 + 0.2 = 1.28
        let expected = sigmoid(1.28);
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_advanced_growth_trigger_bounds() {
        // Result should always be between 0 and 1
        let result = calculate_advanced_growth_trigger(10.0, 10.0, 10.0, 2.0, 0.8, 0.5);
        assert!(result > 0.0 && result < 1.0);
        assert!(result > 0.99); // Should be very close to 1
        
        let result2 = calculate_advanced_growth_trigger(-10.0, -10.0, -10.0, 2.0, 0.8, 0.5);
        assert!(result2 > 0.0 && result2 < 1.0);
        assert!(result2 < 0.01); // Should be very close to 0
    }
}
