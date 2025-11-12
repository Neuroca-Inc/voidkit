/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};

/// Calculates a proxy for Brain-Derived Neurotrophic Factor (BDNF) levels,
/// which can be used to trigger structural plasticity.
///
/// This proxy is based on the principle that correlated, rewarded activity
/// promotes structural growth.
///
/// # Arguments
/// * `spike_times_pre` - Spike times of the pre-synaptic neuron
/// * `spike_times_post` - Spike times of the post-synaptic neuron
/// * `rewards` - An array of reward signals (Nx2 array with time, reward value)
/// * `time_window` - The time window (in ms) to consider for correlated activity
///
/// # Returns
/// The calculated BDNF proxy value
#[pyfunction]
#[pyo3(signature = (spike_times_pre, spike_times_post, rewards, time_window=50.0))]
pub fn calculate_bdnf_proxy(
    spike_times_pre: PyReadonlyArray1<f64>,
    spike_times_post: PyReadonlyArray1<f64>,
    rewards: PyReadonlyArray2<f64>,
    time_window: f64,
) -> PyResult<f64> {
    let pre_times = spike_times_pre.as_slice()?;
    let post_times = spike_times_post.as_slice()?;
    let rewards_array = rewards.as_array();
    
    let mut bdnf_proxy = 0.0;

    for &t_pre in pre_times {
        // Find post-synaptic spikes within the time window
        let post_in_window: Vec<f64> = post_times
            .iter()
            .filter(|&&t_post| t_post > t_pre && t_post <= t_pre + time_window)
            .copied()
            .collect();

        if !post_in_window.is_empty() {
            // Find the reward associated with this correlated activity
            let mut max_reward = 0.0;
            for i in 0..rewards_array.shape()[0] {
                let reward_time = rewards_array[[i, 0]];
                let reward_value = rewards_array[[i, 1]];
                
                if reward_time > t_pre && reward_time <= t_pre + time_window {
                    max_reward = f64::max(max_reward, reward_value);
                }
            }
            
            bdnf_proxy += max_reward * post_in_window.len() as f64;
        }
    }

    Ok(bdnf_proxy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Note: Tests requiring Python/numpy integration are disabled pending
    // test infrastructure updates for PyO3 0.22 API
    
    /*
    #[test]
    fn test_calculate_bdnf_proxy_basic() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let pre_times = vec![10.0, 20.0, 30.0];
            let post_times = vec![15.0, 25.0, 35.0];
            let rewards = vec![
                [12.0, 1.0],
                [22.0, 2.0],
                [32.0, 1.5],
            ];
            
            let pre_array = numpy::PyArray1::from_vec(py, pre_times);
            let post_array = numpy::PyArray1::from_vec(py, post_times);
            let rewards_array = numpy::PyArray2::from_vec2(py, &rewards).unwrap();
            
            let result = calculate_bdnf_proxy(
                pre_array.readonly(),
                post_array.readonly(),
                rewards_array.readonly(),
                50.0,
            ).unwrap();
            
            // Each pre-spike has a post-spike within window and gets reward
            // 10->15 (reward 1.0 * 1 spike) + 20->25 (reward 2.0 * 1 spike) + 30->35 (reward 1.5 * 1 spike)
            assert_relative_eq!(result, 4.5, epsilon = 1e-6);
        });
    }

    #[test]
    fn test_calculate_bdnf_proxy_no_correlation() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let pre_times = vec![10.0, 20.0];
            let post_times = vec![100.0, 200.0]; // Far apart
            let rewards = vec![[15.0, 1.0]];
            
            let pre_array = numpy::PyArray1::from_vec(py, pre_times);
            let post_array = numpy::PyArray1::from_vec(py, post_times);
            let rewards_array = numpy::PyArray2::from_vec2(py, &rewards).unwrap();
            
            let result = calculate_bdnf_proxy(
                pre_array.readonly(),
                post_array.readonly(),
                rewards_array.readonly(),
                50.0,
            ).unwrap();
            
            assert_relative_eq!(result, 0.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_calculate_bdnf_proxy_empty_spikes() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let pre_times: Vec<f64> = vec![];
            let post_times: Vec<f64> = vec![];
            let rewards = vec![[15.0, 1.0]];
            
            let pre_array = numpy::PyArray1::from_vec(py, pre_times);
            let post_array = numpy::PyArray1::from_vec(py, post_times);
            let rewards_array = numpy::PyArray2::from_vec2(py, &rewards).unwrap();
            
            let result = calculate_bdnf_proxy(
                pre_array.readonly(),
                post_array.readonly(),
                rewards_array.readonly(),
                50.0,
            ).unwrap();
            
            assert_relative_eq!(result, 0.0, epsilon = 1e-10);
        });
    }
    */
}
