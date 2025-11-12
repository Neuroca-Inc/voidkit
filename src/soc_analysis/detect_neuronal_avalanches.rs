/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's
ethical principles. Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Neuronal Avalanche Detection
//! 
//! Detects neuronal avalanches from spike trains.

use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use std::collections::HashMap;

/// Detects neuronal avalanches from a spike train.
/// 
/// An avalanche is a continuous sequence of time bins with at least one spike,
/// preceded and succeeded by an empty time bin.
/// 
/// # Arguments
/// * `spike_times` - Array of spike times
/// * `bin_width` - Width of time bins (default: 1.0)
/// 
/// # Returns
/// Dictionary containing:
/// - 'sizes': List of avalanche sizes (number of spikes)
/// - 'durations': List of avalanche durations (number of bins)
#[pyfunction]
#[pyo3(signature = (spike_times, bin_width = 1.0))]
pub fn detect_neuronal_avalanches(
    spike_times: PyReadonlyArray1<f64>,
    bin_width: f64,
) -> PyResult<HashMap<String, Vec<i32>>> {
    let times = spike_times.as_array();
    
    if times.is_empty() {
        let mut result = HashMap::new();
        result.insert("sizes".to_string(), Vec::new());
        result.insert("durations".to_string(), Vec::new());
        return Ok(result);
    }
    
    // Find max time and create bins
    let max_time = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let n_bins = ((max_time + bin_width) / bin_width).ceil() as usize;
    
    // Bin the spike times
    let mut binned_spikes = vec![0i32; n_bins];
    for &spike_time in times.iter() {
        let bin_idx = (spike_time / bin_width).floor() as usize;
        if bin_idx < n_bins {
            binned_spikes[bin_idx] += 1;
        }
    }
    
    // Detect avalanches
    let mut sizes = Vec::new();
    let mut durations = Vec::new();
    
    let mut in_avalanche = false;
    let mut current_size = 0i32;
    let mut current_duration = 0i32;
    
    for &n_spikes in &binned_spikes {
        if n_spikes > 0 {
            if !in_avalanche {
                in_avalanche = true;
            }
            current_size += n_spikes;
            current_duration += 1;
        } else {
            if in_avalanche {
                in_avalanche = false;
                sizes.push(current_size);
                durations.push(current_duration);
                current_size = 0;
                current_duration = 0;
            }
        }
    }
    
    // Handle avalanche that extends to the end
    if in_avalanche {
        sizes.push(current_size);
        durations.push(current_duration);
    }
    
    let mut result = HashMap::new();
    result.insert("sizes".to_string(), sizes);
    result.insert("durations".to_string(), durations);
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_neuronal_avalanches() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Create a spike train with clear avalanches
            // Avalanche 1: spikes at 1.5, 2.5 (bin 1, 2) -> size=2, duration=2
            // Gap at bin 3
            // Avalanche 2: spikes at 4.5, 5.5, 6.5 (bin 4, 5, 6) -> size=3, duration=3
            let spikes = vec![1.5, 2.5, 4.5, 5.5, 6.5];
            let spikes_py = numpy::PyArray1::from_vec_bound(py, spikes);
            
            let result = detect_neuronal_avalanches(spikes_py.readonly(), 1.0).unwrap();
            
            let sizes = result.get("sizes").unwrap();
            let durations = result.get("durations").unwrap();
            
            assert_eq!(sizes.len(), 2);
            assert_eq!(sizes[0], 2);
            assert_eq!(sizes[1], 3);
            assert_eq!(durations[0], 2);
            assert_eq!(durations[1], 3);
        });
    }

    #[test]
    fn test_empty_spike_train() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let spikes = vec![];
            let spikes_py = numpy::PyArray1::from_vec_bound(py, spikes);
            
            let result = detect_neuronal_avalanches(spikes_py.readonly(), 1.0).unwrap();
            
            assert!(result.get("sizes").unwrap().is_empty());
            assert!(result.get("durations").unwrap().is_empty());
        });
    }
}
