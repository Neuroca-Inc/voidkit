/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray2, IntoPyArray};
use ndarray;

/// Detects bursts of spikes in a spike train.
///
/// A burst is defined as a sequence of spikes where the inter-spike interval
/// is less than or equal to max_interspike_interval.
///
/// # Arguments
/// * `spike_times` - A 1D array of spike times
/// * `max_interspike_interval` - The maximum time between spikes to be considered part of a burst
/// * `min_spikes_in_burst` - The minimum number of spikes required to form a burst
///
/// # Returns
/// An array of the start and end times of the detected bursts (Nx2 array)
#[pyfunction]
#[pyo3(signature = (spike_times, max_interspike_interval=10.0, min_spikes_in_burst=3))]
pub fn detect_bursts<'py>(
    py: Python<'py>,
    spike_times: PyReadonlyArray1<f64>,
    max_interspike_interval: f64,
    min_spikes_in_burst: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let times = spike_times.as_slice()?;
    
    if times.len() < min_spikes_in_burst {
        // Return empty 2D array
        return Ok(ndarray::Array2::from_shape_vec((0, 2), vec![])
            .unwrap()
            .into_pyarray_bound(py));
    }

    // Calculate interspike intervals
    let mut interspike_intervals = Vec::with_capacity(times.len() - 1);
    for i in 0..times.len() - 1 {
        interspike_intervals.push(times[i + 1] - times[i]);
    }

    let is_in_burst: Vec<bool> = interspike_intervals
        .iter()
        .map(|&interval| interval <= max_interspike_interval)
        .collect();

    let mut bursts: Vec<[f64; 2]> = Vec::new();
    let mut current_burst_start_idx: Option<usize> = None;

    for i in 0..is_in_burst.len() {
        if is_in_burst[i] && current_burst_start_idx.is_none() {
            current_burst_start_idx = Some(i);
        } else if !is_in_burst[i] && current_burst_start_idx.is_some() {
            let start_idx = current_burst_start_idx.unwrap();
            let burst_length = i - start_idx + 1;
            
            if burst_length >= min_spikes_in_burst {
                bursts.push([times[start_idx], times[i]]);
            }
            current_burst_start_idx = None;
        }
    }

    // Handle the case where the spike train ends with a burst
    if let Some(start_idx) = current_burst_start_idx {
        let burst_length = times.len() - start_idx;
        if burst_length >= min_spikes_in_burst {
            bursts.push([times[start_idx], times[times.len() - 1]]);
        }
    }

    // Convert to numpy array
    if bursts.is_empty() {
        Ok(ndarray::Array2::from_shape_vec((0, 2), vec![])
            .unwrap()
            .into_pyarray_bound(py))
    } else {
        let flat: Vec<f64> = bursts.iter().flat_map(|b| vec![b[0], b[1]]).collect();
        Ok(ndarray::Array2::from_shape_vec((bursts.len(), 2), flat)
            .unwrap()
            .into_pyarray_bound(py))
    }
}

#[cfg(test)]
mod tests {

    // Note: Tests requiring Python/numpy integration are disabled pending
    // test infrastructure updates for PyO3 0.22 API
    
    /*
    #[test]
    fn test_detect_bursts_basic() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Spike train with a clear burst: spikes at 0, 5, 10, 15 (ISI=5)
            // Then a gap, then another burst at 100, 105, 110 (ISI=5)
            let spike_times = vec![0.0, 5.0, 10.0, 15.0, 100.0, 105.0, 110.0];
            let times_array = numpy::PyArray1::from_vec(py, spike_times.clone());
            
            let result = detect_bursts(py, times_array.readonly(), 10.0, 3).unwrap();
            let result_array = result.readonly();
            let result_slice = result_array.as_array();
            
            assert_eq!(result_slice.shape()[0], 2); // Two bursts
            
            // First burst: 0 to 15
            assert_relative_eq!(result_slice[[0, 0]], 0.0, epsilon = 1e-10);
            assert_relative_eq!(result_slice[[0, 1]], 15.0, epsilon = 1e-10);
            
            // Second burst: 100 to 110
            assert_relative_eq!(result_slice[[1, 0]], 100.0, epsilon = 1e-10);
            assert_relative_eq!(result_slice[[1, 1]], 110.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_detect_bursts_no_bursts() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Spikes too far apart
            let spike_times = vec![0.0, 50.0, 100.0, 150.0];
            let times_array = numpy::PyArray1::from_vec(py, spike_times);
            
            let result = detect_bursts(py, times_array.readonly(), 10.0, 3).unwrap();
            let result_array = result.readonly();
            
            assert_eq!(result_array.as_array().shape()[0], 0);
        });
    }

    #[test]
    fn test_detect_bursts_too_few_spikes() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let spike_times = vec![0.0, 5.0]; // Only 2 spikes
            let times_array = numpy::PyArray1::from_vec(py, spike_times);
            
            let result = detect_bursts(py, times_array.readonly(), 10.0, 3).unwrap();
            let result_array = result.readonly();
            
            assert_eq!(result_array.as_array().shape()[0], 0);
        });
    }

    #[test]
    fn test_detect_bursts_min_spikes_threshold() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Exactly 3 spikes in burst
            let spike_times = vec![0.0, 5.0, 10.0];
            let times_array = numpy::PyArray1::from_vec(py, spike_times.clone());
            
            let result = detect_bursts(py, times_array.readonly(), 10.0, 3).unwrap();
            let result_array = result.readonly();
            let result_slice = result_array.as_array();
            
            assert_eq!(result_slice.shape()[0], 1);
            assert_relative_eq!(result_slice[[0, 0]], 0.0, epsilon = 1e-10);
            assert_relative_eq!(result_slice[[0, 1]], 10.0, epsilon = 1e-10);
        });
    }
    */
}
