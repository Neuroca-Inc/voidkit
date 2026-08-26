/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Calculate descriptive statistics for a given dataset.
*/

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Calculate descriptive statistics for a given dataset.
///
/// # Arguments
///
/// * `data` - Input data array (1D)
/// * `nan_policy` - How to handle NaN values ('propagate', 'omit', 'raise')
/// * `ddof` - Delta Degrees of Freedom for std/var calculations
///
/// # Returns
///
/// Dictionary containing:
/// * 'count': Number of observations
/// * 'mean': Arithmetic mean
/// * 'median': Median value
/// * 'std': Standard deviation
/// * 'var': Variance
/// * 'min': Minimum value
/// * 'max': Maximum value
/// * 'q1': First quartile (25th percentile)
/// * 'q3': Third quartile (75th percentile)
/// * 'iqr': Interquartile range (q3 - q1)
///
/// # Examples
///
/// ```python
/// from voidkit_rust import descriptive_stats
///
/// data = [1, 2, 3, 4, 5]
/// stats = descriptive_stats(data, ddof=1)
/// print(f"Mean: {stats['mean']}, Std: {stats['std']}")
/// ```
#[pyfunction]
#[pyo3(signature = (data, nan_policy="propagate", ddof=0))]
pub fn descriptive_stats<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    nan_policy: &str,
    ddof: usize,
) -> PyResult<Bound<'py, PyDict>> {
    // Validate nan_policy
    let valid_policies = ["propagate", "omit", "raise"];
    if !valid_policies.contains(&nan_policy) {
        return Err(PyValueError::new_err(format!(
            "Invalid nan_policy '{}'. Must be one of: propagate, omit, raise",
            nan_policy
        )));
    }

    let data_array = data.as_array();
    if data_array.is_empty() {
        return Err(PyValueError::new_err("Input data is empty"));
    }

    // Handle NaN values based on policy
    let mut clean_data: Vec<f64> = Vec::new();
    let mut has_nan = false;

    for &val in data_array.iter() {
        if val.is_nan() {
            has_nan = true;
            if nan_policy == "raise" {
                return Err(PyValueError::new_err("NaN values found in data"));
            } else if nan_policy == "omit" {
                continue;
            }
        }
        clean_data.push(val);
    }

    if clean_data.is_empty() && nan_policy == "omit" {
        return Err(PyValueError::new_err(
            "All values are NaN and nan_policy='omit'",
        ));
    }

    // If propagate and has NaN, return all NaN stats
    if has_nan && nan_policy == "propagate" {
        let result = PyDict::new_bound(py);
        result.set_item("count", clean_data.len())?;
        result.set_item("mean", f64::NAN)?;
        result.set_item("median", f64::NAN)?;
        result.set_item("std", f64::NAN)?;
        result.set_item("var", f64::NAN)?;
        result.set_item("min", f64::NAN)?;
        result.set_item("max", f64::NAN)?;
        result.set_item("q1", f64::NAN)?;
        result.set_item("q3", f64::NAN)?;
        result.set_item("iqr", f64::NAN)?;
        return Ok(result);
    }

    // Calculate statistics
    let count = clean_data.len();
    let mean = clean_data.iter().sum::<f64>() / count as f64;

    // Variance and standard deviation
    let variance = if count > ddof {
        let sum_sq_diff: f64 = clean_data.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq_diff / (count - ddof) as f64
    } else {
        f64::NAN
    };
    let std_dev = variance.sqrt();

    // Sort data for median and quartiles
    let mut sorted_data = clean_data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = percentile(&sorted_data, 50.0);
    let q1 = percentile(&sorted_data, 25.0);
    let q3 = percentile(&sorted_data, 75.0);
    let iqr = q3 - q1;

    let min = sorted_data.first().copied().unwrap();
    let max = sorted_data.last().copied().unwrap();

    // Create result dictionary
    let result = PyDict::new_bound(py);
    result.set_item("count", count)?;
    result.set_item("mean", mean)?;
    result.set_item("median", median)?;
    result.set_item("std", std_dev)?;
    result.set_item("var", variance)?;
    result.set_item("min", min)?;
    result.set_item("max", max)?;
    result.set_item("q1", q1)?;
    result.set_item("q3", q3)?;
    result.set_item("iqr", iqr)?;

    Ok(result)
}

/// Calculate percentile using linear interpolation
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return f64::NAN;
    }

    if sorted_data.len() == 1 {
        return sorted_data[0];
    }

    let rank = (p / 100.0) * (sorted_data.len() - 1) as f64;
    let lower_idx = rank.floor() as usize;
    let upper_idx = rank.ceil() as usize;

    if lower_idx == upper_idx {
        sorted_data[lower_idx]
    } else {
        let lower_val = sorted_data[lower_idx];
        let upper_val = sorted_data[upper_idx];
        let fraction = rank - lower_idx as f64;
        lower_val + fraction * (upper_val - lower_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_relative_eq!(percentile(&data, 0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(percentile(&data, 50.0), 3.0, epsilon = 1e-10);
        assert_relative_eq!(percentile(&data, 100.0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_and_std() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;

        assert_relative_eq!(mean, 2.5, epsilon = 1e-10);

        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = variance.sqrt();

        assert_relative_eq!(variance, 5.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(std, (5.0 / 3.0_f64).sqrt(), epsilon = 1e-10);
    }
}
